#!/usr/bin/env python3
"""Status page for a pyrt-transient host: the latest observations, their
processing status, and the latest GRB observation.

  status_page.py [--data-dir DIR] [--public-dir DIR] [--out-dir DIR]
                 [--daemon-log FILE] [--rows N] [--title TEXT]
                 [--min-interval SECONDS]

Defaults: ~/transient_work, ~/public_html, <public-dir>/observations,
~/logs/transient_daemon.log and 60 s, overridable with PYRT_STATUS_DATA_DIR,
PYRT_STATUS_PUBLIC_DIR, PYRT_STATUS_OUT_DIR, PYRT_STATUS_DAEMON_LOG,
PYRT_STATUS_TITLE and PYRT_STATUS_MIN_INTERVAL.

It writes <out-dir>/transient_status.json, and <out-dir>/index.html when
that page changes. The page is a fixed shell: its JavaScript loads the JSON
when opened and every 30 seconds after that, and redraws in place. The data
lives outside the web root (the observations, the daemon log, the RTS2
database), so the JSON still has to be written on the server.
tools/pipeline_entry.py asks for that at the start and end of every frame.

Requests are merged: at most one refresh per --min-interval. A request
during that interval is served by one delayed run, so the latest state
always reaches the page within about a minute. The per-observation sites it
links are <public-dir>/obs_<id>/.

Targets: names and types come from the RTS2 database on every run (see
pipeline_entry.lookup_targets), because they change. The only thing cached
(<data-dir>/.status_cache.json) is what an observation's own ECSV header
says, keyed by that file's name.

Status of an observation:
- "running" while a frame of it is being processed (the markers
  pipeline_entry.py writes);
- otherwise the daemon's last result for it: done, failed or timeout;
- "-" if that result is no longer in the part of the log that is read.
"""
import argparse
import fcntl
import html
import json
import math
import os
import re
import socket
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import pipeline_entry  # noqa: E402  (one header reader, target lookup and GRB test for both)

LOG_TAIL_BYTES = 4 << 20
GRB_SEARCH_CHUNK = 100
STATUS_JSON = "transient_status.json"
_TS = r"(\d{4}-\d\d-\d\d \d\d:\d\d:\d\d),\d+"
STATUS_RE = re.compile(_TS + r" - INFO: Status: (.*)$")
JOB_RE = re.compile(_TS + r" - (?:INFO|ERROR): Batch (\S+?)(?: \[\d+/\d+\])?: job \S+ "
                    r"(succeeded|failed \(exit -?\d+\)|timed out|failed: .*)$")
CANDIDATE_COLUMNS = ("ALPHA_J2000", "DELTA_J2000", "MAG_CALIB", "quality_score",
                     "n_detections", "candidate_type",
                     # The follow-up exposure recommendation the pipeline
                     # writes (followup/enrichment.py): how long to integrate
                     # for its target SNR, and the magnitude it planned for.
                     # Absent in observations processed before that existed.
                     "followup_exptime_s", "followup_mag")


def _quality(value):
    """quality_score as a finite float: JSON with NaN would not parse in the browser."""
    try:
        q = float(value)
    except (TypeError, ValueError):
        return 0.0
    return q if math.isfinite(q) else 0.0


def read_ipac(path, columns=CANDIDATE_COLUMNS):
    """Rows of an astropy ascii.ipac table as {column: str}, for `columns`."""
    try:
        lines = Path(path).read_text(errors="replace").splitlines()
    except OSError:
        return []
    header = next((line for line in lines if line.startswith("|")), None)
    if header is None:
        return []
    bars = [i for i, c in enumerate(header) if c == "|"]
    names = [header[a + 1:b].strip() for a, b in zip(bars, bars[1:])]
    wanted = {name: k for k, name in enumerate(names) if name in columns}
    rows = []
    for line in lines:
        if not line.strip() or line[0] in "|\\":
            continue
        rows.append({name: line[bars[k] + 1:bars[k + 1] + 1].strip()
                     for name, k in wanted.items()})
    return rows


def read_daemon_log(path):
    """Last status line, last job result per observation, recent failures."""
    info = {"status": None, "last_job": {}, "failures": []}
    try:
        with open(path, "rb") as fh:
            fh.seek(0, os.SEEK_END)
            fh.seek(max(0, fh.tell() - LOG_TAIL_BYTES))
            text = fh.read().decode("utf-8", "replace")
    except OSError:
        return info
    for line in text.splitlines()[1:]:  # the first one may be cut
        m = STATUS_RE.match(line)
        if m:
            info["status"] = {"time": m.group(1), "text": m.group(2)}
            continue
        m = JOB_RE.match(line)
        if m:
            when, obs, raw = m.groups()
            outcome = "done" if raw == "succeeded" else "timeout" if raw == "timed out" else "failed"
            info["last_job"][obs] = {"time": when, "outcome": outcome}
            if outcome != "done":
                info["failures"].append({"time": when, "obs_id": obs, "outcome": raw})
    info["failures"] = info["failures"][-10:][::-1]
    return info


def running_frames(data_dir):
    frames = []
    for marker in (data_dir / ".running").glob("*.json"):
        try:
            entry = json.loads(marker.read_text())
            os.kill(int(entry["pid"]), 0)
        except (OSError, ValueError, KeyError):
            # Finished, or left behind by a frame the daemon killed at its
            # timeout (SIGKILL leaves the marker). Remove it: without that,
            # one killed frame sits in .running/ for ever and shows as
            # "running" on every page until something else refreshes.
            try:
                marker.unlink()
            except OSError:
                pass
            continue
        frames.append(entry)
    return sorted(frames, key=lambda e: e.get("started", 0))


def _first_ecsv(obs_dir):
    try:
        with os.scandir(obs_dir) as it:
            for entry in it:
                if entry.name.endswith(".ecsv") and not entry.name.endswith("_transients.ecsv"):
                    return entry.name
    except OSError:
        pass
    return None


def _frame_time(name):
    """'20260909011221-634-i-020-df.ecsv' -> '2026-09-09 01:12:21'."""
    s = name[:14]
    return f"{s[:4]}-{s[4:6]}-{s[6:8]} {s[8:10]}:{s[10:12]}:{s[12:14]}" if s.isdigit() else ""


def _mtime(path):
    """st_mtime, or 0 for a path that is gone (cleaned up meanwhile)."""
    try:
        return os.stat(path).st_mtime
    except OSError:
        return 0


def header_facts(obs_dir, cache):
    """What the observation's own ECSV header says, cached by that file.

    A file's header does not change. If the file is gone (the directory
    was cleaned up or reused), the header is read again.
    """
    cached = cache.get(obs_dir.name)
    if cached and (obs_dir / cached["file"]).exists():
        return cached
    ecsv = _first_ecsv(obs_dir)
    if not ecsv:
        return {"file": "", "target": None, "object": "", "grb_ra": None}
    meta = pipeline_entry.read_header(obs_dir / ecsv)
    facts = {"file": ecsv, "target": pipeline_entry.target_id(meta), "object": meta.get("OBJECT", ""),
             "grb_ra": meta.get("GRB_RA"), "grb_dec": meta.get("GRB_DEC"), "grb_err": meta.get("GRB_ERR")}
    cache[obs_dir.name] = facts
    return facts


def classify(facts, targets):
    """(current target name, GRB-type?) from the database, else the header."""
    target = targets.get(facts["target"]) if facts["target"] is not None else None
    meta = {"OBJECT": facts["object"], **({"GRB_RA": facts["grb_ra"]} if facts.get("grb_ra") else {})}
    return (target["name"] if target else facts["object"]), pipeline_entry.is_grb(meta, target)


def describe(obs_dir, facts, targets, with_candidates=0):
    name, grb = classify(facts, targets)
    try:
        processed = json.loads((obs_dir / "detection_metadata.json").read_text()).get("processed_files", [])
    except (OSError, ValueError):
        processed = []
    frames = sorted(processed)
    rows = read_ipac(obs_dir / "candidates.tbl")
    for row in rows:
        row["q"] = _quality(row.get("quality_score"))
    summary = {"obs_id": obs_dir.name[4:], "object": name, "grb": grb, "target": facts["target"],
               "grb_ra": facts.get("grb_ra"), "grb_dec": facts.get("grb_dec"), "grb_err": facts.get("grb_err"),
               "frames": len(frames), "first_frame": _frame_time(frames[0]) if frames else "",
               "last_frame": _frame_time(frames[-1]) if frames else "",
               "updated": _mtime(obs_dir),
               "candidates": len(rows), "reliable": sum(r["q"] >= 1 for r in rows)}
    if with_candidates:
        summary["top"] = sorted(rows, key=lambda r: -r["q"])[:with_candidates]
    return summary


def collect(data_dir, public_dir, daemon_log, rows):
    cache_path = data_dir / ".status_cache.json"
    try:
        cache = json.loads(cache_path.read_text())
    except (OSError, ValueError):
        cache = {}
    try:
        with os.scandir(data_dir) as it:
            obs_dirs = [(_mtime(e.path), Path(e.path)) for e in it
                        if e.name.startswith("obs_") and e.is_dir()]
    except OSError:
        obs_dirs = []
    obs_dirs = [p for m, p in sorted(obs_dirs, reverse=True) if m]

    targets = {}

    def facts_for(dirs):
        facts = [header_facts(d, cache) for d in dirs]
        missing = {f["target"] for f in facts if f["target"] is not None} - targets.keys()
        targets.update(pipeline_entry.lookup_targets(missing))
        return facts

    shown = obs_dirs[:rows]
    latest = [describe(d, f, targets) for d, f in zip(shown, facts_for(shown)) if d.is_dir()]
    latest_grb = None
    for start in range(0, len(obs_dirs), GRB_SEARCH_CHUNK):  # newest first
        chunk = obs_dirs[start:start + GRB_SEARCH_CHUNK]
        for d, f in zip(chunk, facts_for(chunk)):
            if classify(f, targets)[1] and d.is_dir():
                latest_grb = describe(d, f, targets, with_candidates=5)
                break
        if latest_grb:
            break

    log = read_daemon_log(daemon_log)
    running = running_frames(data_dir)
    running_obs = {r.get("obs_id") for r in running}
    for obs in latest + ([latest_grb] if latest_grb else []):
        job = log["last_job"].get(obs["obs_id"])
        obs["status"] = ("running" if obs["obs_id"] in running_obs else
                         job["outcome"] if job else "-")
        obs["last_job"] = job["time"] if job else ""
        obs["site"] = (f"../obs_{obs['obs_id']}/index.html"
                       if (public_dir / f"obs_{obs['obs_id']}" / "index.html").exists() else "")
    try:
        tmp = cache_path.with_name(".status_cache.tmp")
        tmp.write_text(json.dumps(cache))
        os.replace(tmp, cache_path)
    except OSError:
        pass
    return {"generated": time.time(), "daemon": log["status"], "targets_db": bool(targets),
            "running": running, "latest_grb": latest_grb, "observations": latest,
            "failures": log["failures"]}


# The page. Everything shown is built by renderStatus() from the JSON; every
# value is escaped. Times are UTC; ages count from the viewer's clock, and the
# daemon's staleness from the moment the data was collected, so a quiet
# afternoon without frames does not look like a dead daemon.
SHELL = r"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>__TITLE__</title>
<style>
:root{--bg:#f7f7f5;--fg:#1d1d1f;--muted:#6b6b70;--card:#fff;--line:#e3e3e0;--grb:#b3261e;
--grb-bg:#fdecea;--ok:#1b7f3b;--run:#1f5fbf;--bad:#b3261e}
@media (prefers-color-scheme:dark){:root{--bg:#141416;--fg:#ececef;--muted:#9a9aa2;--card:#1d1d21;
--line:#2e2e34;--grb:#ff8a80;--grb-bg:#3a1d1b;--ok:#6fd08c;--run:#8ab4ff;--bad:#ff8a80}}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--fg);
font:14px/1.45 system-ui,-apple-system,"Segoe UI",sans-serif}
main{max-width:1100px;margin:0 auto;padding:24px 16px 48px}
h1{font-size:22px;margin:0 0 4px}h2{font-size:16px;margin:28px 0 10px}
.meta,.muted{color:var(--muted)}.card{background:var(--card);border:1px solid var(--line);
border-radius:10px;padding:14px 16px}.grb{border-left:4px solid var(--grb)}
.scroll{overflow-x:auto}table{border-collapse:collapse;width:100%;background:var(--card)}
th,td{padding:6px 10px;border-bottom:1px solid var(--line);text-align:left;white-space:nowrap}
th{font-weight:600;color:var(--muted);font-size:12px;text-transform:uppercase;letter-spacing:.03em}
td.obj{white-space:normal;min-width:180px}.num{text-align:right;font-variant-numeric:tabular-nums}
.tag{display:inline-block;padding:1px 7px;border-radius:9px;font-size:12px;font-weight:600}
.t-grb{background:var(--grb-bg);color:var(--grb)}.s-done{color:var(--ok)}.s-running{color:var(--run)}
.s-failed,.s-timeout,.warn{color:var(--bad)}a{color:var(--run)}ul{padding-left:18px;margin:6px 0}
</style></head>
<body><main>
<h1 id="title">__TITLE__</h1>
<p id="error" class="warn"></p>
<div id="app"><p class="muted">Loading&hellip;</p></div>
<noscript><p>This page needs JavaScript. The data is in <a href="transient_status.json">transient_status.json</a>.</p></noscript>
<p class="muted">Archive validation: <a href="../grb_replay_validation/">GRB replay results</a>
&middot; <a href="transient_status.json">status as JSON</a></p>
</main>
<script>
"use strict";
function esc(v) {
  return String(v === null || v === undefined ? "" : v).replace(/[&<>"']/g,
    c => ({"&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;"}[c]));
}
function ago(sec) {
  sec = Math.max(0, Math.floor(sec));
  for (const [unit, size] of [["d", 86400], ["h", 3600], ["min", 60]])
    if (sec >= size) return Math.floor(sec / size) + " " + unit + " ago";
  return "just now";
}
function utc(ts) { return new Date(ts * 1000).toISOString().slice(0, 16).replace("T", " "); }
function logTime(t) { return Date.parse(t.replace(" ", "T") + "Z") / 1000; }
function status(s) { return '<span class="s-' + esc(s) + '">' + esc(s) + "</span>"; }
function link(href, text) { return href ? "<a href='" + esc(href) + "'>" + esc(text) + "</a>" : esc(text); }

function renderStatus(s, now) {
  const out = [];
  let daemon = '<span class="warn">no daemon status found in the log</span>';
  if (s.daemon) {
    const age = s.generated - logTime(s.daemon.time);
    daemon = "daemon: " + esc(s.daemon.text) + (age > 300 ?
      ' <span class="warn">(no daemon status for ' + ago(age).replace(" ago", "") +
      " before this update: daemon down?)</span>" : "");
  }
  const db = s.targets_db ? "" :
    ' &middot; <span class="warn">target database unavailable, names from file headers</span>';
  out.push("<p class='meta'>Data from " + utc(s.generated) + " UTC (" + ago(now - s.generated) +
           ") &middot; " + daemon + db + "</p>");

  out.push("<h2>Latest GRB observation</h2>");
  const g = s.latest_grb;
  if (g) {
    const pos = g.grb_ra ? " &middot; trigger position " + esc(g.grb_ra) + ", " + esc(g.grb_dec) +
      " (&plusmn;" + esc(g.grb_err) + "&deg;)" : "";
    const site = g.site ? " &middot; " + link(g.site, "candidate page") : "";
    out.push("<div class='card grb'><strong>" + (esc(g.object) || "obs " + esc(g.obs_id)) + "</strong>" +
      "<div class='muted'>obs " + esc(g.obs_id) + " &middot; target " + esc(g.target) + " &middot; " +
      esc(g.frames) + " frames, " + esc(g.first_frame) + " &ndash; " + esc(g.last_frame) + " UTC" + pos +
      " &middot; " + status(g.status) + site + "</div>" +
      "<p>" + esc(g.candidates) + " candidates, " + esc(g.reliable) + " with quality &ge; 1</p>");
    if (g.top && g.top.length) {
      // "follow-up": how long the pipeline says to integrate for its target
      // SNR, and the magnitude it planned for. Older observations have none.
      out.push("<div class='scroll'><table><tr><th>RA</th><th>Dec</th><th class='num'>mag</th>" +
               "<th class='num'>quality</th><th class='num'>detections</th><th>type</th>" +
               "<th class='num'>follow-up</th></tr>");
      for (const c of g.top) {
        const secs = Number(c.followup_exptime_s);
        const planned = Number(c.followup_mag);
        const followup = secs > 0
          ? esc(secs.toFixed(0)) + " s" + (planned > 0 ? "<span class='when'>for mag " + esc(planned.toFixed(2)) + "</span>" : "")
          : "<span class='dim'>&mdash;</span>";
        out.push("<tr><td>" + esc(c.ALPHA_J2000) + "</td><td>" + esc(c.DELTA_J2000) + "</td><td class='num'>" +
                 esc(c.MAG_CALIB) + "</td><td class='num'>" + Number(c.q).toFixed(2) + "</td><td class='num'>" +
                 esc(c.n_detections) + "</td><td>" + esc(c.candidate_type) + "</td><td class='num'>" +
                 followup + "</td></tr>");
      }
      out.push("</table></div>");
    }
    out.push("</div>");
  } else {
    out.push("<p class='muted'>No GRB observation found.</p>");
  }

  const stale = now - s.generated > 900;   // no frame has refreshed the data for a while
  if (stale) out.push("<p class='warn'>This data is " + ago(now - s.generated) +
    ". The page is refreshed while frames are being processed, so it stands still when the telescope is idle.</p>");

  out.push("<h2>Running now</h2>");
  if (stale) {
    out.push("<p class='muted'>Nothing has been processed since " + utc(s.generated) + " UTC.</p>");
  } else if (s.running.length) {
    out.push("<ul>");
    for (const r of s.running)
      out.push("<li>obs " + esc(r.obs_id) + " &middot; " + esc(r.object) + " &middot; " + esc(r.frame) +
               " &middot; started " + ago(now - r.started) +
               (r.grb ? " <span class='tag t-grb'>GRB priority</span>" : "") + "</li>");
    out.push("</ul>");
  } else {
    out.push("<p class='muted'>Nothing was being processed at " + utc(s.generated) + " UTC.</p>");
  }

  out.push("<h2>Latest observations</h2><div class='scroll'><table><tr><th>obs</th><th>target</th>" +
           "<th class='num'>frames</th><th>frames (UTC)</th><th class='num'>candidates</th>" +
           "<th class='num'>q&nbsp;&ge;&nbsp;1</th><th>status</th><th>updated</th></tr>");
  for (const o of s.observations) {
    const span = o.frames ? esc(o.first_frame.slice(5, 16)) + " &ndash; " + esc(o.last_frame.slice(11, 16)) : "";
    out.push("<tr><td>" + link(o.site, o.obs_id) + "</td><td class='obj'>" + esc(o.object) +
             (o.grb ? " <span class='tag t-grb'>GRB</span>" : "") + "</td><td class='num'>" + esc(o.frames) +
             "</td><td>" + span + "</td><td class='num'>" + esc(o.candidates) + "</td><td class='num'>" +
             esc(o.reliable) + "</td><td>" + status(o.status) + "</td><td class='muted'>" +
             ago(now - o.updated) + "</td></tr>");
  }
  out.push("</table></div>");

  if (s.failures.length) {
    out.push("<h2>Recent failures</h2><ul>");
    for (const f of s.failures)
      out.push("<li><span class='muted'>" + esc(f.time) + "</span> obs " + esc(f.obs_id) + ": " + esc(f.outcome) + "</li>");
    out.push("</ul>");
  }
  return out.join("\n");
}

if (typeof module !== "undefined") module.exports = {renderStatus};  // for the tests (node)

if (typeof document !== "undefined") {
  let last = null;
  const draw = () => { if (last) document.getElementById("app").innerHTML = renderStatus(last, Date.now() / 1000); };
  async function load() {
    try {
      const r = await fetch("transient_status.json?t=" + Date.now(), {cache: "no-store"});
      if (!r.ok) throw new Error("HTTP " + r.status);
      last = await r.json();
      document.getElementById("error").textContent = "";
      if (last.title) { document.title = last.title; document.getElementById("title").textContent = last.title; }
    } catch (e) {
      document.getElementById("error").textContent =
        "Could not load transient_status.json (" + e.message + ")" + (last ? "; showing the last data loaded." : ".");
    }
    draw();
  }
  load();
  setInterval(load, 30000);
}
</script>
</body></html>
"""


def _write(path, text):
    tmp = path.with_name(f".{path.name}.tmp")
    tmp.write_text(text)
    os.replace(tmp, path)


def generate(data_dir, public_dir, daemon_log, rows=40, title=None, out_dir=None):
    data_dir, public_dir = Path(data_dir), Path(public_dir)
    out_dir = Path(out_dir) if out_dir else public_dir / "observations"
    title = title or f"Transient pipeline on {socket.gethostname()}"
    status = collect(data_dir, public_dir, Path(daemon_log), rows)
    status["title"] = title
    out_dir.mkdir(parents=True, exist_ok=True)
    _write(out_dir / STATUS_JSON, json.dumps(status, indent=1, default=str, allow_nan=False))
    page = SHELL.replace("__TITLE__", html.escape(title))
    index = out_dir / "index.html"
    try:
        current = index.read_text()
    except OSError:
        current = None
    if current != page:
        _write(index, page)
    return status


def main(argv=None):
    home = Path.home()
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--data-dir", default=os.environ.get("PYRT_STATUS_DATA_DIR", home / "transient_work"))
    ap.add_argument("--public-dir", default=os.environ.get("PYRT_STATUS_PUBLIC_DIR", home / "public_html"))
    ap.add_argument("--out-dir", default=os.environ.get("PYRT_STATUS_OUT_DIR"))
    ap.add_argument("--daemon-log", default=os.environ.get("PYRT_STATUS_DAEMON_LOG",
                                                           home / "logs" / "transient_daemon.log"))
    ap.add_argument("--rows", type=int, default=40)
    ap.add_argument("--title", default=os.environ.get("PYRT_STATUS_TITLE"))
    ap.add_argument("--min-interval", type=float,
                    default=float(os.environ.get("PYRT_STATUS_MIN_INTERVAL", "60")))
    args = ap.parse_args(argv)

    data_dir = Path(args.data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)
    out_json = Path(args.out_dir or Path(args.public_dir) / "observations") / STATUS_JSON
    lock_path, again = data_dir / ".status_page.lock", data_dir / ".status_page.again"
    error = data_dir / ".status_page.error"
    with open(lock_path, "w") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            again.touch()  # the run holding the lock will go once more
            return
        while True:
            wait = _mtime(out_json) + args.min_interval - time.time()
            if wait > 0:
                time.sleep(wait)  # requests meanwhile only leave the flag
            again.unlink(missing_ok=True)
            try:
                generate(data_dir, args.public_dir, args.daemon_log, args.rows, args.title, args.out_dir)
                error.unlink(missing_ok=True)
            except Exception:  # runs detached from the pipeline: keep the reason somewhere
                import traceback
                error.write_text(f"{datetime.now(timezone.utc):%Y-%m-%d %H:%M:%S} UTC\n"
                                 + traceback.format_exc())
            if not again.exists():
                break


if __name__ == "__main__":
    main()
