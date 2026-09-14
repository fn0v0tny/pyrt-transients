#!/usr/bin/env python3
"""Write the replay-validation front page from the latest archive replay.

  replay_site_index.py <site dir> --replay <replay dir> [--code TEXT]
                       [--describe NAME=TEXT ...] [--archive-name FILE]

<site dir> holds the front page (index.html), the per-burst candidate pages
obs_<burst>/ and the replay runs replay_*/. For example, on lascaux50,
~/public_html/grb_replay_validation.

<replay dir> holds one sub-directory per configuration (historical/,
vetting/, ...), each with the summary.json that tools/replay_archive.py
writes.

The page shows only the current replay: every burst under each
configuration, and links to the per-burst pages, the full run tables and
the earlier runs. A front page written by hand (anything without this
tool's generator tag) is kept once as --archive-name
(july_2026_study.html) and linked from the footer. Nothing is deleted.
"""
import argparse
import html
import json
import shutil
import statistics
import sys
import time
from pathlib import Path

GENERATOR = '<meta name="generator" content="pyrt-transient replay_site_index">'
DESCRIPTIONS = {
    "historical": "the production detection settings: reference catalogues ATLAS, Gaia and USNO-B.",
    "vetting": ("stricter checks against constant sources: Gaia's full catalogue, ATLAS, Pan-STARRS and "
                "USNO-B; a catalogue star within 3&Prime; counts as a match; a star listed without "
                "photometry does not make a source new; and a new source must exceed a variability floor."),
}


def _e(value):
    return html.escape(str(value if value is not None else ""))


def load_runs(replay_dir):
    """{configuration: {burst: result}} for every sub-directory with a summary.json."""
    runs = {}
    for summary in sorted(Path(replay_dir).glob("*/summary.json")):
        try:
            results = json.loads(summary.read_text())
        except (OSError, ValueError):
            continue
        runs[summary.parent.name] = {r["name"]: r for r in results if "name" in r}
    return runs


def _seconds(s):
    s = int(round(s))
    return f"{s} s" if s < 120 else f"{s // 60} min {s % 60:02d} s" if s < 3600 else f"{s / 3600:.1f} h"


def detection_time(result):
    """'83 s after the trigger' when the trigger time is known, else after the first image."""
    t = result.get("target", {}).get("t_first_s")
    if t is None:
        return ""
    if result.get("t0_source") == "list":
        return f"{_seconds(t)} after the trigger"
    return f"{_seconds(max(0.0, t - (result.get('first_frame_after_t0_s') or 0.0)))} after the first image"


def _cell(result):
    if result is None:
        return '<span class="dim">not run</span>'
    if result.get("error"):
        return '<span class="dim">replay error</span>'
    t = result.get("target", {})
    other = result.get("spurious_final")
    other_txt = f'<span class="other">{other} other candidate{"" if other == 1 else "s"}</span>' if other is not None else ""
    if not t.get("recovered"):
        return f'<span class="dim">not recovered</span>{other_txt}'
    q, sep = t.get("quality_final"), t.get("sep_final_arcsec")
    facts = [f'image {_e(t.get("k_first"))} of {_e(result.get("n_epochs"))}']
    if q is not None:
        facts.append(f"Q {q:.1f}")
    if sep is not None:
        facts.append(f"{sep:.1f}&Prime;")
    when = detection_time(result)
    return (f'<span class="hitline">{" &middot; ".join(facts)}</span>'
            f'{f"<span class=when>{when}</span>" if when else ""}{other_txt}')


CSS = """
:root{--bg:#0b0f1a;--surface:#131b2e;--surface-2:#1a2540;--text:#e6e9f2;--muted:#8790a8;--faint:#566078;
--accent:#e2a542;--accent-soft:#3a2f1a;--accent-line:#7a6335;--rule:#212c48;--rule-soft:#1a2338;
--serif:Georgia,"Iowan Old Style","Palatino Linotype",serif;
--sans:-apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,Helvetica,Arial,sans-serif;
--mono:ui-monospace,"SF Mono","Cascadia Code",Menlo,Consolas,monospace}
@media (prefers-color-scheme:light){:root:not([data-theme="dark"]){--bg:#f5f2ea;--surface:#fff;--surface-2:#edeadf;
--text:#201d17;--muted:#635c4d;--faint:#948d7a;--accent:#a6690f;--accent-soft:#f3e4c8;--accent-line:#d3ac66;
--rule:#ddd5c0;--rule-soft:#e8e2d1}}
:root[data-theme="light"]{--bg:#f5f2ea;--surface:#fff;--surface-2:#edeadf;--text:#201d17;--muted:#635c4d;
--faint:#948d7a;--accent:#a6690f;--accent-soft:#f3e4c8;--accent-line:#d3ac66;--rule:#ddd5c0;--rule-soft:#e8e2d1}
*{box-sizing:border-box}
body{margin:0;background:var(--bg);color:var(--text);font:16px/1.5 var(--sans);-webkit-font-smoothing:antialiased}
.page{max-width:1000px;margin:0 auto;padding:4rem 1.25rem 5rem}
.eyebrow{font:0.72rem var(--mono);letter-spacing:.14em;text-transform:uppercase;color:var(--faint);margin-bottom:.9rem}
h1{font:400 2.3rem/1.15 var(--serif);margin:0 0 .9rem;text-wrap:balance}
h2{font:400 1.25rem var(--serif);margin:3rem 0 .4rem}
.lede{color:var(--muted);max-width:68ch;margin:0}.lede strong{color:var(--text);font-weight:600}
.statrow{display:grid;grid-template-columns:repeat(auto-fit,minmax(190px,1fr));gap:1px;background:var(--rule);
border:1px solid var(--rule);border-radius:10px;overflow:hidden;margin:2.5rem 0 1rem}
.stat{background:var(--surface);padding:1.3rem 1.2rem}
.stat .n{font:600 1.9rem/1 var(--mono);font-variant-numeric:tabular-nums;display:block;margin-bottom:.5rem}
.stat .n.accent{color:var(--accent)}.stat .n small{color:var(--faint);font-weight:400;font-size:.6em}
.stat .label{font-size:.8rem;color:var(--muted);line-height:1.35}
.configs{display:grid;grid-template-columns:repeat(auto-fit,minmax(280px,1fr));gap:1rem;margin:1.5rem 0 0}
.config{border-left:2px solid var(--accent-line);padding:.2rem 0 .2rem 1rem;font-size:.88rem;color:var(--muted)}
.config b{font-family:var(--mono);color:var(--text);font-weight:600}
.sub{font-size:.88rem;color:var(--muted);margin:0 0 1.1rem;max-width:72ch}
.scroll{overflow-x:auto}
table{width:100%;border-collapse:collapse;background:var(--surface);border:1px solid var(--rule);border-radius:10px;overflow:hidden}
th{font:0.66rem var(--mono);letter-spacing:.06em;text-transform:uppercase;color:var(--faint);text-align:left;
background:var(--surface-2);padding:.7rem 1rem;white-space:nowrap}
td{padding:.8rem 1rem;border-top:1px solid var(--rule-soft);vertical-align:top;font-size:.86rem}
.grbname{font:600 .92rem var(--mono);color:var(--text)}
a.grbname{text-decoration:none;border-bottom:1px solid var(--accent-line)}a.grbname:hover{color:var(--accent);border-color:var(--accent)}
.coords,.when,.other{display:block;font:0.72rem var(--mono);color:var(--faint);margin-top:.15rem;font-variant-numeric:tabular-nums}
.hitline{font-family:var(--mono);font-size:.82rem;font-variant-numeric:tabular-nums}
.dim{color:var(--faint);font-family:var(--mono);font-size:.82rem}
.num{font-family:var(--mono);font-variant-numeric:tabular-nums;text-align:right}
.pill{display:inline-flex;align-items:center;gap:.4rem;font:0.7rem var(--mono);padding:.25rem .6rem;border-radius:100px;white-space:nowrap}
.pill::before{content:"";width:6px;height:6px;border-radius:50%;background:currentColor}
.pill.hit{background:var(--accent-soft);color:var(--accent);border:1px solid var(--accent-line)}
.pill.miss{color:var(--faint);border:1px solid var(--rule)}
a{color:var(--accent)}
footer{margin-top:3.5rem;padding-top:1.5rem;border-top:1px solid var(--rule);font-size:.85rem;color:var(--muted)}
footer p{margin:0 0 .7rem;max-width:75ch}
@media (max-width:700px){h1{font-size:1.8rem}.page{padding-top:2.5rem}}
"""


def render_page(site_dir, replay_dir, runs, code, describe, archive_name):
    site_dir, replay_dir = Path(site_dir), Path(replay_dir)
    configs = list(runs)
    bursts = []
    for results in runs.values():
        bursts += [n for n in results if n not in bursts]
    rel = replay_dir.relative_to(site_dir) if replay_dir.is_relative_to(site_dir) else replay_dir

    stats = []
    for c in configs:
        res = runs[c].values()
        rec = [r for r in res if r.get("target", {}).get("recovered")]
        stats.append(f'<div class="stat"><span class="n accent">{len(rec)}<small> / {len(runs[c])}</small></span>'
                     f'<span class="label">afterglows recovered &middot; {_e(c)}</span></div>')
        ks = [r["target"]["k_first"] for r in rec if r["target"].get("k_first")]
        if ks:
            stats.append(f'<div class="stat"><span class="n">{statistics.median(ks):g}</span>'
                         f'<span class="label">median image of first detection &middot; {_e(c)}</span></div>')
        stats.append(f'<div class="stat"><span class="n">{sum(r.get("spurious_final") or 0 for r in res)}</span>'
                     f'<span class="label">other candidates at the last image, all fields &middot; {_e(c)}</span></div>')

    config_notes = "".join(
        f'<div class="config"><b>{_e(c)}</b> &mdash; {describe.get(c, "")}</div>' for c in configs)
    head = "".join(f"<th>{_e(c)}</th>" for c in configs)
    rows = []
    for name in bursts:
        any_result = next(runs[c][name] for c in configs if name in runs[c])
        page = site_dir / f"obs_{name}" / "index.html"
        label = (f'<a class="grbname" href="obs_{_e(name)}/index.html">{_e(name)}</a>' if page.exists()
                 else f'<span class="grbname">{_e(name)}</span>')
        coords = (f'<span class="coords">{any_result["ra"]:.5f}&deg; {any_result["dec"]:+.5f}&deg;</span>'
                  if isinstance(any_result.get("ra"), (int, float)) else "")
        hit = any(runs[c].get(name, {}).get("target", {}).get("recovered") for c in configs)
        rows.append(f"<tr><td>{label}{coords}</td><td class=num>{_e(any_result.get('n_epochs', ''))}</td>"
                    + "".join(f"<td>{_cell(runs[c].get(name))}</td>" for c in configs)
                    + f'<td><span class="pill {"hit" if hit else "miss"}">{"recovered" if hit else "missed"}</span></td></tr>')

    run_links = " &middot; ".join(f'<a href="{_e(rel)}/{_e(c)}/index.html">{_e(c)}</a>' for c in configs)
    earlier = sorted((p.name for p in site_dir.glob("replay_*") if p.is_dir() and p.resolve() != replay_dir.resolve()),
                     reverse=True)
    earlier_links = ", ".join(f'<a href="{_e(n)}/">{_e(n)}</a>' for n in earlier)
    archive = (f'<p>The earlier hand-written study of eighteen fields (July 2026, older code) is kept as '
               f'<a href="{_e(archive_name)}">{_e(archive_name)}</a>.</p>' if (site_dir / archive_name).exists() else "")
    generated = time.strftime("%Y-%m-%d %H:%M UTC", time.gmtime())
    n_rec = {c: sum(bool(r.get("target", {}).get("recovered")) for r in runs[c].values()) for c in configs}
    summary = " and ".join(f"<strong>{n_rec[c]} of {len(runs[c])}</strong> ({_e(c)})" for c in configs)

    return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
{GENERATOR}
<title>GRB archive replay &middot; pyrt-transient</title>
<style>{CSS}</style></head>
<body><div class="page">
<header>
  <div class="eyebrow">pyrt-transient &middot; archive replay validation</div>
  <h1>{len(bursts)} GRB nights, replayed through the current pipeline</h1>
  <p class="lede">Each night's archived images are fed to the transient pipeline one at a time, as if they were
  arriving live, and the candidates are compared with the afterglow position from the GCN circulars (a match
  within 5&Prime;). The afterglow was recovered in {summary}. Code: {_e(code)}.</p>
</header>
<div class="statrow">{"".join(stats)}</div>
<div class="configs">{config_notes}</div>

<h2>Every field</h2>
<p class="sub">Field names open the candidate page built with the current code. For a recovered afterglow:
the image at which it first passed the quality gate, its final quality score Q and its distance from the GCN
position. The time counts from the GRB trigger where the trigger time is known, otherwise from the first
image. &ldquo;Other candidates&rdquo; are the rest of the candidate list at the last image.</p>
<div class="scroll"><table>
<tr><th>field</th><th class="num">images</th>{head}<th></th></tr>
{"".join(rows)}
</table></div>

<footer>
  <p>Full tables per configuration: {run_links}. Replayed with <code>tools/replay_archive.py</code>;
  this page by <code>tools/replay_site_index.py</code>, {generated}.</p>
  {f"<p>Earlier replay runs: {earlier_links}.</p>" if earlier else ""}
  {archive}
</footer>
</div></body></html>
"""


def write_front_page(site_dir, replay_dir, code="", describe=None, archive_name="july_2026_study.html"):
    site_dir = Path(site_dir)
    index = site_dir / "index.html"
    runs = load_runs(replay_dir)
    if not runs:
        raise SystemExit(f"no */summary.json under {replay_dir}")
    try:
        current = index.read_text()
    except OSError:
        current = None
    if current is not None and GENERATOR not in current and not (site_dir / archive_name).exists():
        shutil.copy2(index, site_dir / archive_name)  # a hand-written page: keep it
    page = render_page(site_dir, replay_dir, runs, code or Path(replay_dir).name,
                       {**DESCRIPTIONS, **(describe or {})}, archive_name)
    tmp = index.with_name(".index.html.tmp")
    tmp.write_text(page)
    tmp.replace(index)
    return runs


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("site_dir", type=Path)
    ap.add_argument("--replay", type=Path, required=True)
    ap.add_argument("--code", default="", help="version text shown on the page, e.g. a commit")
    ap.add_argument("--describe", action="append", default=[], metavar="NAME=TEXT",
                    help="description of a configuration (HTML allowed)")
    ap.add_argument("--archive-name", default="july_2026_study.html")
    args = ap.parse_args(argv)
    describe = dict(d.split("=", 1) for d in args.describe if "=" in d)
    runs = write_front_page(args.site_dir, args.replay, args.code, describe, args.archive_name)
    for cfg, results in runs.items():
        n = sum(bool(r.get("target", {}).get("recovered")) for r in results.values())
        print(f"{cfg}: {n}/{len(results)} recovered", file=sys.stderr)


if __name__ == "__main__":
    main()
