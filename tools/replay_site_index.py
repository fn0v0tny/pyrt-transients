#!/usr/bin/env python3
"""Put the latest archive replay on the replay-validation front page.

  replay_site_index.py <site dir> --replay <replay dir> [--code TEXT]

<site dir> is the directory holding the front page (index.html), the per-burst
candidate pages obs_<burst>/ and the replay runs replay_*/. For example, on
lascaux50, ~/public_html/grb_replay_validation.

<replay dir> holds one sub-directory per configuration (historical/,
vetting/, ...), each with the summary.json that tools/replay_archive.py
writes.

The front page is hand-written analysis, so it is not regenerated. This
writes one "current version" section between two markers:
- a stat row;
- a table of every burst under each configuration;
- links to the per-burst pages that exist, to the full run pages and to the
  earlier runs.

The first time, the section goes right after the page's </header>, and the
original page is kept as index.html.bak-<date>. Later runs replace the
section in place. The section uses the page's own classes (group, sub,
statrow, grbtable, row, pill, ...).
"""
import argparse
import html
import json
import shutil
import sys
import time
from pathlib import Path

START, END = "<!-- current-replay:start -->", "<!-- current-replay:end -->"


def _e(value):
    return html.escape(str(value if value is not None else ""))


def load_runs(replay_dir):
    """{config name: {burst: result}} for every sub-directory with a summary.json."""
    runs = {}
    for summary in sorted(Path(replay_dir).glob("*/summary.json")):
        try:
            results = json.loads(summary.read_text())
        except (OSError, ValueError):
            continue
        runs[summary.parent.name] = {r["name"]: r for r in results if "name" in r}
    return runs


def _cell(result):
    """One configuration's outcome for one burst, as a short HTML fragment."""
    if result is None:
        return '<span class="num dim">not run</span>'
    if result.get("error"):
        return '<span class="num dim">error</span>'
    t = result.get("target", {})
    other = result.get("spurious_final")
    other = f" &middot; {other} other" if other is not None else ""
    if not t.get("recovered"):
        return f'<span class="num dim">missed{other}</span>'
    q = t.get("quality_final")
    sep = t.get("sep_final_arcsec")
    return (f'<span class="num">image {_e(t.get("k_first"))} / {_e(result.get("n_epochs"))}'
            f'{" &middot; Q " + format(q, ".1f") if q is not None else ""}'
            f'{" &middot; " + format(sep, ".1f") + "&Prime;" if sep is not None else ""}{other}</span>')


def render_section(site_dir, replay_dir, runs, code):
    site_dir, replay_dir = Path(site_dir), Path(replay_dir)
    configs = list(runs)
    bursts = []
    for results in runs.values():
        for name in results:
            if name not in bursts:
                bursts.append(name)
    rel = replay_dir.relative_to(site_dir) if replay_dir.is_relative_to(site_dir) else replay_dir

    def recovered(cfg):
        return sum(bool(r.get("target", {}).get("recovered")) for r in runs[cfg].values())

    stats = "".join(
        f'<div class="stat"><span class="n accent">{recovered(c)}<span style="color:var(--text-faint);'
        f'font-weight:400;"> / {len(runs[c])}</span></span><span class="label">afterglows recovered, '
        f'{_e(c)} configuration &middot; {sum(r.get("spurious_final") or 0 for r in runs[c].values())} '
        f'other candidates in total</span></div>' for c in configs)
    head = "".join(f'<div class="num" style="text-align:left">{_e(c)}</div>' for c in configs)
    cols = " ".join(["1.2fr", "0.55fr"] + ["1.6fr"] * len(configs) + ["0.8fr"])
    rows = []
    for name in bursts:
        any_result = next(runs[c][name] for c in configs if name in runs[c])
        page = site_dir / f"obs_{name}" / "index.html"
        label = (f'<a class="grbname" href="obs_{_e(name)}/index.html">{_e(name)}</a>' if page.exists()
                 else f'<span class="grbname">{_e(name)}</span>')
        coords = (f'<span class="coords">{any_result["ra"]:.5f}&deg; {any_result["dec"]:+.5f}&deg;</span>'
                  if isinstance(any_result.get("ra"), (int, float)) else "")
        hit = any(runs[c].get(name, {}).get("target", {}).get("recovered") for c in configs)
        rows.append(f'<div class="row cur"><div>{label}{coords}</div>'
                    f'<div class="num">{_e(any_result.get("n_epochs", ""))}</div>'
                    + "".join(f"<div>{_cell(runs[c].get(name))}</div>" for c in configs)
                    + f'<div class="pill {"hit" if hit else "miss"}">{"recovered" if hit else "missed"}</div></div>')
    run_links = " &middot; ".join(f'<a href="{_e(rel)}/{_e(c)}/index.html">{_e(c)}</a>' for c in configs)
    earlier = sorted((p.name for p in site_dir.glob("replay_*") if p.is_dir() and p != replay_dir),
                     reverse=True)
    earlier_links = ", ".join(
        f'<a href="{_e(n)}/">{_e(n)}</a>' for n in earlier) or "none"
    generated = time.strftime("%Y-%m-%d %H:%M UTC", time.gmtime())
    return f"""{START}
<style>.row.cur{{grid-template-columns:{cols}}}</style>
<section class="group" id="current">
  <h2>Current version <span class="count">{_e(code)}</span></h2>
  <p class="sub">The same archive replayed through the current code, one image at a time, scored against the GCN
  afterglow positions (5&Prime; match). &ldquo;Image k / n&rdquo; is the image at which the afterglow first
  passed the quality gate; &ldquo;other&rdquo; counts the remaining candidates at the last image. Field names link
  to candidate pages built with the current code. Full numbers: {run_links}. Generated {generated}.</p>
  <div class="statrow" style="grid-template-columns: repeat({max(1, len(configs))}, 1fr); margin: 1.2rem 0 1.4rem;">{stats}</div>
  <div class="grbtable">
    <div class="row headrow cur"><div>field</div><div class="num">images</div>{head}<div></div></div>
    {"".join(rows)}
  </div>
  <p class="sub" style="margin-top:1rem">Earlier replay runs: {earlier_links}. Everything below this section is the
  July 2026 study (eighteen fields, older code), kept for its per-field analysis.</p>
</section>
{END}"""


def update_front_page(site_dir, replay_dir, code="", backup=True):
    site_dir = Path(site_dir)
    index = site_dir / "index.html"
    page = index.read_text()
    runs = load_runs(replay_dir)
    if not runs:
        raise SystemExit(f"no */summary.json under {replay_dir}")
    section = render_section(site_dir, replay_dir, runs, code or Path(replay_dir).name)
    if START in page and END in page:
        before, rest = page.split(START, 1)
        page = before + section + rest.split(END, 1)[1]
    else:
        if backup:
            stamp = time.strftime("%Y%m%d")
            bak = site_dir / f"index.html.bak-{stamp}"
            if not bak.exists():
                shutil.copy2(index, bak)
        at = page.find("</header>")
        if at < 0:
            at = page.find("<body>")
            at = at + len("<body>") if at >= 0 else 0
        else:
            at += len("</header>")
        page = page[:at] + "\n" + section + "\n" + page[at:]
    tmp = index.with_name(".index.html.tmp")
    tmp.write_text(page)
    tmp.replace(index)
    return runs


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("site_dir", type=Path)
    ap.add_argument("--replay", type=Path, required=True)
    ap.add_argument("--code", default="", help="version text shown in the heading, e.g. a commit")
    ap.add_argument("--no-backup", action="store_true")
    args = ap.parse_args(argv)
    runs = update_front_page(args.site_dir, args.replay, args.code, backup=not args.no_backup)
    for cfg, results in runs.items():
        n = sum(bool(r.get("target", {}).get("recovered")) for r in results.values())
        print(f"{cfg}: {n}/{len(results)} recovered", file=sys.stderr)


if __name__ == "__main__":
    main()
