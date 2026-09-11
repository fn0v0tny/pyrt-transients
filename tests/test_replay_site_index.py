"""tools/replay_site_index.py: the latest replay goes into one marked section
of the hand-written replay-validation front page, and nothing else changes."""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "tools"))
import replay_site_index as rsi  # noqa: E402

FRONT = """<!doctype html><html><head><style>.row{}</style></head><body>
<div class="page">
  <header class="masthead"><h1>Eighteen GRB fields</h1></header>
  <div class="statrow">July numbers</div>
  <section class="group"><h2>Afterglow recovered</h2></section>
</div></body></html>
"""


def _result(name, recovered, k=4, n=20, q=39.3, sep=1.02, other=1, error=None):
    if error:
        return {"name": name, "error": error}
    return {"name": name, "ra": 336.75081, "dec": 12.46227, "n_epochs": n,
            "candidates_final": other + recovered, "spurious_final": other,
            "target": {"recovered": recovered, "k_first": k if recovered else None,
                       "quality_final": q if recovered else None,
                       "sep_final_arcsec": sep if recovered else None}}


def _site(tmp_path, vetting=True):
    site = tmp_path / "grb_replay_validation"
    site.mkdir()
    (site / "index.html").write_text(FRONT)
    (site / "obs_GRB250813B").mkdir()
    (site / "obs_GRB250813B" / "index.html").write_text("page")
    (site / "replay_2026-09-04b").mkdir()
    replay = site / "replay_2026-09-11b"
    runs = {"historical": [_result("GRB250813B", True), _result("GRB240414A", False),
                           _result("GRB180325A<x>", False, error="RuntimeError()")],
            "vetting": [_result("GRB250813B", True, k=5, other=0), _result("GRB240414A", False, other=1)]}
    for cfg, results in runs.items():
        if cfg == "vetting" and not vetting:
            continue
        (replay / cfg).mkdir(parents=True)
        (replay / cfg / "summary.json").write_text(json.dumps(results))
    return site, replay


def test_section_goes_after_the_header_and_keeps_the_rest(tmp_path):
    site, replay = _site(tmp_path)

    rsi.update_front_page(site, replay, code="503d11e")
    page = (site / "index.html").read_text()

    assert page.count(rsi.START) == 1
    assert page.index("</header>") < page.index(rsi.START) < page.index("July numbers")
    assert "Afterglow recovered" in page and "Eighteen GRB fields" in page
    assert "503d11e" in page
    assert '<a class="grbname" href="obs_GRB250813B/index.html">GRB250813B</a>' in page
    assert '<span class="grbname">GRB240414A</span>' in page          # no page for it
    assert "GRB180325A&lt;x&gt;" in page and "error" in page
    assert "image 4 / 20 &middot; Q 39.3 &middot; 1.0&Prime; &middot; 1 other" in page
    assert "image 5 / 20" in page
    assert 'href="replay_2026-09-11b/historical/index.html"' in page
    assert 'href="replay_2026-09-04b/"' in page and 'href="replay_2026-09-11b/"' not in page


def test_a_second_run_replaces_the_section_and_keeps_one_backup(tmp_path):
    site, replay = _site(tmp_path)
    rsi.update_front_page(site, replay, code="code-AAA")
    rsi.update_front_page(site, replay, code="code-BBB")
    page = (site / "index.html").read_text()

    assert page.count(rsi.START) == 1 and page.count(rsi.END) == 1
    assert "code-BBB" in page and "code-AAA" not in page
    backups = list(site.glob("index.html.bak-*"))
    assert len(backups) == 1 and backups[0].read_text() == FRONT


def test_a_configuration_missing_from_the_run_is_left_out(tmp_path):
    site, replay = _site(tmp_path, vetting=False)

    runs = rsi.update_front_page(site, replay)

    assert list(runs) == ["historical"]
    page = (site / "index.html").read_text()
    assert "vetting" not in page and "1<span" in page    # 1 of 3 recovered


def test_counts_in_the_stat_row(tmp_path):
    site, replay = _site(tmp_path)
    runs = rsi.load_runs(replay)
    section = rsi.render_section(site, replay, runs, "x")

    # historical: 1 + 1 other, the errored burst counts none; vetting: 0 + 1
    assert "afterglows recovered, historical configuration &middot; 2 other candidates in total" in section
    assert "afterglows recovered, vetting configuration &middot; 1 other candidates in total" in section
