"""tools/replay_site_index.py: the replay-validation front page shows the
current replay only; a hand-written page it replaces is archived once."""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "tools"))
import replay_site_index as rsi  # noqa: E402

JULY = "<!doctype html><html><body><h1>Eighteen GRB fields</h1><p>July analysis</p></body></html>"


def _result(name, recovered, k=4, n=20, q=39.3, sep=1.02, other=1, t_first=83.0, t0="first_frame_minus_30s",
            error=None):
    if error:
        return {"name": name, "error": error}
    return {"name": name, "ra": 336.75081, "dec": 12.46227, "n_epochs": n, "t0_source": t0,
            "first_frame_after_t0_s": 30.0 if t0 != "list" else 32.7,
            "candidates_final": other + recovered, "spurious_final": other,
            "target": {"recovered": recovered, "k_first": k if recovered else None,
                       "t_first_s": t_first if recovered else None,
                       "quality_final": q if recovered else None,
                       "sep_final_arcsec": sep if recovered else None}}


def _site(tmp_path, vetting=True):
    site = tmp_path / "grb_replay_validation"
    site.mkdir()
    (site / "index.html").write_text(JULY)
    (site / "obs_GRB250813B").mkdir()
    (site / "obs_GRB250813B" / "index.html").write_text("page")
    (site / "replay_2026-09-04b").mkdir()
    replay = site / "replay_2026-09-11b"
    runs = {"historical": [_result("GRB250813B", True),
                           _result("GRB250702F", True, k=4, n=17, t_first=67.7, t0="list"),
                           _result("GRB240414A", False),
                           _result("GRB180325A<x>", False, error="RuntimeError()")],
            "vetting": [_result("GRB250813B", True, k=5, other=0), _result("GRB240414A", False, other=1)]}
    for cfg, results in runs.items():
        if cfg == "vetting" and not vetting:
            continue
        (replay / cfg).mkdir(parents=True)
        (replay / cfg / "summary.json").write_text(json.dumps(results))
    return site, replay


def test_the_page_shows_the_current_replay_and_archives_the_old_one(tmp_path):
    site, replay = _site(tmp_path)

    rsi.write_front_page(site, replay, code="503d11e")
    page = (site / "index.html").read_text()

    assert (site / "july_2026_study.html").read_text() == JULY
    assert "July analysis" not in page and 'href="july_2026_study.html"' in page
    assert rsi.GENERATOR in page and "503d11e" in page
    assert "4 GRB nights" in page
    assert '<a class="grbname" href="obs_GRB250813B/index.html">GRB250813B</a>' in page
    assert '<span class="grbname">GRB240414A</span>' in page          # no page for it
    assert "GRB180325A&lt;x&gt;" in page and "replay error" in page
    assert "image 4 of 20 &middot; Q 39.3 &middot; 1.0&Prime;" in page
    assert "53 s after the first image" in page                       # 83 s from first image - 30 s
    assert "68 s after the trigger" in page                           # 67.7 s, trigger time known
    assert "<strong>2 of 4</strong> (historical) and <strong>1 of 2</strong> (vetting)" in page
    assert 'href="replay_2026-09-11b/historical/index.html"' in page
    assert 'href="replay_2026-09-04b/"' in page and 'href="replay_2026-09-11b/"' not in page
    assert "stricter checks against constant sources" in page


def test_a_second_run_rewrites_the_page_but_not_the_archive(tmp_path):
    site, replay = _site(tmp_path)
    rsi.write_front_page(site, replay, code="code-AAA")
    rsi.write_front_page(site, replay, code="code-BBB")
    page = (site / "index.html").read_text()

    assert "code-BBB" in page and "code-AAA" not in page
    assert (site / "july_2026_study.html").read_text() == JULY
    assert sorted(p.name for p in site.glob("*.html")) == ["index.html", "july_2026_study.html"]


def test_a_configuration_missing_from_the_run_is_left_out(tmp_path):
    site, replay = _site(tmp_path, vetting=False)

    runs = rsi.write_front_page(site, replay)

    assert list(runs) == ["historical"]
    page = (site / "index.html").read_text()
    assert "<th>vetting</th>" not in page and "<th>historical</th>" in page


def test_stat_row_and_custom_descriptions(tmp_path):
    site, replay = _site(tmp_path)
    rsi.write_front_page(site, replay, describe={"vetting": "custom text"})
    page = (site / "index.html").read_text()

    assert "afterglows recovered &middot; historical" in page
    # historical: one other candidate each for three bursts, none for the errored one
    assert '<span class="n">3</span><span class="label">other candidates at the last image, all fields &middot; historical' in page
    assert '<span class="n">4</span><span class="label">median image of first detection &middot; historical' in page
    assert "custom text" in page and "stricter checks" not in page


def test_detection_time_wording():
    assert rsi.detection_time({"t0_source": "list", "target": {"t_first_s": 3700}}) == "1.0 h after the trigger"
    assert rsi.detection_time({"t0_source": "first_frame_minus_30s", "first_frame_after_t0_s": 30.0,
                               "target": {"t_first_s": 758.0}}) == "12 min 08 s after the first image"
    assert rsi.detection_time({"target": {}}) == ""
