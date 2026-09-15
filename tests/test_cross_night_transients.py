"""tools/cross_night_transients.py: candidates seen on more than one night."""
import importlib.util
import json
import math

import pytest
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
FIXTURE = REPO / "tests" / "210619B"
needs_fixture = pytest.mark.skipif(not (FIXTURE / "candidates.tbl").exists(),
                                   reason="tests/210619B fixture data not present")
SPEC = importlib.util.spec_from_file_location("cross_night_transients",
                                              REPO / "tools" / "cross_night_transients.py")
xn = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(xn)


def _make_obs(data_dir, obs_id, day, empty=False, dmag=0.0, keep_frame=False):
    """A copy of the 210619B observation moved to the night of 2026-09-<day>.

    empty: the field was observed but nothing was flagged (header-only
    candidates table, no lightcurves); the afterglow's rows are dropped
    from the frame too unless keep_frame (seen, but not flagged).
    dmag: shift every magnitude in the lightcurve files.
    """
    obs_dir = data_dir / f"obs_{obs_id}"
    obs_dir.mkdir(parents=True)

    def ren(text):
        return text.replace("20210619", f"202609{day - 1:02d}").replace("20210620", f"202609{day:02d}")

    ecsvs = sorted(f for f in FIXTURE.glob("*.ecsv")
                   if not f.name.endswith(("_transients.ecsv", "_lightcurve.ecsv")))
    # The third frame is the first one that holds the afterglow.
    frame = ecsvs[2].read_text()
    if empty and not keep_frame:
        # The field was observed but the afterglow was not there: drop its
        # rows from the frame's detection table too (the crawler reads the
        # frames to tell a miss from an unflagged detection).
        kept, names = [], None
        for ln in frame.splitlines():
            if ln.startswith("#") or not ln.strip():
                kept.append(ln); continue
            parts = ln.split()
            if names is None:
                names = parts; kept.append(ln); continue
            ra, dec = float(parts[names.index("ALPHA_J2000")]), float(parts[names.index("DELTA_J2000")])
            if abs(ra - 319.7181) * 0.83 + abs(dec - 33.8504) > 0.003:
                kept.append(ln)
        frame = "\n".join(kept) + "\n"
    (obs_dir / ren(ecsvs[2].name)).write_text(frame)
    (obs_dir / "detection_metadata.json").write_text(json.dumps(
        {"processed_files": [ren(f.name) for f in ecsvs]}))
    table = (FIXTURE / "candidates.tbl").read_text().splitlines()
    if empty:
        (obs_dir / "candidates.tbl").write_text("\n".join(ln for ln in table if ln.startswith(("|", "\\"))) + "\n")
        return obs_dir
    for lc in FIXTURE.glob("*_lightcurve.ecsv"):
        text = lc.read_text()
        if dmag:
            lines = text.splitlines()
            head = [ln for ln in lines if ln.startswith("#")]
            body = [ln for ln in lines if not ln.startswith("#")]
            cols = body[0].split(" ")
            k = cols.index("MAG_CALIB")
            out = [body[0]]
            for ln in body[1:]:
                parts = ln.split(" ")
                if parts[k] not in ("99.0", "nan"):
                    parts[k] = f"{float(parts[k]) + dmag:.4f}"
                out.append(" ".join(parts))
            text = "\n".join(head + out) + "\n"
        (obs_dir / lc.name).write_text(text)
    (obs_dir / "candidates.tbl").write_text(ren("\n".join(table)) + "\n")
    return obs_dir


def test_night_of_frame_time():
    t = xn.frame_time("20260910001523-133-N-010-df.ecsv")
    assert t.isoformat() == "2026-09-10T00:15:23+00:00"
    assert xn.night_of(t) == "2026-09-09"  # after midnight still belongs to the evening's night
    assert xn.frame_time("stack_of_things.ecsv") is None


def test_cluster_joins_neighbours_only():
    pts = [{"ra": 10.0, "dec": 20.0}, {"ra": 10.0 + 1.0 / 3600 / 0.94, "dec": 20.0},
           {"ra": 10.0, "dec": 20.0 + 2.0 / 3600}, {"ra": 10.1, "dec": 20.0}]
    groups = sorted(sorted(g) for g in xn.cluster(pts, radius=3.0))
    assert groups == [[0, 1, 2], [3]]


@needs_fixture
def test_page_groups_same_source_on_two_nights(tmp_path):
    data, public = tmp_path / "work", tmp_path / "html"
    _make_obs(data, "1", 10)
    _make_obs(data, "2", 12)
    _make_obs(data, "3", 12)  # second observation of the same night
    site = public / "obs_1"
    (site / "cutouts").mkdir(parents=True)
    (site / "index.html").write_text("x")
    (site / "cutouts" / "transient_319.718_33.850_montage.webp").write_bytes(b"x")

    assert xn.main(["--data-dir", str(data), "--public-dir", str(public), "--days", "0", "-q"]) == 0

    out = json.loads((public / "new_transients" / "new_transients.json").read_text())
    assert out["n_observations"] == 3
    # Three sources clear the default quality floor of 0.02 (36.9, 0.053, 0.027);
    # the forced row (0.63) is dropped. The bright, well-covered one scores first.
    assert len(out["groups"]) == 3
    scores = [g["score"] for g in out["groups"]]
    assert scores == sorted(scores, reverse=True) and [g["rank"] for g in out["groups"]] == [1, 2, 3]
    g = out["groups"][0]
    assert round(g["max_q"], 3) == 36.929 and g["field"] == "GRB 210620.000 GCN #1056757"
    assert g["score_parts"]["brightness"] == pytest.approx(18 - g["mag_bright"], abs=0.01)
    assert g["score_parts"]["coverage"] == pytest.approx(2 + math.log2(1 + g["n_points"]), abs=0.01)
    assert g["nights"] == ["2026-09-09", "2026-09-11"] and g["obs_ids"] == ["1", "2", "3"]
    assert abs(g["ra"] - 319.7181) < 1e-3 and abs(g["dec"] - 33.8504) < 1e-3
    # The 210619B afterglow fades by 1.2 mag (first third vs last third) inside the night.
    assert g["trend"] == "fading" and g["grb_field"] is True
    assert g["changed"] is True and g["change_kind"] == "within a night"
    assert g["delta_mag"] == pytest.approx(1.2, abs=0.2) and g["score_parts"]["change"] == pytest.approx(min(2.0, g["delta_mag"]), abs=0.01)
    assert g["n_missed_before"] == 0 and g["appeared"] is False
    assert all(h["changed"] is False and h["trend"] == "steady" for h in out["groups"][1:])
    # The forced NUMBER 0 row at the pointing is dropped, so it forms no group.
    assert not any(d["forced"] for d in g["detections"])
    assert g["detections"][0]["site"] == "../obs_1/index.html"
    assert g["detections"][0]["montage"].endswith("_montage.webp")
    assert "lightcurve" not in g["detections"][0]
    assert g["detections"][1]["site"] == ""

    # The stitched lightcurve: every epoch of the fixture's lightcurve file per night,
    # band taken from the frame name ('N'), and one SVG panel per night on the page.
    d0 = g["detections"][0]
    assert len(d0["points"]) >= 5 and all(pt[3] == "N" for pt in d0["points"])
    assert all(10 < pt[1] < 13 and pt[2] is not None for pt in d0["points"])

    page = (public / "new_transients" / "index.html").read_text()
    assert "319.71811 +33.85042" in page and "../obs_1/index.html" in page and "grb" in page
    assert page.count("<svg class='lc'") == 3 and page.count("2026-09-09 ·") == 3 and page.count("2026-09-11 ·") == 3
    assert page.count("<details class='field'") == 1 and "<b>GRB 210620.000 GCN #1056757</b> — 3 sources" in page

    # A rerun serves everything from the cache; with the forced target row kept
    # (quality 0.63 in the fixture) the pointing becomes a group of its own.
    assert xn.main(["--data-dir", str(data), "--public-dir", str(public), "--days", "0",
                    "--include-forced", "--min-quality", "0.5", "-q"]) == 0
    out = json.loads((public / "new_transients" / "new_transients.json").read_text())
    assert len(out["groups"]) == 2
    assert all(d["forced"] for d in out["groups"][1]["detections"])
    assert (data / xn.CACHE_NAME).exists()


def test_read_lightcurve_bands_and_fallback(tmp_path):
    lc = tmp_path / "x_lightcurve.ecsv"
    lc.write_text("# %ECSV 1.0\n# ---\n# datatype:\n# - {name: MAG_CALIB}\n"
                  "NUMBER MAG_CALIB MAGERR_CALIB mjd source_file filter phot_filter\n"
                  "1 15.5 0.1 61000.5 20260909011221-634-i-020-df.ecsv i Sloan_r\n"
                  "0 99.0 99.0 61000.6 20260909011300-634-i-020-df.ecsv i Sloan_r\n"
                  "2 15.7 0.2 61000.4 20260909011000-634-i-020-df.ecsv \"\" \"\"\n")
    pts = xn.read_lightcurve(lc, "N")
    assert pts == [[61000.4, 15.7, 0.2, "Sloan_i"], [61000.5, 15.5, 0.1, "Sloan_r"]]
    assert xn.read_lightcurve(tmp_path / "missing.ecsv") == []
    obs = {"obs_id": "9", "filter": "", "first_frame": "20260909011221-634-i-020-df.ecsv"}
    p = {"id": "nothing", "obs": obs, "mag": 16.0, "magerr": 0.05,
         "source_file": "20260909011221-634-i-020-df"}
    assert xn.detection_points(p, tmp_path) == [[pytest.approx(61292.05024, abs=1e-4), 16.0, 0.05, "Sloan_i"]]


@needs_fixture
def test_max_cards_puts_the_rest_in_a_table(tmp_path):
    data, public = tmp_path / "work", tmp_path / "html"
    _make_obs(data, "1", 10)
    _make_obs(data, "2", 12)
    assert xn.main(["--data-dir", str(data), "--public-dir", str(public), "--days", "0",
                    "--per-field", "1", "-q"]) == 0
    page = (public / "new_transients" / "index.html").read_text()
    out = json.loads((public / "new_transients" / "new_transients.json").read_text())
    assert len(out["groups"]) == 3
    assert page.count("<svg class='lc'") == 1 and "2 more in the table" in page
    assert page.count("<tr id='g2'><td>2</td>") == 1 and page.count("<tr id='g3'><td>3</td>") == 1  # ranks in the table

    # The page-wide cap wins over the per-field one, and the table can be cut too.
    assert xn.main(["--data-dir", str(data), "--public-dir", str(public), "--days", "0",
                    "--max-cards", "2", "--max-rows", "0", "-q"]) == 0
    page = (public / "new_transients" / "index.html").read_text()
    assert page.count("<svg class='lc'") == 2 and "1 more in the table" in page


def test_score_and_dominant_field():
    s = xn.transient_score(mag_bright=12.0, n_nights=3, n_points=15, max_q=99.0)
    assert s == {"brightness": 6.0, "coverage": 8.0, "quality": 2.0, "change": 0.0, "score": 16.0}
    assert xn.transient_score(None, 2, 1, 0.0)["score"] == 3.0
    s = xn.transient_score(12.0, 3, 15, 99.0, delta_mag=1.0, appeared=True, disappeared=True)
    assert s["change"] == 8.0 and s["score"] == 24.0
    assert xn.transient_score(12.0, 3, 15, 99.0, delta_mag=5.0)["change"] == 6.0
    assert xn.transient_score(12.0, 3, 15, 99.0, delta_mag=5.0, fading=True)["change"] == 2.0
    assert xn.transient_score(5.0, 1, 0, 0.0)["brightness"] == 10.0
    dets = [{"obs": {"object": "A"}}, {"obs": {"object": "B"}}, {"obs": {"object": "B"}}, {"obs": {"object": ""}}]
    assert xn.dominant_field(dets) == "B"
    assert xn.dominant_field([{"obs": {"object": ""}}]) == ""


def test_bin_points_caps_marks_per_night():
    d = {"obs_id": "1"}
    pts = [([61000.0 + k / 1440, 15.0 + k / 100, 0.1, "N"], d) for k in range(120)]
    binned = xn.bin_points(pts, 40)
    assert len(binned) == 40 and sum(n for _, _, n in binned) == 120
    assert all(n == 3 for _, _, n in binned) and binned[0][0][2] == pytest.approx(0.1 / 3 ** 0.5)
    assert xn.bin_points(pts[:5], 40) == [(pt, d, 1) for pt, d in pts[:5]]
    two = xn.bin_points(pts + [([61000.02, 14.0, 0.1, "Sloan_r"], d)], 40)
    assert len(two) == 41 and sum(1 for pt, _, _ in two if pt[3] == "Sloan_r") == 1


@needs_fixture
def test_recent_window_excludes_old_nights(tmp_path):
    data, public = tmp_path / "work", tmp_path / "html"
    _make_obs(data, "1", 10)
    _make_obs(data, "2", 12)
    assert xn.main(["--data-dir", str(data), "--public-dir", str(public), "--days", "0.001", "-q"]) == 0
    out = json.loads((public / "new_transients" / "new_transients.json").read_text())
    assert out["n_observations"] == 0 and out["groups"] == []


def test_covers_uses_the_frame_wcs():
    facts = {"center": [100.0, 20.0], "cd": [-0.0003, 0.0, 0.0, 0.0003], "size": [1024, 1024]}
    assert xn.covers(facts, 100.0, 20.0)                                          # no CRPIX: centre assumed
    assert xn.covers(facts, 100.0 + 0.1 / math.cos(math.radians(20.0)), 20.1)     # 333 px off centre
    assert not xn.covers(facts, 100.0, 20.2)                                      # 667 px: beyond the edge
    assert not xn.covers({"center": [None, None], "cd": None, "size": None}, 100.0, 20.0)
    # CRVAL sits at CRPIX, which need not be the centre: with the reference
    # point in a corner the same sky position is outside the frame.
    corner = dict(facts, crpix=[1.0, 1.0])
    assert not xn.covers(corner, 100.0, 20.2) and not xn.covers(corner, 100.0, 19.9)
    assert xn.covers(corner, 100.0 - 0.1 / math.cos(math.radians(20.0)), 20.1)   # +333 px in x and y


@needs_fixture
def test_change_and_appearance_from_field_history(tmp_path):
    data, public = tmp_path / "work", tmp_path / "html"
    _make_obs(data, "1", 5, empty=True)            # field observed, nothing found (limit 17.1)
    _make_obs(data, "2", 7, empty=True)
    _make_obs(data, "3", 10)                       # source at 11.4
    _make_obs(data, "4", 12, dmag=1.5)             # a magnitude and a half fainter
    _make_obs(data, "5", 14, empty=True)           # gone again
    _make_obs(data, "6", 16, empty=True, keep_frame=True)   # in the frame, not flagged: not a miss
    assert xn.main(["--data-dir", str(data), "--public-dir", str(public), "--days", "0", "-q"]) == 0
    out = json.loads((public / "new_transients" / "new_transients.json").read_text())
    g = out["groups"][0]
    assert g["n_missed_before"] == 2 and g["n_missed_after"] == 1 and g["n_missed_between"] == 0
    assert g["n_present_unflagged"] == 1
    # The other two fixture sources sit outside the covered part of the frames.
    assert all(h["n_missed_before"] == 0 and h["n_present_unflagged"] == 0 for h in out["groups"][1:])
    assert (data / xn.CELLS_DIR / "obs_1.bin").exists()
    page = (public / "new_transients" / "index.html").read_text()
    assert "detected but not flagged on 1 other night(s)" in page
    assert g["appeared"] is True and g["disappeared"] is True
    assert g["limit_before"] == pytest.approx(17.1, abs=0.05)
    assert g["changed"] is True and g["change_sigma"] > 5
    assert g["nightly"]["2026-09-11"][0] - g["nightly"]["2026-09-09"][0] == pytest.approx(1.5, abs=0.01)
    assert g["trend"] == "fading"
    assert g["score_parts"]["change"] == pytest.approx(min(2.0, g["delta_mag"]) + 4.0 + 1.0, abs=0.01)
    assert [m["night"] for m in g["missed"]] == ["2026-09-04", "2026-09-06", "2026-09-13"]
    page = (public / "new_transients" / "index.html").read_text()
    assert "tag appeared" in page and "tag disappeared" in page and "tag fading" in page
    assert "Changed and new sources (3)" in page and "<a href='#g1'>1</a>" in page  # dmag shifted every source
    assert "not in 2 earlier frame(s)" in page and "../obs_1/index.html" in page
    # The faint sources (17.5 mag) are below the 1 mag margin of the 17.1 limit: not "appeared".
    assert all(not h["appeared"] for h in out["groups"][1:])


def test_within_night_change_ignores_single_bad_epochs():
    def det(mags, night="2026-09-09"):
        return {"night": night, "points": [[61000.0 + k / 1440, m, 0.02, "N"] for k, m in enumerate(mags)]}
    delta, sig = xn.within_night_change([det([15.0, 15.01, 14.99, 15.0, 15.02, 15.0, 14.98, 15.01, 15.0])])
    assert abs(delta) < 0.05
    delta, sig = xn.within_night_change([det([15.0] * 8 + [12.0])])   # one wild epoch out of nine
    assert abs(delta) < 0.05 or sig < 5
    delta, sig = xn.within_night_change([det([15.0, 15.0, 15.0, 15.5, 16.0, 16.0])])
    assert delta == pytest.approx(1.0, abs=0.01) and sig > 5
    assert xn.within_night_change([det([15.0, 16.0, 17.0])]) == (0.0, 0.0)   # fewer than four epochs


def test_slow_decline_over_many_nights_is_a_trend():
    # 0.08 mag/day: no single night-to-night step reaches 0.3 mag, the trend does.
    dets = []
    for k in range(8):
        mjd0 = 61300 + 2 * k
        dets.append({"night": f"2026-09-{10 + 2 * k:02d}", "mag": 15.0, "magerr": 0.05,
                     "points": [[mjd0 + j / 1440, 15.0 + 0.08 * 2 * k + (0.03 if j % 2 else -0.03), 0.05, "N"]
                                for j in range(8)]})
    stats = xn.nightly_stats(dets)
    fit = xn.night_trend(stats)
    assert fit["slope"] == pytest.approx(0.08, abs=0.005) and fit["sigma"] > 5 and fit["span_days"] == pytest.approx(14, abs=0.1)
    assert xn.night_trend({k: v for k, v in list(stats.items())[:2]}) is None
    steps = xn.change_between_nights(stats)
    assert steps[0] == pytest.approx(1.12, abs=0.02)   # first against last night, still caught


def test_detection_cells_round_trip(tmp_path):
    cells = {xn.cell_of(100.0, 20.0, 20.0), xn.cell_of(100.1, 20.1, 20.0)}
    xn.save_cells(tmp_path, "obs_x", cells)
    assert xn.load_cells(tmp_path, "obs_x") == cells
    assert xn.load_cells(tmp_path, "obs_missing") is None
    assert xn.detected_in(cells, 100.0 + 1.0 / 3600 / math.cos(math.radians(20.0)), 20.0, 20.0)  # 1" away
    assert not xn.detected_in(cells, 100.0, 20.0 + 10.0 / 3600, 20.0)                           # 10" away


def test_ensemble_offsets_remove_a_bad_night():
    # Six constant stars; night 2 is 0.4 mag off for all of them (zero point).
    dets_by_star = []
    for k in range(6):
        dets = []
        for n in range(4):
            off = 0.4 if n == 1 else 0.0
            dets.append({"night": f"2026-09-{10 + n:02d}", "mag": 15.0 + k, "magerr": 0.05,
                         "points": [[61000 + n + j / 1440, 15.0 + k + off + 0.01 * (j % 2), 0.03, "N"] for j in range(6)]})
        dets_by_star.append(("F", dets))
    offsets = xn.ensemble_offsets(dets_by_star)
    assert offsets[("F", "2026-09-11", "N")] == pytest.approx(0.3, abs=0.01)   # 0.4 above the star's 4-night mean
    dets = dets_by_star[0][1]
    assert xn.change_between_nights(xn.nightly_stats(dets, "N"))[0] == pytest.approx(0.4, abs=0.02)
    xn.apply_offsets(dets, "F", offsets)
    assert xn.change_between_nights(xn.nightly_stats(dets, "N"))[0] < 0.15
    assert xn.ensemble_offsets(dets_by_star[:3]) == {}                          # too few sources to vote


def test_nightly_stats_keep_bands_apart():
    dets = [{"night": "2026-09-10", "mag": 15.0, "magerr": 0.05,
             "points": [[61000.0, 15.0, 0.03, "Sloan_r"]] * 4 + [[61000.1, 15.6, 0.03, "Sloan_i"]] * 4}]
    assert xn.bands_of(dets) == ["Sloan_i", "Sloan_r"]
    assert set(xn.nightly_stats(dets, "Sloan_r")) == {"2026-09-10"} and xn.nightly_stats(dets, "Sloan_r")["2026-09-10"][0] == 15.0
    assert xn.nightly_stats(dets, "Sloan_i")["2026-09-10"][0] == pytest.approx(15.6)
    assert xn.nightly_stats(dets, "V") == {}


def test_within_night_change_does_not_mix_bands():
    # r frames then g frames in one night: a colour, not a change.
    det = {"night": "2026-09-09", "points": [[61000.0 + k / 1440, 12.6, 0.02, "Sloan_r"] for k in range(6)]
           + [[61000.01 + k / 1440, 13.96, 0.02, "Sloan_g"] for k in range(6)]}
    delta, sig = xn.within_night_change([det])
    assert abs(delta) < 0.05
    # The same in one band is a change.
    det = {"night": "2026-09-09", "points": [[61000.0 + k / 1440, 12.6 + 0.2 * k, 0.02, "Sloan_r"] for k in range(6)]}
    delta, sig = xn.within_night_change([det])
    assert delta == pytest.approx(0.8, abs=0.05) and sig > 5


def test_band_names_are_normalised(tmp_path):
    assert xn.normalise_band("i") == "Sloan_i" and xn.normalise_band("Sloan_i") == "Sloan_i"
    assert xn.normalise_band("C") == "N" and xn.normalise_band("N") == "N" and xn.normalise_band("V") == "V"
    lc = tmp_path / "x_lightcurve.ecsv"
    lc.write_text("# %ECSV 1.0\n# ---\n# datatype:\n# - {name: MAG_CALIB}\n"
                  "NUMBER MAG_CALIB MAGERR_CALIB mjd source_file\n"
                  "1 15.5 0.1 61000.5 20260909011221-634-i-020-df.ecsv\n")
    assert xn.read_lightcurve(lc, "N")[0][3] == "Sloan_i"


@needs_fixture
def test_interesting_sources_take_the_cards_first(tmp_path):
    data, public = tmp_path / "work", tmp_path / "html"
    _make_obs(data, "1", 5, empty=True)
    _make_obs(data, "2", 10)
    _make_obs(data, "3", 12, dmag=1.5)
    # One card per field: it must go to the changed/appeared afterglow (rank 1 anyway),
    # so use a page cap of 1 with a steadier field ordering check on the JSON.
    assert xn.main(["--data-dir", str(data), "--public-dir", str(public), "--days", "0",
                    "--per-field", "1", "-q"]) == 0
    out = json.loads((public / "new_transients" / "new_transients.json").read_text())
    carded = [g for g in out["groups"] if g["card"]]
    assert len(carded) == 1 and carded[0]["appeared"] and carded[0]["changed"]


def test_a_single_epoch_or_one_outlier_does_not_make_a_change():
    def det(night, mags, day):
        return {"night": night, "mag": 15.0, "magerr": 0.05,
                "points": [[61000 + day + k / 1440, m, 0.03, "N"] for k, m in enumerate(mags)]}
    # Night with one epoch 1.5 mag off: not a night.
    stats = xn.nightly_stats([det("2026-09-10", [15.0] * 6, 0), det("2026-09-12", [13.5], 2)], "N")
    assert list(stats) == ["2026-09-10"]
    # One wild epoch among six: the median and its error ignore it.
    stats = xn.nightly_stats([det("2026-09-10", [15.0] * 6, 0), det("2026-09-12", [15.0, 15.02, 13.0, 14.98, 15.01, 15.0], 2)], "N")
    d, sig = xn.change_between_nights(stats)
    assert d < 0.05
    # The band-blind summary still lists the single-epoch night.
    assert len(xn.nightly_stats([det("2026-09-10", [15.0] * 6, 0), det("2026-09-12", [13.5], 2)], min_epochs_off=True)) == 2
    # Within a night: five epochs are too few for the thirds test; one outlier in nine is ignored.
    assert xn.within_night_change([det("2026-09-10", [15.0, 15.0, 15.0, 13.0, 15.0], 0)]) == (0.0, 0.0)
    delta, sig = xn.within_night_change([det("2026-09-10", [15.0] * 8 + [13.0], 0)])
    assert abs(delta) < 0.05


def test_frame_positions_skip_nan_rows(tmp_path):
    f = tmp_path / "x.ecsv"
    f.write_text("# %ECSV 1.0\n# ---\nNUMBER ALPHA_J2000 DELTA_J2000\n1 100.0 20.0\n2 nan nan\n3 100.1 20.1\n")
    assert xn.frame_positions(f) == [(100.0, 20.0), (100.1, 20.1)]
    assert xn.cell_of(float("nan"), 20.0, 20.0) is None
    assert len(xn.scan_cells(tmp_path, 20.0)) == 2


def test_insane_magnitudes_and_bad_frames_are_dropped(tmp_path):
    lc = tmp_path / "x_lightcurve.ecsv"
    lc.write_text("# %ECSV 1.0\n# ---\nNUMBER MAG_CALIB MAGERR_CALIB mjd source_file\n"
                  "1 15.5 0.1 61000.5 a-r-x.ecsv\n2 -28.0 0.1 61000.6 a-r-x.ecsv\n3 15.4 0.9 61000.7 a-r-x.ecsv\n")
    assert [pt[1] for pt in xn.read_lightcurve(lc, "N")] == [15.5]
    # Six stars of one field; the frame at mjd 61000.5 is 2 mag off for all of them.
    clusters = []
    for k in range(6):
        pts = [[61000.0 + j / 1440, 15.0 + k, 0.03, "N"] for j in range(6)] + [[61000.5, 17.0 + k, 0.03, "N"]]
        clusters.append(("F", [{"night": "2026-09-10", "mag": 15.0 + k, "magerr": 0.03, "points": pts}]))
    bad = xn.frame_offsets(clusters)
    assert bad[("F", "N", round(61000.5 / xn.FRAME_KEY_DAYS))] == pytest.approx(2.0, abs=0.01)
    assert sum(xn.drop_bad_frames(d, f, bad) for f, d in clusters) == 6
    assert all(len(d[0]["points"]) == 6 for _, d in clusters)


def test_witness_efficiency_disqualifies_a_blind_observation(tmp_path):
    # Twelve sources of one field; observation A detected all, observation B none.
    facts = lambda oid: {"obs_id": oid, "center": [100.0, 20.0], "crpix": [512.0, 512.0],
                         "cd": [-0.0003, 0.0, 0.0, 0.0003], "size": [1024, 1024], "maglim": 18.0, "n_det": 500,
                         "nights": ["2026-09-01"], "object": "F", "first": ""}
    clusters, cells_a = [], set()
    for k in range(12):
        ra, dec = 100.0 + 0.01 * k, 20.0 + 0.005 * k
        cells_a.add(xn.cell_of(ra, dec, 20.0))
        clusters.append(("F", [{"night": "2026-09-05", "ra": ra, "dec": dec, "q": 1.0, "mag": 15.0, "magerr": 0.03,
                                "points": [[61000.0 + j / 1440, 15.0, 0.03, "N"] for j in range(4)]}]))
    xn.save_cells(tmp_path, "obs_A", cells_a)
    xn.save_cells(tmp_path, "obs_B", set())
    observations = [{"facts": facts("A")}, {"facts": facts("B")}]
    w = xn.witness_efficiency(clusters, observations, tmp_path, {})
    assert all(d for _, d in w["A"]) and not any(d for _, d in w["B"])
    assert xn.witness_ok(w, "A", 15.0) is True and xn.witness_ok(w, "B", 15.0) is False
    # No peers within two magnitudes: the observation as a whole decides (12 sources tested).
    assert xn.witness_ok(w, "A", 9.0) is True and xn.witness_ok(w, "B", 9.0) is False
    # Fewer than five peers and fewer than ten sources in all: no verdict.
    assert xn.witness_ok(xn.witness_efficiency(clusters[:4], observations, tmp_path, {}), "A", 15.0) is None
    assert xn.witness_ok(w, "C", 15.0) is None
