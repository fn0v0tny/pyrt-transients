"""validation/replay.py + summary.py with a stub strategy, so the snapshot
bookkeeping, latency and completeness maths are tested without catalogues."""
import numpy as np
import pytest
from astropy.table import Table

from pyrt_transient import PipelineConfig
from pyrt_transient.validation import replay, summary


def _epoch(i, ctime):
    t = Table({"ALPHA_J2000": [10.0], "DELTA_J2000": [20.0]})
    t.meta = {"filename": f"e{i}.ecsv", "CTIME": ctime, "EXPTIME": 10.0, "FIELD": 0.3,
              "CTRRA": 10.0, "CTRDEC": 20.0}
    return t


class StubStrategy:
    """Reports target A from k>=3 with score growing with k, a spurious
    candidate from k>=5, and target B never."""
    def __init__(self, data_dir, config):
        self.catalog_loader = None

    def run(self, tables, config=None, params=None, **kw):
        k = len(tables)
        rows = []
        if k >= 3:
            rows.append({"ALPHA_J2000": 10.0001, "DELTA_J2000": 20.0, "quality_score": 0.1 * k,
                         "n_detections": k, "transient_id": "A", "candidate_type": "new"})
        if k >= 5:
            rows.append({"ALPHA_J2000": 10.05, "DELTA_J2000": 20.05, "quality_score": 1.0,
                         "n_detections": 3, "transient_id": "S", "candidate_type": "new"})
        return (Table(rows=rows) if rows else Table()), {}


def test_incremental_replay_snapshots_and_latency(tmp_path):
    tables = [_epoch(i, 1000.0 + 11 * i) for i in range(8)]
    targets = [{"id": "A", "ra": 10.0, "dec": 20.0}, {"id": "B", "ra": 10.2, "dec": 20.2}]
    cfg = PipelineConfig()
    snaps, cands, _ = replay.incremental_replay(
        tables, cfg, tmp_path, targets, params=replay.query_params_for(tables),
        strategy_factory=StubStrategy,
    )
    assert [s["k"] for s in snaps] == list(range(1, 9))
    assert snaps[1]["n_candidates"] == 0
    assert snaps[2]["targets"]["A"]["sep_arcsec"] < 1.0
    assert snaps[2]["targets"]["B"] is None
    assert snaps[4]["n_spurious"] == 1 and snaps[3]["n_spurious"] == 0
    fr = replay.first_recovery(snaps, "A")
    assert fr["k"] == 3
    assert replay.first_recovery(snaps, "A", min_quality=0.45)["k"] == 5
    assert replay.first_recovery(snaps, "B") is None
    assert len(cands) == 2


def test_recovery_table_and_summary(tmp_path):
    tables = [_epoch(i, 1000.0 + 11 * i) for i in range(6)]
    cfg = PipelineConfig()
    sources = [
        {"source_id": "A", "ra": 10.0, "dec": 20.0, "m0": 16.0, "alpha": 1.0, "t0": 970.0, "t_ref": 1005.0},
        {"source_id": "B", "ra": 10.2, "dec": 20.2, "m0": 19.0, "alpha": 0.0, "t0": 970.0, "t_ref": 1005.0},
    ]
    targets = [{"id": s["source_id"], "ra": s["ra"], "dec": s["dec"]} for s in sources]
    snaps, _, _ = replay.incremental_replay(tables, cfg, tmp_path, targets,
                                            params=replay.query_params_for(tables), strategy_factory=StubStrategy)
    truth_rows = []
    for s in sources:
        for i, t in enumerate(tables):
            truth_rows.append({"source_id": s["source_id"], "epoch_file": t.meta["filename"],
                               "t_mid": 1005.0 + 11 * i, "t_since_t0": 35 + 11 * i,
                               "mag_true": s["m0"] + 0.1 * i, "mag_obs": s["m0"], "magerr": 0.05,
                               "maglim": 17.0, "maglimit": 19.1, "p_detect": 1.0,
                               "detected": s["source_id"] == "A" or i < 2})
    rec = summary.recovery_table(sources, Table(rows=truth_rows), snaps, cfg.detection.min_quality)
    a, b = rec[rec["source_id"] == "A"][0], rec[rec["source_id"] == "B"][0]
    assert a["recovered_final"] and not b["recovered_final"]
    assert a["k_first"] == 3 and a["t_first_s"] == pytest.approx(1005.0 + 22 - 970.0)
    assert a["n_epochs_detected"] == 6 and b["n_epochs_detected"] == 2
    assert a["dmag_peak_vs_maglim"] == pytest.approx(-1.0)
    comp = summary.completeness_by_bin(rec, "dmag_peak_vs_maglim", [-2, 0, 4])
    assert comp["n_recovered"][0] == 1 and comp["n"][0] == 1
    assert comp["n_recovered"][1] == 0 and comp["n"][1] == 1
    sp = summary.spurious_vs_k([snaps])
    assert sp["max"][-1] == 1
    head = summary.summarise(rec, sp, cfg.detection.min_quality)
    assert head["n_recovered"] == 1 and head["latency_epochs"]["median"] == 3
    summary.plot_completeness(comp, tmp_path / "c.pdf")
    summary.plot_latency(rec, tmp_path / "l.pdf", cfg.detection.min_quality)
    summary.plot_spurious(sp, tmp_path / "s.pdf")
    summary.dump_json(head, tmp_path / "h.json")
    assert (tmp_path / "c.pdf").exists() and (tmp_path / "h.json").exists()


def test_wilson_interval_edges():
    lo, hi = summary.wilson_interval(0, 10)
    assert lo == pytest.approx(0.0) and 0 < hi < 0.3
    lo, hi = summary.wilson_interval(10, 10)
    assert 0.7 < lo < 1 and hi == pytest.approx(1.0)


def test_pass2_merges_on_member_distance_not_centroid(tmp_path):
    """Regression for GRB 220403B (2026-09-04): three candidate rows of one
    source within 1.1" of each other plus one outlying blob row 1.8" away
    (wide adaptive radius, links to one of them in pass 1). Centroid-based
    merging left two 2-row components below min_n_detections=3; member-level
    single linkage must keep all four together."""
    import numpy as np
    from astropy.table import Table
    from pyrt_transient import PipelineConfig
    from pyrt_transient.core.epochs import prepare_epoch_detections
    from pyrt_transient.detection.blind_multicatalog import clustering

    ra0, dec0 = 191.471, 89.185
    cosd = np.cos(np.radians(dec0))
    # (dRA arcsec, dDec arcsec, epoch, fwhm, errxy)
    rows = [(0.0, 0.0, 0, 3.0, 0.02), (1.0, 0.3, 1, 3.0, 0.02), (-0.8, 0.5, 2, 3.0, 0.02),
            (1.7, 0.6, 3, 14.0, 0.0)]      # the blob: 1.8" off, zero centroid error
    tables = []
    for k, (dra, ddec, ep, fwhm, err) in enumerate(rows):
        t = Table({"ALPHA_J2000": [ra0 + dra / 3600 / cosd], "DELTA_J2000": [dec0 + ddec / 3600],
                   "X_IMAGE": [500.0 + dra], "Y_IMAGE": [500.0 + ddec], "ERRX2_IMAGE": [err], "ERRY2_IMAGE": [err],
                   "FWHM_IMAGE": [fwhm], "FLAGS": [0], "MAG_CALIB": [19.0], "MAGERR_CALIB": [0.1],
                   "quality_score": [1.0], "candidate_type": ["new"], "reference_catalog": ["gaia"]})
        t.meta = {"filename": f"e{ep}.ecsv", "CTIME": 1000.0 + 60 * ep, "EXPTIME": 10.0,
                  "CD1_1": -1.18 / 3600, "CD2_2": 1.18 / 3600, "ASTVAR": 1.0, "ASTSIGMA": 0.5, "MAGLIM": 20.0}
        tables.append(t)
        t.write(tmp_path / f"e{ep}_transients.ecsv", format="ascii.ecsv", overwrite=True)
    cfg = PipelineConfig(); cfg.detection.vsx_filter_enabled = False
    cfg.detection.new_source_variability_floor = True   # constant 19.0 mag rows would otherwise score 0
    dets = prepare_epoch_detections(tables)
    cands, lcs = clustering.combine_with_lightcurves(tmp_path, tables, dets, position_match_radius=2.0,
                                                     min_n_detections=3, config=cfg)
    assert len(cands) == 1
    assert int(cands["n_detections"][0]) == 4


def test_split_component_keeps_distinct_epochs_whole_and_splits_conflicts():
    import numpy as np
    from astropy.table import Table
    from pyrt_transient.detection.blind_multicatalog.clustering import split_component_by_epoch
    ra0, dec0 = 191.471, 89.185; cosd = np.cos(np.radians(dec0))
    def tab(rows):
        return Table({"ALPHA_J2000": [ra0 + r[0] / 3600 / cosd for r in rows], "DELTA_J2000": [dec0 + r[1] / 3600 for r in rows],
                      "epoch_id": [r[2] for r in rows], "quality_score": [r[3] for r in rows]})
    # blob (highest score) 1.8" off; the three real rows near the origin; all epochs distinct
    t = tab([(0.0, 0.0, 13, 1.0), (1.9, 0.5, 15, 1.0), (0.4, 0.2, 16, 1.0), (2.6, 1.0, 41, 1.5)])
    assert split_component_by_epoch([0, 1, 2, 3], t, 2.0) == [[0, 1, 2, 3]]
    # a genuine conflict (two rows in epoch 13) still splits
    t2 = tab([(0.0, 0.0, 13, 1.0), (0.3, 0.1, 13, 0.9), (0.2, 0.0, 15, 1.0)])
    subs = split_component_by_epoch([0, 1, 2], t2, 2.0)
    assert len(subs) == 2 and sorted(map(len, subs)) == [1, 2]


def test_filter_epochs_by_pointing_handles_the_ra_wrap():
    """A field straddling RA=0 used to lose every epoch: the median RA and
    the planar difference put each frame ~180 deg from the field centre, so
    the function returned [] and query_params_for crashed on tables[0]."""
    def _t(ra, dec):
        t = Table({"ALPHA_J2000": [ra], "DELTA_J2000": [dec]})
        t.meta = {"CTRRA": ra, "CTRDEC": dec, "filename": f"{ra}.ecsv"}
        return t
    tables = [_t(359.95, 20.0), _t(0.02, 20.0), _t(359.99, 20.02)]
    kept = replay.filter_epochs_by_pointing(tables, max_sep_arcmin=10.0)
    assert len(kept) == 3
    # a genuine outlier is still dropped, wrap or no wrap
    kept = replay.filter_epochs_by_pointing(tables + [_t(7.5, 20.0)], max_sep_arcmin=10.0)
    assert len(kept) == 3


def test_filter_epochs_by_pointing_still_drops_a_stray_field():
    """The case the filter exists for (GRB 250813B's first frame, 7.5 deg
    away) away from the wrap."""
    def _t(ra, dec):
        t = Table({"ALPHA_J2000": [ra], "DELTA_J2000": [dec]})
        t.meta = {"CTRRA": ra, "CTRDEC": dec, "filename": f"{ra}.ecsv"}
        return t
    tables = [_t(243.918, 14.399), _t(243.930, 14.402), _t(236.4, 14.4)]
    kept = replay.filter_epochs_by_pointing(tables, max_sep_arcmin=10.0)
    assert len(kept) == 2
