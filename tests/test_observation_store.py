"""Unit tests for io/observation_store.py (rewrite.md Phase 3 step 6) --
using tmp_path fixtures, no astropy/FITS involved. This is the payoff of
the extraction: fast, no real data needed.

Run with: python3 -m pytest tests/test_observation_store.py -v
"""
import json
import sys

import pytest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from pyrt_transient.io.observation_store import (
    ObservationStore,
    clean_observation_id,
    extract_observation_id,
)


def test_clean_observation_id_strips_decimal_and_invalid_chars():
    assert clean_observation_id("94249.01") == "94249"
    assert clean_observation_id("abc def!!") == "abc_def"
    assert clean_observation_id("__leading_trailing__") == "leading_trailing"
    assert clean_observation_id("") == "unknown"
    assert clean_observation_id("...") == "unknown"


def test_extract_observation_id_falls_back_to_filename(tmp_path):
    # No such file -> open_ecsv_file raises inside the try, falls back to
    # cleaning the filename stem.
    missing = tmp_path / "obs_72006_epoch1.ecsv"
    result = extract_observation_id(str(missing))
    assert result == "obs_72006_epoch1" or "72006" in result


def test_obs_dir_created_on_construction(tmp_path):
    store = ObservationStore(tmp_path, "72006")
    assert store.obs_dir == tmp_path / "obs_72006"
    assert store.obs_dir.exists()


def test_load_existing_tables_no_metadata_returns_empty(tmp_path):
    store = ObservationStore(tmp_path, "72006")
    tables, processed = store.load_existing_tables()
    assert tables == []
    assert processed == set()


def test_mark_processed_then_already_processed(tmp_path):
    store = ObservationStore(tmp_path, "72006")
    assert store.already_processed("a.ecsv") is False

    store.mark_processed("a.ecsv")
    assert store.already_processed("a.ecsv") is True
    assert store.already_processed("b.ecsv") is False


def test_mark_processed_accumulates_across_calls(tmp_path):
    store = ObservationStore(tmp_path, "72006")
    store.mark_processed("a.ecsv")
    store.mark_processed("b.ecsv")

    metadata_file = store.obs_dir / "detection_metadata.json"
    metadata = json.loads(metadata_file.read_text())
    assert set(metadata["processed_files"]) == {"a.ecsv", "b.ecsv"}
    assert metadata["total_files"] == 2
    assert metadata["observation_id"] == "72006"


def test_load_existing_tables_skips_unparseable_file(tmp_path):
    store = ObservationStore(tmp_path, "72006")
    store.mark_processed("bad.ecsv")
    # Not a real ECSV -- open_ecsv_file is expected to fail on this, and
    # load_existing_tables must catch that and skip it, not raise.
    (store.obs_dir / "bad.ecsv").write_text("not a real ecsv file\n")

    tables, processed = store.load_existing_tables()
    assert tables == []  # failed to parse, skipped
    assert processed == {"bad.ecsv"}  # metadata read still succeeds


def test_should_run_analysis_new_detection_added(tmp_path):
    store = ObservationStore(tmp_path, "72006")
    should, reason = store.should_run_analysis(new_detection_added=True)
    assert should is True
    assert "New detection" in reason


def test_should_run_analysis_no_existing_results(tmp_path):
    store = ObservationStore(tmp_path, "72006")
    should, reason = store.should_run_analysis(new_detection_added=False)
    assert should is True
    assert "No existing results" in reason


def test_should_run_analysis_skips_when_results_exist(tmp_path):
    store = ObservationStore(tmp_path, "72006")
    (store.obs_dir / "candidates.tbl").write_text("dummy")
    should, reason = store.should_run_analysis(new_detection_added=False)
    assert should is False
    assert "no new data" in reason


def test_analysis_lock_creates_and_keeps_lock_file(tmp_path):
    """The lock file must survive release: flock() locks the inode, so
    unlinking on release let a third process open a fresh inode and enter
    the critical section while a second was still blocked on the old one."""
    store = ObservationStore(tmp_path, "72006")
    lock_path = store.obs_dir / ".analysis.lock"
    assert not lock_path.exists()

    with store.analysis_lock():
        assert lock_path.exists()

    assert lock_path.exists()
    # ...and it can be re-acquired immediately by the same path.
    with store.analysis_lock():
        assert lock_path.exists()


def test_load_existing_tables_orders_epochs_by_observation_time(tmp_path):
    """glob() is filesystem order; consumers of [0]/[-1] need time order."""
    from astropy.table import Table
    store = ObservationStore(tmp_path, "72006")
    # Names chosen so alphabetical order != time order.
    for name, ctime in (("a.ecsv", 3000.0), ("b.ecsv", 1000.0), ("c.ecsv", 2000.0)):
        t = Table({"X_IMAGE": [1.0]})
        t.meta.update({"CTIME": ctime, "EXPTIME": 10.0})
        t.write(store.obs_dir / name, format="ascii.ecsv")
        store.mark_processed(name)
    tables, processed = store.load_existing_tables()
    assert [t.meta["CTIME"] for t in tables] == [1000.0, 2000.0, 3000.0]
    assert processed == {"a.ecsv", "b.ecsv", "c.ecsv"}
    # An undated table sorts first, never as "the newest".
    u = Table({"X_IMAGE": [1.0]})
    u.write(store.obs_dir / "z.ecsv", format="ascii.ecsv")
    store.mark_processed("z.ecsv")
    tables, _ = store.load_existing_tables()
    assert "CTIME" not in tables[0].meta and tables[-1].meta["CTIME"] == 3000.0


def test_two_stores_same_base_dir_share_metadata(tmp_path):
    # Simulates two sequential pipeline_magic invocations against the same
    # observation -- each constructs its own ObservationStore instance, but
    # state persists via the filesystem, not in-memory.
    store1 = ObservationStore(tmp_path, "72006")
    store1.mark_processed("a.ecsv")

    store2 = ObservationStore(tmp_path, "72006")
    assert store2.already_processed("a.ecsv") is True
    tables, processed = store2.load_existing_tables()
    assert processed == {"a.ecsv"}


def _write_epoch(path, obsid, ra, dec, ctime):
    from astropy.table import Table
    t = Table({"ALPHA_J2000": [ra], "DELTA_J2000": [dec], "MAG_CALIB": [15.0]})
    t.meta.update({"OBSID": obsid, "CTRRA": ra, "CTRDEC": dec, "CTIME": ctime, "EXPTIME": 10.0})
    t.write(path, format="ascii.ecsv", overwrite=True)


def test_resolve_observation_id_groups_by_pointing(tmp_path):
    from pyrt_transient.io.observation_store import ObservationStore, resolve_observation_id
    base = tmp_path / "work"
    e1 = tmp_path / "e1.ecsv"
    _write_epoch(e1, "71883.00", 243.918, 14.399, 1_623_000_000)
    # first epoch: nothing to join, raw OBSID
    obs_id, reason = resolve_observation_id(e1, base, radius_arcmin=10.0)
    assert obs_id == "71883"
    store = ObservationStore(base, obs_id)
    from pyrt_transient.io.ecsv import open_ecsv_file
    store.record_pointing(open_ecsv_file(str(e1), verbose=False).meta)
    # same field 20 min later under a different telescope OBSID -> joins
    e2 = tmp_path / "e2.ecsv"
    _write_epoch(e2, "71885.01", 243.930, 14.402, 1_623_001_200)
    obs_id2, reason2 = resolve_observation_id(e2, base, radius_arcmin=10.0)
    assert obs_id2 == "71883" and reason2.startswith("pointing")
    # a different field the same night keeps its own ID even with the same OBSID
    e3 = tmp_path / "e3.ecsv"
    _write_epoch(e3, "71883.00", 204.282, 14.465, 1_623_001_500)
    obs_id3, reason3 = resolve_observation_id(e3, base, radius_arcmin=10.0)
    assert obs_id3 == "71883b" and "another pointing" in reason3
    # ... and the same field a week later is a new observation
    e4 = tmp_path / "e4.ecsv"
    _write_epoch(e4, "80000.00", 243.918, 14.399, 1_623_600_000)
    assert resolve_observation_id(e4, base, radius_arcmin=10.0, max_gap_hours=12.0)[0] == "80000"
    # grouping off -> raw OBSID always
    assert resolve_observation_id(e2, base)[0] == "71885"


def test_record_pointing_keeps_running_mean_and_last_time(tmp_path):
    import json
    from pyrt_transient.io.observation_store import ObservationStore, POINTING_FILE
    store = ObservationStore(tmp_path, "x")
    store.record_pointing({"CTRRA": 10.0, "CTRDEC": 20.0, "CTIME": 100.0, "EXPTIME": 10.0})
    store.record_pointing({"CTRRA": 10.2, "CTRDEC": 20.0, "CTIME": 400.0, "EXPTIME": 10.0})
    info = json.loads((store.obs_dir / POINTING_FILE).read_text())
    assert info["n_epochs"] == 2 and info["ra"] == pytest.approx(10.1)
    assert info["first_mid_time"] == 105.0 and info["last_mid_time"] == 405.0
    store.record_pointing({"CTIME": 500.0})  # no pointing: ignored
    assert json.loads((store.obs_dir / POINTING_FILE).read_text())["n_epochs"] == 2


def test_record_pointing_averages_ra_across_the_wrap(tmp_path):
    """A field straddling RA=0 used to average to ~180 deg -- a stored centre
    half the sky from the field, after which every later epoch failed
    resolve_observation_id's radius test and never joined the observation."""
    import json
    from pyrt_transient.io.observation_store import ObservationStore, POINTING_FILE
    store = ObservationStore(tmp_path, "wrap")
    store.record_pointing({"CTRRA": 359.9, "CTRDEC": 20.0, "CTIME": 100.0, "EXPTIME": 10.0})
    store.record_pointing({"CTRRA": 0.1, "CTRDEC": 20.0, "CTIME": 400.0, "EXPTIME": 10.0})
    info = json.loads((store.obs_dir / POINTING_FILE).read_text())
    assert info["n_epochs"] == 2
    assert info["ra"] == pytest.approx(0.0, abs=1e-9) or info["ra"] == pytest.approx(360.0, abs=1e-9)
    # ... and the mean stays inside the field, not 180 deg away
    from pyrt_transient.io.observation_store import _sep_arcmin
    assert _sep_arcmin(info["ra"], info["dec"], 0.0, 20.0) < 1.0


def test_resolve_observation_id_prefers_the_epochs_own_obsid(tmp_path):
    """Two OBSIDs of one field can end up as two stores (created before
    either had written a pointing.json). Later epochs must then stay with
    their own OBSID's store rather than following whichever running mean
    happens to sit a fraction of an arcminute closer."""
    from pyrt_transient.io.observation_store import ObservationStore, resolve_observation_id
    from pyrt_transient.io.ecsv import open_ecsv_file
    base = tmp_path / "work"
    # Store "500" is a hair closer to the incoming epoch than store "501".
    ObservationStore(base, "500").record_pointing(
        {"CTRRA": 100.000, "CTRDEC": 10.0, "CTIME": 1_623_000_000.0, "EXPTIME": 10.0})
    ObservationStore(base, "501").record_pointing(
        {"CTRRA": 100.010, "CTRDEC": 10.0, "CTIME": 1_623_000_000.0, "EXPTIME": 10.0})
    e = tmp_path / "e.ecsv"
    _write_epoch(e, "501.00", 100.001, 10.0, 1_623_000_600)
    obs_id, _ = resolve_observation_id(e, base, radius_arcmin=10.0)
    assert obs_id == "501"


def test_registration_and_analysis_are_tracked_separately(tmp_path):
    """An epoch whose run died mid-analysis is registered but not analysed:
    it must still be loaded by the next run (it is on disk and part of the
    observation) while NOT taking the "results already exist" shortcut."""
    store = ObservationStore(tmp_path, "72006")
    store.mark_processed("epoch1.ecsv")
    assert store.already_processed("epoch1.ecsv")
    assert not store.already_analyzed("epoch1.ecsv")
    store.mark_analyzed("epoch1.ecsv")
    assert store.already_analyzed("epoch1.ecsv")
    # marking analysed implies registration, and neither loses the other
    store.mark_analyzed("epoch2.ecsv")
    store.mark_processed("epoch3.ecsv")
    assert store.already_processed("epoch2.ecsv")
    assert store.already_analyzed("epoch1.ecsv") and store.already_analyzed("epoch2.ecsv")
    assert not store.already_analyzed("epoch3.ecsv")


def test_metadata_written_before_the_analyzed_split_still_reads(tmp_path):
    """Old stores have only 'processed_files'; everything in them had been
    analysed, so they must not all look unanalysed and re-run."""
    store = ObservationStore(tmp_path, "old")
    (store.obs_dir / "detection_metadata.json").write_text(
        json.dumps({"processed_files": ["a.ecsv", "b.ecsv"]}))
    assert store.already_analyzed("a.ecsv") and store.already_analyzed("b.ecsv")
    store.mark_processed("c.ecsv")
    assert store.already_processed("c.ecsv")
    assert not store.already_analyzed("c.ecsv")
    assert store.already_analyzed("a.ecsv"), "the seeded history must survive the first write"
