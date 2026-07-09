"""Unit tests for io/observation_store.py (rewrite.md Phase 3 step 6) --
using tmp_path fixtures, no astropy/FITS involved. This is the payoff of
the extraction: fast, no real data needed.

Run with: python3 -m pytest tests/test_observation_store.py -v
"""
import json
import sys
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


def test_analysis_lock_creates_and_removes_lock_file(tmp_path):
    store = ObservationStore(tmp_path, "72006")
    lock_path = store.obs_dir / ".analysis.lock"
    assert not lock_path.exists()

    with store.analysis_lock():
        assert lock_path.exists()

    assert not lock_path.exists()


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
