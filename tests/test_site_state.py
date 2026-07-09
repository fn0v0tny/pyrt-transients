"""Unit tests for web/site_state.py (rewrite.md Phase 4) -- tmp_path
fixtures, no real frontend generation needed.

Run with: python3 -m pytest tests/test_site_state.py -v
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from pyrt_transient.web.site_state import is_up_to_date, load_site_state, record_generated


def test_not_up_to_date_when_no_state_and_no_index(tmp_path):
    website_dir = tmp_path / "site"
    candidates_file = tmp_path / "candidates.tbl"
    candidates_file.write_text("v1")

    up_to_date, current_hash = is_up_to_date(website_dir, candidates_file)
    assert up_to_date is False
    assert current_hash is not None


def test_not_up_to_date_when_index_missing_even_with_matching_state(tmp_path):
    website_dir = tmp_path / "site"
    website_dir.mkdir()
    candidates_file = tmp_path / "candidates.tbl"
    candidates_file.write_text("v1")

    _, current_hash = is_up_to_date(website_dir, candidates_file)
    record_generated(website_dir, current_hash)
    # No index.html written -> still not up to date even though the hash matches.
    up_to_date, _ = is_up_to_date(website_dir, candidates_file)
    assert up_to_date is False


def test_up_to_date_after_record_generated_with_index(tmp_path):
    website_dir = tmp_path / "site"
    website_dir.mkdir()
    candidates_file = tmp_path / "candidates.tbl"
    candidates_file.write_text("v1")

    _, current_hash = is_up_to_date(website_dir, candidates_file)
    record_generated(website_dir, current_hash)
    (website_dir / "index.html").write_text("<html></html>")

    up_to_date, _ = is_up_to_date(website_dir, candidates_file)
    assert up_to_date is True

    state = load_site_state(website_dir)
    assert state["candidates_md5"] == current_hash


def test_changed_candidates_invalidates_gate(tmp_path):
    website_dir = tmp_path / "site"
    website_dir.mkdir()
    candidates_file = tmp_path / "candidates.tbl"
    candidates_file.write_text("v1")

    _, hash_v1 = is_up_to_date(website_dir, candidates_file)
    record_generated(website_dir, hash_v1)
    (website_dir / "index.html").write_text("<html></html>")
    assert is_up_to_date(website_dir, candidates_file)[0] is True

    # Candidates file changes (e.g. new epoch processed) -> gate re-opens.
    candidates_file.write_text("v2 -- different content")
    up_to_date, hash_v2 = is_up_to_date(website_dir, candidates_file)
    assert up_to_date is False
    assert hash_v2 != hash_v1


def test_record_generated_noop_when_hash_is_none(tmp_path):
    website_dir = tmp_path / "site"
    # Should not raise even though website_dir doesn't exist yet.
    record_generated(website_dir, None)
    assert load_site_state(website_dir) is None
