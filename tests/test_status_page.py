"""tools/status_page.py on a synthetic data directory shaped like
lascaux50's transient_work, public_html and daemon log, with the RTS2
target database replaced by a dict."""
import json
import os
import sys
import time
from pathlib import Path

import pytest
from astropy.table import Table

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "tools"))
import pipeline_entry  # noqa: E402
import status_page  # noqa: E402

TARGETS = {  # tar_id -> current stars.targets row
    1057: {"type": "O", "name": "V* V1057 Cyg"},
    1129: {"type": "O", "name": "Gaia18arn <microlensing>"},
    53278: {"type": "G", "name": "GRB 260228.871 (SVOM/ECLAIRs #02)"},
}


@pytest.fixture
def targets_db(monkeypatch):
    db = dict(TARGETS)
    queried = []

    def lookup(ids):
        queried.append(set(ids))
        return {i: db[i] for i in ids if i in db}

    monkeypatch.setattr(pipeline_entry, "lookup_targets", lookup)
    return db, queried


def _obs(data, public, obs_id, target, header_obj, age_h, candidates=(), frames=3):
    d = data / f"obs_{obs_id}"
    d.mkdir()
    (d / f"2026091101{obs_id[-2:]}00-001-r-060-df.ecsv").write_text(
        "# %ECSV 1.0\n# ---\n# meta: !!omap\n"
        f"# - {{TARGET: {target}}}\n# - {{OBSID: '{obs_id}.00'}}\n# - {{OBJECT: '{header_obj}'}}\n"
        "# - {GRB_DEC: 26.04}\nNUMBER\n1\n")
    names = [f"202609110{i}0000-00{i}-r-060-df.ecsv" for i in range(1, frames + 1)]
    (d / "detection_metadata.json").write_text(json.dumps({"processed_files": names[::-1]}))
    if candidates:
        Table(rows=list(candidates), names=("ALPHA_J2000", "DELTA_J2000", "MAG_CALIB", "quality_score",
                                            "n_detections", "candidate_type")
              ).write(d / "candidates.tbl", format="ascii.ipac")
    (public / f"obs_{obs_id}").mkdir()
    (public / f"obs_{obs_id}" / "index.html").write_text("site")
    t = time.time() - age_h * 3600
    os.utime(d, (t, t))
    return d


def _setup(tmp_path):
    data, public = tmp_path / "work", tmp_path / "public"
    data.mkdir(), public.mkdir()
    _obs(data, public, "104215", 1057, "V* V1057 Cyg", age_h=0.1,
         candidates=[(311.1, 44.2, 17.1, 0.4, 3, "new")])
    _obs(data, public, "104210", 1129, "Gaia18arn", age_h=30)
    _obs(data, public, "99995", 53278, "old header name", age_h=300,
         candidates=[(237.44, 26.03, 16.2, 3.5, 12, "new"), (237.47, 26.05, 18.9, 0.6, 4, "variable")])
    log = tmp_path / "transient_daemon.log"
    log.write_text(
        "2026-09-11 20:00:00,000 - INFO: first line may be cut\n"
        "2026-09-11 20:35:05,000 - INFO: Batch 104210 [1/2]: job job_1 succeeded\n"
        "2026-09-11 20:36:05,000 - ERROR: Batch 104210 [2/2]: job job_2 failed (exit 1)\n"
        "2026-09-11 20:37:05,000 - INFO: Batch 99995 [1/1]: job job_3 succeeded\n"
        "2026-09-11 20:38:05,000 - ERROR: Batch 104215: job job_4 timed out\n"
        "2026-09-11 20:38:32,000 - INFO: Status: 1 active jobs, 0 pending (debounce), 10 completed, 2 failed\n")
    return data, public, log


def test_page_lists_latest_observations_status_and_latest_grb(tmp_path, targets_db):
    data, public, log = _setup(tmp_path)
    (data / ".running").mkdir()
    (data / ".running" / "1.json").write_text(json.dumps(
        {"pid": os.getpid(), "obs_id": "104215", "object": "V* V1057 Cyg", "grb": False,
         "frame": "x-df.ecsv", "started": time.time() - 30}))
    (data / ".running" / "2.json").write_text(json.dumps({"pid": 2 ** 22 + 12345, "obs_id": "1"}))

    status = status_page.generate(data, public, log, rows=2, title="D50 test")
    page = (public / "observations" / "index.html").read_text()

    assert not (public / "index.html").exists()
    assert [o["obs_id"] for o in status["observations"]] == ["104215", "104210"]
    by_id = {o["obs_id"]: o for o in status["observations"]}
    assert by_id["104215"]["status"] == "running"        # live marker beats the log
    assert by_id["104210"]["status"] == "failed"         # last job of the batch
    assert [r["obs_id"] for r in status["running"]] == ["104215"]   # dead pid dropped

    grb = status["latest_grb"]                            # older than the rows shown
    assert grb["obs_id"] == "99995" and grb["status"] == "done"
    assert grb["object"] == "GRB 260228.871 (SVOM/ECLAIRs #02)"   # the database name
    assert grb["candidates"] == 2 and grb["reliable"] == 1
    assert grb["top"][0]["quality_score"] == "3.5"
    assert "Gaia18arn &lt;microlensing&gt;" in page      # escaped
    assert "../obs_104210/index.html" in page and "../grb_replay_validation/" in page
    assert "1 active jobs" in page
    assert json.loads((public / "observations" / "transient_status.json").read_text()
                      )["latest_grb"]["obs_id"] == "99995"


def test_names_and_types_are_looked_up_every_run(tmp_path, targets_db):
    db, queried = targets_db
    data, public, log = _setup(tmp_path)
    status_page.generate(data, public, log)

    db[1129] = {"type": "O", "name": "Gaia18arn, renamed"}
    db[1057] = {"type": "G", "name": "EP 260911.850 trigger"}   # reassigned
    status = status_page.generate(data, public, log)

    by_id = {o["obs_id"]: o for o in status["observations"]}
    assert by_id["104210"]["object"] == "Gaia18arn, renamed"
    assert status["latest_grb"]["obs_id"] == "104215"
    cache = json.loads((data / ".status_cache.json").read_text())
    assert "grb" not in cache["obs_104210"] and cache["obs_104210"]["target"] == 1129
    assert [f["outcome"] for f in status["failures"]] == ["timed out", "failed (exit 1)"]


def test_header_names_are_used_when_the_database_is_unavailable(tmp_path, monkeypatch):
    monkeypatch.setattr(pipeline_entry, "lookup_targets", lambda ids: {})
    data, public, log = _setup(tmp_path)

    status = status_page.generate(data, public, log)

    assert status["latest_grb"] is None     # "old header name" has no GRB marker
    assert "target database unavailable" in (public / "observations" / "index.html").read_text()


def test_an_empty_host_still_gets_a_page(tmp_path, targets_db):
    status_page.generate(tmp_path / "none", tmp_path / "public", tmp_path / "no.log")

    page = (tmp_path / "public" / "observations" / "index.html").read_text()
    assert "No GRB observation found" in page and "no daemon status" in page


def test_main_collapses_overlapping_runs(tmp_path, monkeypatch):
    data, public, log = _setup(tmp_path)
    calls = []

    def fake_generate(*args, **kw):
        calls.append(1)
        if len(calls) == 1:   # another frame asks for a refresh meanwhile
            (data / ".status_page.again").touch()

    monkeypatch.setattr(status_page, "generate", fake_generate)
    status_page.main(["--data-dir", str(data), "--public-dir", str(public), "--daemon-log", str(log)])

    assert len(calls) == 2 and not (data / ".status_page.again").exists()


def test_an_observation_deleted_while_the_page_is_built_is_left_out(tmp_path, targets_db, monkeypatch):
    # lascaux50, 2026-09-11: a cleanup removed obs dirs while the page ran.
    import shutil
    data, public, log = _setup(tmp_path)
    real = status_page.header_facts

    def vanishing(obs_dir, cache):
        if obs_dir.name == "obs_104210" and obs_dir.exists():
            shutil.rmtree(obs_dir)
        return real(obs_dir, cache)

    monkeypatch.setattr(status_page, "header_facts", vanishing)
    status = status_page.generate(data, public, log)

    assert [o["obs_id"] for o in status["observations"]] == ["104215", "99995"]


def test_an_error_is_recorded_not_lost(tmp_path, monkeypatch):
    data, public, log = _setup(tmp_path)

    def broken(*args, **kw):
        raise RuntimeError("boom")

    monkeypatch.setattr(status_page, "generate", broken)
    status_page.main(["--data-dir", str(data), "--public-dir", str(public), "--daemon-log", str(log)])

    assert "boom" in (data / ".status_page.error").read_text()
