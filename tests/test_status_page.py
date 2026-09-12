"""tools/status_page.py on a synthetic data directory shaped like
lascaux50's transient_work, public_html and daemon log, with the RTS2
target database replaced by a dict. The page's own JavaScript is run
under node when it is installed."""
import json
import os
import shutil
import subprocess
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
         candidates=[(311.1, 44.2, 17.1, float("nan"), 3, "new")])
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


def _generate(tmp_path, **kw):
    data, public, log = _setup(tmp_path)
    return data, public, log, status_page.generate(data, public, log, **kw)


def test_status_lists_latest_observations_and_latest_grb(tmp_path, targets_db):
    data, public, log = _setup(tmp_path)
    (data / ".running").mkdir()
    (data / ".running" / "1.json").write_text(json.dumps(
        {"pid": os.getpid(), "obs_id": "104215", "object": "V* V1057 Cyg", "grb": False,
         "frame": "x-df.ecsv", "started": time.time() - 30}))
    (data / ".running" / "2.json").write_text(json.dumps({"pid": 2 ** 22 + 12345, "obs_id": "1"}))

    status_page.generate(data, public, log, rows=2, title="D50 test")
    status = json.loads((public / "observations" / "transient_status.json").read_text())  # strict JSON

    assert not (public / "index.html").exists()
    assert status["title"] == "D50 test" and status["targets_db"] is True
    assert [o["obs_id"] for o in status["observations"]] == ["104215", "104210"]
    by_id = {o["obs_id"]: o for o in status["observations"]}
    assert by_id["104215"]["status"] == "running"        # live marker beats the log
    assert by_id["104210"]["status"] == "failed"         # last job of the batch
    assert by_id["104210"]["site"] == "../obs_104210/index.html"
    assert [r["obs_id"] for r in status["running"]] == ["104215"]   # dead pid dropped
    assert status["daemon"]["text"].startswith("1 active jobs")

    grb = status["latest_grb"]                            # older than the rows shown
    assert grb["obs_id"] == "99995" and grb["status"] == "done"
    assert grb["object"] == "GRB 260228.871 (SVOM/ECLAIRs #02)"   # the database name
    assert grb["candidates"] == 2 and grb["reliable"] == 1
    assert grb["top"][0]["q"] == 3.5


def test_the_page_is_a_fixed_shell_that_loads_the_json(tmp_path, targets_db):
    data, public, log, _ = _generate(tmp_path, title="D50 <test>")
    index = public / "observations" / "index.html"
    page = index.read_text()

    assert "fetch(\"transient_status.json" in page and "setInterval(load" in page
    assert "<title>D50 &lt;test&gt;</title>" in page
    assert "../grb_replay_validation/" in page
    before = index.stat().st_mtime_ns
    time.sleep(0.01)
    status_page.generate(data, public, log, title="D50 <test>")
    assert index.stat().st_mtime_ns == before            # only the JSON is rewritten


def test_names_and_types_are_looked_up_every_run(tmp_path, targets_db):
    db, queried = targets_db
    data, public, log, _ = _generate(tmp_path)

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
    _, _, _, status = _generate(tmp_path)

    assert status["targets_db"] is False
    assert status["latest_grb"] is None     # "old header name" has no GRB marker


def test_an_empty_host_still_gets_a_page(tmp_path, targets_db):
    status = status_page.generate(tmp_path / "none", tmp_path / "public", tmp_path / "no.log")

    assert (tmp_path / "public" / "observations" / "index.html").exists()
    assert status["latest_grb"] is None and status["daemon"] is None and status["observations"] == []


def test_an_observation_deleted_while_the_page_is_built_is_left_out(tmp_path, targets_db, monkeypatch):
    # lascaux50, 2026-09-11: a cleanup removed obs dirs while the page ran.
    data, public, log = _setup(tmp_path)
    real = status_page.header_facts

    def vanishing(obs_dir, cache):
        if obs_dir.name == "obs_104210" and obs_dir.exists():
            shutil.rmtree(obs_dir)
        return real(obs_dir, cache)

    monkeypatch.setattr(status_page, "header_facts", vanishing)
    status = status_page.generate(data, public, log)

    assert [o["obs_id"] for o in status["observations"]] == ["104215", "99995"]


def _main(data, public, log, *extra):
    status_page.main(["--data-dir", str(data), "--public-dir", str(public), "--daemon-log", str(log), *extra])


def test_requests_are_merged_into_one_run_per_interval(tmp_path, monkeypatch):
    data, public, log = _setup(tmp_path)
    (public / "observations").mkdir()
    (public / "observations" / "transient_status.json").write_text("{}")   # refreshed just now
    slept, calls = [], []
    monkeypatch.setattr(status_page.time, "sleep", slept.append)

    def fake_generate(*args, **kw):
        calls.append(1)
        if len(calls) == 1:   # another frame asks for a refresh meanwhile
            (data / ".status_page.again").touch()

    monkeypatch.setattr(status_page, "generate", fake_generate)
    _main(data, public, log, "--min-interval", "60")

    assert len(slept) == 2 and 55 < slept[0] <= 60      # waited out the interval, twice
    assert len(calls) == 2 and not (data / ".status_page.again").exists()


def test_a_request_while_another_run_holds_the_lock_leaves_a_flag(tmp_path, monkeypatch):
    import fcntl
    data, public, log = _setup(tmp_path)
    monkeypatch.setattr(status_page, "generate", lambda *a, **k: pytest.fail("must not run"))
    with open(data / ".status_page.lock", "w") as held:
        fcntl.flock(held, fcntl.LOCK_EX | fcntl.LOCK_NB)
        _main(data, public, log)

    assert (data / ".status_page.again").exists()


def test_an_error_is_recorded_not_lost(tmp_path, monkeypatch):
    data, public, log = _setup(tmp_path)

    def broken(*args, **kw):
        raise RuntimeError("boom")

    monkeypatch.setattr(status_page, "generate", broken)
    _main(data, public, log)

    assert "boom" in (data / ".status_page.error").read_text()


@pytest.mark.skipif(shutil.which("node") is None, reason="node not installed")
def test_the_page_script_renders_the_json_escaped(tmp_path, targets_db):
    _, public, _, _ = _generate(tmp_path, title="D50 test")
    page = (public / "observations" / "index.html").read_text()
    script = tmp_path / "page.js"
    script.write_text(page.split("<script>", 1)[1].split("</script>", 1)[0])
    status_json = public / "observations" / "transient_status.json"

    out = subprocess.run(
        ["node", "-e", "const {renderStatus} = require(process.argv[1]);"
                       "const s = JSON.parse(require('fs').readFileSync(process.argv[2]));"
                       "process.stdout.write(renderStatus(s, s.generated + 90));",
         str(script), str(status_json)],
        capture_output=True, text=True, check=True).stdout

    assert "GRB 260228.871 (SVOM/ECLAIRs #02)" in out and "3.50" in out
    assert "Gaia18arn &lt;microlensing&gt;" in out and "<microlensing>" not in out
    assert "../obs_104210/index.html" in out
    assert "(1 min ago)" in out
    assert "s-failed" in out and "Recent failures" in out


def test_a_marker_left_by_a_killed_frame_is_removed(tmp_path, targets_db):
    # The daemon kills a frame at its timeout, so the marker stays behind and
    # the observation showed as "running" until something else refreshed.
    data, public, log = _setup(tmp_path)
    (data / ".running").mkdir()
    dead = data / ".running" / "999.json"
    dead.write_text(json.dumps({"pid": 2 ** 22 + 4242, "obs_id": "104215"}))

    status = status_page.generate(data, public, log)

    assert status["running"] == [] and not dead.exists()


@pytest.mark.skipif(shutil.which("node") is None, reason="node not installed")
def test_stale_data_is_flagged_and_running_is_not_shown(tmp_path, targets_db):
    _, public, _, _ = _generate(tmp_path, title="D50 test")
    page = (public / "observations" / "index.html").read_text()
    script = tmp_path / "page.js"
    script.write_text(page.split("<script>", 1)[1].split("</script>", 1)[0])
    status_json = public / "observations" / "transient_status.json"

    out = subprocess.run(
        ["node", "-e", "const {renderStatus} = require(process.argv[1]);"
                       "const s = JSON.parse(require('fs').readFileSync(process.argv[2]));"
                       "s.running = [{obs_id: '104215', object: 'x', frame: 'f.ecsv', started: s.generated}];"
                       "process.stdout.write(renderStatus(s, s.generated + 3 * 3600));",
         str(script), str(status_json)],
        capture_output=True, text=True, check=True).stdout

    assert "This data is 3 h ago" in out and "Nothing has been processed since" in out
    assert "obs 104215 &middot; x" not in out


def _obs_with_followup(data, public, obs_id, target, age_h):
    """An observation whose candidates carry the follow-up recommendation."""
    d = _obs(data, public, obs_id, target, "GRB 260228.871 (SVOM)", age_h)
    Table(rows=[(237.44, 26.03, 16.2, 3.5, 12, "new", 42.5, 16.3),
                (237.47, 26.05, 18.9, 0.6, 4, "variable", float("nan"), float("nan"))],
          names=("ALPHA_J2000", "DELTA_J2000", "MAG_CALIB", "quality_score", "n_detections",
                 "candidate_type", "followup_exptime_s", "followup_mag")
          ).write(d / "candidates.tbl", format="ascii.ipac", overwrite=True)
    return d


def test_the_follow_up_exposure_reaches_the_page(tmp_path, targets_db):
    data, public = tmp_path / "work", tmp_path / "public"
    data.mkdir(), public.mkdir()
    _obs_with_followup(data, public, "99995", 53278, age_h=1)
    log = tmp_path / "transient_daemon.log"
    log.write_text("2026-09-11 20:38:32,000 - INFO: Status: 0 active jobs, 0 pending (debounce), 1 completed, 0 failed\n")

    status = status_page.generate(data, public, log)

    top = status["latest_grb"]["top"]
    assert top[0]["followup_exptime_s"] == "42.5" and top[0]["followup_mag"] == "16.3"
    assert top[1]["followup_exptime_s"] in ("nan", "null", "")   # not computed for this one


@pytest.mark.skipif(shutil.which("node") is None, reason="node not installed")
def test_the_page_shows_the_follow_up_column(tmp_path, targets_db):
    data, public = tmp_path / "work", tmp_path / "public"
    data.mkdir(), public.mkdir()
    _obs_with_followup(data, public, "99995", 53278, age_h=1)
    log = tmp_path / "transient_daemon.log"
    log.write_text("2026-09-11 20:38:32,000 - INFO: Status: 0 active jobs, 0 pending, 1 completed, 0 failed\n")
    status_page.generate(data, public, log)
    page = (public / "observations" / "index.html").read_text()
    script = tmp_path / "page.js"
    script.write_text(page.split("<script>", 1)[1].split("</script>", 1)[0])

    out = subprocess.run(
        ["node", "-e", "const {renderStatus} = require(process.argv[1]);"
                       "const s = JSON.parse(require('fs').readFileSync(process.argv[2]));"
                       "process.stdout.write(renderStatus(s, s.generated + 10));",
         str(script), str(public / "observations" / "transient_status.json")],
        capture_output=True, text=True, check=True).stdout

    assert "<th class='num'>follow-up</th>" in out
    assert "43 s" in out and "for mag 16.30" in out     # rounded for display
    assert "&mdash;" in out                             # the candidate without one
