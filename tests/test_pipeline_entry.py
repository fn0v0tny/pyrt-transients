"""tools/pipeline_entry.py: GRB frames keep the daemon's priority, routine
frames are niced, the target is judged by its current RTS2 database entry,
and the running marker lives exactly as long as the frame's pipeline run."""
import os
import subprocess
import sys
import types
from pathlib import Path

import pytest

TOOLS = Path(__file__).resolve().parent.parent / "tools"
sys.path.insert(0, str(TOOLS))
import pipeline_entry  # noqa: E402

HEADER = """# %ECSV 1.0
# ---
# datatype:
# - {name: NUMBER, datatype: int32, format: '%-6d'}
# meta: !!omap
# - {DATE-OBS: '2026-02-28T22:36:59.264'}
# - {TARGET: 53278}
# - {OBSID: '99995.00'}
# - {GRB_RA: 237.4411}
# - {OBJECT: 'GRB 260228.871 (SVOM/ECLAIRs #02)'}
# schema: astropy-2.0
NUMBER
1
"""


def test_header_values_are_read_without_yaml_quotes(tmp_path):
    ecsv = tmp_path / "frame-df.ecsv"
    ecsv.write_text(HEADER)

    meta = pipeline_entry.read_header(ecsv)

    assert meta["OBJECT"] == "GRB 260228.871 (SVOM/ECLAIRs #02)"
    assert meta["GRB_RA"] == "237.4411"
    assert pipeline_entry.target_id(meta) == 53278
    assert pipeline_entry.obs_id_from_meta(meta) == "99995"
    assert pipeline_entry.read_header(tmp_path / "missing.ecsv") == {}


# Rows as stars.targets had them on lascaux50, 2026-09-11.
@pytest.mark.parametrize("type_id, name, expected", [
    ("G", "GRB 260228.871 (SVOM/ECLAIRs #02)", True),
    ("G", "EP 260806.960 trigger #01709277716", True),
    ("G", "IceCube 260111.110 trigger #ICECUBE_EVENT_1768162279", True),
    ("G", "renamed to anything", True),               # the type decides
    ("O", "GRB 260310A/AT2026fgk", True),             # manual follow-up
    ("O", "GRB260226A", True),
    ("O", "EP 260321a / SN 2026gzf", True),
    ("O", "Gaia18arn (Arnica) - microlensing event", False),
    ("O", "IGR J21335+5105", False),                  # INTEGRAL X-ray binary
    ("O", "AT 2026kid", False),
])
def test_database_type_and_current_name_decide(type_id, name, expected):
    target = {"type": type_id, "name": name}
    assert pipeline_entry.is_grb({"OBJECT": "stale header name"}, target) is expected


def test_a_renamed_target_follows_the_database_not_the_header():
    header = {"OBJECT": "GRB 260101A", "GRB_RA": "10.0"}
    assert pipeline_entry.is_grb(header, {"type": "O", "name": "field 12"}) is False
    assert pipeline_entry.is_grb(header, None) is True   # no database: header decides


@pytest.mark.parametrize("meta, expected", [
    ({"OBJECT": "field 12", "GRB_RA": "10.0"}, True),
    ({"OBJECT": "GRB 260310A/AT2026fgk"}, True),
    ({"OBJECT": "V* V1057 Cyg"}, False),
    ({}, False),
])
def test_without_the_database_the_header_decides(meta, expected):
    assert pipeline_entry.is_grb(meta) is expected


def test_lookup_parses_psql_rows_and_survives_a_missing_database(monkeypatch):
    calls = []

    def fake_run(cmd, **kw):
        calls.append(cmd)
        return subprocess.CompletedProcess(cmd, 0, stdout="53278\tG\tGRB 260228.871 (SVOM)\n1129\tO\tGaia18arn\n")

    monkeypatch.setattr(pipeline_entry.subprocess, "run", fake_run)
    rows = pipeline_entry.lookup_targets([53278, 1129, None, 53278])
    assert rows == {53278: {"type": "G", "name": "GRB 260228.871 (SVOM)"},
                    1129: {"type": "O", "name": "Gaia18arn"}}
    assert "in (1129,53278)" in calls[0][-1]

    def failing_run(cmd, **kw):
        return subprocess.CompletedProcess(cmd, 2, stdout="", stderr="could not connect")

    monkeypatch.setattr(pipeline_entry.subprocess, "run", failing_run)
    assert pipeline_entry.lookup_targets([1]) == {}
    monkeypatch.setattr(pipeline_entry, "TARGETS_DB", "")
    assert pipeline_entry.lookup_targets([1]) == {}


def _priority_in_child(grb):
    env = {k: v for k, v in os.environ.items() if not k.endswith("_NUM_THREADS")}
    code = ("import os, pipeline_entry as e; e.apply_priority(%r); "
            "print(os.getpriority(os.PRIO_PROCESS, 0), os.environ['OPENBLAS_NUM_THREADS'])" % grb)
    out = subprocess.run([sys.executable, "-c", code], cwd=TOOLS, env=env,
                         capture_output=True, text=True, check=True).stdout.split()
    return int(out[0]), out[1]


def test_routine_frames_are_niced_and_grb_frames_are_not():
    base = os.getpriority(os.PRIO_PROCESS, 0)

    assert _priority_in_child(False) == (max(base, 10), "2")
    assert _priority_in_child(True) == (base, "4")


def test_the_running_marker_lasts_as_long_as_the_pipeline(tmp_path, monkeypatch):
    ecsv = tmp_path / "frame-df.ecsv"
    ecsv.write_text(HEADER)
    decided = []
    monkeypatch.setattr(pipeline_entry, "DATA_DIR", tmp_path)
    monkeypatch.setattr(pipeline_entry, "lookup_targets",
                        lambda ids: {53278: {"type": "G", "name": "GRB 260228.871 (current name)"}})
    monkeypatch.setattr(pipeline_entry, "apply_priority", decided.append)
    monkeypatch.setattr(pipeline_entry, "refresh_status_page", lambda: None)
    monkeypatch.setattr(sys, "argv", ["pipeline_magic.py", str(ecsv), "frame.fits"])
    markers = []

    def fake_pipeline():
        markers.extend(p.read_text() for p in (tmp_path / ".running").glob("*.json"))
        raise SystemExit(1)   # the pipeline exits on errors

    monkeypatch.setitem(sys.modules, "pyrt_transient.pipeline_magic",
                        types.SimpleNamespace(main=fake_pipeline))
    with pytest.raises(SystemExit):
        pipeline_entry.main()

    assert decided == [True]
    assert len(markers) == 1
    assert '"obs_id": "99995"' in markers[0] and "current name" in markers[0]
    assert list((tmp_path / ".running").glob("*.json")) == []
