"""The daemon refreshes the status page on its own, so the page does not
stand still while the telescope is idle (lascaux50, 2026-09-11: the page
showed the same data for eight hours after the last frame of the night)."""
import sys
import types
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from pyrt_transient import transient_daemon as td  # noqa: E402


class _Daemon:
    """Only what print_status touches."""

    def __init__(self, turns):
        self.running = True
        self._turns = turns
        self.active_jobs = self.processed_count = self.failed_count = 0
        self._debounce_files = {}
        self._debounce_lock = self.jobs_lock = _NullLock()
        self.refreshed = 0

    def refresh_status_page(self):
        self.refreshed += 1


class _NullLock:
    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


def _run_turns(monkeypatch, turns, refresh_s, minute=60.0):
    """print_status for `turns` one-minute turns, with a clock that jumps a minute."""
    daemon = _Daemon(turns)
    now = [1000.0]

    def sleep(_seconds):
        now[0] += minute
        daemon._turns -= 1
        if daemon._turns <= 0:
            daemon.running = False

    monkeypatch.setattr(td, "STATUS_REFRESH_S", refresh_s)
    monkeypatch.setattr(td.time, "sleep", sleep)
    monkeypatch.setattr(td.time, "time", lambda: now[0])
    td.TransientDaemon.print_status(daemon)
    return daemon.refreshed


def test_the_page_is_refreshed_on_its_own_schedule(monkeypatch):
    # Ten one-minute turns, refreshing every five: the first turn and the sixth.
    assert _run_turns(monkeypatch, turns=10, refresh_s=300) == 2


def test_every_turn_does_not_refresh(monkeypatch):
    assert _run_turns(monkeypatch, turns=3, refresh_s=3600) == 1


def test_refreshing_can_be_switched_off(monkeypatch):
    assert _run_turns(monkeypatch, turns=10, refresh_s=0) == 0


def test_the_refresh_runs_the_status_page_detached(monkeypatch, tmp_path):
    started = {}

    def fake_popen(cmd, **kw):
        started["cmd"], started["kw"] = cmd, kw
        return types.SimpleNamespace()

    monkeypatch.setattr(td.subprocess, "Popen", fake_popen)
    td.TransientDaemon.refresh_status_page(types.SimpleNamespace())

    script = Path(td.__file__).resolve().parent.parent / "tools" / "status_page.py"
    if not script.exists():
        pytest.skip("tools/status_page.py not installed next to the package")
    assert started["cmd"] == [sys.executable, str(script)]
    assert started["kw"]["start_new_session"] is True
