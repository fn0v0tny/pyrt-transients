"""clustering.epoch_is_cached: a DEGRADED epoch result is kept for
degraded_epoch_retry_s before it is recomputed, so an outage lasting days
does not turn every run into a full recompute of the observation."""
import os
import tempfile
import time
from pathlib import Path

from pyrt_transient.detection.blind_multicatalog import clustering

_DEGRADED = "# %ECSV 1.0\n# ---\n# meta: !!omap\n# - {DEGRADED: SkyBoT down}\n# datatype:\n# - {name: a, datatype: int64}\na\n1\n"
_CLEAN = "# %ECSV 1.0\n# ---\n# datatype:\n# - {name: a, datatype: int64}\na\n1\n"


def test_degraded_epoch_is_kept_within_retry_interval():
    with tempfile.TemporaryDirectory() as d:
        p = Path(d) / "e_transients.ecsv"
        p.write_text(_DEGRADED)
        assert clustering.epoch_is_cached(p) is False, "default: recompute every run"
        assert clustering.epoch_is_cached(p, degraded_retry_s=0) is False
        assert clustering.epoch_is_cached(p, degraded_retry_s=3600) is True
        old = time.time() - 7200
        os.utime(p, (old, old))
        assert clustering.epoch_is_cached(p, degraded_retry_s=3600) is False


def test_clean_epoch_is_cached_regardless_of_interval():
    with tempfile.TemporaryDirectory() as d:
        p = Path(d) / "e_transients.ecsv"
        p.write_text(_CLEAN)
        assert clustering.epoch_is_cached(p) is True
        assert clustering.epoch_is_cached(p, degraded_retry_s=3600) is True
        assert clustering.epoch_is_cached(Path(d) / "missing.ecsv", degraded_retry_s=3600) is False
