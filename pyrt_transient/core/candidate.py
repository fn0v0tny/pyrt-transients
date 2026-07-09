"""Candidate dataclass -- the common schema every detection strategy
(BlindMulticatalogStrategy today, a future subtraction-based strategy) will
emit, replacing the ad-hoc astropy.table.Table row + three separate ID
schemes (transient_id string, _get_cid, _make_id) in use today.

The three `used_*_fallback` flags correspond to three silent-fallback sites
found in the codebase:
  - used_fallback_astrometry: combine_results' sky-coordinate KDTree
    clustering fell back to pixel coordinates after an exception.
  - used_motion_fit_fallback: _compute_motion_features' linear proper-motion
    fit hit np.linalg.LinAlgError/ValueError on a degenerate case.
  - used_catalog_context_fallback: _add_catalog_context_safe's optimized
    local-statistics lookup failed and default (zero/inf) values were used
    instead.
These make previously-invisible degraded-quality states visible on the
Candidate itself instead of only in a log line.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, Optional


@dataclass
class Candidate:
    ra: float
    dec: float
    mjd: float
    mag: float
    mag_err: float
    epoch_id: int
    reference_catalog: str
    candidate_type: str
    features: Dict[str, Any] = field(default_factory=dict)
    quality_score: float = 0.0
    transient_id: Optional[str] = None

    # Silent-fallback visibility flags -- default False, set True at the
    # call site that took the degraded path.
    used_fallback_astrometry: bool = False
    used_motion_fit_fallback: bool = False
    used_catalog_context_fallback: bool = False

    def assign_id(self) -> str:
        """Assign (and return) transient_id from ra/dec, rounded to 3 decimal
        places -- matches the format already in use throughout the pipeline
        (e.g. "transient_319.718_33.850").
        """
        self.transient_id = f"transient_{self.ra:.3f}_{self.dec:.3f}"
        return self.transient_id
