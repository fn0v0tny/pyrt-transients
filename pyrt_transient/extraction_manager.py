"""Field-centre bookkeeping over a set of detection tables.

This used to also carry reference-frame selection, coordinate transforms
and a plotting method, none of which worked (`reference_idx` was never
set, so every method using it raised) and none of which had a caller.
The working version of that machinery is detection/reference_frame.py's
ReferenceFrameSelector; only `.field_center` was ever used from here.
"""

from typing import List, Tuple

import numpy as np
from astropy.table import Table


class ImageExtractionManager:
    """Median field centre of a set of detection tables (CTRRA/CTRDEC, or
    CRVAL1/2 when the pyrt centre keys are absent)."""

    def __init__(self, detection_tables: List[Table]):
        self.detection_tables = detection_tables
        self.field_center = self._compute_field_center()

    def _compute_field_center(self) -> Tuple[float, float]:
        ras, decs = [], []
        for det in self.detection_tables:
            meta = getattr(det, "meta", None) or {}
            ra = meta.get("CTRRA", meta.get("CRVAL1"))
            dec = meta.get("CTRDEC", meta.get("CRVAL2"))
            if ra is None or dec is None:
                continue
            ras.append(float(ra))
            decs.append(float(dec))
        if not ras:
            raise ValueError("no detection table carries CTRRA/CTRDEC or CRVAL1/CRVAL2")
        return float(np.median(ras)), float(np.median(decs))
