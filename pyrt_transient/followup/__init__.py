"""Candidate enrichment: optional, config-gated post-processing on a
detection strategy's output (FUTURE_IDEAS.md, "Candidate enrichment:
forced photometry and observing strategy").

Nothing here is imported by a detection strategy -- enrichment runs *on*
`run()`'s output (a candidates Table + lightcurves dict), so any strategy's
output can pass through it. `exposure.py` is the physics/model layer,
`enrichment.py` the pipeline glue.
"""

from pyrt_transient.followup.enrichment import (
    recommend_exposures,
    run_enrichment,
    write_exposure_report,
)

__all__ = ["recommend_exposures", "run_enrichment", "write_exposure_report"]
