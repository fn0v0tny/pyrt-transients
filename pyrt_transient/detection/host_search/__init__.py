"""Host-galaxy search for a trigger's error circle (GRB / EP / SVOM).

The catalogue cross-match that drives the rest of the GRB pipeline finds a
transient that is *new* or *brighter* than its catalogue entry. A
transient fainter than the galaxy it sits on -- a nuclear tidal disruption
event, a supernova in a bright host -- is neither: EP261006a (AT 2026aetp,
r = 20.2 host-subtracted at 5.6 h on a 17.7 mag galaxy, GCN 45831) was a
constant 'galaxy' candidate on every D50 night. This package answers a
different question for such triggers: *which galaxy in the error circle is
the host?*

Two independent pieces of evidence, both cheap:

- a prior: every catalogued galaxy inside the circle, ranked by how many
  galaxies at least that bright a random circle of that size contains
  (galaxies.rank_by_prior);
- a measurement: one PSF-matched subtraction of the error-circle region of
  the night's stack against an archival survey image, the kernel fitted on
  the field's stars, and the residual at every galaxy's nucleus compared
  with the other galaxies' residuals (stack.py, subtract.py). On EP261006a
  the host is rank 1 of 35 from the first 300 s frame, a 3 sigma outlier
  after four hours, a 28 sigma one five nights later (FUTURE_IDEAS,
  "First-night host identification").

search.maybe_run_host_search is the pipeline hook; it never raises and
writes host_candidates.ecsv / host_candidates.json into the observation
directory. templates.prefetch_template / the `prefetch` CLI fetch the
survey template ahead of the first frame (templates are the slowest
external step, 10-30 s per field, and the same one serves every night).
"""
from pyrt_transient.detection.host_search.search import maybe_run_host_search, prefetch_async  # noqa: F401
