# Future ideas / known gaps

Things intentionally deferred, evaluated-but-not-adopted, or discovered
along the way and not yet acted on. Grouped by theme rather than by when
they came up.

## Dead code to remove

- **`OptimizedTransientAnalyzer` / `OptimizedMultiDetectionAnalyzer`**
  (`transient_analyser.py`) — fully superseded by
  `detection/blind_multicatalog/` + `BlindMulticatalogStrategy`. Confirmed
  via grep: nothing in the live pipeline calls them anymore, only their own
  definitions and the package-level re-export in `__init__.py` reference
  them. Kept for now as a backward-compatible import path, not because
  anything needs them.
- **`forced_photometry.py`** — confirmed not called anywhere in the real
  candidate pipeline (`pipeline_magic.py` never invokes it). See "Forced
  photometry integration" below for what finishing this properly would look
  like, as opposed to just deleting it.
- **`_add_strategy_fields`'s `strategy_v2` import** — `from strategy_v2
  import determine_grb_strategy` has never resolved; `strategy_v2` doesn't
  exist anywhere in this package, only as a stray file in an old pre-split
  scratch directory outside it. Every `strategy_*` column in real output is
  always `None` as a result. This has never actually worked.
- **`frontend_generator.py`'s forced-photometry merge** (~line 663) — `from
  forced_photometry import _get_cid` is missing the `pyrt_transient.`
  prefix, silently caught by a bare `except Exception: lc_key =
  candidate_id`, so the "merge forced photometry lightcurves" frontend
  feature always falls back and never uses its intended key.

None of these are urgent — they're inert, not broken (nothing depends on
their currently-broken behavior) — but they're real feature work waiting to
be picked up, not just a cleanup pass:
1. Port `strategy_v2.py` properly into an `enrichment/` module, deciding
   explicitly what to do when `grb_t0` isn't configured (make the
   trigger-time-vs-first-observation fallback an explicit, visible flag
   rather than a silent substitution).
2. Fix `forced_photometry.py`'s `_get_cid`/`_make_id` to use
   `Candidate.transient_id`/`Candidate.assign_id()` instead of yet another ID
   scheme, and wire `filter_ps1_known_sources` into
   `BlindMulticatalogStrategy.run()`'s output as an optional, config-gated
   post-processing step — not hardcoded into the strategy itself, since
   enrichment should be something any strategy's output can pass through.
3. Fix `frontend_generator.py`'s broken import once (2) lands.

## GRB replay driver (proper version)

A real batch-replay pass across many GRBs (18, in practice — see below) was
run manually via ad-hoc shell scripts during initial validation, not the
clean reusable module the architecture wants:

- `replay_driver.py` — for each GRB, for `k` in `1..len(epochs)`, call
  `BlindMulticatalogStrategy().run(epochs[:k], config)`, apply
  `select_by_grb_prior`, write
  `data/results/<grb_id>/epochs/epoch_<k>/snapshot.json`. This is what the
  ad-hoc "latency pass" scripts approximated by hand (snapshotting
  `candidates.tbl` after each epoch to find how many images were needed
  before the known transient first appeared as a reliable candidate) — a
  real `replay_driver.py` would make this a first-class, reusable
  capability instead of a one-off shell script.
- `core/prior.py` — `select_by_grb_prior(candidates, prior_ra, prior_dec,
  prior_radius_arcsec)`, operating on `list[Candidate]`.
- Once both exist: confirm replay output for all known GRBs matches each
  one's real-time-pipeline final candidate list at `k=len(epochs)` — a
  second, independent confirmation of behavior preservation, via a
  different code path than the regression-baseline harness.

### Observation-ID fragmentation (found during the 18-GRB replay, real risk)

`extract_observation_id()` derives the ID from each epoch's own ECSV field
metadata, not from any external identity. Re-deriving astrometry fresh for
several GRBs revealed that this can assign **different observation IDs to
different epochs of the same physical field** — e.g. one GRB's 20 epochs
split across three different IDs, and two GRBs observed the same night with
adjacent field numbers ending up sharing an ID for part of their data. Since
multi-epoch confidence-building requires all of a field's epochs to
accumulate under one `ObservationStore`, this silently fragments a single
real observation into several 1-2-epoch stubs that never accumulate enough
evidence — not a crash, just quietly worse detection recall.

Worth investigating: whether this is specific to re-deriving astrometry
fresh (vs. the original real-time pipeline, which apparently stays on one
ID all night — possibly because it's driven by a stable per-night session
identifier upstream that a from-scratch replay doesn't have access to), and
if so, whether the replay driver above needs its own explicit
GRB-to-observation-ID mapping instead of trusting fresh field-metadata
derivation epoch-by-epoch.

## Deferred stdpipe adoption

- **`stdpipe.artefacts.filter_sextractor_detections`** — an unsupervised
  IsolationForest pre-filter over `FLUX_RADIUS`/`FWHM`/`FLUX_MAX`-to-
  `FLUX_AUTO` ratio with spatial detrending, no training data needed.
  Evaluated side-by-side against a real fixture: re-running SExtractor with
  its own `.sex`/`.param` file (the columns it needs — `FLUX_RADIUS`,
  `FLUX_MAX`, `FLUX_AUTO` — aren't in the production detection ECSV; pyrt's
  own SExtractor invocation requests a fixed 11-column list that excludes
  them, a config choice, not a SExtractor limitation) gave 2387 raw
  detections, 349 (14.6%) flagged as likely artefacts; cross-matched
  against a real candidate list, 10/148 (~6.8%) of real candidates would be
  flagged too. The known GRB afterglow position survived correctly
  (0.41″ away, classified "good"). Worth adding as a pre-filter stage ahead
  of `core/scoring.py`, but needs either (a) requesting the extra SExtractor
  params in `phcat.py`'s config (a production behavior change in `pyrt`
  itself, needing its own review) or (b) a second, parallel SExtractor pass
  with the fuller param list just for this filter (extra compute cost per
  epoch).
- **`stdpipe.realbogus`/`realbogus_features`** — supervised real/bogus
  classification using cutout-image morphology, not just catalog features.
  Needs labeled real/bogus training data; a multi-GRB replay (confirmed
  transients plus confirmed artifacts across many epochs) is exactly what
  accumulates that labeled set. A first real replay across 18 GRBs has now
  been run manually (see above) — revisit this once the replay driver is a
  real reusable module and can accumulate that data systematically.
- **`stdpipe.simulation`** — `add_hot_pixels`, `create_satellite_trail`,
  `add_cosmic_rays`, `generate_realbogus_training_data`. Useful for richer
  synthetic test fixtures than hand-built tables. Adopt opportunistically
  whenever a unit test needs a more realistic fixture — not a prerequisite
  for anything above.

## Radius-computation unification

`clustering.py`'s `compute_per_detection_radius` was not switched to
`core/radii.py`'s unified `compute_adaptive_radius` when the KDTree
replacements landed — only the match-mechanism swap was in scope at the
time. Wiring it in is a deliberate behavior change beyond "replace the match
mechanism" (same category of decision as the original sky/pixel radius
unification), not a mechanical follow-up.

## Tuning API

- `core/scoring.py`'s `compute_quality_score` is already the single source
  of truth for the quality-score formula — a tuning API would import it
  directly rather than reimplementing anything.
- Cache raw `Candidate.features` per GRB per epoch as the tuning API's
  input, generated as a byproduct of a real replay-driver run rather than a
  separate feature-extraction pass.

## New detection strategy: image subtraction

Build a `detection/subtraction/` package implementing the same
`DetectionStrategy` contract (`run(detection_tables, config) -> ...`) as
`BlindMulticatalogStrategy`, so it's a genuinely swappable alternative, not
a special case elsewhere in the pipeline:

- `SubtractionStrategy(DetectionStrategy)` plus the actual differencing,
  using `detection/reference_frame.py`'s `ReferenceFrameSelector` for
  template selection.
- Decide explicitly between two reference-template strategies rather than
  defaulting to one: **own-epoch template** (pick the best of your own
  prior epochs — zero external dependency, but needs enough prior epochs to
  have a clean one) vs. **external-survey template**
  (`stdpipe.templates.get_ps1_image_and_mask` /
  `get_ls_image_and_mask` — works on the very first observation of a field,
  but depends on survey coverage and network access). Worth supporting both
  behind a config flag rather than picking one permanently.
- Score via the same `core/scoring.py`, or a subtraction-specific scoring
  function if the feature set genuinely differs — but still emitting
  something with `quality_score` populated on the same schema, so
  `pipeline_magic.py`, the frontend, and any tuning API don't need to know
  which strategy produced a given candidate. This also finally lets
  `frontend_generator.py`'s `sn_score`-vs-`quality_score` `elif` collapse
  into a single `quality_score` read.

## `frontend_generator.py` split

Lowest priority, fully independent of everything else. Currently one large
file; natural seams to split along:
- `web/disk_housekeeping.py` — directory size, old-file cleanup, disk-space
  checks.
- `web/atomic_sync.py` — the manifest-based lightcurve sync logic
  (hardlink/symlink/copy fallback chain).
- `web/cutout_rendering.py` — cutout image generation, WCS fallbacks.
- `web/page_templating.py` — `index.html`/`candidates.json`/`info.json`
  generation.
- `FrontendGenerator` itself becomes a thin coordinator over these four,
  the same shape `BlindMulticatalogStrategy` is over its own submodules.

## Real-time throughput headroom

Measured on real production hardware during the 18-GRB replay: combined
astrometry + detection time per image, run single-threaded/serially, is
roughly 20-30s. Real image arrival cadence varies a lot by field/exposure
setup — some fields arrive every ~11-13s, others every ~2 minutes+. For the
fastest-cadence fields, a single worker alone would fall behind over a
sustained run. In practice the daemon absorbs this via
`MAX_PARALLEL_PROCESSES` concurrent workers — but that only helps when
*different* observations are in flight at once: `ObservationStore`'s file
lock serializes concurrent epochs of the *same* observation by design, so
parallelism doesn't speed up a single fast-cadence field's own throughput.
If a single field's cadence is fast enough to matter on its own, the actual
lever is speeding up per-epoch detection itself (catalog-query caching,
`compute_quality_score` cost), not adding workers.

## Deployment note (not a `pyrt_transient` bug, but easy to lose)

`pyrt`'s IRAF-based aperture-photometry step (`phcat.py:call_iraf`) requires
the `TERM` environment variable to be set — IRAF's `cl` aborts with `ERROR:
Environment variable 'TERM' not found` otherwise. Interactive shells have
this set already; non-interactive automation (cron, one-off SSH commands,
some daemon-launch contexts) may not. `login.cl`/`uparm` self-initialize
fine on first run and don't need any other manual setup.
