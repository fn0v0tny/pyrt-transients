# Future ideas / known gaps

Things intentionally deferred, evaluated-but-not-adopted, or discovered
along the way and not yet acted on. Grouped by theme rather than by when
they came up.

## Candidate enrichment: forced photometry and observing strategy

Two features worth adding as optional, config-gated post-processing steps
on `BlindMulticatalogStrategy.run()`'s output — not hardcoded into the
strategy itself, since enrichment should be something any detection
strategy's output can pass through:

1. **Forced photometry from ATLAS and PanSTARRS** — pull historical
   lightcurves for each candidate position from PS1 DR2 and ATLAS forced
   photometry to help confirm or reject it (e.g. drop candidates with a
   long PS1 detection history unless newly brighter). Should use
   `Candidate.transient_id`/`Candidate.assign_id()` as its lookup key so
   it shares one ID scheme with the rest of the pipeline.
   `frontend_generator.py`'s `_load_forced_lightcurves()` already reads a
   `lightcurves.json` if one is present in the observation directory, so
   the frontend picks this up automatically once something writes that
   file.
2. **Telescope strategy suggester** — annotate each candidate with a
   recommended follow-up observing strategy (exposure time, filter,
   EMCCD on/off) based on its previous-epoch magnitude and time since
   the GRB trigger. Should make the trigger-time-vs-first-observation
   fallback (when `grb_t0` isn't configured) an explicit, visible flag on
   the output rather than a silent substitution.

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

Root cause confirmed: the real telescope's own `OBSID` metadata field changes
per filter/observing-block within a single night, not per GRB (e.g. one
GRB's z-band epochs carried `OBSID=71883.00`, then its i-band epochs carried
`OBSID=71885.01`) — and that block-scoped ID can coincidentally collide with
a *different* GRB's assigned slot. `extract_observation_id()` was never
wrong to read it; the real data just doesn't identify "this GRB's campaign"
the way a replay across many separate epochs needs it to.

**Fixed for the replay** by forcing each GRB's own name (e.g. `"GRB210610A"`)
into every epoch's ECSV `OBSID` field before running detection, guaranteeing
one unambiguous ID per GRB with zero collision risk across the whole batch.
This is a replay-side workaround, not a `pyrt_transient` code change — a real
`replay_driver.py` (above) should build this in as a first-class step
(inject/override the observation_id explicitly per GRB) rather than trusting
per-epoch metadata derivation.

## Significance-gate and blending false negatives (found via GCN cross-check)

Cross-checking the 18-GRB replay against real GCN circular photometry (T0,
reported magnitude, time since trigger) turned up three genuinely real,
well-positioned, GCN-confirmed afterglows that never became reported
candidates, for three different concrete reasons — worth tuning/investigating
rather than accepting as expected misses:

1. **A strict significance gate excludes real, positionally-correct
   detections.** `catalog.py:_process_detections_for_candidates` has
   `bad_snr = det_errs >= (1.091 / siglim)` — with the default `siglim=5.0`,
   anything with `MAGERR_CALIB >= 0.218 mag` (roughly S/N < 5) is excluded
   before it can even be considered a candidate. Two GRBs (151027B, and
   210312B in most epochs) had a detection sitting within a few arcsec of
   the catalogued position, with a calibrated magnitude matching GCN's
   independently-reported brightness closely — but `MAGERR_CALIB` was above
   this threshold in nearly every epoch, so the real source never reached
   the candidate stage at all. Worth revisiting whether `siglim=5.0` is too
   strict a default, or whether marginal (3-5σ) detections should be kept
   with a lower quality-score weight instead of hard-excluded outright.
2. **Blending can suppress the magnitude-change check.** GRB200410A's
   detections pass the SNR gate fine (`MAGERR_CALIB` 0.10-0.22 mag,
   consistently) but never become candidates either — every epoch shows two
   very close (~1″) detections with `FLAGS` bits indicating a blend. Likely
   explanation: the pipeline sees a persistent, unchanged-brightness point
   source at a known catalog position (the blended pair read as "this star,
   same as always") and correctly-by-its-own-logic never flags it as new or
   changed, since the added GRB flux is folded into the blend rather than
   standing out. Worth considering a specific check for "blended flag +
   still brighter than expected" rather than treating any blended detection
   as automatically uninteresting.
3. **A recurring `ERRX2_IMAGE = ERRY2_IMAGE = 0.0` anomaly.** Several
   otherwise-real detections (151027B, 210312B in most epochs, 240414A in
   all epochs) have centroid position errors that are exactly zero — real
   SExtractor centroid uncertainties are essentially never exactly 0. This
   doesn't appear to be the direct cause of any of the three misses above
   (traced through `core/radii.py`'s handling: it gets clamped to the
   minimum allowed radius, not an outright failure), but it's a systematic
   data-quality signature worth investigating in `pyrt`'s own aperture-
   photometry step, since it may also be quietly inflating `MAGERR_CALIB`
   for the same detections that then fail the significance gate above.

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
