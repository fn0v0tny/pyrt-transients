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

   **Exposure time implemented** (`followup/exposure.py` +
   `followup/enrichment.py`, config-gated on `FollowupConfig`, wired into
   `pipeline_magic.py` only): a port of the standalone
   `new_exposure_calculator2.py` empirical noise model, driven by the meta
   of the most recent real epoch (`EXPTIME`/`BGSIGMA`/`MAGZERO`/`FWHM`/
   `GAIN`). Every candidate row gains `followup_mag` +
   `followup_exptime_s`; the top-scoring one gets a `followup_exposure.json`
   report. Per this section's own framing it runs on `run()`'s output, not
   inside any strategy. Note the dead `strategy_*` columns visible in old
   `candidates.tbl` outputs (`strategy_exp_s`, `strategy_snr`,
   `strategy_filters`, `strategy_emccd`, `strategy_time_since_trigger_s`,
   ..., all `None`) are the never-resolved `strategy_v2` attempt behind
   `add_strategy_fields_fn` — deliberately not reused, since only the
   exposure half is real now and half-filled legacy names would be worse
   than new ones. Nothing in the frontend reads them (checked
   `template/app.js`/`index.html`; only the captured sample
   `template/candidates.json` carries the nulls).

   **Still deferred**: filter choice, EMCCD on/off, and the time-since-
   trigger term (with its explicit `grb_t0`-fallback flag). These extend
   `followup/`, not the dead hook. Also not wired into
   `pipeline_magic_sn.py`: `recommend_exposures` is strategy-agnostic
   (it takes a candidates table, a lightcurves dict and one reference
   epoch), but a *diff* epoch is the wrong reference — its meta is borrowed
   from the science frame and its background isn't a single exposure's, so
   the SN side needs the underlying science epoch passed deliberately
   rather than whatever `SubtractionStrategy` happened to consume.

   **Reviewed after landing** (design review + a partial automated pass
   that ran out of budget before verifying anything; its unverified leads
   were checked by hand). Fixed: the reference epoch was picked by list
   position, but `ObservationStore.load_existing_tables()` globs the
   directory, so after the lock re-load "last" was filesystem order — now
   chosen by mid-exposure `CTIME` (the same time `core/epochs.py` stamps
   as `obs_time`); enrichment ran inside the pipeline's `try` *before*
   `save_results` with only `ExposureModelError` caught, so any other
   exception would have discarded the detection results — now
   `run_enrichment` never raises; masked `MAG_CALIB`/`quality_score` cells
   are filled to NaN; invalid `target_snr`/bracket is an `error` report,
   not a solver crash. Design changes: the planning magnitude is the
   latest lightcurve point only when it is well measured
   (`max_planning_magerr`), else an inverse-variance mean of the last few,
   with the magnitude error propagated into an `exptime_range_s`; only
   points from epochs in the reference frame's `FILTER` are used when any
   exist, else `filter_mismatch` is flagged; and the model is checked
   against the reference frame's own measured errors on every run
   (`model_check`, warns above 0.1 dex rms) — the instrument-specific
   constants are otherwise invisible. Still open: no saturation/bright
   limit (flagged in the report as "SNR is not the binding constraint"
   when clamped at `min_exptime_s`), no fading extrapolation, and the
   dead `add_strategy_fields_fn` hook is still threaded through
   `clustering.py`/`lightcurve.py`/both strategies.

   **Two real bugs found in the standalone calculator while porting**, both
   measured against this repo's real fixtures rather than reasoned about
   (asserted in `tests/test_followup_exposure.py`; the same fixes were
   applied to the standalone script in `~/Downloads` afterwards, which
   also gained a `--selftest` that runs this check against a frame's own
   photometry):
   - *Mismatched zero points.* `break_magnitude` carries `+ZERO` (10) but
     `calculate_exptime`/`predict_performance` passed a bare
     `magnitude - magzero`, so the two halves of the formula sat 10 mag
     apart. Against every measured source in `tests/210619B`, the model
     with the offset reproduces real `MAGERR_CALIB` at median +0.01 dex /
     rms 0.031 (matching its own quoted 0.047); without it, median
     −2.53 dex — errors ~330x too small. Visible symptom: asking for SNR 10
     on mag 18.5 against a real 10s frame (`MAGLIM` 17.08) answered
     "0.0 seconds" (really ~7e-4 s, and `fsolve` "verified" it, since the
     residual really does have a root there).
   - *Non-inverse sky/BGSIGMA conversion.* `sky_brightness_from_bgsigma`
     subtracts `GAIN**2 * RN**2` from an already-gain-scaled variance while
     `bgsigma_from_sky_brightness` returns the electron-side sigma, so
     round-tripping a header `BGSIGMA` at its own `EXPTIME` gives
     `GAIN*BGSIGMA` (25.78 → 20.88 on a real frame), biasing predicted
     errors by −0.06 dex. The port converts variance to electrons,
     subtracts the (already-electron) readout term there, and converts
     back, so `bgsigma_at(EXPTIME_ref) == BGSIGMA` exactly.

   Two smaller deviations, same rationale: `GAIN` is read from the frame's
   own header (real pyrt ECSV meta has it) instead of the hardcoded 0.81,
   and the solver is a bracketed `brentq` over `[min_exptime, max_exptime]`
   rather than `fsolve` from a 300s guess — "not reachable within
   `max_exptime_s`" is then a reportable answer instead of an absurd
   number. A frame whose sky term comes out non-physical (a co-add or
   rescaled image: `tests/2026kid`'s real frames all do, `BGSIGMA` 2.59 ADU
   over a 2280s effective exposure) raises `ExposureModelError` with the
   reason rather than returning a number or a bare `None`.

## Whole-project review (2026-09-03)

Three review passes now exist and they cover different things. A design
review of `followup/` (fixes applied). An automated diff review of the
working tree against `b8564a9` (stacking rework, catalog blending change,
`validation/`, `tools/`, stdpipe patches, `followup/`) that produced ten
verified findings. And a read of every file in `pyrt_transient/` (~15k
lines), `tools/` and the tests. Not reviewed by any pass: `template/app.js`
and `index.html`, the bundled `fotfit.py`/`termfit.py`, `paper/`.

Everything below was found by one of those passes. **Fixed** items were
fixed the same day (tests in `tests/test_review_fixes.py`,
`test_observation_store.py`, `test_catalog_blending.py`, `test_stacking.py`);
**open** items are recorded here so the next person doesn't re-derive them.

### Fixed

- **Lock file was unlinked on release, defeating the lock**
  (`io/observation_store.py`, `AnalysisLock.__exit__`). `flock()` locks an
  inode, not a path: A unlinks on release while B is blocked on the old
  inode, C opens a fresh inode and acquires immediately, B and C re-cluster
  the same observation concurrently. The file is now never unlinked.
- **Colour corrections were silently off.** `transients.py` did a bare
  `import fotfit`, and `fotfit.py` a bare `import termfit`; neither resolves
  in an installed/console-script context, so `fotfit` was `None` and
  `simple_color_model` returned the raw catalogue r magnitude whenever
  `RESPONSE` carried a colour term. The fixtures only carry spatial terms
  (`PX`, `P3R`, ...) so the frozen baseline never saw it. Both import the
  bundled copies now; a test asserts `PC=0.5` moves a g−r=1 star by 0.5 mag.
- **Epoch order was filesystem order.** `load_existing_tables()` globbed,
  so `detection_tables[0]`/`[-1]` were arbitrary on every run after the
  first: the catalog query box, the follow-up reference epoch, the SN
  pipeline's SkyBoT/PM epoch (`pipeline_magic_sn.py`, `ref_meta`) and the
  stack's most-recent-N selection all read an arbitrary epoch. Tables are
  now sorted by mid-exposure `CTIME` (`epoch_sort_key`), undated ones
  first; `pipeline_magic.py` also drops any `IS_STACK` table *before*
  reading `[0]` (a stack ECSV has `FIELD=0.0`, which collapsed the query
  box to zero width).
- **`mark_processed` truncated a shared `.tmp` before a non-blocking lock**
  and did its read-modify-write outside the analysis lock, so two runs
  could install an empty metadata file or lose each other's entry. Now a
  blocking flock on `.metadata.lock` (never unlinked) around the whole
  read-modify-write, per-process temp file, atomic rename.
- **Every real stack build failed** (`detection/stacking.py`): `-o
  stack.fits.tmp` goes to Montage `mAdd`, which re-appends `.fits`, so
  pyrt-combine's own header update couldn't find its output. Temp name is
  `stack.tmp.fits`; the `_area.fits` companion and any partial output are
  cleaned up on every failure path (frontend globs `*.fits` as epochs).
  A rebuild also used to replace `stack.fits` as soon as pyrt-combine
  succeeded, so a `build_stack_ecsv` failure left a new N-epoch image paired
  with the old M-epoch catalogue; both are now built under `stack.new.*`
  and installed together only when both exist.
- **Blended detections flagged every unresolved catalogue pair as
  "brightening"** (`catalog.py`, `_check_magnitude_changes_cached`): the
  combined flux was compared against each match individually, and a pair is
  always brighter than its fainter member. A blended detection is now judged
  once, against the summed flux of all its matches (strict `siglim` either
  way, then the relaxed `new_source_siglim` bar for a brightening excess).
  The admission gate uses `min(new_source_siglim, siglim)` so a config with
  the former higher can't make blends stricter than unblended sources.
- **Daemon dropped queued images on shutdown** (`transient_daemon.py`):
  the signal handler said "fire pending debounce timers" and cancelled them.
  Pending batches are now run before the executor shuts down.
- **Core pipeline required the optional `[frontend]` extra**:
  `extraction_manager.py` and `blind_multicatalog/plotting.py` imported
  `matplotlib.pyplot` at module level. Now lazy; plots are skipped with a
  warning when it's absent.
- **A VizieR outage at the last step discarded the run**: `apply_vsx_filter`
  in `clustering.combine_with_lightcurves` was unguarded and the caller
  exits before `save_results`. Now logged, candidates kept unfiltered.
- `ObservationStore` no longer falls back to the shared `base_dir` when
  `obs_<id>` can't be created (it pooled every observation into one
  directory); frontend failures log at ERROR with traceback instead of INFO;
  `unix_to_mjd` returns NaN instead of the raw unix time on failure.

### Fixed in the second pass

- **`mark_processed()` now runs after `save_results`** in `pipeline_magic.py`,
  so a run that fails mid-analysis gets the epoch re-processed next time
  instead of skipped as "results should already exist".
- **INI config loader is driven by `dataclasses.fields()`** (`from_file`/
  `to_file` in `config_trans.py`): every field of every section round-trips
  (test mutates all of them); lists accept comma-separated or JSON form,
  bad values are logged and ignored. The pyrt leftovers `parse_arguments`/
  `load_config`/`DEFAULT_CONFIG_FILE` are gone.
- **Frontend cleanup keys on `st_mtime`** (not `st_atime`, which
  `relatime`/`noatime` mounts don't maintain) and deletes oldest files only
  until back under budget, instead of wiping all but 50 cutouts.
- **`transients.py` legacy CLI deleted.** `open_ecsv_file` lives in
  `io/ecsv.py` (logs instead of printing; no bare `except`),
  `simple_color_model` in `core/color_model.py` (with `parse_response` and
  `has_colour_terms`); `transients.py` is a two-line re-export shim.
- **Non-r band without colour terms now warns** once per epoch
  (`catalog_match._warn_if_band_uncorrected`): the catalogue comparison is
  against Sloan r, so an i/z frame whose `RESPONSE` has no `C/D/E/F` term is
  compared uncorrected. A warning, not a fix -- the fix is a `RESPONSE` with
  colour terms from the photometry step.
- **Duplicate SkyBoT in the SN pipeline**: Step 5 is skipped when
  `vsx_filter_enabled` (already applied per epoch, at each epoch's own JD,
  in `combine_results`); the standalone `reject_known_asteroids` remains
  for the disabled case.
- **Dead code removed**: `add_strategy_fields_fn` (four files),
  `ImageExtractionManager`'s never-working `reference_idx` methods and
  plotting (only `.field_center` survives, and it raises instead of
  returning NaN when no table has a centre).
- **Daemon**: paths, pipeline command, extra pipeline args, parallelism and
  debounce come from `PYRT_TRANSIENT_*` environment variables; the pipeline
  command defaults to the installed `pyrt-transient-pipeline` entry point;
  requests are read until EOF instead of one `recv(4096)`; the debounce key
  uses the pipeline's own `extract_observation_id`; counters are updated
  under the lock.
- **`sep-x` dropped from `pyproject.toml`** (stdpipe declares it itself).
- **CI**: `.github/workflows/tests.yml` runs the unit tests on push/PR;
  `tools/check_baseline.py` (needs Gaia/VizieR) is a `workflow_dispatch`
  option. Unverified in GitHub Actions itself -- the `pyrt` install from
  GitHub is the step most likely to need adjusting.

### Open

- **`quality_score` ∝ `mag_range²`** (`core/scoring.py`,
  `variability_factor × mag_range_factor`). For a "new" source the range is
  photometric scatter, so the score rewards noise: 0.5 mag scatter scores
  25× a steady 0.1 mag source. `new_source_variability_floor` exists but
  defaults off. Same lever as the GCN recall work, pulled from the other
  side -- a science decision (it changes the frozen baseline), not a patch.
- **Catalogue comparison is always against Sloan r**
  (`catalog.py`, `magnitudes[idx, 1]`); now warned about (above), not fixed.
- Not reviewed at all: `template/app.js`, `index.html`, `paper/`.

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

## Detection recall: false-negative mechanisms found via GCN cross-check

Cross-checking all 18 replayed GRBs against real GCN circular photometry
(T0, reported magnitude, time since trigger) turned up **seven** fields with
a genuinely real, well-positioned, GCN-confirmed afterglow present in our
own raw detection catalogs that never became a reported candidate. One of
those seven (GRB220403B) has since been recovered as a genuine 5th
detection by extending the replay window rather than changing any
threshold — see below. Of the remaining six, three (GRB151027B, GRB211024B,
GRB210410A) turned out on direct measurement to be genuinely below this
system's single-exposure `MAGLIM` depth — a physical limit, not a pipeline
bug — leaving three still-open, threshold-fixable mechanisms: the
significance gate (GRB210312B), blending (GRB200410A), and a compounding
astrometry failure (GRB180325A).

### Implemented

1. **Split the significance gate by whether a detection has a catalog
   match.** `catalog.py:_process_detections_for_candidates` had a single
   `bad_snr = det_errs >= (1.091 / siglim)` gate (`siglim=5.0`, i.e.
   `MAGERR_CALIB >= 0.218 mag` excluded outright) applied identically
   whether or not the detection matched a known catalog source. Matched
   detections keep that strict bar (avoids false "brightening" flags from
   ordinary photometric scatter on catalogued stars), but genuinely *new*
   detections (no catalog match at all — the case a real, previously
   uncatalogued GRB afterglow falls into) now use a separate, much more
   permissive `DetectionConfig.new_source_siglim` (default `1.5`, i.e.
   `MAGERR_CALIB >= 0.727`). This alone doesn't over-admit noise: a source
   repeating consistently across many epochs earns real confidence via (2)
   below; isolated noise doesn't repeat at the same position.
2. **`compute_lightcurve_score_factor` now has an epoch-consistency
   confidence term.** Previously `n_detections`/`n_epochs` had *zero*
   influence on `quality_score` — they were purely a hard pass/fail gate
   (`min_n_detections`) in `clustering.py`, so a candidate right at the
   threshold scored identically to one confirmed 4x more often. Added a
   root-sum-square-style term (`sqrt(n_detections / min_n_detections)`,
   normalized to 1.0 at the threshold) so real, repeated confirmation is
   rewarded continuously rather than only checked as a binary cutoff.
   `min_n_detections` itself lowered from 5 to 3 to admit candidates to
   scoring sooner.
3. **Added a final `min_quality` gate on the fully-computed score.**
   `min_quality` was only ever checked early, in `combine_results`, on the
   base score alone (before any lightcurve/consistency information exists).
   Nothing filtered on the *final* score after (1) and (2) are applied. On
   the local fixture this dropped 3 previously-reported candidates whose
   final scores (0.014-0.09) were well below `min_quality=0.2` once properly
   computed, while the real confirmed afterglow's score only went up
   (37→56.5) — confirming the score now discriminates real repeated signal
   from marginal noise far more cleanly than before.
4. **Verified `maglim_filter_multiplier` (the pre-existing MAGLIM-based
   filter in `catalog_match.py`) should *not* be loosened.** A single
   exposure's own `MAGLIM` is a genuine physical noise floor, not a tunable
   heuristic — a source meaningfully fainter than it is not reliably
   recoverable from that one exposure, no matter how the significance/
   consistency thresholds above are tuned. Made the multiplier configurable
   (was hardcoded `1.1`) for future experiments against stacked/co-added
   images, but left the default unchanged. Checking it directly against
   GCN-confirmed magnitudes clarified which misses are genuinely
   MAGLIM-limited versus which are gate/threshold problems (see below) —
   worth checking for any future non-detection before assuming it's a
   tunable gate.

Net effect of (1)-(3), verified via `check_baseline.py` at each step (all
changes are deliberate, documented diffs, not regressions) — real
consistently-confirmed candidates score noticeably higher, weak/noise
candidates that used to sneak through on a technicality (enough detections,
low individual quality) now get filtered by the final gate instead.

### Re-classified after checking MAGLIM directly (not fixed by (1)-(3))

Checking each case's `MAG_CALIB / MAGLIM` ratio at the known position
directly (not just assuming "significance gate" from the symptom) gives a
much more precise diagnosis than the original pass:

- **GRB211024B, GRB210410A**: ratio consistently ~1.13-1.35 across every
  epoch — genuinely fainter than this system's own single-exposure depth
  for these fields. Not fixable by (1)-(3) or by any per-epoch threshold
  at all. **Confirmed still true even with real, meaningfully more epochs
  and working image stacking** — see "Validated: real production-data
  stacking/replay pass" below, which is the actual conclusive test of the
  "or simply accepting these are below this telescope/exposure
  combination's single-frame sensitivity" possibility raised here
  originally.
- **GRB151027B**: same ~1.13-1.35 ratio in the original 18-epoch replay,
  but **this one turned out to be replay-window-limited, not depth-limited
  — reclassify alongside GRB220403B below.** Recovered cleanly once run
  against 76 real epochs (still no stacking needed) — see "Validated: real
  production-data stacking/replay pass" below for the full result. The
  original MAGLIM-ratio measurement wasn't wrong, it just couldn't
  distinguish "genuinely too faint for one exposure" from "too faint for
  the specific 18 exposures replayed so far" without a real deeper-replay
  test to compare against.
- **GRB220403B**: ratio mixed, ~1.07-1.26 — borderline, some epochs pass.
  Consistent with the earlier finding that it *does* get flagged `"new"` in
  2/20 epochs, just never by all three catalogs at once. Primary blocker is
  `min_catalogs_fraction=1.0` (see below), with MAGLIM as a compounding
  factor in some epochs.
- **GRB200410A**: ratio consistently ~1.03-1.08 — comfortably brighter than
  MAGLIM in every epoch. Confirms this one was never a significance/MAGLIM
  problem at all; it's purely the blending mechanism below.
- **GRB210312B**: mixed, ~1.06-1.39 depending on epoch quality.

### Validated: extending the replay window recovers the min_catalogs_fraction case

**GRB220403B is now a confirmed 5th recovered detection** (0.73″ separation,
quality 12.5), without touching `min_catalogs_fraction` itself. The original
20-image replay never got past 2/20 epochs individually flagged `"new"` by
only one catalog at a time. Extending the replay to 57 images (raw frames
21-60 astrometrized and fed through the same forced-OBSID pipeline) let
different catalogs agree in different epochs; the `n_detections`-weighted
scoring from the significance-gate work above accumulated those into a
clean final candidate. A fresh one-epoch-at-a-time replay pinpointed the
exact crossing point: first past `min_quality=0.2` at epoch 17/57 (quality
2.76 there, growing to 12.5 by epoch 42/57 as detections kept accumulating).
This confirms the mechanism described below is real, and that **more
epochs, not a looser `min_catalogs_fraction`, is the working fix** for this
specific failure mode — lowering the default remains unnecessary now that
there's a demonstrated alternative that doesn't risk admitting noise.

1. **Unanimous cross-catalog agreement (`min_catalogs_fraction=1.0`) is too
   strict for faint sources in a single epoch.** GRB220403B's afterglow
   *does* get flagged `candidate_type="new"` by an individual catalog in
   some epochs (quality ~1.0-1.3, positionally right on the known
   coordinates) — but by only one of the three queried catalogs (`usno`,
   then `atlas@localhost` in a later epoch — never the same one twice, and
   never all three at once in the same epoch). At mag ~19, catalogs with
   shallower or patchier coverage (Gaia in particular) may simply have no
   counterpart to compare against in any one epoch — that's a coverage gap,
   not evidence the source isn't real. **Resolved for this case** by
   extending the replay window (see above) rather than by lowering the
   default fraction. `DetectionConfig.min_catalogs_fraction` is still
   config-exposed for future tuning if a case turns up that more epochs
   can't fix, but the default (`1.0`) is unchanged.
2. **Blending can suppress the magnitude-change check entirely.**
   GRB200410A's detections pass every other gate cleanly (SNR, MAGLIM) but
   never become candidates — every epoch shows two very close (~1″)
   detections with `FLAGS` bits indicating a blend. Likely explanation: the
   pipeline sees a persistent, unchanged-brightness point source at a known
   catalog position (the blended pair reads as "this star, same as always")
   and correctly-by-its-own-logic never flags it as new or changed, since
   the added GRB flux is folded into the blend rather than standing out.
   Fixed a real, separate bug found alongside this
   (`_check_magnitude_changes_cached` returned early on the *first*
   non-significant catalog match instead of checking all matches before
   deciding — verified via baseline diff, real fix, but empirically did not
   flip this specific case), so the actual blend-handling mechanism is still
   unresolved. **Concrete improvement**: a specific check for "blended flag
   + combined flux brighter than the catalogued star alone would predict"
   rather than treating any blended detection as automatically
   uninteresting.
3. **A recurring `ERRX2_IMAGE = ERRY2_IMAGE = 0.0` anomaly**, present in
   most epochs of 151027B, 210312B, 210410A, 220403B, 180325A, and all
   epochs of 240414A and 211024B. Real SExtractor centroid uncertainties
   are essentially never exactly 0. Traced through `core/radii.py`'s
   handling: it gets clamped to the minimum allowed radius rather than
   causing an outright failure, so it isn't the direct cause of any
   mechanism above — but it's suspiciously correlated with the same
   detections that are also MAGLIM-limited, which raises the question of
   whether whatever produces the degenerate centroid error is *also*
   quietly affecting photometry quality for the same measurements. Worth
   investigating directly in `pyrt`'s own aperture-photometry step
   (`phcat.py`/`dophot3`).
4. **Image stacking/co-addition** — the only real fix for the
   MAGLIM-limited cases above. Reaching ~3 magnitudes deeper (roughly what
   151027B/211024B/210410A would need) needs on the order of 10²·⁴≈250
   stacked frames for comparable per-exposure noise — not something simply
   extending the replayed epoch count achieves on its own, since each
   individual epoch is still evaluated (and gated) independently either
   way. **Not actually a "substantial, separate capability" to build from
   scratch** — `pyrt-combine` (the `combine-images` tool) already exists
   and is already used in production: `tests/2026kid/skel.hdr` is itself a
   real combined-image header from it (`COMBINER=combine-images`,
   `NCOMBINE=19`). Concrete proposal: once enough same-field, same-filter
   epochs have accumulated (e.g. 10-20 — far short of the ~250 needed for
   the full 3-mag MAGLIM gain, but still a real, worthwhile depth
   improvement over any single exposure), run `pyrt-combine` on them and
   run transient detection on the **summed image in parallel** with the
   existing per-epoch detection, not instead of it — a stacked image
   trades time resolution for depth, so it complements per-epoch detection
   (which alone can catch fast/moving/single-epoch phenomena a stack would
   blur together) rather than replacing it. This would also give
   `template_source="own_epoch"` (see "New detection strategy: image
   subtraction" above) a meaningfully deeper, lower-noise template option
   for fields where enough genuinely target-free epochs exist to stack —
   though note stacking epochs that *all* already contain the target
   doesn't help remove it (see that section's target-aware
   `ReferenceFrameSelector` note); it only helps build a deeper *clean*
   baseline, the same "not enough clean epochs" case that selector already
   detects and warns about.

   **Implemented** (`detection/stacking.py`, wired into `pipeline_magic.py`
   only — the GRB/`blind_multicatalog` pipeline, not the SN/subtraction
   one): runs automatically once `stacking_min_epochs` real epochs exist,
   but only as a try-harder fallback (skipped once an existing candidate
   already scores at or above `stacking_score_threshold`, so it doesn't
   spend `pyrt-combine`'s runtime on every single run once something
   convincing has already been found). Reuses
   `detection/subtraction/extraction.py::build_diff_ecsv` for the
   detect+calibrate step (calling it with the stack as both the "diff" and
   the "science" image — a stack, unlike a subtraction diff, keeps its own
   real stars, so self-calibration is valid). The stack epoch is just one
   more table handed to `BlindMulticatalogStrategy.run()`, so it's gated by
   the existing cross-epoch clustering logic exactly like a real epoch —
   **deliberately not changed here**.

   **Deferred**: a stack-only candidate (zero independent per-epoch
   support — exactly the 151027B/211024B/210410A case this feature targets)
   still needs `min_n_detections=3` distinct epoch-detections to clear
   `clustering.py`'s admission gate, same as any real epoch. A single deep
   stack is, on its own, stronger evidence than a single ordinary epoch —
   the confirmed direction for later is a `min_n_detections` (or an
   equivalent admission bar) that scales down with stack depth (e.g. a
   20-image stack needing less independent corroboration than a 6-image
   one). Not implemented now — every stack today, regardless of
   `NCOMBINE`, is gated identically to a real epoch.

### Validated: real production-data stacking/replay pass on the three MAGLIM-limited GRBs

Ran the actual shipped `detection/stacking.py` (not a simulation) against
real, freshly-produced epochs for all three GRBs the MAGLIM-ratio check
above flagged, using production's own `dophot3`/`phcat` pipeline
(`~/bin/get_ecsv.py`'s recipe, run without IRAF via `pyrt-phcat -I`) to
process additional raw frames beyond what the original 18-epoch replay
used:

- **GRB151027B: recovered, 76 real epochs, no stacking needed.** A
  pre-existing deeper production backup (`transient_work.bak/obs_17125`,
  76 real epochs vs. the original replay's 18) already had this — running
  `BlindMulticatalogStrategy` against the full set found the real afterglow
  at **2.95″ from its GCN position, quality_score=33.35**. This is exactly
  the same "more replay, not more per-exposure depth" mechanism already
  confirmed for GRB220403B above — GRB151027B was never actually
  depth-limited, just replay-window-limited (see the reclassification in
  "Re-classified after checking MAGLIM directly" above).
- **GRB210410A: not recovered, even with 66 real epochs (20→66, 46 newly
  processed from raw frames) plus a working stack** (`MAGLIM` gain
  +1.2 to +1.75 mag across runs, `NCOMBINE` 20-62 depending on which
  filter/exptime-consistent majority group was available that run — see
  detection/stacking.py's filter/exptime grouping). Baseline replay alone
  found 9 candidates with more epochs available (vs. 1 at 20 epochs); none
  within 7′ of the real position.
- **GRB211024B: not recovered, even with 80 real epochs (20→80) plus a
  working stack** (`MAGLIM` gain +2.5 to +3.15 mag, `NCOMBINE` up to 80).
  Baseline replay found 8 candidates, including one at
  **quality_score=73.35** — by far the highest score seen in this entire
  validation, exceeding even the confirmed GRB151027B recovery. **This is
  not the GRB** — checked directly against three independent primary GCN
  circulars (Swift-BAT #30989, Swift-XRT enhanced #30994, ground-based
  optical afterglow #30984 — all three mutually consistent to sub-arcsec),
  the real position is 4.00-5.02′ from every candidate found, over 100x
  the true ~2″ localization uncertainty. A concrete example that
  `quality_score` measures consistency/significance, not correct
  identification — a very high score is not, by itself, evidence a
  candidate is the actual target; always cross-check against the real GCN
  position before trusting it.

**Conclusion**: this is the conclusive version of the "or simply accepting
these are below this telescope/exposure combination's single-frame
sensitivity" possibility raised in the original MAGLIM-ratio finding.
GRB210410A and GRB211024B remain genuinely unrecoverable at the depths
reached here (up to 80 epochs stacked) — consistent with the original
~250-frame estimate for the full 3-magnitude gain these two would need,
which is far beyond what either GRB has raw frames available for
(66 total for 210410A, 175 total for 211024B). GRB151027B, by contrast, was
never actually a depth problem.

**A real production-environment bug found and fixed along the way**:
production's live `stdpipe` checkout (`/storage/home/fnovotny/src/stdpipe/`
on the host that ran this validation) crashed every SEP source-detection
call with `TypeError: sum_circle() got an unexpected keyword argument
'clip_sigma'` — `get_objects_sep`'s non-optimal aperture-photometry path
unconditionally calls the plain `sep.sum_circle()` with kwargs only the
newer `sum_circle_optimal` accepts. Not reproducible against the pinned
local dev `stdpipe`; this is a real, independent drift on that specific
host, exactly the kind of thing this repo's README already warns about
("stdpipe is under active development... pin to a specific commit").
Fixed in `detection/subtraction/extraction.py`'s
`_patch_sep_sum_circle_clip_kwargs()` — a narrow compatibility shim
(retries `sep.sum_circle` once without the unsupported kwargs if the first
call raises exactly that `TypeError`), not a stdpipe patch or a
reimplementation of its extraction logic.

Investigated replacing this shim with a real `sep-x` dependency instead
(the package `clip_sigma`/`clip_iters` actually belong to). Found a
*second*, independent stdpipe bug that makes that impossible on our side
alone: `photometry_measure.py`'s `_HAS_SEP_OPTIMAL` detection does
`import sep; hasattr(sep, 'sum_circle_optimal')` — it never imports
`sep_x` at all, so it stays `False` even with `sep-x` installed
(confirmed directly in a clean venv with both packages present side by
side). `sep-x`'s optimal path is unreachable from this stdpipe checkout
regardless of what's installed downstream. `sep-x` was still added to
`pyproject.toml` (harmless, ships a prebuilt wheel, no build step) so
this package benefits automatically once stdpipe fixes its own
detection — but the compatibility shim remains the only thing that
actually prevents the crash today, and stays in place. Both bugs
documented together in the draft GitHub issue for stdpipe.

**A related, separate discovery**: three GRBs' catalog entries in
`grb_detection2.txt` turned out to be data errors, not pipeline results —
GRB210722A's coordinates are duplicated from GRB210610B (real position is
in Cetus), GRB090726's position is off by ~12 arcmin from the real
GCN-confirmed one, and GRB210610A's *source directory* actually contains
images centered on GRB210610B's catalogued position, not its own (204.28°,
+14.47° — nothing was ever detected within 20″ of that position in any of
its 11 real epochs; the closest raw detection was 38° away). Three
independent errors out of eighteen entries is a high enough rate that the
whole catalog is worth a systematic re-validation (cross-check every row's
RA/Dec and source directory against its own GCN localization) rather than
treating these three as isolated one-offs.

## Constant new sources are structurally excluded (found by injection)

`pyrt_transient/validation/` (catalogue-level injection + incremental replay,
`tools/inject_recover.py`) was run on the 210619B fixture with synthetic
point sources at 14–20 mag, constant or fading. Every fading source above the
frame limit was recovered; **no constant source was, at any brightness**,
although each was a per-epoch `new` candidate with base score ~1 in every
epoch. With `min_quality=0` they all reappear — at final score 0.003–0.01,
ranked ~100–160 of 168, while the real afterglow scores 14.6. Cause:
`compute_lightcurve_score_factor` multiplies by `min(mag_range/0.5, 3)` *and*
by `mag_range` again, so a light curve whose range is only photometric
scatter (~0.03 mag for a bright star) gets a factor ~2e-3.

Why it is not simply a bug: the same `min_quality=0` run shows ~160
*persistent* `new`-type candidates on this one field — faint stars missing
from Gaia/USNO-B (catalogue incompleteness at 18–19 mag) that repeat in
every epoch exactly like a constant real source would. The `mag_range` term
is currently the only thing rejecting them. So the pipeline as configured is
a **variability detector for uncatalogued sources**, not a new-source
detector, and a bright afterglow in a plateau phase, a nova at maximum, or a
slow supernova would be lost on a short campaign.

**Root causes (found 2026-09-03 by tracing the 177 "persistent" sources).**
They are *not* faint stars at the catalogue limit — they are 11.5–18 mag
(median 16.5), real, present in 8/8 later epochs, and USNO-B has an entry
within 3" for 150/151 of the `new`-typed ones. Two defects produce them:

1. **Gaia is ~9% incomplete for bright stars as queried.** `pyrt`'s
   `_get_gaia_data` ADQL applies `ruwe < 1.4`,
   `visibility_periods_used >= 8` and non-null BP/RP with positive
   flux-over-error — calibrator cuts. 55 of 583 unflagged D50 detections
   brighter than 17 mag have no Gaia entry within 3" in the cached table;
   an independent `astroquery` cone search finds every one of them in DR3
   at 0.3–1.2" (e.g. G=14.52 at 0.35"). Not a coverage gap (uniform density
   across the frame), not SIP (the pipeline's TAN-only catalogue→pixel
   transform differs from the SIP solution by <0.3 px), not proper motion.
2. **USNO-B's photometric validity fraction is exactly 0** (`valid_stars`
   needs ≥2 Sloan bands; USNO-B has B1/R1/B2/R2/I only), so
   `_check_magnitude_changes_cached` skips every match and falls through to
   `return True, "new"`. On epoch 1 USNO-B typed 1611/1611 candidates `new`
   although all 583 bright detections have a USNO-B entry within 3". With
   `min_catalogs_fraction=1.0`, USNO-B was therefore a no-op and the
   unanimity rule reduced to "Gaia alone".

Implemented, all opt-in with historical defaults (`check_baseline.py`
unchanged): `catalogs: [gaia_full]` (same query, cuts stripped, in
`CatTransients._get_gaia_full_data`), `unphotometered_match_is_new: false`,
`catalog_match_floor_arcsec: {usno: 2.0}`, `new_source_variability_floor:
true` (`compute_lightcurve_score_factor` clips the two Δm factors at ≥1 for
`new`-typed candidates only). Validation numbers below.

3. **The 1-px match-radius floor (1.18" on D50) is tighter than the
   frame's own astrometric scatter.** Matching every unflagged D50
   detection brighter than 18 mag in all 9 epochs of 210619B against
   `gaia_full` on the sky: median offset 0.5", p90 1.0–1.7", p99 ~2.5", and
   7–20% of stars lie beyond 1.2" *at every distance from the frame edge*
   (it is not an edge effect, though the 10–30 px strip is worst at 20%).
   So in any single epoch ~8% of perfectly ordinary stars are "new" for a
   catalogue whose radius floor is 1 px — with the Δm term gone, the two
   edge/corner stars that survived as persistent candidates
   (`transient_319.524_33.708`, `transient_319.904_34.018`, Q 0.25–0.5) are
   exactly this. `catalog_match_floor_arcsec: {gaia: 3.0, usno: 3.0}`
   absorbs it; a real transient within 3" of a catalogue star is then typed
   `brightening` rather than `new`, which is still a candidate.

**Validation (2026-09-03, `tools/inject_recover.py` + `tools/replay_driver.py`,
210619B, gaia+usno locally; `local_test_output/inject_{default,vetting_full,vetting3}`).**

| config | detectable recovered | constant | fading | spurious/field | afterglow Q |
|---|---|---|---|---|---|
| default | 162/239 (68%) | 12/83 | 150/156 | 0 | 56.5 |
| gaia_full + unphotometered off + usno floor 2" + variability floor | 112/118 (95%) | 40/42 | 72/76 | 2 (edge stars, Q 0.25-0.5) | 57.2 |
| same with floor {gaia: 3, usno: 3} | 111/118 (94%) | 39/42 | 72/76 | 0 | 57.6 |
| gaia_full + atlas@vizier (vetting, floor 3") | 113/118 (96%) | 40/42 | 73/76 | 4 (galaxies, Q 0.35-0.65) | 55.0 |
| gaia_full + atlas@vizier + usno (vetting, floor 3") | 111/118 (94%) | 39/42 | 72/76 | 0 | 57.6 |
| gaia_full + atlas@vizier + panstarrs@vizier (vetting, floor 3") | 113/118 (96%) | 40/42 | 73/76 | 0 | 55.0 |
| gaia_full + atlas@vizier + sdss (no coverage here) | = gaia_full + atlas@vizier | | | 4 | 55.0 |

The four Gaia+ATLAS leftovers are 17.8-18.7 mag sources with FWHM 1.5-2x
stellar, seen in 3-7/9 epochs, with no Gaia or ATLAS entry within 10" but a
USNO-B entry at 0.9-2.6": galaxies. Gaia and refcat2 are point-source
catalogues; USNO-B's photographic plates include extended objects. So the
recommendation is Gaia + ATLAS for photometry and completeness, plus
Pan-STARRS (`panstarrs@vizier`) as the galaxy veto where it has coverage
(0 spurious, 113/118 — the best configuration measured) and USNO-B as the
positional-only fallback veto everywhere else (harmless now that it cannot
type things "new"). The last row shows the no-coverage path: an uncovered
catalogue in the list changes nothing instead of zeroing the field.
The right long-term galaxy veto is PS1 (broken query, see above) or
HyperLEDA/LS where covered; a morphology term would also catch these
(fwhm_ratio 1.5-2), but that is the score again.

Latency unchanged (median 3 epochs); un-injected replay reports the same two
real candidates as the baseline in every configuration. Not yet run with
`atlas@localhost`; production should re-measure with all three catalogues
before switching the default.

**Which catalogues to use instead of USNO-B (characterised 2026-09-03 on
210619B, unflagged D50 detections, counterpart within 3").**

| catalogue | valid photometry | bright (<17) matched | faint (17-19) matched | depth (r p95) | note |
|---|---|---|---|---|---|
| gaia_full | 98.3% | 582/583 | 826/834 | 19.4 (mlim 20) | Sloan via Jordi+2010 transform |
| atlas@vizier | 100% | 583/583 | 827/834 | 19.7 | real g,r,i,z; 9 s query |
| usno | 0% | 583/583 | 826/834 | 19.4 | positional only, ~0.85 mag photometric scatter vs D50 |
| panstarrs | — | — | — | — | **query broken**: astroquery MAST rejects pyrt's `nDetections.gt` filter (`InvalidQueryError ... Did you mean 'nDetections'?`) — pyrt-side fix |
| legacysurvey | — | — | — | — | no DR10 coverage of this field (b ≈ -13°) |
| sdss | — | — | — | — | no coverage |

Also note PS1's filters are registered as `g,r,i,z,y`, not `Sloan_*`, so
even once the query works `precompute_photometric_data` (which looks for
`Sloan_g/r/i/z`) would find zero valid stars — the same USNO-B failure
mode — unless the column names are mapped.

**Fixed in this package (2026-09-03), pyrt still needs the upstream fix:**
`CatTransients._get_panstarrs_data` overrides pyrt's MAST query with the
current `column=[("op", value)]` criteria syntax and an explicit numeric
column list (without it astroquery 0.4.11 fails parsing with `could not
convert string to float: 'None'`), keeps stars with incomplete bands, and
adds `Sloan_*` aliases. A second catalogue `panstarrs@vizier` (II/349/ps1,
DR1 mean photometry) is much faster (4958 rows in 1 s for a 0.3° box vs
136 rows in 2 s for a 0.03° MAST cone) and is the one to use for vetting.
Both return None (→ "unavailable") for query boxes entirely south of
Dec −30 without touching the network (`CatTransients.ps1_covers`).

**Catalogue availability bug (found while adding PS1, affects every
catalogue).** `catalog_match.find_transients_multicatalog` put a failed or
empty catalogue into `results` as `_empty_candidates_table()`, and
`combine_results` uses `len(transients)` as the unanimity denominator — so
any catalogue with no coverage of the field (PS1 south of −30, SDSS/LS
off-footprint, a timed-out download, USNO-B's occasional zero-coverage
fields) vetoed every source and the field silently produced zero
candidates. Now: no rows / exception → recorded as unavailable, excluded
from `results`, warning logged; all unavailable → error logged. Verified by
replaying 210619B with `[gaia_full, atlas@vizier, sdss]` (SDSS has no
coverage here): result identical to `[gaia_full, atlas@vizier]` instead of
empty. Baseline unchanged (both default catalogues available there).

**Southern hemisphere (Dec < -30, no Pan-STARRS) — probed 2026-09-03 on
VizieR / NOIRLab, 0.1° box at (50, -45):**

| catalogue | id | rows | galaxies | coverage | note |
|---|---|---|---|---|---|
| Legacy Survey DR10 | `legacysurvey` (already implemented, NOIRLab TAP) | 27144 in 0.3° / 20 s | yes (tractor type) | south at \|b\| ≳ 18°, north Dec < +32 | fails cleanly ("no data") at (120, -50) → excluded by the availability fix |
| SkyMapper DR4 | VizieR `II/379/smssdr4` | 62 / 1 s | `ClassStar` | whole sky Dec < +2, incl. plane | g,r,i,z PSF mags to ~20-21; the natural PS1 twin — a 20-line copy of `panstarrs@vizier` |
| DES DR2 | VizieR `II/371/des_dr2` | 1358 / 3 s | yes | 5000 deg², high-lat south | deep (r~24), overkill |
| VHS DR5 | VizieR `II/367` | 260 / 2 s | `Mclass` | most of the south | near-IR; useful where nothing optical exists |
| GSC 2.4 | VizieR `I/353` | 315 / 1 s | `Class` (star/galaxy) | all sky | photographic like USNO-B but with Gaia-tied astrometry and a galaxy class — candidate replacement for `usno` as the universal floor |

Plan: implement `skymapper@vizier` (mirror of `panstarrs@vizier`) and let
the catalogue list carry both; with the availability fix each field uses
whichever covers it, at one cached query per 1° cell. Then re-run the
injection harness on a real southern D50 field (none in the local
fixtures) before recommending defaults.

Directions still open (changes the score, needs its own baseline
review):
- Replace raw `mag_range` with a significance-normalised variability
  (chi-squared of the light curve against a constant), which separates real
  scatter from photometric noise and stops rewarding noisy faint sources.
- Reject persistence with evidence rather than with the score: a
  `new` candidate brighter than the local catalogue completeness limit that
  has no counterpart in any catalogue is far more suspicious than one at
  the catalogue floor; forced photometry against PS1/ATLAS at the position
  (see "Candidate enrichment") answers it directly.
- Until then, keep a note on any campaign shorter than a few decay times
  that the constant-source case is blind by construction.

**Isolation penalty from undetected catalogue stars (also found by injection).**
Six of 156 fading injected sources detected in every epoch were never
reported. Each sits within ~2" of a catalogue star fainter than the frame
(Gaia G≈19.9 / USNO-B R≈19.2 in the traced case) — outside the adaptive
match radius, so still typed `new`, but `nearest_source_dist` (distance to
the nearest *catalogue* entry, not the nearest detection) is ~1–2", the
isolation factor `clip(d/10, 0, 1)` becomes 0.1–0.2, one catalogue's
per-epoch base score lands just under `min_quality=0.2` (0.189 for USNO-B
vs 0.373 for Gaia in the traced epoch), and `min_catalogs_fraction=1.0`
then drops the source. ~2% of random positions in this field are within 2"
of a catalogue entry, matching the miss rate; denser fields will lose more.
Fix direction: compute the isolation distance against sources *detected in
the frame* (or against catalogue entries brighter than the frame limit), not
against the full catalogue. Changes scores → baseline review.

Also surfaced by the same runs: the 210619B fixture has 9 epochs (not 22 —
the 22 `.ecsv` files include the derived `_transients` tables).

## External services must degrade, never crash (SkyBoT outage, 2026-09-04)

The GCN-position archive replay on lascaux50 failed on 14/15 bursts within
seconds: IMCCE's SkyBoT returned no table (`astroquery` raises `ValueError:
No table found`, which `apply_skybot_filter` did not catch — only the
`KeyError` of the empty-result stdpipe bug), and the exception propagated
from `combine_results` through `strategy.run()` and killed the epoch. The
previous day's run had passed because the service was up. Fixed: any
exception from the SkyBoT call is logged and treated as "nothing
rejected". Same rule as catalogue availability: a remote service that is
down must cost a filter, not the detection. Remaining unguarded remote
calls worth checking with the same eye: HyperLEDA and TNS in
`pipeline_magic_sn.py`.

**Open, found in review 2026-09-09 — the graceful degradation opened a
hole on the SN side.** `pipeline_magic_sn.py:1009` decides
`per_epoch_skybot` from `strategy_name`, `vsx_filter_enabled` and the
header times alone — never from whether the per-epoch cross-match actually
ran. When it is true, the final `reject_known_asteroids` at line 1018 is
skipped as redundant. But `apply_skybot_filter` now returns `ok=False` and
passes candidates through unfiltered on an outage
(`stdpipe_filters.py:118`), so during an outage like the 2026-09-04 one
the per-epoch pass rejects nothing *and* the final pass is skipped:
known asteroids are reported as candidates. Before the degradation fix the
final pass would still have caught them (the epoch died instead, which is
its own bug — the fix was right, this is the missing half). The flag needs
to be gated on the epochs having really been SkyBoT-filtered, e.g. no
SkyBoT `DEGRADED` reason on any epoch, rather than on configuration alone.
Deliberately left for later: deciding what "really filtered" means across a
mixed set of epochs (some filtered, some not) is the actual design
question, and partial coverage should probably still run the final pass.

## Archive replay results (2026-09-04, `tools/replay_archive.py` on lascaux50)

15 campaigns / 14 distinct bursts with fixed-OBSID tables, all epochs
(3-60), GCN afterglow positions, 5" match, run under both configurations
(`~/etc/pyrt-transient/replay_{historical,vetting}.yaml`); website
`~/public_html/grb_replay_validation/replay_2026-09-04b/{historical,vetting}/index.html`,
summaries copied to `local_test_output/archive_replay_2026-09-04b/`.

| burst | ep | hist k_first / Q | vet k_first / Q | other cand. H/V | outcome |
|---|---|---|---|---|---|
| 250813B | 20 | 4 / 11.2 | 5 / 39.3 | 1/1 | recovered |
| 250702F | 17 | 3 / 45.6 | 3 / 45.6 | 1/1 | recovered (52 s after trigger) |
| 240414A | 20 | - | - | 1/1 | not recovered, i band, cause open |
| 230818A | 20 | 3 / 20.6 | 3 / 19.8 | 4/1 | recovered |
| 220403B | 57 | 17 / 19.8 | 17 / 19.8 | 8/2 | recovered (window) |
| 211024B | 60 | - | - | 5/2 | below m_lim |
| 210722A (=210610B late) | 20 | 5 / 8.3 | 4 / 8.4 | 0/1 | recovered |
| 210619B | 20 | 5 / 126.8 | 5 / 131.5 | 3/1 | recovered |
| 210610A (=210610B) | 11 | 7 / 0.30 | 3 / 1.36 | 1/0 | recovered |
| 210410A | 60 | 5 / 26.8 | 7 / 26.9 | 0/1 | marginal (every frame 2-4 mag below m_lim, 3.5" off, nothing in PS1/Gaia within 6") |
| 210312B | 60 | - | - | 2/1 | significance gate |
| 200410A | 60 | - | - | 2/1 | blending |
| 180325A | 3 | - | - | 0/0 | 3 usable frames |
| 151027B | 58 | 44 / 35.7 | 26 / 35.7 | 3/3 | marginal (as 210410A) |
| 090726 | 20 | 6 / 0.74 | 3 / 1.38 | 1/1 | recovered |

Totals: 10/15 campaigns (9/14 bursts) in both; non-afterglow candidates
32 (H) vs 17 (V). Superseded numbers: the earlier "5/18 detected" summary
scored against trigger positions. Open: 240414A (i-band, never traced),
and the "marginal" pair are reported through the mag_range noise term —
with a chi2/significance-based variability term they would not be.

## Cross-epoch clustering is fragile to one junk row (GRB 220403B, 2026-09-04)

Under the vetting catalogues the 220403B afterglow (19 mag, 46 light-curve
detections by epoch 57) was tracked from epoch 17 (Q 2.5) to epoch 41
(Q 9.4) and then vanished from the candidate list entirely — not below the
gate, gone. Traced with the union-find components captured at k=41/42:

- Epoch 42 (a bad 20 s frame, MAGLIM 14.6) contributes a per-epoch `new`
  candidate 1.8" from the afterglow: FWHM 13.9 px, ERRX2=ERRY2=0, masked
  MAG_CALIB, base score 1.5 (higher than the real rows' ~1.0 because the
  masked magnitude drops the magnitude factor to neutral).
- Pass 1 links it (0.97") to the epoch-15 row; pass 2 compared component
  *centroids* at 2": centroid([13,16]) vs centroid([15,42]) = 2.21" → no
  merge → two 2-row components, both below `min_n_detections=3` → the
  source with 31 real detections is discarded. (The candidate-row count,
  not the light-curve count, is what `min_n_detections` gates.)
- After merging was fixed (member-level single linkage at the base radius
  plus radius-weighted centroids, `combine_with_lightcurves` pass 2/2b), the
  4-row component reached `split_component_by_epoch`, whose greedy split
  seeds on the highest-scoring row — the blob — and re-validates distance
  to *its* centroid, stranding the real rows again in a 2-row subcluster.
  The split only exists to enforce one-detection-per-epoch, and this
  component had four distinct epochs, so it now returns conflict-free
  components whole and only splits genuine conflicts.

Result: 220403B kept at k=42 (Q 14.2, 31 det) and k=57 (Q 19.8, 46 det).
Unit tests: `test_pass2_merges_on_member_distance_not_centroid`,
`test_split_component_keeps_distinct_epochs_whole_and_splits_conflicts`.
Still open: the representative row of a group is its highest-scoring
member, so the reported position sits on the blob (1.8" off) rather than
on the 30 real detections — the light-curve mean position would be the
better `transient_id`/reported coordinate; and a masked-magnitude row
should not out-score real ones (neutral factor 1.0 vs. <1 for a real faint
row).

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

**Phase A implemented** (consuming already-differenced epochs; see
`tests/2026kid/` — a real multi-night HOTPANTS/PS1 campaign for AT2026kid,
and the subtraction-branch plan). `detection/subtraction/` now has:

- `SubtractionStrategy(DetectionStrategy)` (`__init__.py`), mirroring
  `BlindMulticatalogStrategy` exactly: per-epoch candidate building →
  `clustering.save_epoch_results` (cached to disk) → cross-epoch
  `clustering.combine_with_lightcurves` (unchanged, reused as-is) →
  lightcurve plotting. Only the per-epoch candidate *source* differs.
- `candidates.py`: builds "new"-typed candidates directly from diff-image
  detections (no cross-catalog matching needed — subtraction already
  removed constant sources). Real-fixture finding: diff-image `.cat` files
  carry no calibration meta at all (no `MAGZERO`/`MAGLIM`/`CTRRA`/etc.,
  `MAG_AUTO` is a raw zp=0 instrumental magnitude) — these get borrowed from
  the matching science epoch (same night/filter, name without the trailing
  `h`) via `borrow_science_meta`/`calibrate_diff_magnitudes`, since
  HOTPANTS's `-n i` normalization keeps the diff image in the same ADU
  units as the science image.
- `artifact_filters.py`: morphology/magnitude filters ported from
  `pipeline_magic_sn.py`, plus a new `reject_dipole_artifacts` (samples the
  diff FITS pixel data directly for a nearby negative-flux counterpart —
  the catalog itself never lists negative detections, single-polarity
  SExtractor).
- Validated end-to-end against the real 7-night fixture: 3782 raw diff
  detections → 4 final clustered candidates, with the real target surviving
  at n_detections=5/7 and ranking #1-2 by quality_score even before the
  SN-specific post-filters (PM-star/galaxy/TNS, still to be layered in via
  `pipeline_magic_sn.py`) are applied.

Reference-template strategy (own-epoch via `ReferenceFrameSelector` vs.
external-survey via `stdpipe.templates`) and the differencing engine itself
(HOTPANTS vs. PyZOGY) are **Phase B**, not built yet — Phase A works
entirely from pre-made diff images. `core/scoring.py`'s existing
`quality_score` schema turned out to need no subtraction-specific changes;
`reference_catalog` is repurposed to record template provenance instead of
a matched-catalog name.

### Phase B implemented: template acquisition, differencing, extraction

`detection/subtraction/templates.py`, `differencing.py`, `extraction.py` --
own-epoch template (via `ReferenceFrameSelector`), PS1 external-survey
template (`stdpipe.templates`, with a real bug found and fixed — see
below), HOTPANTS and PyZOGY differencing engines, and SEP-based extraction/
calibration, all validated against the real `tests/2026kid/` fixture end to
end: real HOTPANTS and real PyZOGY differencing, real PS1 photometric
calibration, real target recovered by both engines.

### Phase B wired into `pipeline_magic_sn.py`: raw science images end to end

`config.detection.diff_input_mode`: `"prebuilt"` (Phase A, default -- the
input is already a diff-image pair) or `"raw"` (Phase B -- the input is a
raw science epoch, and `_ensure_diff_epochs` builds the
template/diff/extraction automatically before handing off to the same
`SubtractionStrategy`). Validated with a real 7-night raw-mode run
(own-epoch template, HOTPANTS, target position supplied via
`--target-positions`): real target recovered at 0.90″, mag 16.70, ranked
#7 of 21 final candidates by `sn_score`.

Two real bugs found and fixed while wiring this up, both concrete
consequences of running the full pipeline for the first time rather than
each module in isolation:

- **`FIELD`/`CTRRA`/`CTRDEC`/`MAGLIM` never reached the diff FITS header.**
  These live only in the science `.ecsv`'s sidecar meta, never in the raw
  FITS file's own header (verified directly) -- `differencing.py` was
  copying just the FITS header, so every Phase-B-built diff image silently
  had `FIELD=0.0`. Concrete, non-hypothetical consequence: a `FIELD=0.0`
  field size turned into a 0-degree HyperLEDA search radius in Step 7,
  which for reasons on VizieR's end returned **983,261 galaxies** and hung
  the whole pipeline for minutes rather than erroring. Fixed:
  `run_diff`/`run_hotpants_diff`/`run_zogy_diff`/`_write_diff` now take a
  `science_meta` dict (the science table's own `.meta`) and merge the
  relevant keys into the diff FITS header before writing.
- **Same OBSID-fragmentation bug, one level deeper.** `derive_observation_id`
  only handled being given a *diff* file (looks up its science sibling) --
  Phase B's raw mode passes the *science* file directly as the input, which
  `find_science_sibling` doesn't recognize (it isn't named with the
  trailing `h` a diff file has), so it silently fell through to the
  generic OBSID-based ID and refragmented every epoch into its own
  `ObservationStore` directory again. Fixed: `derive_observation_id` now
  checks the given path's own meta for `OBJECT`/`TARGET` first, before
  falling back to the science-sibling lookup.

**Also found (documented, not fixed -- needs real-data tuning, not a
guessed replacement number)**: `apply_morphology_filter`'s
`max_ellipticity=0.4` default was implicitly tuned against the real
fixture's external SExtractor-based diff catalogs. Measured directly on
one real diff image extracted via `extraction.py`'s stdpipe/SEP path: 57%
of all genuine SEP detections (43/75) had `ELLIPTICITY >= 0.4`, including
the real AT2026kid target itself (0.594) -- silently dropped by the filter
as a result. A second, related SEP-specific quirk: `FWHM_IMAGE` comes back
exactly `0` for a large fraction of marginal diff-image detections (38/75
on the same image), which pushed the per-epoch median FWHM to 0 and
silently disabled the `fwhm_ratio` half of the morphology check entirely
(division guard fell back to a neutral 1.0 for every row). Exposed
`morphology_max_ellipticity`/`morphology_fwhm_ratio_min`/`_max` as config
(`DetectionConfig`, wired into both `SubtractionStrategy`'s internal
per-epoch filtering and `pipeline_magic_sn.py`'s Step 3) so this can be
tuned without a code change -- confirmed loosening `max_ellipticity` to
0.65 recovers the real target -- but the right production value needs
calibrating against more real SEP-extracted data, not one field's worth.

**Important, non-obvious finding**: own-epoch template differencing
reveals the *change* relative to the reference epoch, not the target's
absolute brightness — not a bug, but easy to misread as one. Verified
directly: AT2026kid's own science-image magnitude was 15.652 on 2026-04-25
and 15.649 on 2026-04-26 (essentially flat, <0.01 mag change) — a genuinely
slowly-evolving source. Differencing 04-26 against 04-25 as an own-epoch
template therefore subtracts away nearly all of the target's actual flux
(since it's nearly identical in both epochs), leaving a much fainter
residual (HOTPANTS: 18.14 mag; PyZOGY: 17.53 mag) than the true total
brightness (~15.65, matching the real PS1-template-based fixture). This is
exactly correct behavior for detecting *new* transients or *sudden*
changes, but means own-epoch differencing is the wrong choice for
continued monitoring of an already-known, slowly-evolving source — an
external, genuinely-quiescent template (PS1/LegacySurvey) is needed there
to recover meaningful absolute photometry.

**Implemented**: `ReferenceFrameSelector` (`detection/reference_frame.py`)
and `get_template_own_epoch` now take optional `target_ra`/`target_dec`.
Previously the selector picked purely on generic image quality
(seeing/depth/source count/center distance) with no way to know whether
the target itself already had real flux in a candidate reference epoch —
confirmed this was a real, not hypothetical, gap: AT2026kid is present at
essentially constant brightness in *every* one of the 7 real campaign
epochs, so the selector could pick any of them as "best quality" with no
warning that the resulting own-epoch template would already contain the
target. Now: when a target position is given, epoch selection prefers a
genuinely target-free epoch if one exists (verified: correctly overrides
even a large seeing advantage on the contaminated epoch), and falls back
to the best-quality epoch with an explicit warning log (plus a
`:target-contaminated` suffix on the returned provenance string) when
every candidate epoch already has the target in it — the AT2026kid case.
`target_ra`/`target_dec` are optional and default to `None`, which
reproduces the exact original quality-only selection (verified via a
dedicated backward-compatibility test) — a caller doing a blind
first-detection search with no known target position yet is unaffected.

**PS1 template retrieval bug found and fixed**: this project's installed
`stdpipe` build's `normalize_ps1_skycell` crashes on real PS1 downloads
(`ValueError: cannot convert float NaN to integer`, inside astropy's own
compressed-tile decompression of certain BLANK-valued integer skycell
masks) — the exact same bug the one-off `subtract_supernova.py` reference
script already found and worked around, by reading the skycell with
`fitsio` instead of astropy for that one step. Reapplied verbatim in
`templates.py`'s `_patch_normalize_ps1_skycell` (monkeypatches
`stdpipe.templates.normalize_ps1_skycell`, only if `fitsio` is
importable) -- **and guarded by signature**: this patch targets the older
`normalize_ps1_skycell(filename, outname=None, verbose=False)` this
project's installed stdpipe build has, which reopens a file from disk. A
newer stdpipe (verified against 0.4.1 from PyPI) changed the signature to
an in-memory `normalize_ps1_skycell(image, header, verbose=False)`, where
this bug is apparently already fixed upstream -- applying the old patch
there unconditionally would break every PS1 call outright rather than
restoring the old bug. `_patch_normalize_ps1_skycell` now inspects the
installed function's parameter names first and only patches the old
`filename`-based form, falling back to stdpipe's own implementation
otherwise (verified both branches: old signature gets patched, newer
`image`/`header` signature is left untouched). Confirmed fixed: a real PS1 fetch got past skycell
normalization after the patch, then correctly reported "missing `swarp`
binary" rather than crashing. Root cause turned out to be trivial: this
machine's `swarp` (apt package `swarp` 2.41.5-1) installs its binary as
`/usr/bin/SWarp` (capitalized), while stdpipe's `reproject_swarp` only
checks `shutil.which('swarp')` (lowercase) — not actually missing, just
unresolvable by name. Fixed locally with a user-writable symlink
(`~/.local/bin/swarp -> /usr/bin/SWarp`, no sudo needed since
`~/.local/bin` is already first on `PATH`). Validated end-to-end on this
machine with a real call: `get_template_ps1` against one of the
210619B fixture epochs' real header/WCS (`CRVAL1/2 = 319.71, 33.83`)
downloaded 6 real PS1 skycells, ran SWarp to reproject+coadd them onto
the science WCS, and returned a real `(1024, 1024)` template image in
1.5s ("SWarp run successfully" / "RESULT: got template (1024, 1024)
provenance: ps1_template") — full PS1-template reprojection path
confirmed working, not just unblocked in theory.

**Verified against real stdpipe 0.4.1 (PyPI latest), not just the older
local checkout.** This machine's installed `stdpipe` (0.1, an old checkout
predating even this project's own declared `stdpipe>=0.3.0` minimum) is
what all the above was originally found and fixed against. To check
whether these workarounds -- and the rest of this package's stdpipe usage
-- still hold up against a current release, built a throwaway, fully
isolated environment (`conda create -n pyrt-transient python=3.11`, kept
separate from the global anaconda3 env every other astro_mates script
shares, so nothing there was touched) with `stdpipe==0.4.1` from PyPI, the
real `pyrt` installed from local source, and `fitsio`. Found and fixed two
real, version-specific breaks along the way:

- **`_patch_normalize_ps1_skycell`** (`templates.py`) needed a signature
  guard -- see above; already covered.
- **`_patch_sep_sum_circle_clip_kwargs`** (`extraction.py`) did a fresh
  top-level `import sep` and patched that. Reading the installed
  `stdpipe.photometry` module directly showed 0.4.1 does `import sep_x as
  sep` internally (falling back to plain `sep` only if `sep_x` isn't
  installed) -- so the plain `sep` package (not installed in the new env
  at all) was never the module `get_objects_sep` actually calls into,
  and the patch was silently patching nothing. Fixed to read
  `stdpipe.photometry`'s own `sep` attribute instead of importing fresh,
  so it patches whichever module stdpipe itself resolved to, on any
  version. `README.md`'s old claim that `stdpipe`'s `_HAS_SEP_OPTIMAL`
  detection "never actually imports `sep_x`" was true of the old checkout
  but is no longer true of 0.4.1 -- updated.

Also surfaced, unrelated to stdpipe itself but found while building the
clean environment: **`pip install pyrt` installs the wrong package.**
PyPI's `pyrt` is an unrelated ray-tracer (camera/geometry/light/material
modules, nothing astronomical) that happens to share a name with the real
`pyrt` this project depends on -- confirmed by downloading and inspecting
the wheel directly. Every install instruction in this repo saying `pip
install pyrt` (`README.md`, `pyproject.toml`'s dependency comment) was
wrong and has been corrected to install from source instead
(`https://github.com/mates14/pyrt` or a local checkout). This is a real
footgun for a completely clean install (this project's own `pip install
-e .` lists bare `"pyrt"` as a dependency, so pip would happily fetch the
wrong one from PyPI if the real one isn't already installed first) --
worth keeping in mind if `pyrt-transient` is ever published/installed
somewhere new.

With both fixes applied, the full test suite (122 tests via `pytest
tests/`) passes unchanged against `stdpipe==0.4.1` + `sep-x==1.5.1` +
`photutils==3.0.0` + `numpy==2.4.6` + `astropy==8.0.1` in the isolated
environment -- not just the pre-existing global-env versions. Every other
stdpipe entry point this package calls (`planar_match`, `spherical_match`,
`get_cat_vizier`, `get_background`, `get_objects_sep`,
`calibrate_photometry`, `filter_transient_candidates`, `run_hotpants`,
`get_ps1_image_and_mask`, `get_ls_image_and_mask`, `reproject_swarp`) was
individually confirmed still present with a compatible keyword-argument
signature in 0.4.1 -- `stdpipe.astrometry.planar_match` in particular
(used by `core/matching.py` for all pixel-space matching) doesn't exist at
all in the old 0.1 checkout, which is why `test_core_matching.py` and
`test_reference_frame.py` fail outright against it but pass clean here.

### Performance: found and fixed a real scaling bottleneck

Profiling the Phase-A validation run (7 epochs, 3782 raw candidates) found
`clustering.py`'s `combine_results` — shared by both strategies — was
building a brand-new single-row `Table()` **one column at a time** for
every surviving cluster (~14,000 `astropy.table.Table.add_column` calls for
436 candidates in one epoch, 6.5s just for that function). This wasn't
particular to subtraction, but subtraction's raw per-epoch candidate counts
(hundreds, no early catalog-cross-match narrowing) hit it much harder than
blind-multicatalog's typically-smaller per-epoch counts — left alone, this
would have made subtraction impractical at real survey scale (a full-frame
image can have thousands of raw diff detections before filtering).

Fixed by collecting per-cluster winning indices and override values in
plain Python lists, then building the result with one fancy-index select
plus at most two whole-column overwrites, instead of one Table() per
cluster. Verified byte-for-byte identical output (same rows, same values in
every column, including the `candidate_type`/`magnitude_difference`
override logic) against the pre-fix version on three real epochs. Net
effect: ~9x faster on `combine_results` alone (6.5s→0.7s for 436
candidates), ~2.5x faster end-to-end for the 7-epoch validation run
(22.5s→9.1s).

Remaining cost after the fix is dominated by `combine_with_lightcurves`'s
own per-component `Table` slicing (inherent to the union-find clustering
approach, not a clear anti-pattern the way the single-row construction was)
and `.ecsv` text I/O — both scale linearly with candidate/epoch count in
this measurement, not superlinearly. Total runtime scaled roughly linearly
with epoch count throughout (1→7 epochs: ~1.0s→9.1s, no blowup). Not yet
measured: Phase B's own differencing cost (HOTPANTS/PyZOGY convolution
scales with image pixel count and kernel size, a genuinely separate cost
this profiling doesn't cover) — worth benchmarking again once Phase B
lands, on a realistic full-frame image size rather than the 1024×1024
fixture.

### Operational gaps found running a real multi-night SN campaign (target 53393)

Running the subtraction pipeline end to end against a real, multi-night
target (53393) surfaced four gaps that are operational rather than
detection-logic bugs — nothing here changed a threshold or a scoring
formula, but each one cost real debugging time and would recur for any
future campaign run the same way.

1. **Silent failure vs. real non-detection is indistinguishable — the
   dangerous one.** The PS1 template bug (see "PS1 template retrieval bug
   found and fixed" above) didn't crash the pipeline — it degraded to "0
   candidates," exactly what a genuine non-detection also looks like. If we
   hadn't gone looking, "no counterpart found" would have been the reported
   result, when the real issue was a broken dependency.
   `_get_survey_template`'s except-and-warn pattern is the right call for a
   truly external failure (network down, no coverage), but there's no
   distinction in the output between "searched and found nothing" and
   "couldn't search." A campaign report should surface "template retrieval
   failed for N/N epochs" as a headline, not a buried `WARNING` in a log.
   **Architecture note**: the root cause is that `Optional[Tuple[...]]` /
   `None` is already overloaded to mean two genuinely different things
   (`_get_survey_template` returns `None` for "coverage/network legitimately
   has nothing" *and* for "an external library crashed retrieving it," and
   `get_template_own_epoch` returns `None` for "no prior epochs" *and* "the
   selected epoch's FITS couldn't be read"). Collapsing distinct failure
   modes into one falsy value is exactly what makes a broken dependency
   indistinguishable from a real non-detection three call-frames up in
   `pipeline_magic_sn.py`. A small result type (e.g. an enum status —
   `OK` / `NO_COVERAGE` / `RETRIEVAL_ERROR` — alongside the existing
   `(image, mask, provenance)` payload) threaded through
   `_get_survey_template` → `get_template_ps1`/`get_template_legacysurvey` →
   `_ensure_diff_epochs` would let the campaign-level report tally
   `RETRIEVAL_ERROR` counts separately from `NO_COVERAGE` ones, without
   changing the per-epoch control flow at all (both still mean "skip this
   epoch's diff, keep going").
2. **No preflight/dependency check.** `hotpants` (missing
   `libcfitsio.so.5`), `fitsio` (needed by this project's own documented PS1
   workaround, `_patch_normalize_ps1_skycell`), and `swarp` (not installed
   at all, and even once installed, found only via a case-sensitive
   `shutil.which('swarp')` that misses Debian's capitalized `SWarp` binary
   — see the swarp finding above) were all silently broken or absent until
   we hit them mid-run. Nothing validates the subtraction toolchain before
   a campaign starts. A `pyrt-transient-sn-pipeline --check-deps` (or a
   check baked into daemon startup) that actually invokes
   `hotpants`/`swarp` and imports `fitsio` would have caught all three in
   seconds instead of after three separate debugging detours.
   **Architecture note**: this deliberately does not belong inside
   `templates.py`/`differencing.py` themselves — every per-call site in
   those modules is correctly designed to degrade gracefully (return `None`,
   log a warning, keep the epoch loop going), and a preflight check must do
   the opposite: fail loudly, once, before any epoch processing starts. That
   makes it a startup-time concern, the same category as config validation,
   not a detection-strategy concern — it belongs alongside wherever
   `DetectionConfig`/CLI args are already validated at daemon or
   `pipeline_magic_sn.py` entry, as one `check_subtraction_deps()` call that
   invokes each external binary with `--version` (or equivalent) and
   `importlib.import_module`s each optional dependency, rather than being
   scattered as a `try/except ImportError` per call site the way it is
   today.
3. **No "campaign" concept above the per-observation daemon.**
   `c0_pipeline.py` organizes everything into sequential `obs_<id>` by
   observing block, not by target — reconstructing "every epoch of target
   53393 across three nights" meant grepping ECSV headers for
   `TARGET`/`OBJECT` and matching timestamps to source directories by hand.
   `derive_observation_id`'s `OBJECT`/`TARGET`-based grouping already
   exists for the subtraction pipeline's own bookkeeping, but there's no
   equivalent tool for a human to say "give me every raw frame and every
   daemon-processed epoch for target 53393" — that logic (re-derived ad hoc
   for this campaign) belongs in a script, not in scrollback.
   **Architecture note**: this should be a read-only index layer, not a
   second mutable store competing with `ObservationStore`. `ObservationStore`
   already owns per-observation locking/consistency and reorganizing it
   around targets instead of `obs_<id>` would be a real, risky rewrite of
   the daemon's core bookkeeping for no detection-logic benefit. The cheaper
   design: a `campaign_index.py` that does what was done by hand here —
   scan ECSV `OBJECT`/`TARGET` fields (reusing `derive_observation_id`'s
   existing field-reading logic rather than re-deriving it a third time)
   across raw-frame and `obs_<id>` directories, and build a
   target→{raw frames, observation dirs} mapping, cached/rebuilt on demand
   rather than maintained live. It sits *above* the daemon as a query tool
   (`pyrt-transient-campaign --target 53393`), the same relationship
   `replay_driver.py` (see "GRB replay driver" above) has to the per-epoch
   detection strategies it drives — a reporting/orchestration layer that
   reads the daemon's output, not a change to how the daemon itself
   organizes work.
4. **Stacking isn't wired into the subtraction pipeline at all.**
   `detection/stacking.py` already calls `pyrt-combine` automatically for
   the `blind_multicatalog` strategy (see "Image stacking/co-addition"
   above) — but the subtraction strategy has no equivalent. This campaign
   required hand-building a shared skeleton (`pyrt-combine
   --skeleton-only`), stacking each night onto it, then hand-running
   `phcat.py`+`dophot3.py` twice per stack to mimic `get_ecsv.py`'s
   calibration. If nightly stacking before `own_epoch`/PS1 differencing is
   something future campaigns will want again (depth considerations
   suggest they will), that whole sequence belongs as a mode in
   `pipeline_magic_sn.py`, not a one-off shell script per campaign.
   **Architecture note**: this is a smaller gap than it looks, precisely
   because `config.detection.diff_input_mode="raw"` already exists (see
   "Phase B wired into `pipeline_magic_sn.py`" above) and treats a raw
   science epoch as the unit of work `_ensure_diff_epochs` builds a
   template/diff/extraction from. A stacked image is, from
   `SubtractionStrategy`'s point of view, just another science-epoch-shaped
   input — the same relationship `blind_multicatalog`'s stacking fallback
   already has to `BlindMulticatalogStrategy.run()` (a stack epoch handed in
   as "just one more table," per the "Implemented" note under "Image
   stacking/co-addition" above). The missing piece isn't new
   differencing/extraction logic, it's a `pipeline_magic_sn.py` step that
   runs `pyrt-combine` per night *before* `_ensure_diff_epochs`, then feeds
   the resulting stack through the exact same `diff_input_mode="raw"` path
   nightly raw epochs already use — reusing the raw-mode machinery rather
   than adding a second, parallel stacking-aware code path through
   `SubtractionStrategy`.

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

## Quality-score alternatives tested against the 15-field replay (2026-09-03)

Harness: `tools/score_eval_collect.py` (runs the strategy with
`min_quality=0` at k = 1..10, 12, ..., 60, final; pickles every candidate
with its light curve; optional 30-source injected campaign per field) and
`tools/score_eval_analyse.py` (labels, score formulae, ranking/calibration
metrics, leave-one-field-out fits). Data: the fixed-OBSID detection tables
of the 18-GRB replay copied to `local_test_output/replay18/raw/` (15 fields
have tables), GCN afterglow positions in `local_test_output/replay18/fields.json`
(the old list held trigger positions for 210312B, 090726, 211024B, ...).
Two catalogue configurations: `default` (gaia + usno, as production minus
ATLAS) and `vetting` (gaia_full + atlas@vizier + usno, README options).
Results: `local_test_output/replay18/score_eval/report.md`.

Scores compared, all from the same features: `q_cur` (shipped), `q_floor`
(shipped + `new_source_variability_floor`), `q_chi2` (mag_range² replaced by
clip(reduced χ², 1, 10)), `llr` (hand-built log-likelihood ratio:
persistence vs. MAGLIMIT-expected detections, morphology, position scatter,
catalogue completeness, variability, fading trend), `p_bayes` (mixture
posterior transient : uncatalogued star : noise, persistence cancelling
between the first two), logistic regression / gradient boosting
(leave-one-field-out), and a logistic calibration of ln q_cur (`q_cal`).

Headline numbers at the final epoch count (7 afterglows, 252 injected,
1281 negatives in `default`; 6 afterglows, 246 injected, 60 negatives in
`vetting`):

| config | score | AUC | recall at the FP count of q_cur>=0.2 | afterglows ranked #1 |
|---|---|---|---|---|
| default | q_cur | 0.79 | 0.84 (constant injected 0.59) | 3/7 |
| default | q_floor | 0.85 | 0.93 (0.86) | 1/7 |
| default | q_chi2 | 0.76 | 0.90 (0.76) | 1/7 |
| default | llr / p_bayes | 0.63 / 0.64 | 0.68 / 0.56 | 1/7, 4/7 |
| default | gb_lofo | 0.99 | 1.00 | 3/7 |
| vetting | q_cur | 0.74 | 0.87 (0.66) | 4/6 |
| vetting | q_floor | 0.86 | 0.97 (0.94) | 4/6 |
| vetting | q_chi2 | 0.87 | 0.96 (0.91) | 4/6 |
| vetting | llr / p_bayes | 0.76 / 0.84 | 0.94 / 0.94 | 3/6, 4/6 |
| vetting | lr_lofo_noshape / gb_lofo | 0.87 / 0.99 | 1.00 / 1.00 | 6/6, 5/6 |

What it says:

- The catalogue set dominates the score formula. With `default` catalogues
  680 negatives per 15 fields pass `min_quality=0.2` (persistent uncatalogued
  stars, median 90% of epochs detected, reduced χ² 2.5), and no formula built
  from these features can separate a constant new source from them; every
  score puts GRB 090726 (18 mag) at rank 12-32. With `vetting` the same
  afterglow is rank 1 at k=3 under every score.
- `q_floor` is at least as good as `q_cur` on everything measured (same
  afterglow ranks and latency, +0.11 AUC, constant-source recall 0.66 -> 0.94
  in `vetting`). `q_chi2` has the best AUC/AP and 0-FP recall among the
  hand-built scores but drops the faint 220403B afterglow (rank 7, never
  rank 1 in 57 epochs, vs rank 1 at k=18 for q_cur/q_floor): the mag_range²
  term promotes faint sources through their photometric scatter, which is
  noise-driven but happens to help faint afterglows.
- Hand-built probabilities are badly calibrated (`p_bayes` predicts ~1 for
  617 default-config candidates of which 22% are real): the per-epoch
  evidence is over-counted because MAGERR_CALIB is underestimated (negatives
  have median reduced χ² 2.5) and epochs are not independent. A calibrated
  mapping of the existing score (`q_cal`, Brier 0.10/0.14, reliability within
  ~0.1 in every bin) is the cheapest honest probability.
- Learned scores look excellent on injected sources but that is largely the
  injection itself (unflagged, point-like donor rows; fading by 2 mag in the
  window). On real afterglows they are no better than q_cur in `vetting` and
  the bright, saturated GRB 210619B (44% flagged epochs) gets P ~ 1e-100 from
  the logistic model in both configs. Do not train on injections alone.

Found on the way (open):

- **The vetting configuration loses GRB 250813B** (14.9 mag afterglow, rank 1
  under `default`): USNO-B has an R=19.7 star 2.1" away (also Gaia G=19.8,
  PS1 r=20.1 at 2.6"), and with `unphotometered_match_is_new: false` plus the
  3" floor that counts as "known star" although the detection is 4.5 mag
  brighter. The positional veto must be magnitude-aware: a USNO-B (B/R/I) or
  Gaia (G) entry can only veto a detection within a plausible magnitude of
  it; otherwise type it `brightening`.
- Injection on GRB 250813B recovers 0/30 sources in both configurations
  although 263/600 injected rows were "detected" (MAGLIM 15.5-15.8, MAGLIMIT
  17.5-17.8): not traced.
- Two masked-value crashes fixed: `lightcurve.py` made `is_variable` a masked
  float when a light curve had masked photometry (vstack TableMergeError,
  i-band 211024B), and the final quality gate in `clustering.py` did
  `int()` on a masked `quality_score` (210312B, 210410A, 200410A with
  vetting catalogues); `scoring._row_features` now treats masked/non-finite
  cells as absent.

## Directions for improving detection (assessment after the 2026-09-03 score study)

Ranked by expected gain; the numbers refer to the score-comparison section
above.

1. **Catalogue vetting as the default, with a magnitude-aware veto.** Worth
   more than any score change (negatives above the gate ~45 -> ~2 per
   field; 18 mag afterglows rank 1 from the third epoch). Implemented
   2026-09-03: `unphotometered_veto_max_brightening_mag` (a USNO-B/Gaia-G
   entry only vetoes a detection within 2 mag of it). Validated on the 15
   replay fields with the vetting catalogues (real frames, k up to 20):
   GRB 250813B recovered (1.0", Q=39, typed `brightening`), afterglows
   found 6 -> 7, candidates 66 -> 68, other candidates above the gate
   42 -> 46 in total; frozen baseline unchanged (defaults untouched). Still needed before flipping the defaults: a galaxy
   veto where Pan-STARRS has no coverage (SkyMapper DR4 or GSC 2.4, see the
   southern-catalogue table above) and a regenerated frozen baseline.
2. **Photometric error model.** Constant stars have a median reduced χ² of
   2.5 against MAGERR_CALIB, so every variability term (mag_range, χ², the
   fading-trend significance, any likelihood-based score) is biased. The
   user notes pyrt-dophot already carries the statistics of the photometric
   model and its scatter (per-frame fit residuals, and more photometric
   parameters can be fitted where needed); the pipeline should derive an
   empirical error floor from those rather than from the SExtractor error
   alone. Do this before any χ²-based score or calibrated probability.
3. **Score:** turn on `new_source_variability_floor` (never hurts a real
   afterglow, same latency, constant-source recall 0.66 -> 0.94 at equal
   false positives); expose a calibrated probability to the frontend (a
   logistic map of ln q, reliable to ~0.1 in every bin). A full
   likelihood/mixture score only after (2).
4. **Forced photometry at candidate positions** against PS1/ATLAS pixel data
   ("Candidate enrichment" above): confirms or rejects a constant new
   source directly instead of waiting for it to fade.
5. **Depth: a running stack of the first N frames searched alongside the
   single frames.** Faint afterglows are limited by per-frame depth, not by
   the score (220403B rank 1 only at k=18, 151027B at 76 epochs).
   `detection/stacking.py` exists but is not run incrementally.
6. **Admission edge cases:** isolation factor computed against catalogue
   entries fainter than the frame (~2% of sources lost); i/z-band fields
   compared against Sloan r with only a warning (apply `core/color_model`);
   blending (200410A) needs subtraction, not thresholds.
7. **Operational:** observation-ID fragmentation still a replay-side
   workaround; astrometry failure starves clustering silently (180325A,
   3/20 frames).
8. **Validation realism before learned scoring:** injected sources are
   unflagged, perfectly point-like and fade by 2 mag in the window; real
   bright afterglows saturate (210619B: 44% flagged epochs) and faint ones
   barely fade. Add realistic flags/morphology from bright donors, an
   afterglow-like decay prior, the corrected error model, and trace why
   250813B recovers 0/30 injected sources in both configurations.

### Done 2026-09-04 on the list above

- **(2) Photometric error model.** Measured on the 15 replay fields
  (constant unflagged stars in >=70% of epochs, matched to the reference
  epoch): faint stars scatter as quoted (rms/MAGERR 1.0-1.1 at 17-19 mag),
  bright stars do not (12-15 mag: rms 0.02-0.04 mag vs quoted 0.003-0.01;
  0.1 mag on the poor night of 220403B). The excess is additive
  (rms^2 = err^2 + floor^2, floor 0.007-0.09 mag, median ~0.02) and mostly
  per-star, not zeropoint jitter (0.006-0.02 mag except 220403B, 0.09).
  WSSRNDF in the dophot header (median 435) is the catalogue-fit chi^2/NDF
  and does not give the light-curve repeatability. Implemented
  `lightcurve.estimate_magnitude_error_floor` (self-calibrated per
  campaign from the epochs themselves; `magerr_floor_auto` /
  `magerr_floor_mag` in DetectionConfig) and applied it to
  `mag_chi2_reduced` / `is_variable` (`mag_chi2_reduced_raw` keeps the old
  value; the weighted mean and quality_score are untouched, baseline
  passes). Offline, with the measured floors, the negatives' variability
  z-score drops from 2.5 to 0.9 and the chi^2 score variant becomes the
  best hand-built score: AUC 0.76 -> 0.84 (default catalogues, above
  q_cur 0.79) and 0.87 -> 0.90 (vetting), AP 0.97, 220403B afterglow rank
  7 -> 3 (`local_test_output/replay18/score_eval_floor/report.md`).
  A dophot-side improvement (fit residual scatter vs magnitude, more
  photometric terms where needed) would replace the campaign estimate
  with a per-frame one.
- **(3) Calibrated probability.** `p_real = sigmoid(a + b ln q)` as a
  candidates column and in the frontend when
  `score_probability_intercept/slope` are set (vetting catalogues
  0.840/0.843, Brier 0.10; historical gaia+usno -2.539/1.101). The
  variability floor is still opt-in (default flip = baseline change).
- **(6) Admission edge cases.** `isolation_max_mag_margin` (opt-in):
  the isolation/density statistics count only catalogue entries brighter
  than MAGLIMIT + margin. Catalogue comparison now uses the frame's own
  band (PHFILTER -> Sloan g/r/i/z, J) where the catalogue has it, with
  Sloan r as the fallback; r-band frames are unchanged. Blending is left
  to subtraction.
- **(7) Observation-ID fragmentation.** `resolve_observation_id`
  (io/observation_store.py): with `observation_grouping_radius_arcmin` set
  in the global config, an epoch whose frame centre lies within that radius
  of an existing observation's pointing (`obs_*/pointing.json`, written by
  `ObservationStore.record_pointing` after every processed epoch) and
  within `observation_grouping_max_gap_hours` of its last epoch joins that
  observation; otherwise the raw OBSID is used. Off by default. The
  daemon's debounce key still uses the raw OBSID (only batching, harmless).
  The reverse collision (two fields sharing one OBSID) is handled too:
  when the raw ID's store points elsewhere, the epoch opens `obs_<id>b`
  (`c`, ...) instead. Real instance: the fixed-OBSID 250813B set carries
  one frame 7.5 deg away (production `obs_95786`), which the replay merged
  into the field.
- **(8) Injection realism.** Bright injected sources (no unflagged
  point-like donor within 0.5 mag) now borrow saturated donors (FLAGS bit
  4, never blends), so they carry the flags and profiles real bright
  transients have. Still open: an afterglow-like decay prior for the
  injected light curves (the fixed alphas 0/0.5/1 with t0 = first frame -
  30 s fade far faster than the real late-time afterglows in the replay),
  The 0/30 recovery on GRB 250813B was that stray first frame: the
  injection drew its positions on it, so every injected source lay
  outside the real frames. `validation.replay.filter_epochs_by_pointing`
  (used by `inject_recover.py` and `score_eval_collect.py`) now drops
  epochs > 10' from the median pointing before anything else.
- **(5) Running stack: already there.** `stacking.maybe_build_stack_table`
  rebuilds the stack every `stacking_rebuild_interval` epochs from
  `stacking_min_epochs` on and appends it as an epoch, so the earlier
  statement that stacking is not run incrementally was wrong. The real
  gap is the trigger: the rebuild is skipped as soon as *any* candidate
  scores >= `stacking_score_threshold` (1.0), which with the historical
  catalogues is nearly always true (persistent stars), so a faint
  afterglow never gets its stack. With the vetting catalogues the rule
  is reasonable; alternatively key the skip on a candidate inside the
  GRB error box rather than on the field's best score.
