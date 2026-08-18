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
importable). Confirmed fixed: a real PS1 fetch got past skycell
normalization after the patch, then correctly reported "missing `swarp`
binary" (this dev machine doesn't have SWarp installed) rather than
crashing — full PS1-template reprojection therefore still needs
validating end-to-end on a host with `swarp` installed.

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
