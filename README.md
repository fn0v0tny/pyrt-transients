# pyrt-transient

Transient detection addon for the [pyrt](https://github.com/mates14/pyrt) astronomical photometry pipeline.

Compares sources extracted by pyrt against multi-catalog reference data (Gaia, ATLAS refcat, USNO-B, PanSTARRS, DESI Legacy Survey) to identify transient candidates: new sources, variable stars brightening or fading beyond a significance threshold, and moving objects — then clusters detections across epochs into lightcurves and ranks them by a composite quality score.

---

## Requirements

- **pyrt** must be installed first — it provides the base `Catalog` class, photometric calibration, and produces the ECSV detection files this package consumes. **PyPI's `pyrt` package is an unrelated ray-tracer (`pip install pyrt` there gets you a rendering library, not this) — install the real one from its actual source** (https://github.com/mates14/pyrt, or a local checkout: `pip install -e /path/to/pyrt`).
- **[stdpipe](https://github.com/karpov-sv/stdpipe)** — used for coordinate matching (`stdpipe.astrometry`) and VSX/SkyBoT candidate filtering (`stdpipe.pipeline.filter_transient_candidates`). Tested against PyPI's `0.4.1` (`pip install stdpipe`) — see "stdpipe version compatibility" below for the two version-specific workarounds this package carries; both are guarded/self-selecting, so an older stdpipe build (this project was originally developed against an older checkout with a different API in a few places) still works too.
- **sep-x** comes in as stdpipe's own dependency (its `photometry` module does `import sep_x as sep`, falling back to plain `sep` on older checkouts); nothing here imports it directly. `detection/subtraction/extraction.py`'s `_patch_sep_sum_circle_clip_kwargs` reads `stdpipe.photometry`'s own `sep` attribute so it patches whichever module stdpipe actually bound.
- Python ≥ 3.10 (stdpipe uses PEP 604 union-type syntax internally; tested on 3.11).

### stdpipe version compatibility

This package carries two workarounds for real bugs found in some `stdpipe` builds, both self-guarding so they're safe to run against any stdpipe version (old or new) without picking one at install time:

- **`_patch_sep_sum_circle_clip_kwargs`** (`detection/subtraction/extraction.py`) — some stdpipe builds' `get_objects_sep` unconditionally passes `clip_sigma`/`clip_iters` to a plain `sep.sum_circle()` call that has never accepted them, crashing every SEP detection. Patched to retry once without those kwargs on exactly that `TypeError`; a `sep`/`sep_x` version that *does* accept them is unaffected (the first call just succeeds, so the retry path never runs).
- **`_patch_normalize_ps1_skycell`** (`detection/subtraction/templates.py`) — an older stdpipe's `normalize_ps1_skycell(filename, outname=None, verbose=False)` crashes on real PS1 skycell downloads (`ValueError` inside astropy's compressed-tile decompression of certain BLANK-valued masks); worked around with a `fitsio`-based reimplementation. Verified against `0.4.1` that this signature changed to an in-memory `normalize_ps1_skycell(image, header, verbose=False)`, where the bug is apparently already fixed upstream — applying the old patch there unconditionally would break every PS1 call, so it now inspects the installed function's parameter names first and only patches the old `filename`-based form, using stdpipe's own implementation otherwise.

Optional, only needed for the subtraction strategy (see "SN search" below):

- **[HOTPANTS](https://github.com/acbecker/hotpants)** — `subtraction_engine: hotpants` (the default), via `stdpipe.subtraction.run_hotpants`.
- **[PyZOGY](https://github.com/dguevel/PyZOGY)** — `subtraction_engine: zogy`.
- **SWarp** — reprojecting an external-survey template (`template_source: ps1`/`legacysurvey`) onto the science WCS, via `stdpipe.templates`. Not needed for `template_source: own_epoch`.
- **[fitsio](https://github.com/esheldon/fitsio)** (`pip install pyrt-transient[ps1]`) — optional; enables `_patch_normalize_ps1_skycell` above. Without it, some PS1 template fetches may fail on an older stdpipe build where they would otherwise succeed (a no-op on a newer one that's already fixed upstream).

---

## Installation

```bash
# Install pyrt first -- NOT `pip install pyrt` (that's an unrelated PyPI
# ray-tracer package). Install the real one from source instead:
pip install -e /path/to/pyrt   # or: pip install git+https://github.com/mates14/pyrt

# Install stdpipe (tested against 0.4.1)
pip install stdpipe

# Install the transient detection package
pip install pyrt-transient

# With optional frontend (cutout images + HTML candidate browser)
pip install pyrt-transient[frontend]

# With the PS1-skycell fitsio workaround (see "stdpipe version compatibility" above)
pip install pyrt-transient[ps1]
```

---

## How it fits with pyrt

```
pyrt (photometry pipeline)
  └── produces:  <obsid>.ecsv    (source detections + WCS + photometric model)
                 <obsid>.fits    (calibrated image)

pyrt-transient (this package)
  └── consumes:  <obsid>.ecsv + <obsid>.fits, one epoch at a time
  └── produces:  candidates.tbl        (ranked transient candidates, all epochs combined)
                 lightcurve_summary.json
                 <public_dir>/obs_<id>/  (HTML browser + cutout images + lightcurve plots)
```

The integration point is the ECSV file format — there is no Python-level dependency on pyrt beyond the `Catalog` base class.

---

## Quick start

### Run on a single epoch

```bash
pyrt-transient-pipeline image.ecsv image.fits --output-dir=/path/to/data/dir
```

Each invocation is incremental: it loads all previously-processed epochs for the same observation from `--output-dir`, adds the new one, and re-runs clustering/lightcurve analysis over the full accumulated set. Re-running on an already-processed file is a cheap no-op.

With frontend generation:

```bash
pyrt-transient-pipeline image.ecsv image.fits --output-dir=/path/to/data/dir --generate-frontend
```

With a config file:

```bash
pyrt-transient-pipeline image.ecsv image.fits --config=config.yaml
```

### Run the daemon (watch a socket for incoming files from pyrt)

```bash
pyrt-transient-daemon
```

The daemon listens on a Unix socket, receives ECSV + FITS paths, and launches `pyrt-transient-pipeline` as a subprocess automatically (up to `PYRT_TRANSIENT_MAX_PARALLEL` concurrent runs). Socket, work/log directories, the pipeline command and its extra arguments (e.g. `--config=`) are set by `PYRT_TRANSIENT_*` environment variables -- see the module docstring of `transient_daemon.py`. It applies a debounce window so a burst of images from the same observation is processed as one batch rather than N overlapping runs — concurrent invocations for the *same* observation directory still serialize on `ObservationStore`'s file lock, so parallelism across observations is what actually helps throughput, not parallelism within one.

### Real-time deployment alongside pyrt

In our own deployment, this package runs continuously alongside pyrt's own image-processing pipeline, not standalone:

1. As each new raw frame arrives, pyrt's own pipeline (astrometry + aperture photometry, e.g. via `dophot3`/`phcat`) calibrates it and writes the `<obsid>.ecsv` + `<obsid>.fits` pair.
2. That same pipeline run opens a connection to `transient_daemon.py`'s Unix socket and sends `{"ecsv_path": ..., "fits_path": ...}` as its very last step, once the pair exists on disk.
3. `transient_daemon.py` (already running as a long-lived background process) receives this, debounces bursts from the same observation, and invokes `pipeline_magic.py <ecsv> <fits>` as a subprocess — which loads every previously-processed epoch for that observation, adds the new one, re-clusters, and rewrites `candidates.tbl` (and the website, if `generate_frontend` is set).

So pyrt and pyrt-transient are two independent long-running processes on the same host, connected only by the ECSV/FITS file pair and a one-line socket message — pyrt never imports this package, and this package never imports pyrt's astrometry/photometry code, only its `Catalog` base class (see "How it fits with pyrt" above). This keeps the transient-detection side fully restartable/upgradable without touching the (much more expensive to get wrong) astrometric pipeline.

---

## SN search (image subtraction)

`pipeline_magic_sn.py` (entry point: `pyrt-transient-sn-pipeline`) is a second, independent pipeline built around `config.detection.strategy`:

- `blind_multicatalog` (default) — the same catalog cross-match approach as `pipeline_magic.py`, plus an SN-specific post-filter chain: morphology, magnitude plausibility, SkyBoT asteroid rejection, high-proper-motion star rejection, HyperLEDA host-galaxy scoring, TNS cross-match, and a composite `sn_score`.
- `subtraction` — differencing against a template (`detection/subtraction/`) instead of catalog cross-matching. Constant sources vanish in the difference image, so essentially any significant residual is a genuine candidate — this sidesteps several structural recall limits catalog cross-matching has (significance gates, unanimous-catalog requirements, single-exposure depth). The same SN post-filter chain above still runs on top, strategy-agnostic.

### Subtraction: two ways to feed it

`config.detection.diff_input_mode` controls what `ecsv_file`/`fits_file` are:

- `prebuilt` (default) — they're already a diff-image pair (e.g. produced by an external pipeline). The science sibling is located automatically via a naming convention (`0429rh.fits` ↔ `0429r.fits`) to borrow calibration meta the diff catalog itself doesn't carry.
- `raw` — they're a raw science epoch. The pipeline builds everything itself: picks a template (`template_source`), differences against it (`subtraction_engine`), detects and calibrates sources on the result (`photometric_catalog`), then feeds that into the same candidate pipeline as `prebuilt` mode.

```yaml
detection:
  strategy: subtraction
  diff_input_mode: raw
  template_source: own_epoch      # or "ps1" / "legacysurvey"
  subtraction_engine: hotpants    # or "zogy"
  photometric_catalog: ps1
  min_n_detections: 2
```

```bash
pyrt-transient-sn-pipeline image.ecsv image.fits \
  --config=sn_config.yaml --output-dir=/path/to/data/dir \
  --target-positions=228.988292,56.309083 --generate-frontend
```

**Template choice matters.** `own_epoch` reuses one of the campaign's own prior epochs (picked by `detection/reference_frame.py`'s `ReferenceFrameSelector`) — no external dependency, but it reveals only the *change* relative to that epoch, not the target's absolute brightness. That's the right tool for a brand-new appearance or a sudden change, and the wrong one for continued monitoring of an already-known, slowly-evolving source (an external, genuinely quiescent template — `ps1`/`legacysurvey` — is needed there for meaningful absolute photometry). Pass `--target-positions` whenever it's known: `ReferenceFrameSelector` uses it to avoid picking a reference epoch that already has the target in it, and warns explicitly when every candidate epoch does. See `FUTURE_IDEAS.md` for the full analysis and the real campaign this was found against.

Both differencing engines write a diff FITS with a `TEMPLATE` header keyword recording which template/provenance was used; `frontend_generator.py` uses this to render a science/template/difference triplet per epoch in the candidate browser (falling back to the single-image view for `blind_multicatalog` candidates, which have no template to show).

---

## Image stacking (GRB pipeline)

Some sources are genuinely fainter than any single exposure's own limiting magnitude (`MAGLIM`) — no amount of per-epoch threshold tuning recovers them, only reaching deeper via co-addition does. `pipeline_magic.py` (the GRB/`blind_multicatalog` pipeline) handles this automatically via `detection/stacking.py`: once enough same-field epochs have accumulated and no candidate found so far scores convincingly, it runs `pyrt-combine` (part of the already-required `pyrt` package — no extra install) over the accumulated epochs and feeds the resulting deeper image into the exact same candidate pipeline as one more epoch, run **in parallel with**, not instead of, per-epoch detection.

It's a try-harder fallback, not a replacement strategy: skipped entirely once an existing candidate already scores at or above `stacking_score_threshold`, so it doesn't spend the extra `pyrt-combine` runtime once something convincing has already been found — but once a stack has been built, it stays included in every subsequent run regardless of later scoring, so a stack-anchored candidate doesn't disappear once it's done its job. Only same-filter, same-exposure-time epochs are ever combined together — `pyrt-combine` does not normalize for either.

```yaml
detection:
  stacking_enabled: true          # default
  stacking_min_epochs: 10         # don't attempt a stack before this many real epochs exist
  stacking_max_epochs: 20         # cap on frames fed to pyrt-combine per stack
  stacking_rebuild_interval: 5    # re-stack only after this many more real epochs accumulate
  stacking_score_threshold: 1.0   # skip stacking once an existing candidate scores at least this well
```

`MAGLIM` for the stack is measured empirically from its own actually-detected sources (checked against the stack's own directly-measured background noise, not the per-source flux-error model `get_objects_sep` uses — that model assumes single-exposure statistics that don't hold once frames have been combined), not derived from a formula or borrowed from any single input epoch.

### Stack-only candidates

The stack is a single epoch for the clustering, so a source that the stack detects but single
frames mostly miss can never collect `min_n_detections` detections. For such stack-only
candidates, `detection/blind_multicatalog/forced.py` measures aperture photometry at the
candidate's position in the frames the stack was built from. It admits the candidate when this
forced lightcurve shows a persistent source:

- SNR ≥ `stack_forced_snr` in at least `stack_forced_min_fraction` of those frames, and
- no single frame carrying more than `stack_forced_max_flux_fraction` of the positive flux.

The second test rejects cosmic rays, hot pixels and satellite glints that were averaged into
the stack (`pyrt-combine` has no per-pixel rejection), because all their flux is in one frame.
The stack ECSV lists its input frames in `STACK_INPUTS`, so the forced measurements use
exactly those frames.

An admitted candidate gets its forced points (`FORCED = True`) as its lightcurve, is scored
like any other candidate, and is marked `admission = stack+forced`. Forced points are never
clustered into new candidates, so the step cannot feed back into itself. It costs about
25 ms per frame, almost all of it reading the image and estimating the background.

On GRB 190919B (`tests/190919B`), the 20-frame stack's afterglow reaches SNR > 3 in 24 of 40
frames and is admitted, while the three single-frame flashes in the same stack reach it in 1
of 40 each and are rejected.

```yaml
detection:
  stack_forced_admission: true        # default
  stack_forced_snr: 3.0
  stack_forced_min_fraction: 0.3
  stack_forced_max_flux_fraction: 0.5
  stack_forced_min_frames: 5
  stack_forced_max_candidates: 50     # stack-only candidates measured per run, by quality
  stack_forced_lc_snr: 2.0            # forced points at or above this form the lightcurve
  stack_forced_aperture_fwhm: 1.0     # aperture radius in units of the frame FWHM
```

---

## Follow-up exposure recommendation

After detection, `pipeline_magic.py` answers "how long does the next frame need to be?" for the candidates it just found, using the conditions of the epoch it just processed — the sky background, seeing, zeropoint and exposure time of that real frame, scaled to a different exposure time via an empirically calibrated noise model (`followup/exposure.py`, RMS 0.047 dex in log(magerror); verified against every measured source in `tests/210619B` at median +0.01 dex, rms 0.031).

Every candidate row gains `followup_mag` (the magnitude planned against) and `followup_exptime_s`. The highest-scoring candidate additionally gets `followup_exposure.json` in the observation directory, with the reference conditions used, the predicted SNR/magerror actually achieved at the recommended time, the limiting magnitude that exposure reaches, and an `exptime_range_s` reflecting the planning magnitude's own uncertainty.

The planning magnitude is the source's *latest* lightcurve point (not its best epoch) when that point is well measured (`max_planning_magerr`); a noisier one — admission goes down to `new_source_siglim`, i.e. ~0.7 mag — is averaged with the preceding points instead, since exposure time scales roughly as 10^(0.8·Δm). Only points from epochs in the reference frame's filter are used when any exist; otherwise the report flags `filter_mismatch` rather than silently mixing bands.

```yaml
followup:
  exposure_enabled: true      # default
  target_snr: 10.0            # what the recommended exposure aims for
  max_planning_magerr: 0.2    # a latest point noisier than this is averaged, not trusted alone
  readout_noise_e: 8.0        # camera-specific; the model's calibration value
  min_exptime_s: 1.0          # the answer is clamped here for very bright candidates
  max_exptime_s: 3600.0       # beyond this the report says "unreachable", not a number
```

This is enrichment, not detection: it runs *on* a strategy's `(candidates, lightcurves)` output, so it never influences what is found, and it cannot abort the run that produced them (`run_enrichment` never raises). Anything it can't do is reported explicitly — a report always carries a `status` of `ok`/`skipped`/`error` plus a reason, so "couldn't plan an exposure" never reads like "no exposure needed". The reference epoch is the most recently *observed* real frame (by mid-exposure time, not file order); stack epochs are skipped, since a co-add's exposure time isn't one a telescope can re-take and its background sits below the readout floor the model needs.

The model's constants are a fit to one instrument, so every run also checks it against the reference frame's own measured photometric errors and records the result as `model_check` (`rms_dex` over the frame's sources; a warning is logged above 0.1 dex). On another camera that number, not a crash, is what tells you the recommendation can't be trusted.

Only exposure time is computed. Filter choice, EMCCD on/off, and the time-since-trigger term of the original "telescope strategy suggester" idea remain deferred (see `FUTURE_IDEAS.md`).

---

## Validation: injection-recovery and replay

`pyrt_transient/validation/` plus two CLIs regenerate the completeness,
latency and purity numbers for a field from its shipped catalogues:

```bash
# Inject synthetic fading point sources into every epoch's catalogue and
# replay the blind-multicatalog strategy one epoch at a time
python tools/inject_recover.py tests/210619B --out local_test_output/inject \
    --realisations 8 --sources 40 --mag-min 14 --mag-max 20 --alphas 0 0.5 1

# Replay a real field and track a known target (default: GRB_RA/GRB_DEC in the meta)
python tools/replay_driver.py tests/210619B --out local_test_output/replay_210619B
```

Injection is at the catalogue level (`validation/injection.py`): positions go
through the frame's own WCS (SIP or ZPN included), scattered by the frame's
astrometric residual (pyrt's ASTSCATT, or ASTSIGMA from pyrt before
mates14/pyrt 97101a7, converted from pixels to arcsec), magnitudes
follow `m(t) = m0 + 2.5 alpha log10(t/t1)` since a trigger `t0`, noise is
Gaussian in flux with the error the `followup.exposure` model predicts for
that frame, every other column is copied from a real, unflagged, point-like
star of the same brightness in the same frame, and a source below the frame's
faintest-detection limit (`MAGLIMIT`) is absent from the catalogue. It tests
the pipeline's recall and latency *given* a detection — not SExtractor, not
blending, not pixel-level artefacts.

Outputs per run: `recovery.ecsv` (one row per injected source: peak magnitude
relative to `MAGLIM`, epochs detected, recovered or not, first epoch and
seconds-since-`t0` at which it was reported, final score and rank),
`spurious.ecsv` (candidates matching neither an injected source nor a known
real target, vs. epoch count), `completeness.pdf`, `latency.pdf`,
`spurious.pdf`, `summary.json`.

### Constant sources and the vetting configuration

With the default configuration a *constant* new source — however bright — is
never reported: the final score multiplies in the light-curve magnitude range
twice, so a light curve whose only variation is noise scores ~0.01 and is
removed by `min_quality`. The injection run that found this also found why
the term is load-bearing: two upstream defects were producing ~180
persistent "new" candidates per field, and the magnitude-range term was the
only thing hiding them.

- `pyrt`'s Gaia query keeps only calibrator-quality stars (`ruwe < 1.4`,
  BP/RP present, ...). Right for photometric calibration, wrong for vetting:
  ~9% of real 12–17 mag stars are absent, and each one is a persistent "new"
  source. `catalogs: [gaia_full, ...]` uses the same query without the cuts.
- USNO-B carries no Sloan bands, so none of its stars has "valid" photometry
  and every match was reported as `new` — USNO-B contributed nothing to the
  unanimity rule. `unphotometered_match_is_new: false` reports such a match
  as matched-with-unknown-photometry (not a candidate) instead — unless the
  detection is brighter than the brightest plate/broad-band magnitude of
  the matched entries by more than
  `unphotometered_veto_max_brightening_mag` (default 2.0), in which case it
  is a `brightening` candidate: a 19.7 mag USNO-B star 2" away cannot be a
  14.9 mag afterglow (GRB 250813B was lost to exactly that veto).

Recommended vetting configuration (validated in `FUTURE_IDEAS.md`,
"Constant new sources"):

```yaml
detection:
  # atlas@vizier when there is no local ATLAS; usno is the galaxy veto that
  # takes over south of Dec -30 where Pan-STARRS reports "unavailable"
  catalogs: [gaia_full, atlas@localhost, panstarrs@vizier, usno]
  unphotometered_match_is_new: false    # a USNO-B match is a star, not a "new" source
  unphotometered_veto_max_brightening_mag: 2.0   # ... unless the detection is >2 mag brighter than it
  catalog_match_floor_arcsec: {gaia: 3.0, atlas: 3.0, panstarrs: 3.0, usno: 3.0}   # D50 vs catalogue scatter: p99 ~2.5"
  new_source_variability_floor: true    # constancy never penalises a "new" source
  isolation_max_mag_margin: 0.5         # isolation counts only catalogue stars brighter than MAGLIMIT + 0.5
  score_probability_intercept: 0.840    # p_real = sigmoid(0.840 + 0.843 ln quality_score), fitted for this
  score_probability_slope: 0.843        # catalogue set on the 15-field replay (historical gaia+usno: -2.539, 1.101)
```

Measured on the 210619B field with catalogue-level injection (4 x 40
sources, 14-20 mag, constant/fading), default vs. this configuration:
recovered fraction of sources detectable in >=3 epochs 68% -> 94%, constant
sources 14% -> 93%, fading sources 96% -> 95%, spurious candidates per field
0 -> 0, real afterglow score 56.5 -> 57.6, median latency 3 epochs both.

Which catalogues (measured on the same field, see `FUTURE_IDEAS.md`): Gaia
(`gaia_full`) and ATLAS refcat2 are each complete to the D50 single-frame
depth with usable photometry, and together give 113/118 recovery — but
both are *point-source* catalogues, and four 18th-mag galaxies in the field
then survive as persistent low-score (0.35–0.65) candidates. USNO-B has
them (photographic plates include extended objects), so keeping `usno` as a
purely positional veto (it can only veto once `unphotometered_match_is_new`
is false) brings the spurious count back to 0 at no cost in recovery. Do not
use USNO-B for photometry (0.85 mag scatter against D50).

Pan-STARRS is the better galaxy-inclusive veto north of Dec −30: with
`[gaia_full, atlas@vizier, panstarrs@vizier]` the same injection gives
113/118 detectable sources recovered (constant 40/42) and 0 spurious
candidates in every realisation — the best of every combination measured. Two
catalogue names are provided: `panstarrs@vizier` (PS1 DR1 mean photometry
from VizieR II/349/ps1 — a 0.3° box in ~1 s, cached, includes galaxies,
`Sloan_*` columns) and `panstarrs` (PS1 DR2 from MAST, deeper and slower;
this package overrides `pyrt`'s implementation, whose MAST criteria syntax
the current API rejects). Both return "unavailable" south of Dec −30
without querying.

### Catalogues without coverage: the pipeline works everywhere

A reference catalogue that has no coverage of the field (Pan-STARRS south
of −30, SDSS and Legacy Survey off-footprint), returned no rows, or failed
to download is now **excluded from the agreement requirement** for that
field, with a warning naming it. Previously such a catalogue was inserted
as an empty candidate table, and because `min_catalogs_fraction: 1.0`
counts every listed catalogue, it vetoed every source: a field outside any
one catalogue's footprint silently produced zero candidates. A catalogue
that *was* searched and flagged nothing still participates (that is a real
veto). If no catalogue at all is available the run logs an error and
produces no candidates. So a catalogue list like
`[gaia_full, atlas@localhost, panstarrs@vizier, usno]` is safe at any
declination: south of −30 it degrades to the other three.

All four default to the historical behaviour, so the regression baseline is
unchanged unless you opt in. Enable `new_source_variability_floor` only
together with the other three: on its own it turns every star missing from
a quality-cut catalogue into a high-scoring candidate.

## Archive validation

`tools/replay_archive.py` replays every burst of a target list over the
fixed-OBSID ECSV archive and writes the validation website (`index.html` +
per-burst `track.png`, `snapshots.json`, `summary.json`); `--resume` skips
bursts already done.

---

## Python API

```python
from pyrt_transient import BlindMulticatalogStrategy, ObservationStore, PipelineConfig
from pyrt_transient.catalog import QueryParams
from pyrt_transient.extraction_manager import ImageExtractionManager
from pyrt_transient.transients import open_ecsv_file

config = PipelineConfig()
store = ObservationStore(config.base_data_dir, observation_id="12345")

detection_tables, _ = store.load_existing_tables()
detection_tables.append(open_ecsv_file("new_epoch.ecsv"))

image_manager = ImageExtractionManager(detection_tables)
ra, dec = image_manager.field_center
params = QueryParams(ra=ra, dec=dec, width=0.5, height=0.5, mlim=20)

strategy = BlindMulticatalogStrategy(data_dir=store.obs_dir, config=config)
candidates, lightcurves = strategy.run(detection_tables, config=config, params=params)

store.save_results(candidates, lightcurves)
print(f"{len(candidates)} candidates found")
```

`BlindMulticatalogStrategy` is the production detection strategy — it's what `pipeline_magic.py` actually calls.

---

## Supported catalogs

| Key | Source | Notes |
|-----|--------|-------|
| `atlas@localhost` | ATLAS refcat (local install) | Sloan griz + J; not reachable outside a host with a local install |
| `gaia` | Gaia DR3 (ESA TAP) | G, BP, RP |
| `usno` | USNO-B1.0 (VizieR) | B1 R1 B2 R2 I; low Sloan coverage |
| `panstarrs` | PanSTARRS DR2 (MAST) | grizy |
| `legacysurvey` | DESI Legacy DR10 (NOIRLab TAP) | grz; requires `pyvo` |

Remote catalogs (all except `atlas@localhost`) are cached to disk via `setup_catalog_cache()`.

---

## Configuration

Detection behaviour is controlled by `DetectionConfig` (post-detection enrichment by `FollowupConfig`) — see `pyrt_transient/config_trans.py` for the full list of fields and defaults (matching radii, adaptive-radius parameters, trail/moving-object thresholds, quality-score weights, which catalogs to query). A YAML config file can be passed via `--config=`:

```yaml
detection:
  min_n_detections: 3
  min_catalogs_fraction: 0.5
  catalogs: [gaia, usno]
base_data_dir: /data/transient_work
base_public_dir: /var/www/transients
generate_frontend: true
```

---

## Package layout

```
pyrt_transient/
├── pipeline_magic.py        Entry point: pyrt-transient-pipeline (thin CLI)
├── pipeline_magic_sn.py     Entry point: pyrt-transient-sn-pipeline (SN search, see above)
├── transient_daemon.py      Entry point: pyrt-transient-daemon
├── config_trans.py          PipelineConfig, DetectionConfig (dataclasses)
│
├── core/                    Pure, dependency-free building blocks
│   ├── matching.py            Radius-based coordinate matching (wraps stdpipe.astrometry)
│   ├── radii.py                Per-detection adaptive matching radius
│   ├── scoring.py               Quality-score computation
│   ├── candidate.py             Candidate dataclass (not yet the strategy return type -- see detection/base.py)
│   ├── epochs.py, union_find.py, timeutil.py, fileutil.py, config_loader.py
│
├── io/                      Filesystem-backed observation state
│   ├── observation_store.py   ObservationStore: processed-file tracking, locking, results
│   ├── logging_setup.py, naming.py
│
├── detection/               Detection strategies
│   ├── base.py                 DetectionStrategy ABC
│   ├── reference_frame.py       ReferenceFrameSelector (multi-image reference-frame selection,
│   │                             optionally target-aware -- see "SN search" above)
│   ├── stacking.py              Image co-addition for the GRB pipeline -- see "Image stacking" above
│   ├── blind_multicatalog/      Production strategy: catalog compare -> cluster -> score -> plot
│   │   ├── __init__.py            BlindMulticatalogStrategy (the orchestrator)
│   │   ├── catalog_query.py       Per-run catalog loading/caching
│   │   ├── catalog_match.py       Per-catalog candidate detection
│   │   ├── stdpipe_filters.py     VSX (positional) / SkyBoT (per-epoch) rejection filters
│   │   ├── clustering.py          Cross-catalog/cross-epoch clustering + lightcurve combination
│   │   ├── lightcurve.py          Lightcurve building and stats
│   │   ├── trail_detection.py     Motion/trail features for moving-object candidates
│   │   └── plotting.py            Lightcurve plots
│   └── subtraction/              SN-search strategy: difference against a template -> detect -> score
│       ├── __init__.py            SubtractionStrategy (the orchestrator)
│       ├── templates.py           Template acquisition: own-epoch / PS1 / LegacySurvey, with cache
│       ├── differencing.py        HOTPANTS / PyZOGY differencing engines
│       ├── extraction.py          SEP-based detection + science-image-calibrated photometry
│       ├── candidates.py          Diff-detection -> Candidate table, science-meta borrowing
│       └── artifact_filters.py    Morphology/magnitude filters + dipole-artifact rejection
│
├── followup/                Candidate enrichment, run on a strategy's output
│   ├── exposure.py             Empirical exposure-time/noise model (magnitude <-> SNR <-> exptime)
│   └── enrichment.py            Annotates candidates + reports on the top one -- see above
│
├── web/                     Frontend generation
│   ├── orchestration.py        generate_frontend() entry point
│   └── site_state.py            Checksum-based regeneration gating
│
├── catalog.py                CatTransients -- Catalog subclass, all detection methods
├── extraction_manager.py      Image + ECSV file management (.field_center)
├── transients.py              CLI utilities
├── frontend_generator.py      HTML candidate browser + cutout images [optional: Pillow]
├── fotfit.py, termfit.py       Bundled photometric/term fitters (from pyrt)
├── template/, template_sn/    Frontend HTML/JS templates
│
tools/
├── generate_baseline.py       Snapshot current pipeline output as a regression baseline
├── check_baseline.py          Diff current output against the baseline; fails on any candidate/
│                               quality_score/timing drift beyond tolerance
└── fixture_runner.py          Shared harness both scripts build on

tests/
└── test_*.py                  Unit tests (core/matching, core/radii, core/scoring,
                                reference_frame, observation_store, site_state,
                                detection_subtraction)
```

See `FUTURE_IDEAS.md` for known gaps, deferred work, and design decisions still open.

---

## License

MIT
