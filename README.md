# pyrt-transient

Transient detection addon for the [pyrt](https://github.com/mates14/pyrt) astronomical photometry pipeline.

Compares sources extracted by pyrt against multi-catalog reference data (Gaia, ATLAS refcat, USNO-B, PanSTARRS, DESI Legacy Survey) to identify transient candidates: new sources, variable stars brightening or fading beyond a significance threshold, and moving objects — then clusters detections across epochs into lightcurves and ranks them by a composite quality score.

---

## Requirements

- **pyrt** must be installed first — it provides the base `Catalog` class, photometric calibration, and produces the ECSV detection files this package consumes.
- **[stdpipe](https://github.com/karpov-sv/stdpipe)** — used for coordinate matching (`stdpipe.astrometry`) and VSX/SkyBoT candidate filtering (`stdpipe.pipeline.filter_transient_candidates`). Pin to a specific commit rather than "latest" — stdpipe is under active development and its API has moved (e.g. `stdpipe.artefacts` didn't exist a few months prior to this being written).
- Python ≥ 3.10 (stdpipe uses PEP 604 union-type syntax internally; tested on 3.11).

Optional, only needed for the subtraction strategy (see "SN search" below):

- **[HOTPANTS](https://github.com/acbecker/hotpants)** — `subtraction_engine: hotpants` (the default), via `stdpipe.subtraction.run_hotpants`.
- **[PyZOGY](https://github.com/dguevel/PyZOGY)** — `subtraction_engine: zogy`.
- **SWarp** — reprojecting an external-survey template (`template_source: ps1`/`legacysurvey`) onto the science WCS, via `stdpipe.templates`. Not needed for `template_source: own_epoch`.
- **[fitsio](https://github.com/esheldon/fitsio)** — optional; works around a real bug in some `stdpipe` builds' PS1 skycell normalization (`normalize_ps1_skycell` raises on certain BLANK-valued compressed masks). Without it, some PS1 template fetches may fail where they would otherwise succeed.

---

## Installation

```bash
# Install pyrt first
pip install pyrt

# Install stdpipe (pin a known-good commit)
pip install git+https://github.com/karpov-sv/stdpipe.git@<commit>

# Install the transient detection package
pip install pyrt-transient

# With optional frontend (cutout images + HTML candidate browser)
pip install pyrt-transient[frontend]
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

The daemon listens on a Unix socket, receives ECSV + FITS paths, and launches `pipeline_magic.py` as a subprocess automatically (up to `MAX_PARALLEL_PROCESSES` concurrent runs). It applies a debounce window so a burst of images from the same observation is processed as one batch rather than N overlapping runs — concurrent invocations for the *same* observation directory still serialize on `ObservationStore`'s file lock, so parallelism across observations is what actually helps throughput, not parallelism within one.

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

Detection behaviour is controlled by `DetectionConfig` — see `pyrt_transient/config_trans.py` for the full list of fields and defaults (matching radii, adaptive-radius parameters, trail/moving-object thresholds, quality-score weights, which catalogs to query). A YAML config file can be passed via `--config=`:

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
