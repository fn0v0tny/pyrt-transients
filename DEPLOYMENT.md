# Deploying pyrt-transient on a telescope machine

pyrt-transient runs in production on one host only, lascaux50 (D50 images).
It does not run at SBT or FRAM yet. This guide covers setting it up on
another machine, and the lascaux50 reference deployment is described at
the end. Everything here was checked against lascaux50 on 2026-09-11,
where commit 74060f1 is deployed.

## 1. How it fits together

```
raw frame -> pyrt chain (astrometry + photometry)   [already runs at the telescope]
          -> <frame>.ecsv + WCS-solved <frame>.fits
          -> one JSON message on a Unix socket
          -> pyrt-transient-daemon                  [long-running service]
               debounces a burst of frames from the same observation, then per frame:
          -> pyrt-transient-pipeline <ecsv> <fits>  [one subprocess per frame]
               adds the epoch to <base_data_dir>/obs_<id>/, re-clusters all epochs,
               stacks and does forced photometry once there are enough epochs,
               writes candidates.tbl and the web page <base_public_dir>/obs_<id>/
```

pyrt never imports this package. The telescope side needs only three
things: the ECSV/FITS pair on disk, one socket message, and read access for
the daemon's user. You can upgrade or restart the transient side without
touching the astrometry chain.

## 2. Before you start

- **Machine:**
  - Linux with Python 3.10 or newer (3.11 is what production runs).
  - lascaux50 has 8 cores and 15 GB RAM. The pipeline takes about 4 min
    per frame late in a 42-frame night there, which is dominated by the
    catalogue matching of all accumulated epochs.
  - Disk: obs_104210 (42 D50 frames) takes 274 MB, including the FITS
    copies. Plan for roughly 7 MB per frame per night, plus the catalogue
    cache.
- **pyrt must already produce ECSVs on the machine.** It must also refit
  the astrometry: run dophot twice with `-a`, as `get_ecsv.sh` does. Only
  then does the ECSV meta carry `ASTSIGMA`/`ASTVAR`, which set the
  matching radii. Both the older flat-layout pyrt (`dophot3.py`) and
  mates14/pyrt main (from 97101a7 on, which also writes `ASTSCATT`) are
  supported.
- **Network:**
  - Outbound HTTPS to VizieR, the Gaia TAP and MAST for catalogue queries.
    Results are cached in `~/catalog_cache` of the daemon's user.
  - Only lascaux50 has a local ATLAS (`atlas@localhost`).
- **Accounts:** a dedicated unprivileged service account, e.g. `pyrt`. You
  need root only for a system-wide systemd unit; a user unit works without
  it, see section 5.
- **Optional:**
  - A web server serving `base_public_dir`, for the candidate pages.
  - HOTPANTS or PyZOGY plus SWarp, only for the SN/subtraction pipeline.

## 3. Install

```bash
sudo -u pyrt -i
python3.11 -m venv ~/venv
. ~/venv/bin/activate

# pyrt from source -- NOT `pip install pyrt` (that is an unrelated ray-tracer).
# Use the same pyrt that writes the ECSVs on this machine.
pip install -e /path/to/pyrt            # or: pip install git+https://github.com/mates14/pyrt

pip install stdpipe                     # tested with 0.4.1

git clone https://github.com/fn0v0tny/pyrt-transients.git ~/src/pyrt-transient
pip install -e "~/src/pyrt-transient[frontend,ps1]"
```

**What the extras are for:**

- `frontend` gives the cutout and HTML pages.
- `ps1` installs `fitsio`, which is needed for the two PS1-skycell
  workarounds in `detection/subtraction/templates.py`.

**Check the install:**

```bash
which pyrt-transient-pipeline pyrt-transient-daemon pyrt-combine
python -c "import pyrt_transient, stdpipe; print(pyrt_transient.__file__)"
cd ~/src/pyrt-transient && python -m pytest -q -p no:cacheprovider
```

- `pyrt-combine` comes from pyrt. Stacking calls it through `PATH`, and
  if it is missing the pipeline only logs a warning and silently never
  stacks. On lascaux50 it is in `/usr/local/bin`.
- Tests that need FITS fixtures skip without them. On lascaux50 the result
  is 225 passed, 30 skipped.
- The FRAM end-to-end test needs `tests/190919B`, which is not in git.
  Copy it from a machine that has it.

## 4. Site configuration

Put the settings in one file, e.g. `/etc/pyrt-transient/site.yaml`, and pass
it to every pipeline run with `--config=` (section 5 does this through the
daemon). `global:` holds the pipeline-wide keys, and `detection:` holds
the keys of `DetectionConfig` in `pyrt_transient/config_trans.py`.

```yaml
global:
  # Set this. The built-in default is lascaux50's /home/fnovotny/transient_work/.
  # The directory must exist; the pipeline does not create it.
  base_data_dir: /srv/pyrt-transient/data
  base_public_dir: /srv/pyrt-transient/public   # default ~/public_html
  generate_frontend: true
detection:
  # The default list starts with atlas@localhost, which exists only on lascaux50.
  # A catalogue that cannot be reached, or does not cover the field, is dropped
  # with a warning (Pan-STARRS south of Dec -30), so this list is safe anywhere.
  catalogs: ["atlas@vizier", gaia, usno]
  min_n_detections: 3
```

Unknown keys are ignored silently, so check what was actually loaded:

```bash
python -c "from pyrt_transient.core.config_loader import load_config_with_yaml_support as l; \
c = l('/etc/pyrt-transient/site.yaml'); print(c.base_data_dir, c.base_public_dir, c.detection.catalogs)"
```

- **Stacking** (`stacking_*` in `DetectionConfig`):
  - It is on by default. It starts after 10 epochs, combines at most 20,
    and rebuilds every 5.
  - It stops once a candidate reaches quality score 1.0.
  - It stacks the largest group of frames sharing `(PHFILTER, EXPTIME)`.
- **Stack-only candidates:** a source that only shows up in the stack is
  admitted when forced photometry finds it at SNR ≥ 3 in at least 30 % of
  the stacked frames, and no single frame contributes more than half its
  flux (`stack_forced_*`).
- **Where the defaults are validated:** the archive replay described in the
  README (10/15 GRB afterglows recovered on D50). Keep them unless a replay
  on your own archive says otherwise.

## 5. Run the daemon as a niced service

**Priority:** give the service a lower CPU and IO priority from the start.
On lascaux50 users complained that the pipeline slowed their work.
Everything the daemon starts inherits the unit's priority, including
pipeline runs, SExtractor and `pyrt-combine`.

**Settings:** the daemon itself is configured by `PYRT_TRANSIENT_*`
environment variables (see the docstring of
`pyrt_transient/transient_daemon.py`).

`/etc/systemd/system/pyrt-transient.service`:

```ini
[Unit]
Description=pyrt-transient detection daemon
After=network-online.target

[Service]
Type=simple
User=pyrt
Group=pyrt
ExecStart=/home/pyrt/venv/bin/pyrt-transient-daemon
Restart=on-failure
RestartSec=10
# The venv's bin must be on PATH: the daemon finds pyrt-transient-pipeline
# and the pipeline finds pyrt-combine there.
Environment=PATH=/home/pyrt/venv/bin:/usr/local/bin:/usr/bin:/bin
Environment=PYTHONUNBUFFERED=1
Environment=PYRT_TRANSIENT_SOCKET=/srv/pyrt-transient/transient_daemon.sock
Environment=PYRT_TRANSIENT_WORK_DIR=/srv/pyrt-transient/work
Environment=PYRT_TRANSIENT_LOG_DIR=/srv/pyrt-transient/logs
Environment=PYRT_TRANSIENT_PIPELINE_ARGS=--config=/etc/pyrt-transient/site.yaml
Environment=PYRT_TRANSIENT_MAX_PARALLEL=2
Environment=OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2 NUMEXPR_NUM_THREADS=2
# Lower priority than interactive work. Keep comments on their own lines:
# systemd does not strip a trailing "# ..." and throws away the whole value
# (lascaux50's CPUQuota=200% has been ignored for exactly this reason).
Nice=10
IOSchedulingClass=best-effort
IOSchedulingPriority=7
CPUQuota=200%
# lascaux50 runs with 1 GB. Raise it if jobs die with exit -9 while stacking.
MemoryMax=2G

[Install]
WantedBy=multi-user.target
```

```bash
sudo mkdir -p /srv/pyrt-transient/{data,public,work,logs} && sudo chown -R pyrt: /srv/pyrt-transient
sudo systemd-analyze verify /etc/systemd/system/pyrt-transient.service
sudo systemctl daemon-reload && sudo systemctl enable --now pyrt-transient
systemctl show pyrt-transient -p Nice -p IOSchedulingClass -p CPUQuotaPerSecUSec -p MemoryMax
```

Check the last command's output:

- `CPUQuotaPerSecUSec=2s` means the quota is applied; `infinity` means
  systemd rejected it.
- `Nice=10` should be listed.

**Without root:** the same unit works as a user unit.

1. Put it in `~/.config/systemd/user/`, without `User=`/`Group=`.
2. Run `systemctl --user enable --now pyrt-transient`.
3. Ask an admin once for `loginctl enable-linger pyrt`, so it survives
   logout and reboot.

`Nice=` and `IOScheduling*` work in user units. `CPUQuota`/`MemoryMax`
need cgroup delegation, so check them with `systemctl --user show`.

**If the unit can't be changed:** do what lascaux50 does, and make the
pipeline entry point lower its own priority before it imports anything.
See `~/bin/pipeline_magic.py` in section 9.

**Where the logs go:**

- The daemon writes to `$PYRT_TRANSIENT_LOG_DIR/transient_daemon.log` and
  logs a status line every minute.
- Each pipeline run writes to `<base_data_dir>/logs/pipeline_<obs>.log`.

## 6. Hook the telescope side

Make this the last step of the per-frame pyrt script, once the final ECSV
and the WCS-solved FITS of the same frame both exist:

```python
#!/usr/bin/env python3
"""Hand one calibrated frame to the pyrt-transient daemon.
usage: notify_transient_daemon.py <frame.ecsv> <frame.fits>"""
import json
import os
import socket
import sys

sock_path = os.environ.get("PYRT_TRANSIENT_SOCKET",
                           os.path.expanduser("~/transient_daemon.sock"))
ecsv, fits = (os.path.abspath(p) for p in sys.argv[1:3])
with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as s:
    s.settimeout(60)
    s.connect(sock_path)
    s.sendall(json.dumps({"ecsv_path": ecsv, "fits_path": fits}).encode())
    s.shutdown(socket.SHUT_WR)
    reply = json.loads(s.recv(4096).decode())
print(reply)
sys.exit(0 if reply.get("success") else 1)
```

- **The reply.** The daemon copies both files into its work directory
  before it answers `{"success": true, "job_id": ...}`, so the caller may
  delete or move its own files afterwards.
- **Permissions.** The socket is created world-writable. The telescope
  user needs to be able to reach its directory, and the daemon user needs
  read access to the files.
- **Socket path length.** Keep `PYRT_TRANSIENT_SOCKET` short. Linux
  limits a Unix socket path to 107 bytes, and a longer one makes both the
  daemon and the client fail with `AF_UNIX path too long`.
- **Separate users.** On lascaux50 the photometry pipeline runs as `mates`
  and the daemon as `fnovotny`. On a new machine, do the same with the
  telescope pipeline user and the service account.
  - Put the socket in a directory both users can reach, e.g.
    `/srv/pyrt-transient/`, or `/run/pyrt-transient/` via
    `RuntimeDirectory=pyrt-transient` in the unit.
  - Make the frames readable by the service account, e.g. through a
    shared group.
- **Order on the telescope side**, as described in the lascaux50 daemon
  documentation (`Transient Daemon Documentation.pdf`):
  1. process the frame;
  2. upload to the photometry database;
  3. send the request;
  4. delete the local copies only after `"success": true`.
- **Manual test** without the Python client:

  ```bash
  echo '{"ecsv_path":"/abs/frame.ecsv","fits_path":"/abs/frame.fits"}' \
    | socat - UNIX-CONNECT:/srv/pyrt-transient/transient_daemon.sock
  ```
- **File names.** The pipeline pairs light-curve rows with images by file
  stem.
  - D50 writes `…-df.ecsv` next to `…-dft.fits`, and the pipeline accepts
    that "t" suffix.
  - Any other naming works as long as the ECSV and FITS stems are equal.
- **Observations.** Frames of one observation are grouped by the OBSID in
  the ECSV meta, or else by the file name.
  - Keep OBSID stable across a night.
  - Set `observation_grouping_radius_arcmin` if one field gets several
    OBSIDs.

## 7. Test the installation

1. **Unit tests:** `python -m pytest -q` in the checkout (section 3).
2. **Smoke test on a past night, in scratch directories.** It needs no
   daemon and doesn't touch the live directories.

   ```bash
   T=/tmp/pyrt-smoke; mkdir -p $T/data $T/public
   printf 'global:\n  base_data_dir: %s/data\n  base_public_dir: %s/public\n  generate_frontend: true\n' $T $T > $T/smoke.yaml
   # append the site's detection: section to smoke.yaml, then feed a night's frames in time order
   for e in /archive/<night>/*.ecsv; do
     nice -n 10 pyrt-transient-pipeline "$e" "${e%.ecsv}.fits" --config=$T/smoke.yaml || echo "FAILED $e"
   done
   ```

   **What working looks like:**

   - every run exits 0;
   - `$T/data/obs_<id>/candidates.tbl` exists from the third epoch on;
   - `stack` files appear after 10 epochs;
   - `$T/public/obs_<id>/index.html` opens with cutouts centred on the
     candidates.

   **Also check:**

   - the log names each catalogue as loaded, or dropped for no coverage;
   - on a night with a known transient or afterglow, it is among the
     candidates.
3. **Live path:** send one frame through the socket with the client above
   and watch `transient_daemon.log`.
   - About 30 s after the last frame (the debounce window) you should see
     `Debounce fired`, then `job … succeeded`.
   - `ps -o ni,cmd -C python3` should show the pipeline at nice 10.
4. **Detection quality.** Once a site has an archive with known events,
   run `tools/replay_archive.py` over it (README, "Archive validation"),
   and repeat the replay before every deploy.

### Monitoring

**Daemon health.** The daemon logs a status line every minute:
`Status: N active jobs, M pending (debounce), X completed, Y failed`. The
lascaux50 daemon documentation recommends alerting when:

- the service has been down for more than 5 minutes;
- more than 10 % of jobs fail within an hour;
- no job has succeeded for 2 hours during observing;
- the work or data directory is more than 80 % full.

**What to grep for:**

- `job … failed (exit N)` and `timed out` in `transient_daemon.log`;
- `Catalog <name> failed for this field` in the pipeline logs. This is a
  catalogue-server outage. The run carries on with the other catalogues
  and marks the epoch degraded, and a later run recomputes it.
  - A failed query is not repeated within the same run.
  - A field with an expired cache entry keeps using it, with the warning
    `Using the expired … cache`.

## 8. Telescope notes

**FRAM (FRAM-Auger, Dec about −45):**

- *Reference:* the `tests/190919B` fixture is a FRAM-Auger burst night
  with the GRB 190919B afterglow.
- *Stacking:*
  - The afterglow is found only from the stack, and it is then admitted
    by forced photometry.
  - The night mixes 40×20 s unfiltered frames with 81×60 s R frames.
    Stacking takes the largest `(PHFILTER, EXPTIME)` group, so check which
    group your cadence produces.
- *Catalogues:* no Pan-STARRS coverage. Use `atlas@vizier`, `gaia` and
  `usno`. The subtraction pipeline also falls back to ATLAS for its
  zeropoint there.
- *R-band frames:* pyrt maps `R` to Johnson_R, which pyrt's
  `atlas@vizier` does not carry, so dophot matches no stars. Pass
  `PHOT_FILTER=Sloan_r` to `pyrt-cat2det`, as `tools/make_pyrt_ecsv.sh`
  shows.
- *Projection:* the NF4 camera needs a ZPN projection. The refitted
  PV/ZPN and SIP terms in the ECSV meta are kept by the matcher and the
  cutouts.
- *mates14/pyrt 73dc607:* its S0+SC astrometric error model returns NaN on
  this field, and the second dophot pass then fails. Until that is fixed
  upstream, use the older pyrt for FRAM or the local fixes described in
  `tools/make_pyrt_ecsv.sh`.

**SBT:**

- *Per-frame chain:* the archive chain in pyrt's `get_ecsv.sh` shows it:
  `proc_images.py` flat-fielding, a per-CCD crop (C1/C2, C3), then
  `solve-field`, `sscat-noradec`, `cat2det.py`, and finally `dophot3.py -a`
  and `-s` with `-M sbt`.
- *Hook:* the transient hook goes after the final `dophot3.py -s`, on the
  `…t.ecsv` it writes and the solved `…t.fits`.
- *Not yet validated:* no SBT night has been run through pyrt-transient.
  Do the smoke test (section 7.2) on a night with a known variable or
  transient before relying on it.

## 9. Updating and rollback

**Deploy with git.** Commit and push, then run this on the host:

```bash
cd ~/src/pyrt-transient
git status --short         # must be empty: edits made on the host get lost otherwise
git fetch origin && git merge --ff-only origin/main
python -m pytest -q -p no:cacheprovider
```

**When to reinstall or restart:**

- Run `pip install -e .` only when `pyproject.toml` dependencies or
  console scripts change.
- A code change needs **no daemon restart**, because every frame starts a
  fresh pipeline process.
- Restart (`systemctl restart pyrt-transient`) only when
  `transient_daemon.py` or the unit's environment changes. Pending
  debounced frames are processed on SIGTERM, not dropped.

**Rollback:** `git reset --hard <previous commit>`. Note the commit before
every deploy.

**Hotfixes made on the host:** commit them, even as a local branch. On
lascaux50 two such edits sat uncommitted for weeks, and a plain pull would
have erased them (see section 10).

## 10. Reference deployment: lascaux50

- **Code:**
  - Venv `~/bin/pyrt_transient_venv` (Python 3.11.2), with an editable
    install of the checkout `~/src/pyrt_transient_src`, now at 74060f1.
  - `stdpipe` is editable from `~/src/stdpipe`; `pyrt` comes from
    `/storage/home/mates/pyrt`.
  - `pyrt-combine` is in `/usr/local/bin`.
- **Service:**
  - Unit `/etc/systemd/system/transient-daemon.service` runs its own older
    copy of the daemon, `~/bin/transient_daemon.py`, not the packaged
    `pyrt-transient-daemon`.
  - For each frame it calls `~/bin/pipeline_magic.py <ecsv> <fits>`,
    with up to 4 in parallel and a 900 s timeout.
  - Observations go to `~/transient_work/obs_<id>`, pages to
    `~/public_html/obs_<id>`, and the log to `~/logs/transient_daemon.log`.
- **Throttle:**
  - The unit's `CPUQuota=200%  # ...` line is ignored by systemd
    (`CPUQuotaPerSecUSec=infinity`). `MemoryLimit=1G` does apply.
  - Changing the unit needs root. Until someone with root moves the
    comment to its own line and adds `Nice=10`, the throttle lives in
    `~/bin/pipeline_magic.py` (since 2026-09-11): nice 10, ionice
    best-effort 7, and `*_NUM_THREADS=2`, all set before the package is
    imported.
  - The previous wrapper is `~/bin/pipeline_magic.py.bak-20260911`.
- **2026-09-11 deploy:**
  - The checkout went from 468c9c1 to 74060f1.
  - Its two uncommitted host edits (templates `_patch_get_skycells`, the
    frontend `-dft` stem key) are now in 74060f1. The originals are kept
    in `git stash` and in `~/src/prod_hotfixes_468c9c1.patch`.
  - To roll back, run `git reset --hard 468c9c1 && git stash pop` in the
    checkout, then restore the old wrapper.
- **Staging and replay:**
  - `tools/deploy_lascaux.sh` rsyncs a working tree to
    `~/src/pyrt-transient-staging`. Tools run from there with the
    production interpreter, and the daemon is not touched.
  - Replay inputs are `~/phdb_fixed_obsid/*.ecsv` (OBSID meta = burst
    name), with targets in
    `~/etc/pyrt-transient/replay_targets_gcn.tsv` and configs in
    `~/etc/pyrt-transient/replay_{historical,vetting}.yaml`.
  - Results go to `~/public_html/grb_replay_validation/replay_<tag>/`.
    Run the replay niced (`nice -n 19 ionice -c3`, single-threaded BLAS).

- **The daemon documentation PDF** (`Transient Daemon Documentation.pdf`)
  describes an earlier version of this daemon. Its install steps, request
  and response format, and troubleshooting still apply. These figures have
  changed:

  | | PDF | now |
  |---|---|---|
  | workers | 3 | 4 |
  | job timeout | 10 min | 900 s |
  | socket | `/run/transient_daemon.sock` | `/home/fnovotny/transient_daemon.sock` |
  | a full queue | rejected "at capacity" | frames are queued, and a 30 s debounce batches each observation |

**Still to do:**

- Move lascaux50 to the packaged daemon and a unit like the one in
  section 5. That needs root.
- Change the `base_data_dir` default in `config_trans.py` from the
  lascaux50 path to `~/transient_work`, and let `setup_pipeline_logging`
  create missing parents.
- Pin `pyrt` and `stdpipe` to commits and check them in the smoke test.
  They are unpinned checkouts owned by two people.
