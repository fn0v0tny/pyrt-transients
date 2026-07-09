"""
Forced photometry queries for SN candidate lightcurves.

Supports:
  - PanSTARRS DR2 detection photometry  (MAST API, no auth)
  - ATLAS forced photometry             (fallingstar-data.com API, token required)
"""

import logging
import time
import numpy as np
from concurrent.futures import ThreadPoolExecutor, as_completed
from astropy.time import Time

log = logging.getLogger('forced_photometry')

# ── PanSTARRS ─────────────────────────────────────────────────────────────────

PS1_TAP_URL = "https://mast.stsci.edu/vo-tap/api/v0.1/ps1dr2/"
PS1_FILTER_COLORS = {'g': '#2ecc71', 'r': '#e74c3c', 'i': '#e67e22', 'z': '#9b59b6', 'y': '#1abc9c'}


def _ps1_detections_at(ra, dec, radius_arcsec=2.0, timeout=60):
    """Return list of PS1 DR2 detections near (ra, dec) via TAP service."""
    import pyvo
    radius_deg = radius_arcsec / 3600.0
    query = f"""
        SELECT d.obstime, f.filtertype AS band,
               d.psfflux, d.psffluxerr, d.infoflag2
        FROM Detection AS d
        JOIN Filter AS f ON d.filterid = f.filterid
        WHERE CONTAINS(POINT('ICRS', d.ra, d.dec),
                       CIRCLE('ICRS', {ra}, {dec}, {radius_deg})) = 1
          AND d.psfqfperfect >= 0.9
          AND d.psfflux > 0
    """
    try:
        tap = pyvo.dal.TAPService(PS1_TAP_URL)
        result = tap.search(query, maxrec=5000).to_table()
        points = []
        for row in result:
            try:
                flux = float(row['psfflux'])
                ferr = float(row['psffluxerr']) if row['psffluxerr'] is not None else 0.0
                mjd  = float(row['obstime'])
                band = str(row['band']).strip().lower()
                iflag = int(row['infoflag2']) if row['infoflag2'] is not None else 0
            except (TypeError, ValueError):
                continue
            if iflag and (iflag & 8):   # extended / galaxy core — skip
                continue
            if flux <= 0:
                continue
            # Convert Jy flux to AB magnitude
            mag = -2.5 * np.log10(flux) + 8.90
            err = 1.0857 * ferr / flux if ferr > 0 else 0.0
            if np.isfinite(mag) and 0 < mag < 30:
                points.append({'mjd': mjd, 'band': band, 'mag': mag, 'mag_err': err,
                                'survey': 'PS1', 'ulim': False})
        return points
    except Exception as e:
        log.debug(f"PS1 TAP query at ({ra:.4f},{dec:.4f}) failed: {e}")
        return []


def query_panstarrs_lightcurves(candidates, logger=None, radius_arcsec=2.0, max_workers=8):
    """Query PS1 DR2 detection photometry for all candidates in parallel.

    Returns dict: candidate_id -> list of photometry points.
    """
    if logger is None:
        logger = log

    ids_coords = []
    for row in candidates:
        cid = _get_cid(row)
        ids_coords.append((cid, float(row['ALPHA_J2000']), float(row['DELTA_J2000'])))

    results = {}
    logger.info(f"PS1: querying {len(ids_coords)} positions …")

    def _query(args):
        cid, ra, dec = args
        return cid, _ps1_detections_at(ra, dec, radius_arcsec=radius_arcsec)

    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futs = {pool.submit(_query, a): a[0] for a in ids_coords}
        for fut in as_completed(futs):
            cid, pts = fut.result()
            results[cid] = pts

    n_with_data = sum(1 for v in results.values() if v)
    logger.info(f"PS1: {n_with_data}/{len(results)} candidates have historical detections")
    return results


# ── ATLAS ─────────────────────────────────────────────────────────────────────

ATLAS_API = "https://fallingstar-data.com/forcedphot"
ATLAS_FILTER_COLORS = {'o': '#f39c12', 'c': '#2980b9', 'i': '#8e44ad'}


def _atlas_submit(ra, dec, mjd_min, mjd_max, token, timeout=30):
    """Submit ATLAS forced photometry job. Returns task_url or None."""
    import requests, re as _re
    headers = {'Authorization': f'Token {token}', 'Accept': 'application/json'}
    data = {'ra': ra, 'dec': dec, 'mjd_min': mjd_min, 'mjd_max': mjd_max,
            'send_email': False, 'use_reduced': False}
    while True:
        r = requests.post(f"{ATLAS_API}/queue/", headers=headers, data=data, timeout=timeout)
        if r.status_code == 201:
            return r.json().get('url')
        elif r.status_code == 429:
            msg = r.json().get('detail', '')
            t_sec = _re.findall(r'available in (\d+) seconds', msg)
            t_min = _re.findall(r'available in (\d+) minutes', msg)
            wait = int(t_sec[0]) if t_sec else (int(t_min[0]) * 60 if t_min else 10)
            time.sleep(wait)
        else:
            r.raise_for_status()
            return None


def _atlas_poll(task_url, token, poll_interval=5, max_wait=300):
    """Poll ATLAS task_url until finished. Returns result_url or raises TimeoutError."""
    import requests
    headers = {'Authorization': f'Token {token}', 'Accept': 'application/json'}
    deadline = time.time() + max_wait
    while time.time() < deadline:
        r = requests.get(task_url, headers=headers, timeout=30)
        r.raise_for_status()
        info = r.json()
        if info.get('finishtimestamp'):
            if info.get('error_msg'):
                log.debug(f"ATLAS task finished with error: {info['error_msg']}")
            return info.get('result_url')
        time.sleep(poll_interval)
    raise TimeoutError(f"ATLAS task did not finish within {max_wait}s")


def _atlas_download(result_url, token):
    import requests
    headers = {'Authorization': f'Token {token}'}
    r = requests.get(result_url, headers=headers, timeout=60)
    r.raise_for_status()
    return r.text


def _parse_atlas_result(text):
    """Parse ATLAS forced photometry text into list of dicts."""
    points = []
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith('#'):
            continue
        parts = line.split()
        if len(parts) < 6:
            continue
        try:
            mjd = float(parts[0])
            mag = float(parts[1])   # 'm' column; 99 = non-detection
            dm  = float(parts[2])
            uJy = float(parts[3])
            duJy = float(parts[4])
            filt = parts[5]         # 'F' column: o, c, i
        except (ValueError, IndexError):
            continue
        # Non-detection: negative flux (noise) or sentinel mag > 50
        ulim = (uJy <= 0 or mag > 50 or mag == 0)
        if ulim:
            # 3-sigma upper limit from flux error: 23.9 - 2.5*log10(3*duJy [uJy])
            if duJy > 0:
                mag = 23.9 - 2.5 * np.log10(max(3 * duJy, 1e-6))
                dm = 0.0
            else:
                continue
        points.append({'mjd': mjd, 'band': filt, 'mag': mag, 'mag_err': dm,
                        'survey': 'ATLAS', 'ulim': ulim})
    return points


def query_atlas_lightcurves(candidates, obs_jd, logger=None, api_token=None,
                             lookback_days=365, max_workers=4, max_wait_per=180):
    """Query ATLAS forced photometry for all candidates.

    Submits all jobs in parallel, then polls until all finish (or timeout).
    Returns dict: candidate_id -> list of photometry points.
    """
    if logger is None:
        logger = log
    if not api_token:
        logger.warning("ATLAS: no API token provided, skipping")
        return {}

    obs_mjd = Time(obs_jd, format='jd').mjd
    mjd_min = obs_mjd - lookback_days
    mjd_max = obs_mjd + 2

    ids_coords = []
    for row in candidates:
        cid = _get_cid(row)
        ids_coords.append((cid, float(row['ALPHA_J2000']), float(row['DELTA_J2000'])))

    # Submit all jobs
    logger.info(f"ATLAS: submitting {len(ids_coords)} forced photometry jobs …")
    task_map = {}  # task_url -> candidate_id

    def _submit(args):
        cid, ra, dec = args
        task_url = _atlas_submit(ra, dec, mjd_min, mjd_max, api_token)
        return cid, task_url

    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futs = [pool.submit(_submit, a) for a in ids_coords]
        for fut in as_completed(futs):
            try:
                cid, task_url = fut.result()
                if task_url:
                    task_map[task_url] = cid
            except Exception as e:
                logger.warning(f"ATLAS submit error: {e}")

    logger.info(f"ATLAS: {len(task_map)} jobs submitted, polling for results …")

    # Poll all jobs
    results = {cid: [] for cid in [a[0] for a in ids_coords]}

    def _fetch(task_url):
        try:
            result_url = _atlas_poll(task_url, api_token, max_wait=max_wait_per)
            if result_url:
                text = _atlas_download(result_url, api_token)
                return task_url, _parse_atlas_result(text)
        except Exception as e:
            logger.warning(f"ATLAS task error: {e}")
        return task_url, []

    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futs = [pool.submit(_fetch, task_url) for task_url in task_map]
        for fut in as_completed(futs):
            task_url, pts = fut.result()
            cid = task_map.get(task_url)
            if cid:
                results[cid] = pts

    n_with_data = sum(1 for v in results.values() if v)
    logger.info(f"ATLAS: {n_with_data}/{len(results)} candidates have lightcurve data")
    return results


# ── Lightcurve magnitude delta ────────────────────────────────────────────────

def compute_lc_mag_delta(lcs, candidates, obs_mag_col=None):
    """Compute brightening delta: historical_mag - current_mag.

    Positive delta = currently brighter than historical baseline = potential SN.

    Priority: ATLAS detections > PS1 detections > ATLAS upper limits (non-detections).
    For upper limits, the delta reflects "at least this much brighter than the limit".

    Returns dict: cid -> {
        'delta':    float  — hist_mag - cur_mag (positive = brightened),
        'hist_mag': float  — median historical magnitude,
        'cur_mag':  float  — current detection magnitude,
        'n_hist':   int    — number of historical measurements used,
        'source':   str    — 'atlas' | 'ps1' | 'atlas_ulim',
    } or None if no data / no current mag.
    """
    if obs_mag_col is None:
        for col in ('MAG_CALIB', 'mag_weighted_mean', 'MAG_AUTO'):
            if col in candidates.colnames:
                obs_mag_col = col
                break

    results = {}
    for row in candidates:
        cid = _get_cid(row)
        lc = lcs.get(cid, {})

        cur_mag = None
        if obs_mag_col and obs_mag_col in row.colnames:
            try:
                cur_mag = float(row[obs_mag_col])
            except (TypeError, ValueError):
                pass
        if cur_mag is None or not np.isfinite(cur_mag):
            results[cid] = None
            continue

        atlas_dets  = [p for p in lc.get('atlas', []) if not p['ulim']]
        ps1_dets    = [p for p in lc.get('ps1',   []) if not p['ulim']]
        atlas_ulims = [p for p in lc.get('atlas', []) if p['ulim']]

        if atlas_dets:
            hist_mag = float(np.median([p['mag'] for p in atlas_dets]))
            source, n_hist = 'atlas', len(atlas_dets)
        elif ps1_dets:
            hist_mag = float(np.median([p['mag'] for p in ps1_dets]))
            source, n_hist = 'ps1', len(ps1_dets)
        elif atlas_ulims:
            # Only upper limits: source was below detection threshold → strong evidence for new event
            hist_mag = float(np.median([p['mag'] for p in atlas_ulims]))
            source, n_hist = 'atlas_ulim', len(atlas_ulims)
        else:
            results[cid] = None
            continue

        delta = hist_mag - cur_mag   # positive = currently brighter
        results[cid] = {
            'delta':    round(delta,    3),
            'hist_mag': round(hist_mag, 3),
            'cur_mag':  round(cur_mag,  3),
            'n_hist':   n_hist,
            'source':   source,
        }

    return results


# ── PS1 filtering ─────────────────────────────────────────────────────────────

def filter_ps1_known_sources(candidates, ps1_lcs, obs_mag_col=None,
                              min_detections=5, brightening_threshold=0.5,
                              logger=None):
    """Remove candidates that are known persistent sources in PS1 DR2.

    Logic:
      - A candidate with >= min_detections PS1 epochs is a "known source".
      - It is kept only if the current detection is at least brightening_threshold
        magnitudes brighter than the brightest (smallest mag) PS1 detection
        in the same or adjacent band — i.e. a real brightening event (SN!).
      - If no current magnitude is available, the candidate is kept.

    Returns (filtered_candidates, n_rejected).
    """
    if logger is None:
        logger = log

    if len(candidates) == 0:
        return candidates, 0

    # Determine magnitude column
    if obs_mag_col is None:
        for col in ('MAG_CALIB', 'mag_weighted_mean', 'MAG_AUTO'):
            if col in candidates.colnames:
                obs_mag_col = col
                break

    keep = []
    n_rejected = 0

    for row in candidates:
        cid = _get_cid(row)
        pts = ps1_lcs.get(cid, [])
        detections = [p for p in pts if not p['ulim']]

        if len(detections) < min_detections:
            # Not enough history → keep (could be genuinely new)
            keep.append(True)
            continue

        # Current magnitude
        cur_mag = None
        if obs_mag_col and obs_mag_col in row.colnames:
            try:
                cur_mag = float(row[obs_mag_col])
            except (TypeError, ValueError):
                pass

        if cur_mag is None or not np.isfinite(cur_mag):
            keep.append(True)
            continue

        # Brightest historical detection (smallest mag value = brightest)
        brightest_ps1 = min(p['mag'] for p in detections)

        # Keep if currently brighter than brightest PS1 by > threshold (brightening)
        if cur_mag < brightest_ps1 - brightening_threshold:
            keep.append(True)
        else:
            keep.append(False)
            n_rejected += 1
            logger.debug(
                f"  PS1 filter: {cid} rejected — {len(detections)} PS1 detections, "
                f"brightest PS1={brightest_ps1:.2f}, current={cur_mag:.2f}"
            )

    mask = np.array(keep, dtype=bool)
    logger.info(
        f"PS1 history filter: {n_rejected} known persistent sources removed, "
        f"{int(mask.sum())} remaining"
    )
    return candidates[mask], n_rejected


# ── Helpers ───────────────────────────────────────────────────────────────────

def _make_id(row):
    ra = float(row['ALPHA_J2000'])
    dec = float(row['DELTA_J2000'])
    return f"tr_{ra*1000:.0f}_{abs(dec)*1000:.0f}"


def _get_cid(row):
    """Return the canonical candidate ID for a table row.

    Preference: transient_id (used by frontend_generator) > candidate_id > position fallback.
    This ensures lightcurves.json keys match what frontend_generator.py uses.
    """
    cols = row.colnames if hasattr(row, 'colnames') else []
    if 'transient_id' in cols:
        v = row['transient_id']
        if v is not None and str(v).strip():
            return str(v)
    if 'candidate_id' in cols:
        v = row['candidate_id']
        if v is not None and str(v).strip():
            return str(v)
    return _make_id(row)


def merge_lightcurves(ps1_lcs, atlas_lcs, candidates):
    """Merge PS1 and ATLAS lightcurves keyed by candidate id."""
    merged = {}
    for row in candidates:
        cid = _get_cid(row)
        merged[cid] = {
            'ps1':  sorted(ps1_lcs.get(cid, []),   key=lambda p: p['mjd']),
            'atlas': sorted(atlas_lcs.get(cid, []), key=lambda p: p['mjd']),
        }
    return merged
