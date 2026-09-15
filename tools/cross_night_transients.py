#!/usr/bin/env python3
"""Transients seen on more than one night: crawl candidates.tbl of past
observations, cluster the candidates on the sky, and write a static page.

  cross_night_transients.py [--data-dir DIR] [--public-dir DIR] [--out-dir DIR]
                            [--days N] [--radius ARCSEC] [--min-nights N]
                            [--min-quality Q] [--per-field N] [--max-cards N]
                            [--max-rows N] [--open-fields N] [--include-forced]
                            [--title TEXT]

Defaults: ~/transient_work, ~/public_html, <public-dir>/new_transients,
30 days, 3", 2 nights, quality 0.02, 10 cards per field, 300 cards in all,
100 table lines per field, 5 field sections open. Environment overrides:
PYRT_STATUS_DATA_DIR, PYRT_STATUS_PUBLIC_DIR (shared with status_page.py).

Only the standard library is used, so it runs with any python3 on the host.

What it does
- Reads every <data-dir>/obs_*/candidates.tbl whose observation has a frame
  within --days (frame names carry the UTC time). Each observation's rows
  and header facts are cached in <data-dir>/.new_transients_cache.json,
  keyed by the mtime of candidates.tbl, so a rerun only re-reads what the
  pipeline rewrote since.
- Drops the forced target row (NUMBER 0: it sits at the pointing of every
  night, so it would always look "consistent"), trails, and rows below
  --min-quality.
- Friends-of-friends clustering with --radius across all observations.
  A group is reported when it was seen on at least --min-nights distinct
  nights (a night is the UTC date of the frame time minus 12 h, so a
  night that spans midnight is one night).
- Stitches a lightcurve across the nights from each observation's
  <transient_id>_lightcurve.ecsv (every epoch, with its band), falling back
  to the one magnitude in candidates.tbl. The page draws it as one panel
  per night with a shared magnitude axis; the JSON carries the points.
- Looks for change: nightly means with their errors, the largest
  night-to-night difference, a straight-line trend through the nightly
  means (mag/day, for a supernova's slow decline over weeks) and the
  largest change inside one night (first third of the epochs against the
  last third), each with its significance ("changed" at >= 0.3 mag and
  5 sigma), and the frames of the same field that cover the position
  without the source (from the first frame's WCS and MAGLIM in the ECSV
  header): "appeared" when earlier frames went 1 mag deeper than the
  source without showing it, "disappeared" likewise afterwards.
- Scores each source: brightness (18 - brightest nightly mean magnitude,
  0..10) + coverage (2 per extra night + log2 of the measured epochs) +
  quality (log10 of the best pipeline quality score, at most 2) + change
  (3 x a significant brightening of at least 1 mag, at most 6, 1 x a
  fading, +4 appeared, +1 disappeared), so the bright, well covered, new
  and brightening come first.
- Writes <out-dir>/new_transients.json and <out-dir>/index.html. The page
  groups the sources by the field most of their observations were taken
  as (the OBJECT header). Each field is a collapsible section with the
  best --per-field sources in full (chart and one row per observation),
  then a table of the others. --max-cards caps the cards on the whole
  page, --max-rows the table lines per field, and a chart never draws
  more than 40 marks per night (epochs are averaged in time bins), so the
  page stays a few MB at most. The JSON has every source ("card" says
  whether it is on the page in full); the epoch points are kept only for
  those, the others carry their count. The page
  links each night's row to the observation's own site
  (<public-dir>/obs_<id>/index.html) and shows its montage and lightcurve
  thumbnails when that site has them.
"""
import csv
import argparse
import html
import json
import math
import os
import re
import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

CACHE_NAME = ".new_transients_cache_v2.json"
HEADER_BYTES = 64 << 10
_META_LINE = re.compile(r"^#\s*-\s*\{([A-Za-z0-9_\-]+):\s*(.*)\}\s*$")
COLUMNS = ("NUMBER", "ALPHA_J2000", "DELTA_J2000", "MAG_CALIB", "MAGERR_CALIB",
           "quality_score", "n_detections", "n_epochs", "candidate_type",
           "source_file", "transient_id", "motion_rate_as_per_hr", "mag_range")
FRAME_RE = re.compile(r"^(\d{14})")


# ----------------------------------------------------------------- readers

def read_ipac(path, columns=COLUMNS):
    """Rows of an astropy ascii.ipac table as {column: str}, for `columns`."""
    try:
        lines = Path(path).read_text(errors="replace").splitlines()
    except OSError:
        return []
    header = next((line for line in lines if line.startswith("|")), None)
    if header is None:
        return []
    bars = [i for i, c in enumerate(header) if c == "|"]
    names = [header[a + 1:b].strip() for a, b in zip(bars, bars[1:])]
    wanted = {name: k for k, name in enumerate(names) if name in columns}
    rows = []
    for line in lines:
        if not line.strip() or line[0] in "|\\":
            continue
        rows.append({name: line[bars[k] + 1:bars[k + 1] + 1].strip()
                     for name, k in wanted.items()})
    return rows


def read_header(ecsv_path):
    """{KEY: value} from the `# - {KEY: value}` lines of an ECSV header."""
    meta = {}
    try:
        with open(ecsv_path, "rb") as fh:
            head = fh.read(HEADER_BYTES).decode("utf-8", "replace")
    except OSError:
        return meta
    for line in head.splitlines():
        if not line.startswith("#"):
            break
        m = _META_LINE.match(line)
        if m:
            meta.setdefault(m.group(1), m.group(2).strip().strip("'\""))
    return meta


def _finite(x):
    """x, or None when it is not a finite number (JSON has no NaN)."""
    return x if isinstance(x, (int, float)) and math.isfinite(x) else None


def _float(value, default=float("nan")):
    try:
        f = float(value)
    except (TypeError, ValueError):
        return default
    return f if math.isfinite(f) else default


def frame_time(name):
    """'20260909011221-634-i-020-df.ecsv' -> aware UTC datetime, or None."""
    m = FRAME_RE.match(name or "")
    if not m:
        return None
    try:
        return datetime.strptime(m.group(1), "%Y%m%d%H%M%S").replace(tzinfo=timezone.utc)
    except ValueError:
        return None


def night_of(t):
    """UTC date of the night: the frame time minus 12 h."""
    return (t - timedelta(hours=12)).strftime("%Y-%m-%d")


def first_science_ecsv(obs_dir):
    try:
        with os.scandir(obs_dir) as it:
            names = sorted(e.name for e in it
                           if e.name.endswith(".ecsv") and not e.name.endswith("_transients.ecsv")
                           and not e.name.endswith("_lightcurve.ecsv"))
    except OSError:
        return None
    return names[0] if names else None


def band_of_frame(name):
    """'20260909011221-634-i-020-df' -> 'i' (the third dash-separated field)."""
    parts = (name or "").split("-")
    return parts[2] if len(parts) > 3 else ""


def normalise_band(band):
    """One name per band: the frame-name letter of an older lightcurve
    ("i") and the calibration column of a newer one ("Sloan_i") are the
    same band and must be compared with each other, not treated as two."""
    b = (band or "").strip()
    if b in ("g", "r", "i", "z", "u"):
        return "Sloan_" + b
    if b in ("C", "clear", "Clear", "N"):
        return "N"
    return b


def read_lightcurve(path, default_band=""):
    """[[mjd, mag, magerr, band], ...] from a <transient_id>_lightcurve.ecsv.

    Rows with MAG_CALIB 99 (not measured) are skipped. The band is
    phot_filter (what the frame was calibrated to), else filter, else the
    band letter in the frame name, else `default_band`.
    """
    try:
        lines = [ln for ln in Path(path).read_text(errors="replace").splitlines()
                 if ln and not ln.startswith("#")]
    except OSError:
        return []
    if not lines:
        return []
    reader = csv.reader(lines, delimiter=" ", skipinitialspace=True)
    header = next(reader)
    col = {name: k for k, name in enumerate(header)}
    if "MAG_CALIB" not in col or "mjd" not in col:
        return []
    points = []
    for row in reader:
        if len(row) < len(header):
            continue
        mag, mjd = _float(row[col["MAG_CALIB"]]), _float(row[col["mjd"]])
        if not (math.isfinite(mag) and math.isfinite(mjd)) or mag > 90:
            continue
        band = ""
        for key in ("phot_filter", "filter"):
            if key in col and row[col[key]] not in ("", "nan", "None"):
                band = row[col[key]]
                break
        if not band and "source_file" in col:
            band = band_of_frame(Path(row[col["source_file"]]).name)
        err = _float(row[col["MAGERR_CALIB"]]) if "MAGERR_CALIB" in col else float("nan")
        points.append([mjd, mag, _finite(err), normalise_band(band or default_band)])
    points.sort()
    return points


def mjd_of(t):
    return t.timestamp() / 86400.0 + 40587.0


# ------------------------------------------------------ detection cells

CELL_ARCSEC = 3.0
CELLS_DIR = ".new_transients_cells"


def cell_of(ra, dec, dec0):
    """Integer cell id of a position on a CELL_ARCSEC grid (RA scaled by
    cos of the observation's centre declination)."""
    c = CELL_ARCSEC / 3600.0
    ix = int(math.floor(ra * math.cos(math.radians(dec0)) / c))
    iy = int(math.floor((dec + 90.0) / c))
    return ix * 4000000 + iy


def frame_positions(path):
    """(ra, dec) of every row of a pyrt detection ECSV, by text parsing."""
    out = []
    try:
        with open(path, errors="replace") as fh:
            names = None
            for line in fh:
                if line.startswith("#") or not line.strip():
                    continue
                parts = line.split()
                if names is None:
                    names = parts
                    try:
                        ia, id_ = names.index("ALPHA_J2000"), names.index("DELTA_J2000")
                    except ValueError:
                        return out
                    continue
                if len(parts) <= max(ia, id_):
                    continue
                try:
                    out.append((float(parts[ia]), float(parts[id_])))
                except ValueError:
                    continue
    except OSError:
        pass
    return out


def scan_cells(obs_dir, dec0):
    """Set of cells holding at least one detection in any frame of the
    observation. What a frame detected is not what candidates.tbl lists:
    a star matched to the catalogue on one night is absent from that
    night's candidates without being undetected."""
    cells = set()
    try:
        with os.scandir(obs_dir) as it:
            frames = [e.path for e in it if e.name.endswith(".ecsv")
                      and not e.name.endswith(("_transients.ecsv", "_lightcurve.ecsv"))]
    except OSError:
        return cells
    for path in frames:
        for ra, dec in frame_positions(path):
            cells.add(cell_of(ra, dec, dec0))
    return cells


def save_cells(data_dir, obs_name, cells):
    import array
    d = data_dir / CELLS_DIR
    d.mkdir(exist_ok=True)
    arr = array.array("q", sorted(cells))
    tmp = d / f"{obs_name}.tmp"
    with open(tmp, "wb") as fh:
        arr.tofile(fh)
    os.replace(tmp, d / f"{obs_name}.bin")


def load_cells(data_dir, obs_name):
    import array
    path = data_dir / CELLS_DIR / f"{obs_name}.bin"
    try:
        arr = array.array("q")
        with open(path, "rb") as fh:
            arr.frombytes(fh.read())
        return set(arr)
    except OSError:
        return None


def detected_in(cells, ra, dec, dec0):
    """Was anything detected within about one cell of (ra, dec)?"""
    c = CELL_ARCSEC / 3600.0
    for dra in (-c, 0.0, c):
        for ddec in (-c, 0.0, c):
            if cell_of(ra + dra / max(1e-6, math.cos(math.radians(dec0))), dec + ddec, dec0) in cells:
                return True
    return False


# ------------------------------------------------------------------- crawl

def scan_observation(obs_dir):
    """Everything the page needs from one observation, as plain JSON."""
    try:
        processed = json.loads((obs_dir / "detection_metadata.json").read_text()).get("processed_files", [])
    except (OSError, ValueError):
        processed = []
    times = sorted(t for t in (frame_time(f) for f in processed) if t)
    ecsv = first_science_ecsv(obs_dir)
    meta = read_header(obs_dir / ecsv) if ecsv else {}
    if not times and ecsv:
        t = frame_time(ecsv)
        times = [t] if t else []
    def _num(key):
        v = _float(meta.get(key), None)
        return v
    facts = {
        # Frame geometry of the first frame, for "was this position in the
        # field?": TAN centre, CD matrix (deg/px) and size, plus the frame's
        # own limiting magnitude for non-detections.
        "center": [_num("CRVAL1"), _num("CRVAL2")],
        "cd": [_num("CD1_1"), _num("CD1_2"), _num("CD2_1"), _num("CD2_2")],
        "size": [_num("IMAGEW") or _num("NAXIS1"), _num("IMAGEH") or _num("NAXIS2")],
        "maglim": _num("MAGLIM"),
        "obs_id": obs_dir.name[4:],
        "object": meta.get("OBJECT", ""),
        "target": meta.get("TARGET", ""),
        "telescope": meta.get("TELESCOP", ""),
        "filter": meta.get("FILTER", ""),
        "grb_ra": _float(meta.get("GRB_RA"), None),
        "grb_dec": _float(meta.get("GRB_DEC"), None),
        "frames": len(processed),
        "first": times[0].strftime("%Y-%m-%d %H:%M:%S") if times else "",
        "first_frame": min(processed) if processed else (ecsv or ""),
        "last": times[-1].strftime("%Y-%m-%d %H:%M:%S") if times else "",
        "nights": sorted({night_of(t) for t in times}),
    }
    rows = []
    for r in read_ipac(obs_dir / "candidates.tbl"):
        ra, dec = _float(r.get("ALPHA_J2000")), _float(r.get("DELTA_J2000"))
        if not (math.isfinite(ra) and math.isfinite(dec)):
            continue
        t = frame_time(r.get("source_file", ""))
        rows.append({
            "number": int(_float(r.get("NUMBER"), -1)),
            "ra": ra, "dec": dec,
            "mag": _float(r.get("MAG_CALIB")), "magerr": _float(r.get("MAGERR_CALIB")),
            "q": _float(r.get("quality_score"), 0.0),
            "n_det": int(_float(r.get("n_detections"), 0)),
            "n_epochs": int(_float(r.get("n_epochs"), 0)),
            "type": r.get("candidate_type", ""),
            "id": r.get("transient_id", ""),
            "motion": _float(r.get("motion_rate_as_per_hr")),
            "mag_range": _float(r.get("mag_range")),
            "source_file": r.get("source_file", ""),
            "night": night_of(t) if t else (facts["nights"][0] if facts["nights"] else ""),
        })
    return {"facts": facts, "rows": rows}


def crawl(data_dir, days, log):
    cache_path = data_dir / CACHE_NAME
    try:
        cache = json.loads(cache_path.read_text())
    except (OSError, ValueError):
        cache = {}
    cutoff = time.time() - days * 86400 if days > 0 else 0
    observations, n_read, n_cached = [], 0, 0
    try:
        with os.scandir(data_dir) as it:
            dirs = sorted(e.name for e in it if e.name.startswith("obs_") and e.is_dir())
    except OSError as e:
        log(f"cannot read {data_dir}: {e}")
        dirs = []
    for name in dirs:
        obs_dir = data_dir / name
        try:
            st = os.stat(obs_dir / "candidates.tbl")
        except OSError:
            continue  # nothing analysed yet (fewer than three epochs), or removed
        entry = cache.get(name)
        if entry and entry.get("mtime") == st.st_mtime:
            n_cached += 1
        else:
            entry = {"mtime": st.st_mtime, **scan_observation(obs_dir)}
            entry.pop("cells_mtime", None)
            cache[name] = entry
            n_read += 1
        # The frame time, not the table's mtime, decides whether a night is
        # "recent": a reprocessed old observation stays old.
        last = entry["facts"].get("last")
        if cutoff and last:
            try:
                t_last = datetime.strptime(last, "%Y-%m-%d %H:%M:%S").replace(tzinfo=timezone.utc)
            except ValueError:
                t_last = None
            if t_last and t_last.timestamp() < cutoff:
                continue
        elif cutoff and st.st_mtime < cutoff:
            continue
        observations.append(entry)
    # Detection cells (every frame's sources) only for the observations in
    # the window: reading the frames is the slow part (about 2 s per
    # observation), and only these are ever asked about.
    n_cells = 0
    for entry in observations:
        name = f"obs_{entry['facts']['obs_id']}"
        if entry.get("cells_mtime") == entry["mtime"] and (data_dir / CELLS_DIR / f"{name}.bin").exists():
            continue
        dec0 = entry["facts"]["center"][1]
        save_cells(data_dir, name, scan_cells(data_dir / name, 0.0 if dec0 is None else dec0))
        entry["cells_mtime"] = entry["mtime"]
        n_cells += 1
        if n_cells % 50 == 0:
            log(f"  detection cells: {n_cells} observations read")
    live = {name for name in dirs}
    cache = {k: v for k, v in cache.items() if k in live}
    try:
        tmp = cache_path.with_suffix(".tmp")
        tmp.write_text(json.dumps(cache))
        os.replace(tmp, cache_path)
    except OSError as e:
        log(f"cache not written: {e}")
    log(f"{len(dirs)} observation dirs, {n_read} read, {n_cached} from cache, "
        + (f"{len(observations)} within {days} days, " if days else "")
        + f"detection cells built for {n_cells}")
    return observations


# ----------------------------------------------------------------- cluster

def angular_sep_arcsec(ra1, dec1, ra2, dec2):
    d2r = math.pi / 180.0
    dra = (ra1 - ra2) * d2r * math.cos(0.5 * (dec1 + dec2) * d2r)
    ddec = (dec1 - dec2) * d2r
    return math.hypot(dra, ddec) / d2r * 3600.0


def cluster(points, radius):
    """Friends-of-friends on (ra, dec) with a sweep in declination.

    points: list of dicts with 'ra', 'dec'. Returns a list of index lists.
    """
    order = sorted(range(len(points)), key=lambda i: points[i]["dec"])
    parent = list(range(len(points)))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    r_deg = radius / 3600.0
    start = 0
    for k, i in enumerate(order):
        pi = points[i]
        while points[order[start]]["dec"] < pi["dec"] - r_deg:
            start += 1
        for j in order[start:k]:
            pj = points[j]
            if angular_sep_arcsec(pi["ra"], pi["dec"], pj["ra"], pj["dec"]) <= radius:
                parent[find(i)] = find(j)
    groups = {}
    for i in range(len(points)):
        groups.setdefault(find(i), []).append(i)
    return list(groups.values())


COVERAGE_FRACTION = 0.45   # of the frame half-size: stay clear of the edges


def covers(facts, ra, dec):
    """Was (ra, dec) inside this observation's first frame? Uses the TAN
    centre and CD matrix from the header; False when they are missing."""
    c, cd, size = facts.get("center"), facts.get("cd"), facts.get("size")
    if not c or not cd or not size or None in c or None in cd or None in size:
        return False
    det = cd[0] * cd[3] - cd[1] * cd[2]
    if abs(det) < 1e-20:
        return False
    dx = (ra - c[0]) * math.cos(math.radians(dec))
    dra = ((dx + 180) % 360) - 180 if abs(dx) > 180 else dx
    ddec = dec - c[1]
    px = (cd[3] * dra - cd[1] * ddec) / det
    py = (-cd[2] * dra + cd[0] * ddec) / det
    return abs(px) < COVERAGE_FRACTION * size[0] and abs(py) < COVERAGE_FRACTION * size[1]


def n_pts_total(dets):
    return sum(len(p["points"]) for p in dets)


MIN_EPOCHS_PER_NIGHT = 3   # a night with fewer epochs in the band cannot testify to a change


def nightly_stats(dets, band=None, min_epochs_off=False):
    """{night: (median mag, error, n, mean mjd)} from every epoch of the
    night, in one band (the points' band; None takes them all). Nights
    with fewer than MIN_EPOCHS_PER_NIGHT epochs are left out (unless
    min_epochs_off, for the band-blind summary). The error is the robust
    scatter over sqrt(n), never below 0.02 mag or below the median quoted
    error over sqrt(n). Two nights are only comparable in the same band:
    a clear frame calibrated to Sloan r and one to Sloan i differ by the
    star's colour, not by variability."""
    by_night = {}
    for p in dets:
        pts = [(pt[1], pt[2], pt[0]) for pt in p["points"] if band is None or pt[3] == band] or (
            [(p["mag"], _finite(p["magerr"]), None)]
            if band is None and math.isfinite(p["mag"]) and p["mag"] < 90 else [])
        if pts:
            by_night.setdefault(p["night"], []).extend(pts)
    out = {}
    for night, pts in by_night.items():
        mags = sorted(m for m, _, _ in pts)
        n = len(mags)
        if n < MIN_EPOCHS_PER_NIGHT and not (band is None and min_epochs_off):
            continue
        # Median and a scatter from the median absolute deviation: one bad
        # epoch (a cosmic ray, a satellite, a frame with a poor zero point)
        # then moves neither the value nor the error. On the first live
        # page 41 of 65 "changed" sources rested on a single-epoch night.
        med = mags[n // 2] if n % 2 else 0.5 * (mags[n // 2 - 1] + mags[n // 2])
        dev = sorted(abs(m - med) for m in mags)
        mad = dev[n // 2] if n % 2 else 0.5 * (dev[n // 2 - 1] + dev[n // 2])
        scatter = 1.4826 * mad
        errs = sorted(e for _, e, _ in pts if e is not None and e > 0)
        quoted = errs[len(errs) // 2] if errs else 0.05
        err = max(0.02, scatter / math.sqrt(n), quoted / math.sqrt(n))
        mjds = [t for _, _, t in pts if t is not None]
        out[night] = (med, err, n, sum(mjds) / len(mjds) if mjds else None)
    return out


def night_trend(stats):
    """Weighted straight line through the nightly means: (slope mag/day,
    slope error, significance, span days). Fading is a positive slope.
    Needs three nights; with two the pairwise test already says it all."""
    pts = [(v[3], v[0], v[1]) for v in stats.values() if v[3] is not None]
    if len(pts) < 3:
        return None
    w = [1.0 / (e * e) for _, _, e in pts]
    sw = sum(w)
    tm = sum(wi * t for wi, (t, _, _) in zip(w, pts)) / sw
    mm = sum(wi * m for wi, (_, m, _) in zip(w, pts)) / sw
    sxx = sum(wi * (t - tm) ** 2 for wi, (t, _, _) in zip(w, pts))
    if sxx <= 0:
        return None
    slope = sum(wi * (t - tm) * (m - mm) for wi, (t, m, _) in zip(w, pts)) / sxx
    err = math.sqrt(1.0 / sxx)
    span = max(t for t, _, _ in pts) - min(t for t, _, _ in pts)
    return {"slope": slope, "slope_err": err, "sigma": abs(slope) / err, "span_days": span}


def change_between_nights(stats):
    """Largest night-to-night difference and its significance."""
    nights = sorted(stats)
    best = (0.0, 0.0)
    for i in range(len(nights)):
        mi, ei = stats[nights[i]][:2]
        for j in range(i + 1, len(nights)):
            mj, ej = stats[nights[j]][:2]
            d = abs(mi - mj)
            sig = d / math.sqrt(ei * ei + ej * ej)
            if sig > best[1]:
                best = (d, sig)
    return best


def bands_of(dets):
    """Bands of the group's points, most epochs first."""
    counts = {}
    for p in dets:
        for pt in p["points"]:
            counts[pt[3]] = counts.get(pt[3], 0) + 1
    return sorted(counts, key=lambda b: (-counts[b], b))


def ensemble_offsets(groups_dets, min_sources=5):
    """{(field, night, band): median offset} of the field's nightly means
    from each source's own mean over its nights, in mag. A night whose
    zero point sits 0.2 mag off for every star (a clear filter under a
    different sky, an airmass term) would otherwise make every star of the
    field "changed". Sources need three nights in the band to vote, a
    night needs min_sources votes to get an offset."""
    votes = {}
    for field, dets in groups_dets:
        for band in bands_of(dets):
            stats = nightly_stats(dets, band)
            if len(stats) < 3:
                continue
            mean_all = sum(v[0] for v in stats.values()) / len(stats)
            for night, v in stats.items():
                votes.setdefault((field, night, band), []).append(v[0] - mean_all)
    out = {}
    for key, vals in votes.items():
        if len(vals) >= min_sources:
            vals.sort()
            out[key] = vals[len(vals) // 2]
    return out


def apply_offsets(dets, field, offsets):
    """Subtract the field's nightly offsets from the points (in place)."""
    for p in dets:
        for pt in p["points"]:
            off = offsets.get((field, p["night"], pt[3]))
            if off:
                pt[1] -= off


def within_night_change(dets):
    """Largest change inside one night: the median of the first third of
    the night's epochs against the median of the last third, with at least
    two epochs per third, in one band at a time. The telescope cycles its filters
    within a night, so across bands the first and last third are simply
    different filters and every star with a colour would "fade". Returns
    (delta with sign, significance) for the most significant night and
    band; delta > 0 means fading."""
    by_night = {}
    for p in dets:
        for pt in p["points"]:
            if pt[0] is not None:
                by_night.setdefault((p["night"], pt[3]), []).append((pt[0], pt[1], pt[2]))
    best = (0.0, 0.0)
    for pts in by_night.values():
        pts.sort()
        n = len(pts)
        if n < 6:
            continue           # fewer than two epochs per third: one point would decide
        k = n // 3
        a, b = sorted(m for _, m, _ in pts[:k]), sorted(m for _, m, _ in pts[-k:])
        ma, mb = a[k // 2] if k % 2 else 0.5 * (a[k // 2 - 1] + a[k // 2]), \
                 b[k // 2] if k % 2 else 0.5 * (b[k // 2 - 1] + b[k // 2])
        var_a = sum((m - ma) ** 2 for m in a) / (k - 1)
        var_b = sum((m - mb) ** 2 for m in b) / (k - 1)
        errs = sorted(e for _, _, e in pts if e is not None and e > 0)
        quoted = errs[len(errs) // 2] if errs else 0.05
        err = max(0.02, math.sqrt((var_a + var_b) / k), quoted * math.sqrt(2.0 / k))
        sig = abs(mb - ma) / err
        if sig > best[1]:
            best = (mb - ma, sig)
    return best


# A change has to be large to matter here: the interest is in new sources
# and outbursts, and a few tenths of a magnitude between nights is within
# what seeing does to the photometry of a crowded field. --min-change sets it.
CHANGE_MIN_MAG = 1.0
CHANGE_MIN_SIGMA = 5.0
LIMIT_MARGIN_MAG = 1.0     # a non-detection counts when the source would have been this far above the limit


def non_detections(group_dets, ra, dec, observations, detected_obs, data_dir, cells_cache):
    """(missed, present): observations covering (ra, dec) in which the
    source is not among the candidates. `missed` are those where no frame
    detected anything at the position (with the frame limit); `present`
    are those where the frames did detect it but the pipeline did not
    flag it (a catalogue-matched night). Both oldest first."""
    missed, present = [], []
    for obs in observations:
        f = obs["facts"]
        if f["obs_id"] in detected_obs or not covers(f, ra, dec):
            continue
        name = f"obs_{f['obs_id']}"
        cells = cells_cache.get(name)
        if cells is None:
            cells = load_cells(data_dir, name) if data_dir is not None else None
            cells_cache[name] = cells if cells is not None else set()
        dec0 = f["center"][1] if f.get("center") and f["center"][1] is not None else dec
        rec = {"obs_id": f["obs_id"], "night": f["nights"][0] if f["nights"] else "",
               "object": f["object"], "maglim": f.get("maglim"), "first": f["first"]}
        (present if cells and detected_in(cells, ra, dec, dec0) else missed).append(rec)
    return sorted(missed, key=lambda d: d["night"]), sorted(present, key=lambda d: d["night"])


def build_groups(observations, args, log, data_dir=None):
    cells_cache = {}
    points = []
    for obs in observations:
        facts = obs["facts"]
        for row in obs["rows"]:
            if row["number"] == 0 and not args.include_forced:
                continue
            if row["type"] == "trail" or row["q"] < args.min_quality:
                continue
            points.append({**row, "obs": facts})
    log(f"{len(points)} candidates kept for clustering")
    clusters = []
    for members in cluster(points, args.radius):
        dets = sorted((points[i] for i in members), key=lambda p: (p["night"], p["obs"]["obs_id"]))
        nights = sorted({p["night"] for p in dets if p["night"]})
        if len(nights) < args.min_nights:
            continue
        for p in dets:
            p["points"] = detection_points(p, data_dir)
        clusters.append((dominant_field(dets), dets))
    offsets = ensemble_offsets([(f, d) for f, d in clusters]) if not args.no_ensemble else {}
    if offsets:
        big = sorted(offsets.items(), key=lambda kv: -abs(kv[1]))[:3]
        log(f"{len(offsets)} field/night/band zero-point offsets removed, largest: "
            + ", ".join(f"{k[0]} {k[1]} {k[2]} {v:+.2f}" for k, v in big))
    groups = []
    for field, dets in clusters:
        apply_offsets(dets, field, offsets)
        nights = sorted({p["night"] for p in dets if p["night"]})
        obs_ids = sorted({p["obs"]["obs_id"] for p in dets})
        # One detection per observation is the norm; the pipeline may split a
        # source into two rows, in which case both are listed.
        w = [max(p["q"], 0.1) for p in dets]
        ra = sum(p["ra"] * wi for p, wi in zip(dets, w)) / sum(w)
        dec = sum(p["dec"] * wi for p, wi in zip(dets, w)) / sum(w)
        scatter = max(angular_sep_arcsec(p["ra"], p["dec"], ra, dec) for p in dets)
        mags = [p["mag"] for p in dets if math.isfinite(p["mag"]) and p["mag"] < 90]
        # Night-to-night comparisons in one band at a time; the band with
        # the most epochs decides the trend and the tags, but the largest
        # significant change of any band counts.
        bands = bands_of(dets) or [None]
        stats_all = nightly_stats(dets, min_epochs_off=True)   # every band, every night: "seen at all" only
        per_band = {}
        for band in bands:
            st = nightly_stats(dets, band)
            per_band[band] = (st, change_between_nights(st), night_trend(st))
        main_band = bands[0]
        stats = per_band[main_band][0] or stats_all
        night_order = sorted(stats)
        night_mags = [stats[n][0] for n in night_order]
        d_nights, s_nights = max((pb[1] for pb in per_band.values()), key=lambda t: t[1], default=(0.0, 0.0))
        d_within, s_within = within_night_change(dets)
        trend_fit = max((pb[2] for pb in per_band.values() if pb[2]), key=lambda f: f["sigma"], default=None)
        # A slow, steady decline or rise (a supernova over weeks) shows as a
        # significant slope even when single steps are small.
        slope_changed = bool(trend_fit) and trend_fit["sigma"] >= CHANGE_MIN_SIGMA and \
            abs(trend_fit["slope"]) * trend_fit["span_days"] >= CHANGE_MIN_MAG
        nights_changed = (d_nights >= CHANGE_MIN_MAG and s_nights >= CHANGE_MIN_SIGMA) or slope_changed
        within_changed = abs(d_within) >= CHANGE_MIN_MAG and s_within >= CHANGE_MIN_SIGMA
        changed = nights_changed or within_changed
        # The change reported is the larger significant one (else the larger).
        cands = [(d, sg) for d, sg, ok in ((d_nights, s_nights, nights_changed), (abs(d_within), s_within, within_changed)) if ok]
        delta_mag, change_sigma = max(cands or [(d_nights, s_nights), (abs(d_within), s_within)])
        change_kind = ("between nights" if nights_changed and (not within_changed or d_nights >= abs(d_within))
                       else "within a night" if within_changed else "")
        if change_kind and len(bands) > 1:
            change_kind += f", {main_band}"
        trend = ""
        if not changed:
            trend = "steady" if len(night_mags) >= 2 or n_pts_total(dets) >= 4 else ""
        elif change_kind == "between nights":
            first, last = night_mags[0], night_mags[-1]
            sig_ends = abs(first - last) / math.sqrt(stats[night_order[0]][1] ** 2 + stats[night_order[-1]][1] ** 2)
            if slope_changed:
                trend = "fading" if trend_fit["slope"] > 0 else "brightening"
            elif abs(first - last) >= CHANGE_MIN_MAG and sig_ends >= CHANGE_MIN_SIGMA:
                trend = "fading" if last > first else "brightening"
            else:
                trend = "variable"
        else:
            trend = "fading" if d_within > 0 else "brightening"
        objects = sorted({p["obs"]["object"] for p in dets if p["obs"]["object"]})
        n_points = sum(len(p["points"]) for p in dets)
        mag_bright = min(night_mags) if night_mags else None

        # Nights the field was observed without this source. A non-detection
        # is constraining when the frame went LIMIT_MARGIN_MAG deeper than
        # the source's magnitude on its nearest detected night.
        missed, present = (non_detections(dets, ra, dec, observations, set(obs_ids), data_dir, cells_cache)
                           if nights else ([], []))
        first_night, last_night = nights[0], nights[-1]
        before = [m for m in missed if m["night"] and m["night"] < first_night]
        after = [m for m in missed if m["night"] and m["night"] > last_night]
        between = [m for m in missed if m["night"] and first_night <= m["night"] <= last_night
                   and m["night"] not in stats]

        def constraining(missed_list, mag):
            return [m for m in missed_list if m["maglim"] is not None and mag is not None
                    and m["maglim"] - LIMIT_MARGIN_MAG >= mag]
        # The limit test uses the brightest band the source was seen in on
        # that night: a frame in another band is not directly comparable,
        # but a star well above the limit in one band is not 1 mag fainter
        # in the next.
        def night_mag(night):
            # Only a night with enough epochs in some band can claim the
            # source was really there (or really gone by then).
            vals = [pb[0][night][0] for pb in per_band.values() if night in pb[0]]
            return min(vals) if vals else None
        appeared = constraining(before, night_mag(first_night))
        disappeared = constraining(after, night_mag(last_night))
        score = transient_score(mag_bright, len(nights), n_points, max(p["q"] for p in dets),
                                delta_mag if changed else 0.0, bool(appeared), bool(disappeared),
                                fading=(trend == "fading"))
        groups.append({
            "field": dominant_field(dets), "score": score["score"], "score_parts": score,
            "mag_bright": mag_bright, "n_points": n_points,
            "delta_mag": round(delta_mag, 3), "change_sigma": round(change_sigma, 1), "changed": changed,
            "change_kind": change_kind,
            "slope_mag_per_day": round(trend_fit["slope"], 4) if trend_fit else None,
            "slope_sigma": round(trend_fit["sigma"], 1) if trend_fit else None,
            "span_days": round(trend_fit["span_days"], 1) if trend_fit else None,
            "nightly": {n: [round(v[0], 3), round(v[1], 3), v[2]] for n, v in sorted(stats.items())},
            "band": main_band or "",
            "bands": bands if bands != [None] else [],
            "offsets_applied": {f"{n} {b}": round(offsets[(field, n, b)], 3) for n in nights for b in bands
                                if (field, n, b) in offsets},
            "n_missed_before": len(before), "n_missed_after": len(after), "n_missed_between": len(between),
            "n_present_unflagged": len(present),
            "appeared": bool(appeared), "disappeared": bool(disappeared),
            "limit_before": max((m["maglim"] for m in before if m["maglim"] is not None), default=None),
            "limit_after": max((m["maglim"] for m in after if m["maglim"] is not None), default=None),
            "missed": [{"night": m["night"], "obs_id": m["obs_id"], "maglim": m["maglim"], "object": m["object"]}
                       for m in before[-5:] + between[:5] + after[:5]],
            "ra": ra, "dec": dec, "scatter_arcsec": scatter,
            "n_nights": len(nights), "nights": nights, "n_obs": len(obs_ids), "obs_ids": obs_ids,
            "n_det": len(dets), "max_q": max(p["q"] for p in dets),
            "sum_q": sum(p["q"] for p in dets),
            "mag_min": min(mags) if mags else None, "mag_max": max(mags) if mags else None,
            "trend": trend, "objects": objects,
            "grb_field": any(p["obs"].get("grb_ra") is not None for p in dets),
            "telescopes": sorted({p["obs"]["telescope"] for p in dets if p["obs"]["telescope"]}),
            "detections": [{
                "obs_id": p["obs"]["obs_id"], "object": p["obs"]["object"], "night": p["night"],
                "telescope": p["obs"]["telescope"], "filter": p["obs"]["filter"],
                "first": p["obs"]["first"], "frames": p["obs"]["frames"],
                "id": p["id"], "ra": p["ra"], "dec": p["dec"],
                "mag": _finite(p["mag"]), "magerr": _finite(p["magerr"]), "q": p["q"],
                "n_det": p["n_det"], "n_epochs": p["n_epochs"], "type": p["type"],
                "forced": p["number"] == 0,
                "points": p["points"],
            } for p in dets],
        })
    groups.sort(key=lambda g: (-g["score"], -g["n_nights"], -g["max_q"]))
    for k, g in enumerate(groups, 1):
        g["rank"] = k
    log(f"{len(groups)} groups seen on at least {args.min_nights} nights, "
        f"{sum(len(d['points']) for g in groups for d in g['detections'])} lightcurve points")
    return groups


def dominant_field(dets):
    """The field (OBJECT) most of the group's observations were taken as."""
    counts = {}
    for p in dets:
        counts[p["obs"]["object"] or ""] = counts.get(p["obs"]["object"] or "", 0) + 1
    return max(sorted(counts), key=counts.get) if counts else ""


def transient_score(mag_bright, n_nights, n_points, max_q, delta_mag=0.0, appeared=False, disappeared=False,
                    fading=False):
    """Brightest, best covered and changing first.

    brightness = 18 - brightest nightly mean magnitude, clipped to 0..10
    coverage   = 2 x (nights - 1) + log2(1 + measured epochs)
    quality    = log10(1 + best pipeline quality score), at most 2
    change     = 3 x the largest significant change in mag when it is a
                 brightening (at most 6), 1 x when it is a fading (at most 2),
                 + 4 if the source appeared (absent from earlier frames deep
                 enough to have shown it), + 1 if it disappeared the same way.
                 New and brightening sources are what the page is for.
    score      = brightness + coverage + quality + change
    """
    brightness = min(10.0, max(0.0, 18.0 - mag_bright)) if mag_bright is not None else 0.0
    coverage = 2.0 * (n_nights - 1) + math.log2(1 + n_points)
    quality = min(2.0, math.log10(1 + max(max_q, 0.0)))
    d = max(0.0, delta_mag)
    change = (min(2.0, d) if fading else min(6.0, 3.0 * d)) + (4.0 if appeared else 0.0) + (1.0 if disappeared else 0.0)
    return {"brightness": round(brightness, 2), "coverage": round(coverage, 2),
            "quality": round(quality, 2), "change": round(change, 2),
            "score": round(brightness + coverage + quality + change, 2)}


def detection_points(p, data_dir):
    """The epochs of one detection: its lightcurve file, else the table's magnitude."""
    band = p["obs"]["filter"] or band_of_frame(p["obs"]["first_frame"])
    if data_dir is not None and p["id"]:
        pts = read_lightcurve(data_dir / f"obs_{p['obs']['obs_id']}" / f"{p['id']}_lightcurve.ecsv", band)
        if pts:
            return pts
    if not (math.isfinite(p["mag"]) and p["mag"] < 90):
        return []
    t = frame_time(p.get("source_file", "")) or frame_time(p["obs"]["first_frame"])
    return [[mjd_of(t) if t else None, p["mag"], _finite(p["magerr"]), normalise_band(band)]]


# ------------------------------------------------------------------- page

def sanitize_candidate_id(candidate_id):
    """As frontend_generator.sanitize_candidate_id names cutout files."""
    s = re.sub(r"[^a-zA-Z0-9_\-.]", "_", str(candidate_id))
    s = re.sub(r"_+", "_", s).strip("_")
    return s or "candidate"


def site_assets(public_dir, det):
    """Relative links into the observation's own site, when they exist."""
    site_dir = public_dir / f"obs_{det['obs_id']}"
    rel = f"../obs_{det['obs_id']}"
    out = {"site": f"{rel}/index.html" if (site_dir / "index.html").exists() else ""}
    if det["id"]:
        cid = sanitize_candidate_id(det["id"])
        for ext in ("webp", "png", "jpg"):
            if (site_dir / "cutouts" / f"{cid}_montage.{ext}").exists():
                out["montage"] = f"{rel}/cutouts/{cid}_montage.{ext}"
                break
        if (site_dir / "lightcurves" / f"{cid}_lightcurve.png").exists():
            out["lightcurve"] = f"{rel}/lightcurves/{cid}_lightcurve.png"
    return out


def sexagesimal(ra, dec):
    h = ra / 15.0
    hh = int(h); mm = int((h - hh) * 60); ss = ((h - hh) * 60 - mm) * 60
    sign = "-" if dec < 0 else "+"
    d = abs(dec); dd = int(d); dm = int((d - dd) * 60); ds = ((d - dd) * 60 - dm) * 60
    return f"{hh:02d}:{mm:02d}:{ss:05.2f} {sign}{dd:02d}:{dm:02d}:{ds:04.1f}"


def _mag(m, e=None):
    if m is None or m > 90:
        return "-"
    return f"{m:.2f}" + (f" ± {e:.2f}" if e is not None and e < 9 else "")


PAGE_CSS = """
body{font-family:system-ui,sans-serif;margin:0;padding:1em 1.5em;background:#f5f6f8;color:#222}
h1{margin:0 0 .2em}
.meta{color:#555;font-size:.9em;margin-bottom:1em}
.group{background:#fff;border:1px solid #ddd;border-radius:6px;padding:.8em 1em;margin-bottom:1em}
.group h2{margin:0 0 .3em;font-size:1.1em}
.group h2 small{font-weight:normal;color:#666}
.tag{display:inline-block;border-radius:3px;padding:0 .4em;font-size:.8em;margin-left:.4em;color:#fff;background:#888}
.tag.fading{background:#c0392b}.tag.brightening{background:#2980b9}.tag.steady{background:#7f8c8d}.tag.variable{background:#8e44ad}
.tag.grb{background:#d35400}.tag.appeared{background:#16a085}.tag.disappeared{background:#2c3e50}
table{border-collapse:collapse;font-size:.9em;width:100%}
th,td{padding:.25em .5em;border-bottom:1px solid #eee;text-align:left;vertical-align:middle}
th{background:#f0f0f0}
td.num{text-align:right;font-variant-numeric:tabular-nums}
svg.lc{display:block;max-width:760px;margin:.2em 0 .6em;background:#fcfcfb;border:1px solid #eee;border-radius:4px}
img.thumb{height:90px;max-width:100%;border:1px solid #ccc;background:#000}
.links a{margin-right:.6em}
.empty{color:#777;font-style:italic}
details.field{margin-bottom:.6em}
details.field>summary{cursor:pointer;padding:.5em .8em;background:#e9ecf1;border-radius:6px;font-size:1.05em}
details.field[open]>summary{margin-bottom:.6em}
details.field .group{margin-left:.5em}
"""


BAND_ORDER = ("N", "C", "clear", "Sloan_r", "r", "R", "Sloan_g", "g", "V", "B",
              "Sloan_i", "i", "I", "Sloan_z", "z")
BAND_COLOURS = ("#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948")


def band_colours(bands):
    """Fixed hue per band: known bands in BAND_ORDER, others after, in name order."""
    known = [b for b in BAND_ORDER if b in bands]
    rest = sorted(b for b in bands if b not in BAND_ORDER)
    return {b: BAND_COLOURS[k % len(BAND_COLOURS)] for k, b in enumerate(known + rest)}


def mjd_to_ut(mjd):
    return (datetime(1858, 11, 17, tzinfo=timezone.utc) + timedelta(days=mjd)).strftime("%Y-%m-%d %H:%M:%S")


MAX_POINTS_PER_NIGHT = 40


def bin_points(pts, max_points):
    """At most `max_points` marks per night: epochs of one band averaged in
    equal time bins. The tooltip says how many epochs a mark stands for."""
    if len(pts) <= max_points:
        return [(pt, d, 1) for pt, d in pts]
    t0 = min(pt[0] for pt, _ in pts)
    t1 = max(pt[0] for pt, _ in pts)
    bins = {}
    for pt, d in pts:
        k = min(max_points - 1, int((pt[0] - t0) / (t1 - t0) * max_points)) if t1 > t0 else 0
        bins.setdefault((pt[3], k), []).append((pt, d))
    out = []
    for (band, _), members in sorted(bins.items(), key=lambda kv: kv[0][1]):
        n = len(members)
        mjd = sum(pt[0] for pt, _ in members) / n
        mag = sum(pt[1] for pt, _ in members) / n
        errs = [pt[2] for pt, _ in members if pt[2] is not None]
        err = (sum(errs) / len(errs)) / math.sqrt(n) if errs else None
        out.append(([mjd, mag, err, band], members[0][1], n))
    return out


def lightcurve_svg(group, width=760, height=230, max_points=MAX_POINTS_PER_NIGHT):
    """One panel per night, a shared inverted magnitude axis, bands by colour."""
    e = html.escape
    nights = {}
    for d in group["detections"]:
        for pt in d["points"]:
            if pt[0] is None:
                continue
            nights.setdefault(d["night"], []).append((pt, d))
    if not nights:
        return ""
    nights = {night: bin_points(pts, max_points) for night, pts in nights.items()}
    mags = [pt[1] for pts in nights.values() for pt, _, _ in pts]
    lo, hi = min(mags), max(mags)
    pad = max(0.25, 0.15 * (hi - lo))
    y0, y1 = lo - pad, hi + pad           # bright (top) .. faint (bottom)
    step = 0.5 if y1 - y0 <= 3 else 1.0 if y1 - y0 <= 8 else 2.0
    bands = sorted({pt[3] for pts in nights.values() for pt, _, _ in pts})
    colours = band_colours(bands)
    left, right, top, bottom, gap = 44, 12, 26, 30, 10
    n = len(nights)
    panel_w = (width - left - right - gap * (n - 1)) / n
    plot_h = height - top - bottom

    def y_of(m):
        return top + (m - y0) / (y1 - y0) * plot_h

    out = [f"<svg class='lc' viewBox='0 0 {width} {height}' width='100%' role='img' "
           f"aria-label='Lightcurve over {n} nights'>"
           "<style>.lc text{font:11px system-ui,sans-serif;fill:#555}.lc .ax{stroke:#ccc}"
           ".lc .gr{stroke:#eee}.lc .eb{stroke-width:1}.lc circle{stroke:#fff;stroke-width:2}"
           "</style>"]
    m = math.ceil(y0 / step) * step
    while m <= y1:
        y = y_of(m)
        out.append(f"<line class='gr' x1='{left}' x2='{width - right}' y1='{y:.1f}' y2='{y:.1f}'/>"
                   f"<text x='{left - 6}' y='{y + 4:.1f}' text-anchor='end'>{m:g}</text>")
        m += step
    out.append(f"<text x='12' y='{top + plot_h / 2:.0f}' transform='rotate(-90 12 {top + plot_h / 2:.0f})' "
               "text-anchor='middle'>mag</text>")
    for k, night in enumerate(sorted(nights)):
        pts = sorted(nights[night], key=lambda q: q[0][0])
        x_left = left + k * (panel_w + gap)
        t0, t1 = pts[0][0][0], pts[-1][0][0]
        span_min = (t1 - t0) * 1440
        inner = panel_w - 16

        def x_of(mjd, t0=t0, t1=t1, x_left=x_left, inner=inner):
            return x_left + 8 + (inner / 2 if t1 == t0 else (mjd - t0) / (t1 - t0) * inner)

        # Label sized to the panel: the full date only when there is room for it.
        label = (night if panel_w >= 200 else night[5:]) + " · " + mjd_to_ut(t0)[11:16]
        if span_min >= 1 and panel_w >= 150:
            label += f" +{span_min:.0f} min"
        out.append(f"<rect x='{x_left:.1f}' y='{top}' width='{panel_w:.1f}' height='{plot_h}' fill='none' class='ax'/>"
                   f"<text x='{x_left + panel_w / 2:.1f}' y='{top - 8}' text-anchor='middle'>{e(label)}</text>")
        for pt, d, n_avg in pts:
            mjd, mag, err, band = pt
            x, y = x_of(mjd), y_of(mag)
            c = colours.get(band, "#888")
            tip = (f"{mjd_to_ut(mjd)} UT  {mag:.2f}" + (f" ± {err:.2f}" if err is not None else "") +
                   f"  {band}  obs_{d['obs_id']}" + (f"  (mean of {n_avg} epochs)" if n_avg > 1 else ""))
            if err is not None and y_of(mag + err) - y_of(mag - err) >= 1.5:  # visible beyond the mark
                out.append(f"<line class='eb' stroke='{c}' x1='{x:.1f}' x2='{x:.1f}' "
                           f"y1='{y_of(mag - err):.1f}' y2='{y_of(mag + err):.1f}'/>")
            out.append(f"<circle cx='{x:.1f}' cy='{y:.1f}' r='4' fill='{c}'><title>{e(tip)}</title></circle>")
    if len(bands) > 1:
        x = left
        for b in bands:
            out.append(f"<circle cx='{x + 5}' cy='{height - 9}' r='4' fill='{colours[b]}'/>"
                       f"<text x='{x + 13}' y='{height - 5}'>{e(b)}</text>")
            x += 26 + 7 * len(b)
    out.append("</svg>")
    return "".join(out)


EXTERNAL_LINKS = (
    ("Aladin", "https://aladin.cds.unistra.fr/AladinLite/?target={ra:.5f}%20{dec:+.5f}&fov=0.05"),
    ("SIMBAD", "https://simbad.cds.unistra.fr/simbad/sim-coo?Coord={ra:.5f}%20{dec:+.5f}&Radius=10&Radius.unit=arcsec"),
    ("TNS", "https://www.wis-tns.org/search?ra={ra:.5f}&decl={dec:+.5f}&radius=10&coords_unit=arcsec"),
    ("PS1", "https://ps1images.stsci.edu/cgi-bin/ps1cutouts?pos={ra:.5f}{dec:+.5f}&filter=r&size=240&output_size=240&format=jpeg"),
    ("Fink (ZTF)", "https://fink-portal.org/?query_type=Conesearch&ra={ra:.5f}&dec={dec:+.5f}&radius=10"),
)


def _mag_span(g):
    lo, hi = g["mag_min"], g["mag_max"]
    return _mag(lo) + (f" – {_mag(hi)}" if hi is not None and hi != lo else "")


def interesting(g):
    """Changed, appeared or disappeared: what the reader wants first."""
    return bool(g.get("changed") or g.get("appeared") or g.get("disappeared"))


def render_card(g, public_dir):
    """One group in full: header, chart, one row per observation."""
    e = html.escape
    ra, dec = g["ra"], g["dec"]
    tag_list = ([g["trend"]] if g["trend"] else []) + (["appeared"] if g["appeared"] else []) + \
               (["disappeared"] if g["disappeared"] else []) + (["grb"] if g["grb_field"] else [])
    tags = "".join(f"<span class='tag {e(t)}'>{e(t)}</span>" for t in tag_list)
    ext = "".join(f"<a href='{e(url.format(ra=ra, dec=dec))}' target='_blank'>{e(name)}</a>"
                  for name, url in EXTERNAL_LINKS)
    sp = g["score_parts"]
    change_txt = (f"largest change {g['delta_mag']:.2f} mag ({g['change_sigma']:.0f}σ, {g['change_kind']})"
                  if g["changed"] else f"largest change {g['delta_mag']:.2f} mag ({g['change_sigma']:.0f}σ), not significant")
    if g.get("slope_mag_per_day") is not None:
        change_txt += (f", trend {g['slope_mag_per_day']:+.3f} mag/day over {g['span_days']:.0f} days "
                       f"({g['slope_sigma']:.0f}σ)")
    missed_txt = ""
    bits = []
    any_missed = g["n_missed_before"] or g["n_missed_after"] or g["n_missed_between"]
    if g["n_missed_before"]:
        bits.append(f"not in {g['n_missed_before']} earlier frame(s) of the field"
                    + (f" (deepest limit {g['limit_before']:.1f})" if g["limit_before"] is not None else ""))
    if g["n_missed_between"]:
        bits.append(f"missed on {g['n_missed_between']} night(s) in between")
    if g["n_missed_after"]:
        bits.append(f"not in {g['n_missed_after']} later frame(s)"
                    + (f" (deepest limit {g['limit_after']:.1f})" if g["limit_after"] is not None else ""))
    if g.get("n_present_unflagged"):
        bits.append(f"detected but not flagged on {g['n_present_unflagged']} other night(s)")
    if bits:
        missed_txt = "<div class='meta'>" + e("; ".join(bits)) + "."
        if any_missed:
            missed_txt += " Missed nights: " + ", ".join(
                f"<a href='../obs_{e(m['obs_id'])}/index.html'>{e(m['night'])}</a>"
                + (f" &gt;{m['maglim']:.1f}" if m["maglim"] is not None else "") for m in g["missed"])
        missed_txt += "</div>"
    parts = [
        f"<div class='group' id='g{g['rank']}'><h2>#{g['rank']} &nbsp; {ra:.5f} {dec:+.5f} "
        f"<small>({e(sexagesimal(ra, dec))})</small>{tags}"
        f"<small> &nbsp; score {g['score']:.1f} (bright {sp['brightness']:.1f} + cover {sp['coverage']:.1f} "
        f"+ q {sp['quality']:.1f} + change {sp['change']:.1f}) · {g['n_nights']} nights, {g['n_obs']} observations, "
        f"{g['n_points']} epochs, mag {e(_mag_span(g))}, {e(change_txt)}, best quality {g['max_q']:.1f}, "
        f"position scatter {g['scatter_arcsec']:.1f}\""
        + (f", also in: {e(', '.join(o for o in g['objects'] if o != g['field']))}"
           if len(g["objects"]) > 1 else "") + "</small></h2>"
        f"<div class='links meta'>{ext}</div>{missed_txt}{lightcurve_svg(g)}"
        "<table><tr><th>Night</th><th>Obs</th><th>Field</th><th>Tel/filter</th><th>First frame</th>"
        "<th>Candidate</th><th>Mag</th><th>Q</th><th>Det/epochs</th><th>Type</th><th>Montage</th><th>Lightcurve</th></tr>"]
    for d in g["detections"]:
        a = site_assets(public_dir, d)
        cand = e(d["id"] or f"{d['ra']:.4f} {d['dec']:+.4f}") + (" <small>(forced)</small>" if d["forced"] else "")
        obs_cell = f"<a href='{e(a['site'])}'>obs_{e(d['obs_id'])}</a>" if a["site"] else f"obs_{e(d['obs_id'])}"
        montage = (f"<a href='{e(a['montage'])}'><img class='thumb' loading='lazy' src='{e(a['montage'])}'></a>"
                   if a.get("montage") else "-")
        lc = (f"<a href='{e(a['lightcurve'])}'><img class='thumb' loading='lazy' src='{e(a['lightcurve'])}'></a>"
              if a.get("lightcurve") else "-")
        parts.append(
            f"<tr><td>{e(d['night'])}</td><td>{obs_cell}</td><td>{e(d['object'] or '-')}</td>"
            f"<td>{e(d['telescope'] or '-')}{('/' + e(d['filter'])) if d['filter'] else ''}</td>"
            f"<td>{e(d['first'])}</td><td>{cand}</td><td class='num'>{e(_mag(d['mag'], d['magerr']))}</td>"
            f"<td class='num'>{d['q']:.1f}</td><td class='num'>{d['n_det']}/{d['n_epochs']}</td><td>{e(d['type'])}</td>"
            f"<td>{montage}</td><td>{lc}</td></tr>")
    parts.append("</table></div>")
    return "".join(parts)


COMPACT_HEADER = ("<table><tr><th>#</th><th>Score</th><th>RA</th><th>Dec</th><th>Nights</th><th>Obs</th>"
                  "<th>Epochs</th><th>Mag</th><th>Δmag</th><th>σ</th><th>Missed</th><th>Best Q</th><th>Trend</th>"
                  "<th>Nights (links)</th></tr>")


def render_row(g):
    """One group as a table line, linking each night to its observation's site."""
    e = html.escape
    sites = " ".join(f"<a href='{e(d['site'])}'>{e(d['night'])}</a>" if d.get("site") else e(d["night"])
                     for d in g["detections"])
    return (f"<tr id='g{g['rank']}'><td>{g['rank']}</td><td class='num'>{g['score']:.1f}</td><td class='num'>{g['ra']:.5f}</td>"
            f"<td class='num'>{g['dec']:+.5f}</td><td class='num'>{g['n_nights']}</td><td class='num'>{g['n_obs']}</td>"
            f"<td class='num'>{g['n_points']}</td><td class='num'>{e(_mag_span(g))}</td>"
            f"<td class='num'>{g['delta_mag']:.2f}</td><td class='num'>{g['change_sigma']:.0f}</td>"
            f"<td class='num'>{g['n_missed_before']}/{g['n_missed_between']}/{g['n_missed_after']}"
            + (f" (+{g['n_present_unflagged']} seen)" if g.get('n_present_unflagged') else "") + "</td>"
            f"<td class='num'>{g['max_q']:.2f}</td>"
            f"<td>{e(' '.join(t for t in (g['trend'], 'appeared' if g['appeared'] else '', 'disappeared' if g['disappeared'] else '') if t))}</td>"
            f"<td>{sites}</td></tr>")


def render_page(groups, observations, args, public_dir, generated):
    e = html.escape
    parts = [f"<!DOCTYPE html><html lang='en'><head><meta charset='utf-8'>"
             f"<meta name='viewport' content='width=device-width,initial-scale=1'>"
             f"<title>{e(args.title)}</title><style>{PAGE_CSS}</style></head><body>"
             f"<h1>{e(args.title)}</h1>"]
    nights = sorted({n for o in observations for n in o["facts"]["nights"]})
    n_cand = sum(len(o["rows"]) for o in observations)
    parts.append(
        f"<div class='meta'>Generated {e(generated)} UTC. {len(observations)} observations with candidates "
        f"on {len(nights)} nights" + (f" ({e(nights[0])} to {e(nights[-1])})" if nights else "") +
        f", {n_cand} candidate rows, {len(groups)} sources seen on at least {args.min_nights} nights. "
        f"Match radius {args.radius:g}\", quality ≥ {args.min_quality:g}" +
        (", last %g days" % args.days if args.days else "") +
        (", forced target rows included" if args.include_forced else ", forced target rows (NUMBER 0) dropped") +
        ".<br>Score = brightness (18 − brightest nightly mean mag, 0..10) + coverage (2 per extra night "
        "+ log2 of measured epochs) + quality (log10 of the best pipeline score, at most 2) + change "
        f"(3 × the largest brightening, or 1 × the largest fading, between nightly means, along a fitted trend, or between "
        f"the first and last third of one night, when it is at least {CHANGE_MIN_MAG:g} mag and {CHANGE_MIN_SIGMA:g}σ, "
        "at most 6 or 2; +4 when the source appeared, "
        "i.e. earlier frames of the field "
        f"went {LIMIT_MARGIN_MAG:g} mag deeper than it without showing it; +1 when it disappeared the same way). "
        f"Night-to-night changes are compared within one band, between nightly medians of at least {MIN_EPOCHS_PER_NIGHT} "
        "epochs, after removing each field's nightly zero-point offset (the median over its sources with three or "
        "more nights); a single epoch never decides. 'Missed' counts observations of the "
        "field covering the position in which no frame detected anything there: before / between / after its "
        "detections; a night where the frames saw the star but the pipeline did not flag it is not a miss. "
        f"Sources are grouped by the field most of their observations were taken as; the best "
        f"{args.per_field} of each field get a card with the stitched lightcurve (at most {args.max_cards} "
        f"cards in all; changed, appeared and disappeared sources take the cards before steady ones), "
        f"the rest of the field is a table of at most {args.max_rows} lines. "
        "The JSON next to this page lists every source. Rows link to each observation's own page; "
        "thumbnails are that page's montage and lightcurve.</div>")
    if not groups:
        parts.append("<p class='empty'>No candidate was seen on more than one night.</p></body></html>")
        return "".join(parts)

    fields = {}
    for g in groups:  # already sorted by score, so each field's list is too
        fields.setdefault(g["field"], []).append(g)
    order = sorted(fields, key=lambda f: -fields[f][0]["score"])

    hot = [g for g in groups if interesting(g)]
    parts.append(f"<div class='group' id='changed'><h2>Changed and new sources ({len(hot)})</h2>")
    if hot:
        parts.append("<table><tr><th>#</th><th>Field</th><th>RA</th><th>Dec</th><th>Tags</th><th>Score</th>"
                     "<th>Mag</th><th>Δmag</th><th>σ</th><th>Nights</th><th>Missed</th></tr>")
        for g in hot:
            tags = " ".join(t for t in (g["trend"], "appeared" if g["appeared"] else "",
                                        "disappeared" if g["disappeared"] else "") if t)
            parts.append(f"<tr><td><a href='#g{g['rank']}'>{g['rank']}</a></td><td>{e(g['field'] or '-')}</td>"
                         f"<td class='num'>{g['ra']:.5f}</td><td class='num'>{g['dec']:+.5f}</td><td>{e(tags)}</td>"
                         f"<td class='num'>{g['score']:.1f}</td><td class='num'>{e(_mag_span(g))}</td>"
                         f"<td class='num'>{g['delta_mag']:.2f}</td><td class='num'>{g['change_sigma']:.0f}</td>"
                         f"<td class='num'>{g['n_nights']}</td>"
                         f"<td class='num'>{g['n_missed_before']}/{g['n_missed_between']}/{g['n_missed_after']}</td></tr>")
        parts.append("</table>")
    else:
        parts.append("<p class='empty'>None at the current thresholds.</p>")
    parts.append("</div>")
    parts.append("<div class='group'><h2>Fields</h2><table><tr><th>Field</th><th>Sources</th><th>Changed</th>"
                 "<th>Appeared</th><th>Best score</th><th>Brightest</th><th>Most nights</th><th>Nights observed</th></tr>")
    for k, f in enumerate(order, 1):
        gs = fields[f]
        f_nights = sorted({n for g in gs for n in g["nights"]})
        bright = min((g["mag_bright"] for g in gs if g["mag_bright"] is not None), default=None)
        parts.append(f"<tr><td><a href='#f{k}'>{e(f or '(no field name)')}</a></td><td class='num'>{len(gs)}</td>"
                     f"<td class='num'>{sum(1 for g in gs if g['changed'])}</td>"
                     f"<td class='num'>{sum(1 for g in gs if g['appeared'])}</td>"
                     f"<td class='num'>{gs[0]['score']:.1f}</td><td class='num'>{e(_mag(bright))}</td>"
                     f"<td class='num'>{max(g['n_nights'] for g in gs)}</td>"
                     f"<td>{len(f_nights)}: {e(f_nights[0])} … {e(f_nights[-1])}</td></tr>")
    parts.append("</table></div>")

    for k, f in enumerate(order, 1):
        gs = fields[f]
        n_cards = sum(1 for g in gs if g.get("card"))
        rest = [g for g in gs if not g.get("card")]
        shown_rest = rest[:args.max_rows] if args.max_rows > 0 else rest
        parts.append(f"<details class='field' id='f{k}'{' open' if k <= args.open_fields else ''}>"
                     f"<summary><b>{e(f or '(no field name)')}</b> — {len(gs)} sources, best score {gs[0]['score']:.1f}"
                     f", {n_cards} shown in full" +
                     (f", {len(rest)} more in the table" if rest else "") + "</summary>")
        for g in sorted(gs, key=lambda g: (not interesting(g), g["rank"])):
            if g.get("card"):
                parts.append(render_card(g, public_dir))
        if rest:
            parts.append(f"<div class='group'><h2>Other sources in {e(f or 'this field')}</h2>" + COMPACT_HEADER)
            parts.extend(render_row(g) for g in shown_rest)
            parts.append("</table>" + (f"<div class='meta'>{len(rest) - len(shown_rest)} more only in the JSON.</div>"
                                       if len(shown_rest) < len(rest) else "") + "</div>")
        parts.append("</details>")
    parts.append("</body></html>")
    return "".join(parts)


# -------------------------------------------------------------------- main

def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--data-dir", default=os.environ.get("PYRT_STATUS_DATA_DIR", "~/transient_work"))
    ap.add_argument("--public-dir", default=os.environ.get("PYRT_STATUS_PUBLIC_DIR", "~/public_html"))
    ap.add_argument("--out-dir", default=None, help="default <public-dir>/new_transients")
    ap.add_argument("--days", type=float, default=30, help="look back this many days (0 = all)")
    ap.add_argument("--radius", type=float, default=3.0, help="match radius in arcsec")
    ap.add_argument("--min-nights", type=int, default=2)
    ap.add_argument("--min-quality", type=float, default=0.02)
    ap.add_argument("--per-field", type=int, default=10, help="sources shown in full per field")
    ap.add_argument("--max-cards", type=int, default=300, help="sources shown in full on the whole page (0 = no cap)")
    ap.add_argument("--max-rows", type=int, default=100, help="table lines per field for the other sources (0 = all)")
    ap.add_argument("--open-fields", type=int, default=5, help="field sections open when the page loads")
    ap.add_argument("--include-forced", action="store_true", help="keep the NUMBER 0 forced target rows")
    ap.add_argument("--min-change", type=float, default=1.0,
                    help="smallest change in mag that counts (default %(default)s)")
    ap.add_argument("--no-ensemble", action="store_true",
                    help="do not remove per-field nightly zero-point offsets before looking for change")
    ap.add_argument("--title", default="Transients seen on more than one night")
    ap.add_argument("-q", "--quiet", action="store_true")
    args = ap.parse_args(argv)
    globals()["CHANGE_MIN_MAG"] = args.min_change

    data_dir = Path(args.data_dir).expanduser()
    public_dir = Path(args.public_dir).expanduser()
    out_dir = Path(args.out_dir).expanduser() if args.out_dir else public_dir / "new_transients"

    def log(msg):
        if not args.quiet:
            print(msg, file=sys.stderr)

    observations = crawl(data_dir, args.days, log)
    groups = build_groups(observations, args, log, data_dir)
    generated = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")

    out_dir.mkdir(parents=True, exist_ok=True)
    for g in groups:
        for d in g["detections"]:
            d.update(site_assets(public_dir, d))
    carded = set()
    budget = args.max_cards if args.max_cards > 0 else len(groups)
    per_field = {}
    # Cards go to the sources the page is for first: those that changed,
    # appeared or disappeared, in score order; the steady ones fill what is
    # left of each field's allowance.
    for g in sorted(groups, key=lambda g: (not interesting(g), g["rank"])):
        n = per_field.get(g["field"], 0)
        if n < args.per_field and budget > 0:
            per_field[g["field"]] = n + 1
            budget -= 1
            carded.add(g["rank"])
    for g in groups:
        g["card"] = g["rank"] in carded
        if not g["card"]:
            for d in g["detections"]:
                d.pop("points", None)  # keeps the JSON small; the count stays in n_points
    payload = {"generated_utc": generated, "parameters": {
        "days": args.days, "radius_arcsec": args.radius, "min_nights": args.min_nights,
        "min_quality": args.min_quality, "include_forced": args.include_forced,
        "per_field": args.per_field, "max_cards": args.max_cards},
        "n_observations": len(observations), "groups": groups}
    (out_dir / "new_transients.json").write_text(json.dumps(payload, indent=1, allow_nan=False, default=str))
    page = render_page(groups, observations, args, public_dir, generated)
    tmp = out_dir / "index.html.tmp"
    tmp.write_text(page)
    os.replace(tmp, out_dir / "index.html")
    log(f"wrote {out_dir / 'index.html'} ({len(page) / 1e6:.1f} MB)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
