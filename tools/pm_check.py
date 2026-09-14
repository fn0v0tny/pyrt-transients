#!/usr/bin/env python3
"""Does proper-motion propagation put the catalogue's high-pm stars on the
detections? Read-only check on processed observation directories.

  pm_check.py [--catalog gaia] [--pm-min 100] [--cache-dir DIR] obs_dir [obs_dir ...]

For one detection table per observation (the middle one) the catalogue is
loaded the way the pipeline loads it (disk cache under --cache-dir, default
~/catalog_cache, else a query), and the matcher runs with propagation off
and on. Reported per observation:

  hpm      catalogue stars above --pm-min mas/yr inside the frame
  @prop    of those, detected stars whose nearest detection sits at the
           propagated position (the fix works for them)
  @cat     ... at the catalogue-epoch position instead (it does not)
  none     not detected, or moved less than 1" so it cannot be told
  new off/on   "new" candidates without / with propagation
  removed/created   candidates only in one of the two runs

Nothing is written except the catalogue cache the pipeline uses anyway.
"""
import argparse
import logging
import sys
import warnings
from pathlib import Path

import numpy as np
from astropy.table import Table

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from pyrt_transient.catalog import CatTransients, QueryParams, setup_catalog_cache  # noqa: E402


def sep_arcsec(ra1, dec1, ra2, dec2):
    return np.hypot((ra1 - ra2) * np.cos(np.radians(dec1)), dec1 - dec2) * 3600.0


def middle_table(obs_dir):
    files = sorted(f for f in Path(obs_dir).glob("*.ecsv")
                   if not f.name.endswith(("_transients.ecsv", "_lightcurve.ecsv")))
    if not files:
        return None
    det = Table.read(str(files[len(files) // 2]))
    return det[np.isfinite(det["MAG_CALIB"])] if "MAG_CALIB" in det.colnames else None


def check(obs_dir, catalog_name, pm_min):
    det = middle_table(obs_dir)
    if det is None or len(det) == 0:
        return None
    ra0, dec0 = float(np.median(det["ALPHA_J2000"])), float(np.median(det["DELTA_J2000"]))
    half = 0.5 * float(det.meta.get("FIELD", 0.3)) * 2 + 0.05
    params = QueryParams(ra=ra0, dec=dec0, width=half, height=half, mlim=20.0)
    cat = CatTransients(catalog=catalog_name, **params.__dict__)
    cat.precompute_photometric_data()
    epoch = cat.observation_epoch(det.meta)
    res = {}
    for prop in (False, True):
        d = det.copy()
        d.meta["propagate_proper_motion"] = prop
        d.meta["adaptive_idlimit_enabled"] = True
        out = cat.get_transient_candidates_optimized(d, idlimit=3.0, mag_change_threshold=1.0,
                                                     siglim=5.0, new_source_siglim=1.5)
        res[prop] = ({int(n) for n, t in zip(out["NUMBER"], out["candidate_type"]) if t == "new"}
                     if len(out) else set())
    pm = np.hypot(np.asarray(cat["pmra"], float) * np.cos(np.radians(cat["decdeg"])),
                  np.asarray(cat["pmdec"], float)) * 3.6e6
    ra_p, dec_p = cat.positions_at_epoch(epoch)
    dra, ddec = np.asarray(det["ALPHA_J2000"]), np.asarray(det["DELTA_J2000"])
    at_prop = at_cat = neither = 0
    examples = []
    for i in np.where(np.nan_to_num(pm) > pm_min)[0]:
        if sep_arcsec(cat["radeg"][i], cat["decdeg"][i], ra0, dec0) > 0.7 * half * 3600:
            continue
        s_cat = sep_arcsec(dra, ddec, cat["radeg"][i], cat["decdeg"][i]).min()
        s_prop = sep_arcsec(dra, ddec, ra_p[i], dec_p[i]).min()
        shift = sep_arcsec(ra_p[i], dec_p[i], cat["radeg"][i], cat["decdeg"][i])
        if min(s_cat, s_prop) > 3.0 or shift < 1.0:
            neither += 1
        elif s_prop < s_cat:
            at_prop += 1
            examples.append(f"{cat['radeg'][i]:.5f} {cat['decdeg'][i]:+.5f} pm {pm[i]:.0f} mas/yr: "
                            f"{s_cat:.2f}\" at catalogue -> {s_prop:.2f}\" propagated")
        else:
            at_cat += 1
            examples.append(f"{cat['radeg'][i]:.5f} {cat['decdeg'][i]:+.5f} pm {pm[i]:.0f} mas/yr: "
                            f"{s_cat:.2f}\" at catalogue -> {s_prop:.2f}\" propagated  <-- WORSE")
    return dict(obs=Path(obs_dir).name, ra=ra0, dec=dec0, epoch=epoch, n_det=len(det),
                hpm=at_prop + at_cat + neither, at_prop=at_prop, at_cat=at_cat, neither=neither,
                new_off=len(res[False]), new_on=len(res[True]),
                removed=len(res[False] - res[True]), created=len(res[True] - res[False]),
                examples=examples)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("obs_dirs", nargs="+")
    ap.add_argument("--catalog", default="gaia")
    ap.add_argument("--pm-min", type=float, default=100.0, help="mas/yr")
    ap.add_argument("--cache-dir", default=str(Path.home() / "catalog_cache"))
    ap.add_argument("-v", "--verbose", action="store_true", help="list every high-pm star")
    args = ap.parse_args(argv)
    logging.disable(logging.CRITICAL)
    warnings.filterwarnings("ignore")
    setup_catalog_cache(args.cache_dir)
    print(f"{'observation':14} {'centre':>16} {'epoch':>6} {'n_det':>5} {'hpm':>4} {'@prop':>5} {'@cat':>4} "
          f"{'none':>4} {'new off':>7} {'new on':>6} {'removed':>7} {'created':>7}")
    tot = dict.fromkeys(("hpm", "at_prop", "at_cat", "neither", "new_off", "new_on", "removed", "created"), 0)
    for obs_dir in args.obs_dirs:
        try:
            r = check(obs_dir, args.catalog, args.pm_min)
        except Exception as e:
            print(f"{Path(obs_dir).name:14} failed: {e}")
            continue
        if r is None:
            print(f"{Path(obs_dir).name:14} no detection table")
            continue
        print(f"{r['obs']:14} {r['ra']:8.3f} {r['dec']:+7.3f} {r['epoch'] or 0:6.1f} {r['n_det']:5d} {r['hpm']:4d} "
              f"{r['at_prop']:5d} {r['at_cat']:4d} {r['neither']:4d} {r['new_off']:7d} {r['new_on']:6d} "
              f"{r['removed']:7d} {r['created']:7d}")
        if args.verbose:
            for line in r["examples"]:
                print("    " + line)
        for k in tot:
            tot[k] += r[k]
    print("TOTAL " + " ".join(f"{k}={v}" for k, v in tot.items()))
    return 0


if __name__ == "__main__":
    sys.exit(main())
