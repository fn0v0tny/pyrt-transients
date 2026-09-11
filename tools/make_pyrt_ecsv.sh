#!/bin/bash
# Build pyrt-style .ecsv detection tables for raw frames that came without
# them (tests/190919B: FRAM-Auger, GRB 190919B, 2019-09-19/20), using the
# pyrt photometry chain the production host runs:
#
#   pyrt-phcat -I (SExtractor only) -> pyrt-cat2det -> pyrt-dophot -a (twice)
#
# By default the installed pyrt package's console scripts are used. Set
# PYRT_DIR to a checkout of the older flat layout (phcat.py, cat2det.py,
# dophot3.py) to run those scripts with $PYTHON instead.
#
# atlas@localhost (dophot's default) is not reachable outside the production
# host, so the reference catalogue defaults to atlas@vizier (ATLAS-RefCat2,
# all-sky, so it also covers this Dec -45 field that Pan-STARRS does not).
#
# dophot runs twice with -a, as production's get_ecsv.sh does: the first
# pass refits the WCS (ZPN for FRAM's NF4 camera), the second recomputes
# ALPHA/DELTA through it. On 20190919235102 that takes the Gaia residuals
# from 1.09" to 0.24" median and fills ASTSIGMA/ASTVAR.
#
# The committed tests/190919B ECSVs were built with PYRT_DIR pointing at the
# flat layout plus the 2026-09-11 local fixes (termfit squared WSSR and
# variance, stepwise non-finite rows, refit_astrometry get_fitdata), so they
# carry the pre-97101a7 ASTSIGMA/ASTVAR meaning and no ASTSCATT. mates14/pyrt
# 73dc607 has those fixes, but its new S0+SC error model returns NaN on this
# field (NaN rows in the default-mask medians of compute_error_model), and
# the second pass then fails writing ASTSIGMA=nan to a FITS header.
#
# FILTER=R frames need PHOT_FILTER=Sloan_r: dophot maps R to Johnson_R,
# which pyrt's atlas@vizier catalogue does not carry, so it matches no stars
# and writes nothing. The override has to go to cat2det (the filter is baked
# into the .det there; dophot's own -f/-b/-j do not change it) and it also
# replaces the ECSV's FILTER meta with the photometric band. tests/190919B
# was built as:
#   tools/make_pyrt_ecsv.sh tests/190919B tests/190919B/*-N-020-df.fits
#   PHOT_FILTER=Sloan_r tools/make_pyrt_ecsv.sh tests/190919B tests/190919B/*-R-060-df.fits
#
# Usage: tools/make_pyrt_ecsv.sh <out_dir> <frame.fits> [<frame.fits> ...]
# Env:   CATALOG (default atlas@vizier), JOBS (parallel frames, default 4),
#        PHOT_FILTER (optional photometric band override, e.g. Sloan_r),
#        PYRT_DIR + PYTHON (optional: use a flat-layout pyrt checkout)
set -euo pipefail

if [ $# -lt 2 ]; then
    sed -n '2,42p' "$0"
    exit 1
fi

OUT_DIR=$(realpath "$1"); shift
CATALOG=${CATALOG:-atlas@vizier}
JOBS=${JOBS:-4}
PYTHON=${PYTHON:-python3}
if [ -n "${PYRT_DIR:-}" ]; then
    PHCAT="$PYTHON $PYRT_DIR/phcat.py"
    CAT2DET="$PYTHON $PYRT_DIR/cat2det.py"
    DOPHOT="$PYTHON $PYRT_DIR/dophot3.py"
else
    for tool in pyrt-phcat pyrt-cat2det pyrt-dophot; do
        command -v "$tool" >/dev/null || { echo "$tool is not on PATH (install pyrt or set PYRT_DIR)" >&2; exit 1; }
    done
    PHCAT=pyrt-phcat
    CAT2DET=pyrt-cat2det
    DOPHOT=pyrt-dophot
fi

if ! command -v sex >/dev/null && ! command -v source-extractor >/dev/null; then
    echo "SExtractor (sex / source-extractor) is not on PATH" >&2
    exit 1
fi
mkdir -p "$OUT_DIR"

one_frame() {
    local fits stem work
    fits=$(realpath "$1")
    stem=$(basename "$fits" .fits)
    if [ -e "$OUT_DIR/$stem.ecsv" ]; then
        echo "skip $stem (exists)"
        return 0
    fi
    # Each frame gets its own work dir: dophot writes dophot.dat and a
    # catalogue cache into the cwd, which parallel jobs would clobber.
    work="$OUT_DIR/.work/$stem"
    mkdir -p "$work"
    ln -sf "$fits" "$work/$stem.fits"
    # dophot exits 0 even when it matched no stars and wrote nothing, hence
    # the explicit checks for its output.
    (
        cd "$work" &&
        $PHCAT -I "$stem.fits" >phcat.log 2>&1 &&
        $CAT2DET ${PHOT_FILTER:+-f "$PHOT_FILTER"} "$stem.cat" >cat2det.log 2>&1 &&
        $DOPHOT -a -C "$CATALOG" "$stem.det" >dophot.log 2>&1 &&
        [ -e "$stem.ecsv" ] && mv "$stem.ecsv" "$stem.det" &&
        $DOPHOT -a -C "$CATALOG" "$stem.det" >dophot_pass2.log 2>&1 &&
        [ -e "$stem.ecsv" ]
    ) || { echo "FAIL $stem (logs in $work)"; return 1; }
    mv "$work/$stem.ecsv" "$OUT_DIR/$stem.ecsv"
    rm -rf "$work"
    echo "ok   $stem"
}
export -f one_frame
export OUT_DIR CATALOG PHCAT CAT2DET DOPHOT

printf '%s\n' "$@" | xargs -P "$JOBS" -I{} bash -c 'one_frame "$@"' _ {}
rmdir "$OUT_DIR/.work" 2>/dev/null || true
