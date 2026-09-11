"""Frame WCS keywords from an ECSV meta dict.

astropy.wcs.WCS(dict) serialises the *whole* dict into FITS header cards,
so any one unrelated meta value it cannot store breaks the WCS:
- a NaN (pyrt writes ASTSIGMA=nan when its astrometric refit has nothing to
  fit);
- a list;
- a string longer than a card (the stack ECSV's STACK_INPUTS).
Each of these sent the catalogue matcher to its legacy fallback, which found
no candidates at all. Building the WCS from the WCS keywords alone avoids the
whole class.
"""

import numpy as np

WCS_KEY_PREFIXES = (
    "WCSAXES", "CTYPE", "CUNIT", "CRPIX", "CRVAL", "CDELT", "CROTA",
    "CD1_", "CD2_", "PC1_", "PC2_", "PV", "A_", "B_", "AP_", "BP_",
    "LONPOLE", "LATPOLE", "EQUINOX", "RADESYS", "NAXIS",
)
# RTS2 writes its own pointing-model WCS under a 'T' alternate code; astropy
# would read those as a second WCS, and they are not the frame's solution.
RTS2_ALTERNATE_KEYS = ("CTYPE1T", "CTYPE2T", "CRVAL1T", "CRVAL2T", "CRPIX1T", "CRPIX2T",
                       "CDELT1T", "CDELT2T", "CROTA2T")


def wcs_header_from_meta(meta) -> dict:
    """The finite, scalar WCS keywords of `meta`, as a dict astropy.wcs.WCS accepts."""
    header = {}
    for key, value in (meta or {}).items():
        key = str(key)
        if key in RTS2_ALTERNATE_KEYS or not key.startswith(WCS_KEY_PREFIXES):
            continue
        if not isinstance(value, (str, bool, int, float, np.integer, np.floating)):
            continue
        if isinstance(value, (float, np.floating)) and not np.isfinite(value):
            continue
        header[key] = value
    return header
