"""Differential colour correction of catalogue magnitudes, from a frame's
pyrt `RESPONSE` string.

`MAG_CALIB` in the ECSV already includes the zeropoint, spatial, airmass,
radial and nonlinearity corrections -- all computed with the source colours
set to zero. To compare a catalogue star with it, the same photometric model
has to be evaluated at that star's *actual* colours and the difference added
to its catalogue magnitude. That's what `simple_color_model` does, using the
bundled `fotfit` (the model pyrt itself fits).
"""

import logging
import re
from functools import lru_cache
from typing import Dict, Optional

import numpy as np

try:
    # The copy bundled with this package. A bare `import fotfit` only
    # resolves when the working directory is the package folder -- in every
    # installed/console-script context it silently gave `fotfit = None`
    # and colour corrections were quietly off.
    from pyrt_transient import fotfit
except ImportError:  # pragma: no cover - a standalone pyrt checkout on sys.path
    try:
        import fotfit
    except ImportError:
        fotfit = None

logger = logging.getLogger("core.color_model")

# fotfit's colour components are named C/D/E/F (g-r, r-i, i-z, z-J); a
# term key containing one of them (PC, PD, P2C, XC, ...) depends on colour.
_COLOUR_TERM_RE = re.compile(r"[CDEF]")


def parse_response(line: str) -> Dict[str, float]:
    """`"Z=25.0,PC=0.1,P2XY=0.3"` -> `{"Z": 25.0, "PC": 0.1, "P2XY": 0.3}`.
    Non-numeric terms (FILTER, SCHEMA) are skipped."""
    terms: Dict[str, float] = {}
    try:
        for chunk in str(line).split(","):
            if "=" not in chunk:
                continue
            term, strvalue = chunk.split("=", 1)
            term = term.strip()
            if term in ("FILTER", "SCHEMA"):
                continue
            try:
                terms[term] = float(strvalue)
            except ValueError:
                continue
    except (ValueError, AttributeError):
        return {}
    return terms


def has_colour_terms(line: Optional[str]) -> bool:
    """Whether the RESPONSE string carries any colour-dependent term."""
    return any(_COLOUR_TERM_RE.search(k) for k in parse_response(line or "") if k != "Z")


def _row(c1, c2, c3, c4):
    # mc, airmass, x, y, colours..., img, y, err, cat_x, cat_y --
    # everything but the colours at the neutral reference so that the
    # difference in `simple_color_model` isolates the colour dependence.
    return np.array([[0.0], [1.0], [0.0], [0.0], [c1], [c2], [c3], [c4],
                     [0], [0.0], [1.0], [0.5], [0.5]])


@lru_cache(maxsize=64)
def _colour_fit(line: str):
    """The prepared fotfit for a RESPONSE string, with its zero-colour
    reference magnitude; `None` when there is no colour correction to apply.

    Building one means a parse plus `fixall`/`fixterm`, and the RESPONSE
    string is constant for a whole epoch -- the per-detection catalogue
    comparison called this tens of thousands of times per epoch. `model()`
    does not mutate the object, so a single instance serves every call.
    """
    terms = parse_response(line)
    terms_no_z = {k: v for k, v in terms.items() if k != "Z"}
    if not terms_no_z or fotfit is None:
        return None

    try:
        ffit = fotfit.fotfit()
        ffit.fixall()
        ffit.fixterm(list(terms_no_z.keys()), values=list(terms_no_z.values()))
        reference_model = ffit.model(ffit.fixvalues, _row(0.0, 0.0, 0.0, 0.0))[0]
        return ffit, reference_model
    except Exception as exc:
        logger.debug(f"fotfit setup failed ({exc!r}); using uncorrected magnitudes")
        return None


def simple_color_model(line, data):
    """Catalogue magnitude with the frame's differential colour correction
    applied (see module docstring).

    Args:
        line: RESPONSE string (e.g. "Z=25.0,PC=0.1,XC=0.3")
        data: tuple (mag, color1, color2, color3, color4)

    Returns the catalogue magnitude corrected to the system `MAG_CALIB` is
    in; unchanged when there are no colour terms or fotfit is unavailable.
    """
    mag, color1, color2, color3, color4 = data

    prepared = _colour_fit(str(line))
    if prepared is None:
        return mag
    ffit, reference_model = prepared

    try:
        actual_model = ffit.model(ffit.fixvalues, _row(color1, color2, color3, color4))[0]
    except Exception as exc:
        logger.debug(f"fotfit colour correction failed ({exc!r}); using uncorrected magnitude")
        return mag
    return mag + (actual_model - reference_model)
