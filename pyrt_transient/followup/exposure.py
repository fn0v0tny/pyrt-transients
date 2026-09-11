"""Exposure-time calculator -- how long the telescope needs to integrate to
measure a given magnitude at a given SNR under a given night's conditions.

Port of the standalone `new_exposure_calculator2.py` (empirical calibration:
`APE = 6.146 +/- 0.002`, RMS residual 0.047 dex in log10(magerror)). The
model is a smooth transition (`sbl`) between the photon-noise regime
(error ~ 10^(0.2*dm), i.e. 1/sqrt(flux)) and the background-noise regime,
with the crossover magnitude set by the sky sigma and the seeing disc area.

Two deliberate deviations from the standalone script, both measured against
this repo's real fixtures rather than assumed -- see
tests/test_followup_exposure.py, which asserts both:

1. `ZERO_OFFSET` is added to the instrumental magnitude. `break_magnitude`
   already carries `+ZERO` (10), but the script's `calculate_exptime` /
   `predict_performance` passed a bare `magnitude - magzero`, so the two
   halves of the formula sat on different zero points. Checked against real
   photometry (`MAG_CALIB`/`MAGERR_CALIB` of every source in
   tests/210619B's D50 frames): with the offset the model reproduces the
   observed errors to median +0.01 dex, rms 0.031 -- matching the
   calibration's own quoted 0.047; without it, it predicts errors ~330x
   too small (median -2.53 dex), which is why the script would answer
   "0.0 seconds" for a target well below a frame's MAGLIM.
2. Sky/`BGSIGMA` conversion is done consistently in ADU. The script's
   `sky_brightness_from_bgsigma` / `bgsigma_from_sky_brightness` are not
   inverses of each other -- round-tripping a header BGSIGMA at its own
   EXPTIME returns `GAIN * BGSIGMA`, not `BGSIGMA` (25.78 -> 20.88 on a
   real frame). Since `break_magnitude` expects ADU, that shifts predicted
   errors by -0.06 dex. Here variance is converted to electrons
   (`(gain*bgsigma)**2`), the readout noise (already electrons) subtracted
   there, and the result converted back -- so `bgsigma_at(EXPTIME_ref)`
   returns the header BGSIGMA exactly.

`GAIN` is read from the frame's own header when present (it is, in real
pyrt ECSV meta) rather than hardcoded, falling back to the calibration's
own 0.81 otherwise.
"""

from dataclasses import dataclass
from typing import Any, Dict, Optional

import numpy as np
from scipy.optimize import brentq

# Fitted/instrumental constants from the original calibration.
APE = 6.146           # fitted photon/background transition parameter
ZERO_OFFSET = 10.0    # zero point offset the fit was made in (see above)
DEFAULT_GAIN = 0.81   # e-/ADU, used only when the header has no GAIN
DEFAULT_READOUT_NOISE_E = 8.0  # effective readout noise, electrons

_REQUIRED_META = ("EXPTIME", "BGSIGMA", "MAGZERO", "FWHM")


class ExposureModelError(ValueError):
    """The reference frame can't drive the model (missing header keywords,
    or conditions the empirical calibration doesn't cover).

    Deliberately an exception rather than a `None` return: FUTURE_IDEAS.md's
    operational-gaps section (item 1) calls out collapsing distinct failure
    modes into one falsy value as the thing that makes a broken dependency
    indistinguishable from a real result. Callers catch it and report the
    message.
    """


def sbl(A: float, B: float, N: float, x):
    """Smooth transition between two linear regimes of slope A and B."""
    return A * x + (B - A) * (abs(N) * np.sqrt(1.0 + x * x / (N * N)) + x) / 2.0


def break_magnitude(bgsigma: float, fwhm_px: float, gain: float = DEFAULT_GAIN) -> float:
    """Instrumental magnitude where photon noise equals background noise."""
    return -2.5 * np.log10(APE * np.pi / 4 * fwhm_px * fwhm_px * (bgsigma * gain) ** 2) + ZERO_OFFSET


def log_magerror(instrumental_mag, bgsigma: float, fwhm_px: float,
                 gain: float = DEFAULT_GAIN):
    """log10 of the expected magnitude error.

    `instrumental_mag` is `mag - MAGZERO + ZERO_OFFSET` (see
    `instrumental_magnitude`), *not* a calibrated magnitude.
    """
    break_mag = break_magnitude(bgsigma, fwhm_px, gain)
    return sbl(0.2, 0.4, 2.5, instrumental_mag - break_mag) + 0.2 * break_mag - 2


def instrumental_magnitude(mag, magzero: float):
    """Calibrated magnitude -> the zero point `log_magerror` is fitted in."""
    return mag - magzero + ZERO_OFFSET


def snr_to_magerror(snr: float) -> float:
    """magerr = 2.5 / (SNR * ln 10)."""
    return 1.0 / (snr * np.log(10) / 2.5)


def magerror_to_snr(magerror: float) -> float:
    return 1.0 / (magerror * np.log(10) / 2.5)


@dataclass
class ReferenceConditions:
    """Observing conditions extracted from one real frame -- "the previous
    image" the next exposure is being planned against."""

    exptime_s: float
    bgsigma_adu: float
    magzero: float
    fwhm_px: float
    gain: float
    readout_noise_e: float
    sky_e_per_s: float
    source: Optional[str] = None
    maglim: Optional[float] = None

    @classmethod
    def from_meta(cls, meta: Dict[str, Any],
                  readout_noise_e: float = DEFAULT_READOUT_NOISE_E,
                  source: Optional[str] = None) -> "ReferenceConditions":
        missing = [k for k in _REQUIRED_META if meta.get(k) is None]
        if missing:
            raise ExposureModelError(
                f"reference frame is missing header keyword(s): {', '.join(missing)}"
            )
        try:
            exptime = float(meta["EXPTIME"])
            bgsigma = float(meta["BGSIGMA"])
            magzero = float(meta["MAGZERO"])
            fwhm = float(meta["FWHM"])
        except (TypeError, ValueError) as exc:
            raise ExposureModelError(f"non-numeric reference header value: {exc}") from exc

        gain = DEFAULT_GAIN
        raw_gain = meta.get("GAIN")
        if raw_gain is not None:
            try:
                if float(raw_gain) > 0:
                    gain = float(raw_gain)
            except (TypeError, ValueError):
                pass

        if not (exptime > 0 and bgsigma > 0 and fwhm > 0 and np.isfinite(magzero)):
            raise ExposureModelError(
                f"non-physical reference conditions: EXPTIME={exptime}, "
                f"BGSIGMA={bgsigma}, FWHM={fwhm}, MAGZERO={magzero}"
            )

        # Background variance in electrons, minus the (exposure-independent)
        # readout term, per second.
        sky_e_per_s = ((gain * bgsigma) ** 2 - readout_noise_e ** 2) / exptime
        if sky_e_per_s <= 0:
            raise ExposureModelError(
                f"reference background (BGSIGMA={bgsigma:.3g} ADU, GAIN={gain:.3g}) is at or "
                f"below the {readout_noise_e:.3g}e- readout floor -- the sky term is "
                f"non-physical, so exposure scaling is undefined. Typical of a "
                f"co-added/rescaled frame rather than a single raw exposure"
            )

        maglim = None
        try:
            if meta.get("MAGLIM") is not None:
                maglim = float(meta["MAGLIM"])
        except (TypeError, ValueError):
            pass

        return cls(
            exptime_s=exptime, bgsigma_adu=bgsigma, magzero=magzero, fwhm_px=fwhm,
            gain=gain, readout_noise_e=readout_noise_e, sky_e_per_s=sky_e_per_s,
            source=source, maglim=maglim,
        )

    def bgsigma_at(self, exptime_s: float) -> float:
        """Background sigma (ADU) at another exposure time. Returns the
        reference BGSIGMA exactly at `exptime_s == self.exptime_s`."""
        variance_e = self.sky_e_per_s * exptime_s + self.readout_noise_e ** 2
        return float(np.sqrt(variance_e) / self.gain)

    def magzero_at(self, exptime_s: float) -> float:
        return float(self.magzero + 2.5 * np.log10(exptime_s / self.exptime_s))

    @property
    def magzero_1s(self) -> float:
        return self.magzero_at(1.0)

    def as_dict(self) -> Dict[str, Any]:
        return {
            "source": self.source,
            "exptime_s": self.exptime_s,
            "bgsigma_adu": self.bgsigma_adu,
            "magzero": self.magzero,
            "magzero_1s": self.magzero_1s,
            "fwhm_px": self.fwhm_px,
            "gain": self.gain,
            "readout_noise_e": self.readout_noise_e,
            "sky_e_per_s": self.sky_e_per_s,
            "maglim": self.maglim,
        }


def predict_magerror(mag: float, exptime_s: float, conditions: ReferenceConditions) -> float:
    """Magnitude error a source of magnitude `mag` would be measured with in
    an `exptime_s` exposure under `conditions`."""
    bgsigma = conditions.bgsigma_at(exptime_s)
    magzero = conditions.magzero_at(exptime_s)
    inst = instrumental_magnitude(mag, magzero)
    return float(10 ** log_magerror(inst, bgsigma, conditions.fwhm_px, conditions.gain))


def required_exptime(mag: float, target_magerror: float, conditions: ReferenceConditions,
                     min_exptime_s: float = 1.0,
                     max_exptime_s: float = 3600.0) -> Optional[float]:
    """Shortest exposure reaching `target_magerror` on a source of magnitude
    `mag`, or `None` if `max_exptime_s` isn't enough.

    Predicted error falls monotonically with exposure time, so this brackets
    and bisects rather than using an unbracketed root finder -- the original
    script's `fsolve` could and did land on physically meaningless
    sub-millisecond solutions.
    """
    if min_exptime_s <= 0 or max_exptime_s <= min_exptime_s:
        raise ExposureModelError(
            f"invalid exposure bracket: [{min_exptime_s}, {max_exptime_s}]"
        )
    if not np.isfinite(mag) or not (target_magerror > 0):
        return None

    target_log = np.log10(target_magerror)

    def residual(log_exptime: float) -> float:
        return float(np.log10(predict_magerror(mag, 10 ** log_exptime, conditions)) - target_log)

    lo, hi = np.log10(min_exptime_s), np.log10(max_exptime_s)
    if residual(lo) <= 0:
        return float(min_exptime_s)
    if residual(hi) > 0:
        return None
    return float(10 ** brentq(residual, lo, hi, xtol=1e-4))


def limiting_magnitude(exptime_s: float, conditions: ReferenceConditions,
                       target_magerror: float, mag_lo: float = 5.0,
                       mag_hi: float = 30.0) -> Optional[float]:
    """Faintest magnitude measurable to `target_magerror` in `exptime_s` --
    the same model read the other way round, for reporting context."""
    target_log = np.log10(target_magerror)

    def residual(mag: float) -> float:
        return float(np.log10(predict_magerror(mag, exptime_s, conditions)) - target_log)

    if residual(mag_lo) > 0 or residual(mag_hi) < 0:
        return None
    return float(brentq(residual, mag_lo, mag_hi, xtol=1e-3))
