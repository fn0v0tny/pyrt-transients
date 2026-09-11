"""Time-conversion helpers."""

from astropy.time import Time


def unix_to_mjd(unix_time):
    """Convert Unix timestamp to Modified Julian Date; NaN if the input
    isn't a valid timestamp (previously the raw unix time was returned,
    i.e. a wrong number in MJD's place rather than a visibly missing one)."""
    try:
        return Time(unix_time, format='unix').mjd
    except (ValueError, TypeError):
        return float("nan")
