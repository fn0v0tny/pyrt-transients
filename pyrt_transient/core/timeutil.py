"""Time-conversion helpers."""

from astropy.time import Time


def unix_to_mjd(unix_time):
    """Convert Unix timestamp to Modified Julian Date."""
    try:
        t = Time(unix_time, format='unix')
        return t.mjd
    except:
        return unix_time
