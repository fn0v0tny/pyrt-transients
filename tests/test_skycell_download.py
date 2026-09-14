"""The get_skycells replacement from _patch_get_skycells must close each
downloaded HDU list even when reading it fails: the fitsio fallback can
raise, and the Legacy Survey invvar mask sits inside a bare except.
stdpipe's download and fitsio are faked, so nothing touches the network."""
import sys
import types

import numpy as np
import pytest
from astropy.io import fits

stdpipe_templates = pytest.importorskip("stdpipe.templates")
from pyrt_transient.detection.subtraction import templates as pyrt_templates  # noqa: E402


class _Download:
    """What fits_open_remote returns: indexable, with filename() and close()."""

    def __init__(self, data=None, error=None):
        self.data, self.error, self.closed = data, error, False

    def filename(self):
        return "/nonexistent/cell.fits"

    def __getitem__(self, ext):
        download = self

        class _Extension:
            header = fits.Header()

            @property
            def data(self):
                if download.error:
                    raise download.error
                return download.data

        return _Extension()

    def close(self):
        self.closed = True


@pytest.fixture
def get_skycells(monkeypatch):
    """install(cell_url, *downloads) -> the patched get_skycells."""
    def unreadable(path, ext=None):
        raise OSError("fitsio cannot read it either")

    monkeypatch.setitem(sys.modules, "fitsio", types.SimpleNamespace(read=unreadable))
    monkeypatch.setattr(pyrt_templates, "_get_skycells_patch_applied", False)
    monkeypatch.setattr(stdpipe_templates, "get_skycells", stdpipe_templates.get_skycells)

    def install(cell_url, *downloads):
        queue = list(downloads)
        monkeypatch.setattr(stdpipe_templates, "find_skycells", lambda *a, **k: [cell_url])
        monkeypatch.setattr(stdpipe_templates, "fits_open_remote", lambda url: queue.pop(0))
        pyrt_templates._patch_get_skycells()
        return stdpipe_templates.get_skycells

    return install


def test_the_download_is_closed_when_the_fitsio_fallback_fails(get_skycells, tmp_path):
    download = _Download(error=ValueError("cannot convert float NaN to integer"))
    fetch = get_skycells("https://ps1/rings.v3.skycell.2011.080.stk.r.unconv.fits", download)

    with pytest.raises(OSError):
        fetch(10.0, 20.0, 0.1, _cachedir=str(tmp_path))

    assert download.closed


def test_the_invvar_download_is_closed_when_masking_fails(get_skycells, tmp_path):
    image = _Download(data=np.ones((4, 4), dtype=np.float32))
    invvar = _Download(data=np.zeros((3, 3)))   # wrong shape: the mask raises
    fetch = get_skycells("https://ls/legacysurvey-0001m002-image-r.fits.fz", image, invvar)

    files = fetch(10.0, 20.0, 0.1, survey="ls", _cachedir=str(tmp_path))

    assert invvar.closed and image.closed
    assert len(files) == 1
