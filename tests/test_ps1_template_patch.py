"""Unit tests for detection/subtraction/templates.py's
_patch_normalize_ps1_skycell -- the fitsio-based workaround for a real bug
in some `stdpipe` builds' PS1 skycell normalization (see FUTURE_IDEAS.md's
"PS1 template retrieval bug found and fixed").

Covers the signature guard: the patch was written against an older stdpipe
`normalize_ps1_skycell(filename, outname=None, verbose=False)` that reopens
a file from disk. A newer stdpipe (verified against 0.4.1 from PyPI)
changed this to an in-memory `normalize_ps1_skycell(image, header,
verbose=False)` -- applying the old patch there doesn't restore the
original bug, it breaks every PS1 call outright. These tests fake both
signatures directly on `stdpipe.templates` (skipping real fitsio/stdpipe
FITS I/O, same boundary test_detection_subtraction.py draws elsewhere) to
verify the guard patches the old form and leaves the new one alone.

No pytest dependency -- run directly with
`python3 tests/test_ps1_template_patch.py`.
"""
import importlib.util
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import stdpipe.templates as stdpipe_templates

from pyrt_transient.detection.subtraction import templates as pyrt_templates


def _reset():
    pyrt_templates._ps1_patch_applied = False


def _have_fitsio():
    """The patched implementation reads skycells with fitsio, so without it
    _patch_normalize_ps1_skycell correctly declines to patch anything. fitsio
    is optional (it is absent from the pyrt311 dev env), so the test that
    asserts the patch IS installed only applies where it is present."""
    return importlib.util.find_spec("fitsio") is not None


def _old_signature(filename, outname=None, verbose=False):
    raise AssertionError("should have been replaced by the patch")


def _new_signature(image, header, verbose=False):
    return image, header


def test_patches_old_filename_based_signature():
    if not _have_fitsio():
        print("test_patches_old_filename_based_signature: SKIP (fitsio not installed)")
        return
    _reset()
    original = stdpipe_templates.normalize_ps1_skycell
    stdpipe_templates.normalize_ps1_skycell = _old_signature
    try:
        pyrt_templates._patch_normalize_ps1_skycell()
        assert stdpipe_templates.normalize_ps1_skycell is not _old_signature, \
            "old filename-based signature should have been patched"
    finally:
        stdpipe_templates.normalize_ps1_skycell = original
        _reset()
    print("test_patches_old_filename_based_signature: PASS")


def test_skips_newer_in_memory_signature():
    _reset()
    original = stdpipe_templates.normalize_ps1_skycell
    stdpipe_templates.normalize_ps1_skycell = _new_signature
    try:
        pyrt_templates._patch_normalize_ps1_skycell()
        assert stdpipe_templates.normalize_ps1_skycell is _new_signature, \
            "newer image/header-based signature must NOT be patched -- " \
            "the old patch would bind an in-memory array to its `filename` param"
    finally:
        stdpipe_templates.normalize_ps1_skycell = original
        _reset()
    print("test_skips_newer_in_memory_signature: PASS")


def test_only_inspects_signature_once_per_process():
    _reset()
    original = stdpipe_templates.normalize_ps1_skycell
    stdpipe_templates.normalize_ps1_skycell = _new_signature
    try:
        pyrt_templates._patch_normalize_ps1_skycell()
        assert pyrt_templates._ps1_patch_applied is True
        # A second call must be a no-op (early return), regardless of what
        # normalize_ps1_skycell looks like now.
        stdpipe_templates.normalize_ps1_skycell = _old_signature
        pyrt_templates._patch_normalize_ps1_skycell()
        assert stdpipe_templates.normalize_ps1_skycell is _old_signature
    finally:
        stdpipe_templates.normalize_ps1_skycell = original
        _reset()
    print("test_only_inspects_signature_once_per_process: PASS")


def test_missing_symbol_does_not_raise():
    """A stdpipe build without normalize_ps1_skycell at all: the patch is
    called from get_template OUTSIDE the try that degrades to "no template",
    so an AttributeError here would kill the whole detection run instead of
    just skipping the PS1 template."""
    _reset()
    original = stdpipe_templates.normalize_ps1_skycell
    del stdpipe_templates.normalize_ps1_skycell
    try:
        pyrt_templates._patch_normalize_ps1_skycell()   # must not raise
        assert not hasattr(stdpipe_templates, "normalize_ps1_skycell"), \
            "nothing to patch -- the symbol must not be invented"
        assert pyrt_templates._ps1_patch_applied is True, \
            "the decision is final for this process, as for a missing fitsio"
    finally:
        stdpipe_templates.normalize_ps1_skycell = original
        _reset()
    print("test_missing_symbol_does_not_raise: PASS")


if __name__ == "__main__":
    test_patches_old_filename_based_signature()
    test_skips_newer_in_memory_signature()
    test_only_inspects_signature_once_per_process()
    test_missing_symbol_does_not_raise()
    print("All detection/subtraction/templates.py PS1-patch tests passed.")
