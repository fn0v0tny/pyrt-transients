"""Compatibility shim.

This module used to be a standalone transient-search CLI inherited from
pyrt. Everything in it except two functions was dead and broken
(`process_single_image` called a two-argument method with three,
`match_and_filter_detections` referenced an undefined name). The two live
functions now live where they belong:

- `open_ecsv_file`      -> pyrt_transient.io.ecsv
- `simple_color_model`  -> pyrt_transient.core.color_model

Import them from there; this module only re-exports them so older call
sites keep working.
"""

from pyrt_transient.core.color_model import fotfit, simple_color_model  # noqa: F401
from pyrt_transient.io.ecsv import open_ecsv_file  # noqa: F401

__all__ = ["open_ecsv_file", "simple_color_model"]
