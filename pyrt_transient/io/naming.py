"""Filename-derivation helper."""

import os
import re

from astropy.table import Table


def get_base_filename(det_table: Table, epoch_index: int) -> str:
    """Get base filename with improved fallback logic."""
    filename = None
    for key in ['filename', 'FITSFILE', 'source_file', 'FILENAME']:
        if key in det_table.meta and det_table.meta[key]:
            filename = det_table.meta[key]
            break

    if filename is None:
        filename = f'epoch_{epoch_index}'

    # Clean filename
    filename = os.path.basename(str(filename))
    filename = os.path.splitext(filename)[0]

    # Remove problematic characters
    filename = re.sub(r'[^a-zA-Z0-9_\-]', '_', filename)
    filename = re.sub(r'_+', '_', filename).strip('_')

    return filename if filename else f'epoch_{epoch_index}'
