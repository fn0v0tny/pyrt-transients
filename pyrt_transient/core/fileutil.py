"""File-hashing helpers, copied from pipeline_magic.py."""

import hashlib
from pathlib import Path


def compute_file_md5(path, block_size=65536):
    """Compute MD5 checksum of a file, returns hex string or None if missing."""
    p = Path(path)
    if not p.exists() or not p.is_file():
        return None
    md5 = hashlib.md5()
    with open(p, 'rb') as f:
        for chunk in iter(lambda: f.read(block_size), b""):
            md5.update(chunk)
    return md5.hexdigest()
