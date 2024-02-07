"""Iterable interfaces to reading and transforming data already saved on disk."""
# __init__ doesn't use the imported objects
# ruff: noqa: F401
from .stripedchunk import StripedChunks
from .sepdir import SepDirChunks

# in case h5py is not installed
try:
    from .schemah5 import SchemaH5
except ModuleNotFoundError:
    pass
