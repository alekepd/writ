"""Iterable interfaces for writing array data to files.

Only 1 function is currently implemented; it facilitates writing to h5 files.
"""
# __init__ doesn't use the imported objects
# ruff: noqa: F401

# in case h5py is not installed
try:
    from .batchh5 import batched_h5_save
except ModuleNotFoundError:
    pass
