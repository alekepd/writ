# (w)rit

A package to write and read array data to and from disk using an iterable interface.

## Warning

This package is experimental. Interfaces are subject to change and tests are
far from complete. Use at your own risk.

## Installation

Install the `writ` package from source by calling `pip install .`, repository's
root directory. 

Note that individual readers or writers may require other libraries (e.g., 
`h5py`).

## Reading and Writing

Interfaces are given for reading data, and interfaces for writing data are
planned. Note that while the modules are written to clearly state requirements,
these tools are developed for file formats as they are needed.

### Example usage

See the individual modules for defiled information on usage. In general, the objects
in `writ.read` allow one to iterate over either collections of files on disk or iterate
through `h5` files. `SchemaH5` does the latter:

```python
from writ.read import SchemaH5
s = SchemaH5("data/aaqaa_stride_10000_9ramp.h5py",schema=['coords'])
for x in s:
    # iteration goes over all possible 'coords' arrays in s
    print(x.shape)
```
