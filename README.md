# (w)rit

A package to write and read array data to and from disk using an iterable interface.

## Warning

This package is experimental. Interfaces are subject to change and tests are
far from complete. Use at your own risk.

## Installation

Install the `writ` package from source by calling `pip install .` inside the
repository's root directory. 

Note that individual readers or writers may require other libraries (e.g., 
`h5py`).

## Reading and Writing

Interfaces are given for reading data, and interfaces for writing data are
planned. Note that while the modules are written to clearly state requirements,
these tools are developed for file formats as they are needed.

### Example usage

See the individual modules for defiled information on usage. In general, the objects
in `writ.read` allow one to iterate over either collections of files on disk or iterate
through `h5` files. `SchemaH5` does the latter. For example, the following loop will
serve all of the `Dataset`s which are named `'coords'`.
```python
from writ.read import SchemaH5
filename = "data/aaqaa_stride_10000_9ramp.h5py"
s = SchemaH5(filename,schema=['coords'])
for x in s:
    # iteration goes over all possible 'coords' arrays in s
    print(x.shape)
```
In the case of `SchemaH5`, these interface can be used to access multiple fields
simultaneously. For example, if both `coords` and `Fs` are present, they can
be access as follows:
```python
from writ.read import SchemaH5
filename = "data/aaqaa_stride_10000_9ramp.h5py"
s = SchemaH5(filename,schema=['coords','Fs'])
for x,y in s:
    # x contains the "coords" entries, y the "Fs" entries
    print(x.shape)
    print(y.shape)
```

Other tools in `writ.read` provide a similar interface over other data formats. For example,
if molecule dynamics is used to simultaneously run `10` replicas of a system, the data could
be saved in arrays of shape `(10,n,...)`, where the outer index (`10`) indexes the replicas
of the system and the second index (`n`) indexes the timestep of the simulation. Each chunk
could correspond to a different range of timesteps. If these arrays are saved on disk in numpy 
`.npy` files, data of this format can be read as follows:
```python
from writ.read import StripedChunks
filename_glob = 'data/ca_aaqaa_coords_*.npy'
s = StripedChunks(filename_glob)
for x in s:
    # each x contains the continuous trajectory of a single replica.
    print(x.shape)
```
Unlike the case of `SchemaH5`, this class does support multiple fields, as the data format does
not allow it.

Data may also be stored on disk in multiple directories, where numpy files are present in 
both directories and paired together. Pairing is done by considering the pattern provided:
`{}` is treated like a wildcard, with its expanded value considered an ID. The ID of any pair
of files served matches.
```python
from writ.read import SepDirChunks 
coord_sk = 'data/cs/chig_coor_{}.npy'
force_sk = 'data/fs/chig_force_{}.npy'
s = SepDirChunks(patterns=[coord_sk,force_sk])
for x,y in s:
    # each x contains the coords of a chunk
    # each y contains the forces of a chunk
    print(x.shape)
    print(y.shape)
```

There is also code for basic analysis, such as time independent coordinate analysis (to see
this object, you need to have installed `deeptime`).
```python
from writ.transform.tic import TICWindow
from writ.read import StripedChunks
filename_glob = 'data/ca_aaqaa_coords_*.npy'
s = StripedChunks(filename_glob)
t = TICWindow(source=s)
# fit the tic transform on source. If you want to fit on a different source (maybe stride),
# pass it to this .fit(*) call.
t.fit() #
for x in tic:
    # each x is a trajectory mapped to tic space.
    print(x.shape)
```
