"""Iterable interface for reading array chunks which contain multiple replicas of data.

Other readers typically serve each chunk found on disk or in a database. Here,
we instead read slices from all found chunks, concatenate them, and serve it.
All files are read repeatedly as various slices are served. At no point is the
entire data set held in memory.

Functionality is provided by StripedChunks.
"""

from typing import Iterator, Iterable, List, Callable, cast
from glob import glob
import numpy as np


class StripedChunks(Iterable[np.ndarray]):
    """Serve array data from files that have multiple lines of data.

    Each file is expected to have the coordinates of multiple replicas (e.g.,
    trajectories). For example, if the following files are present:
        tr_0.npy tr_1.npy tr2.npy
    each might be of the shape (5,2,10,3). We assume that the first shape index (here 5)
    specifies an individual "run" of data: the file includes 5 datasets, each of shape
    (2,10,3). During the first iteration, we first read the slice [0,:,:,] from each
    file. We then concatenate those two slices into an array of the shape (4,10,3), and
    serve it. During the second iteration we read the slice [1,:,:,:] from each file,
    concatenate them, and serve them. This is then continued for leading indices 2-4.

    Note that the internal concatenation is performed the second axis.

    Stride is implemented in a memory intensive way: the entire unstrided slice
    is created and then strided. This is more robust than striding chunks, but may cause
    memory problems.

    Note that we do not include an include_id option, as the served objects do not
    correspond to individual files.
    """

    def __init__(
        self,
        pattern: str,
        stride: int = 1,
        loader: Callable[[str], np.ndarray] = lambda x: cast(np.ndarray, np.load(x)),
    ) -> None:
        """Initialize.

        Arguments:
        ---------
        pattern:
            Wildcard pattern used to look for files via blob (likely contains *).
        stride:
            Amount by which to stride the data.
        loader:
            Callable which takes a filename as input and returns a numpy ndarray.

        """
        self.pattern = pattern
        self.stride = stride
        self.loader = loader

    def __iter__(self) -> Iterator[np.ndarray]:
        """At each iteration we iterated over files, slice, concatenate, and serve."""
        filenames = self._get_filenames()
        if len(filenames) == 0:
            return
        n_replicas = self.loader(filenames[0]).shape[0]

        for rep in range(n_replicas):
            # without .copy() this will hold the entire numpy chunk in memory via a view
            ts = [self.loader(fn)[rep].copy() for fn in filenames]
            yield np.array(np.concatenate(ts)[:: self.stride])

    def _get_filenames(self) -> List[str]:
        """Return list of filenames to load."""
        return sorted(glob(self.pattern))
