"""Provides a tool to break down large arrays found during iteration.

During iteration a chunk of coordinates may be served which is too large
for some downstream analysis. This objects in this module allow you to break 
arrays down into smaller arrays governed by some maximum size. Importantly, 
data is never fused _between_ arrays from iterates, so if two frames are neighbors
in the served arrays, they were also neighbors in the source data.

Functionality is provided in Breaker.
"""

from typing import (
    Iterable,
    Iterator,
    Collection,
    Optional,
    TypeVar,
)
from itertools import repeat
from ..util import chunker, safezip

It = TypeVar("It", bound=Iterable)


class Breaker(Iterable[It]):
    """Break down Tuples of Collections which exceed a given size.

    Striding can also be done simultaneously. This can be very useful
    for dealing with large h5 datasets.

    When the individual arrays served by SchemaH5 are too large, this
    class can be used to avoid reading too much data at once. For example,
        source: Final = s.SchemaH5(H5_DATA_PATH,
                                   schema=[COORD_LABEL,FORCE_LABEL],
                                   transform=lambda x: x)
        safe_source = s.Breaker(source=source, chunk_size=MAX_AA_CHUNK_SIZE)
    makes sure that the arrays that would be served from source upon iteration
    are broken into chunks no larger than MAX_AA_CHUNK_SIZE along their leading
    dimension. Note the transform=lambda x: x option--- it is important, as
    the default transform will read all data into memory, making this class
    useless. Iterating over safe_source provides the smaller chunks, each of
    which is a numpy; iterating over source with this transform option will
    serve h5py.Datasets.

    Stride is provided as an option here, as striding after chunking is not
    the same as striding before chunking. This object strides in a smart manner,
    returning chunks with frames selected as though you had strided before chunking.
    """

    def __init__(
        self,
        source: Iterable[It],
        chunk_size: Optional[int] = None,
        stride: int = 1,
        mask: Optional[Collection[bool]] = None,
    ) -> None:
        """Initialize.

        Arguments:
        ---------
        source:
            Iterable of AtomicData objects to pull from.
        chunk_size:
            Maximum size of chunk to return. Smaller chunks may be returned
            and the end of input blocks.
        stride:
            Amount to stride data by. Note that the frames are returned as though
            the stride was applied before chunking.
        mask:
            If not None, then the boolean entries in mask determine which
            elements in incoming tuples should have the chunking logic implied. True
            implies an entry should be chunked, False implies it should not. If not
            chunked, that element is repeated for each chunk served. None corresponds
            to a mask of all True.

        """
        self.source = source
        self.chunk_size = chunk_size
        self.stride = stride
        self.mask = mask

    def __iter__(self) -> Iterator[It]:
        """Leave it."""
        # each pull is a sequence of values.
        for pull in self.source:
            if self.mask is None:
                # apply chunker to everything in this case
                iterables = [
                    chunker(x, size=self.chunk_size, stride=self.stride) for x in pull
                ]
            else:
                # apply chunker to thing marked with true in mask
                iterables = [
                    chunker(x, size=self.chunk_size, stride=self.stride)
                    if y
                    else repeat(x)
                    for x, y in zip(pull, self.mask)
                ]

            # iterated along chunker output
            for chunk in safezip(*iterables, mask=self.mask):
                yield chunk  # type: ignore   # mypy doesn't understand safezip
