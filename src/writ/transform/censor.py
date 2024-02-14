"""Provides a way to censor parts of iterates that fail a test.

Censoring is applied via slices; in the case of multidimensional arrays, this
means it is applied along the left-most axis.  When removing frames, the input
iterate's content is broken into parts so that no served content contains
adjacent entries that were not adjacent in the source.

For example:
```
data = ([(np.array([0.9, 0.9, 0.6, 0.3, 0.9, 0.1]),),
         (np.array([0.3, 0.7, 0.8, 0.6, 0.6, 0.6]),)])

list(Censor(source=data,censor=lambda x: x[0]<0.7))
#[(array([0.6, 0.3]),),
# (array([0.1]),),
# (array([0.3]),),
# (array([0.6, 0.6, 0.6]),)]
```
Note that `0.1` from the first source chunk and `0.3` from the second source 
chunk were both preserved, but were not served in a iterate together as they were
not in the same source iterate.

Functionality is provided via the Censor class.
"""
from typing import (
    Iterable,
    Iterator,
    Generic,
    TypeVar,
    Sequence,
    Collection,
    List,
    Protocol,
    Callable,
    Union,
    Tuple,
    cast,
)
import numpy as np
from ..util import tuple_remove, safezip, indices_to_mask

A = TypeVar("A")


def _mask_to_slices(mask: Sequence[bool]) -> List[slice]:
    """Transform sequence of booleans into slices reflecting True entries.

    For example:

    ```
    g=[True,False,True,True,True,False,True,False]
    h=_mask_to_slice(g)
    for x in h:
        print(g[x])
    # [True], [True, True, True,], [True]
    ```

    Arguments:
    ---------
    mask:
        Collection of booleans to be analyzed.

    Returns:
    -------
    A list of slice objects. If the input object is sliced using each
    entry, all the True elements are returned in contiguous chunks.

    """
    if len(mask) == 0:
        return []
    else:
        start_type = mask[0]
        size = len(mask)
        # we first find each place in the boolean array where we switch from true to
        # false or vice versa--- this is in events.
        differences = np.diff(np.array(mask))
        events = cast(List[int], [x + 1 for x in np.nonzero(differences)[0]])
        # We now modify the events list to start with the beginning of a True region.
        # if the first element is True, we add a entry for the beginning of the array.
        if start_type:
            aligned_events = [0, *events]
        else:
            aligned_events = events
        source = iter(aligned_events)
        intervals = []
        # we now group through pairs of the change events. The first marks the start
        # of True region, the second marks the end of one. If we cannot find a second
        # event in a pair, that means the mask ended true.
        while True:
            try:
                start = next(source)
            except StopIteration:
                # no more intervals exist
                break
            try:
                end = next(source)
            except StopIteration:
                end = size
            intervals.append(slice(start, end))
        return intervals


class SupportsSlice(Protocol):
    """Class that supports type preserving slicing."""

    def __getitem__(self: A, key: slice) -> A:
        """Type preserving slicing."""
        ...


Sl = TypeVar("Sl", bound=SupportsSlice)


def _mask_ibreak(sliceable: Sl, mask: Sequence[bool]) -> Iterator[Sl]:
    """Serve slices of a source object that capture values given by a mask.

    ```
    For example:
    m = [True, False, True, True, True, False]
    d = [0, 1, 2, 3, 4, 5]
    list(_mask_ibreak(m,d))
    # [0], [2, 3, 4]
    ```

    Arguments:
    ---------
    sliceable:
        Object that is sliced during seriving.
    mask:
        Collection of boolean values that indicates which elements to serve.

    Returns:
    -------
    Generator that serves slices of sliceable.

    Notes:
    -----
    Type hinting here assumes that slicing preserves type. This is almost always true,
    but sometimes things like h5py.Dataset may subtlety change types under this
    operation.

    """
    slices = _mask_to_slices(mask)
    for s in slices:
        yield sliceable[s]


class Censor(Generic[Sl]):
    """Removes entries from individual chunks while fail a test.

    This class does _not_ drop entire iterates which fail a test. Instead, it filters
    out individual parts of iterates that fail a vectorized test. These parts are the
    slices found by indexing; in the case of ndarrays, this is indexing along the
    leading (left-most) axis.

    Note:
    ----
    This object assumes that inputs are tuples and that censoring is applied to
    specified element(s) of this tuple.

    Example:
    -------
    If my input iterates arrays of floats, I can filter along their leading
    axis for "frames" that are larger than `0.7` as follows:
    ```
    data = ([(np.array([0.9, 0.9, 0.6, 0.3, 0.9, 0.1]),),
             (np.array([0.3, 0.7, 0.8, 0.6, 0.6, 0.6]),)])

    list(Censor(source=data,censor=lambda x: x[0]<0.7))
    #[(array([0.6, 0.3]),),
    # (array([0.1]),),
    # (array([0.3]),),
    # (array([0.6, 0.6, 0.6]),)]
    ```
    Note that the input iterates are broken into parts that preserve only the values
    passing the test given in the censor argument; as a result, if two entries are
    adjacent in a served iterate, they were also adjacent in the original data.

    Attributes:
    ----------
    source:
        Source iterable to draw iterates from.
    censor:
        Callable that generates a boolean collection when given input.
        Input to this callable may either be a subset of or individuals in
        the values present in each pull of the source iterate; see
        input_index. Produced collection should be the same length as the
        objects that are censored; see apply_to.
    input_index:
        Used to prepare input to censor. If None, the entire iterate is passed;
        else it is used to index the incoming iterate.
    discard_input:
        Whether to remove the object that was used as input when serving the result.
        This may only be set to true when input_index is an integer.
    apply_to:
        Which entries in each pull to apply the indexing to. This is done after
        possible application of discard_input. Note that if an entry is not
        specified, it is repeated when the remaining entries are chunked and
        served as smaller iterates.

    These attributes may be modified at runtime to change behavior.

    """

    def __init__(
        self,
        source: Iterable[Sequence[Sl]],
        censor: Callable[..., Sequence[bool]],
        input_index: Union[None, int, slice] = None,
        discard_input: bool = False,
        apply_to: Union[None, Collection[int], Collection[bool]] = None,
    ) -> None:
        """Initialize.

        Arguments:
        ---------
        source:
            Source iterable to draw iterates from.
        censor:
            Callable that generates a boolean collection when given input.
            Input to this callable may either be a subset of or individuals in
            the values present in each pull of the source iterate; see
            input_index. Produced coollection should be the same length as the
            objects that are censored; see apply_to.
        input_index:
            Used to prepare input to censor. If None, the entire iterate is passed;
            else it is used to index the incoming iterate.
        discard_input:
            Whether to remove the object that was used as input when serving the result.
            This may only be set to true when input_index is an integer.
        apply_to:
            Which entries in each pull to apply the indexing to. This is done after
            possible application of discard_input. Note that if an entry is not
            specified, it is repeated when the remaining entries are chunked and
            served as smaller iterates.

        """
        self.source = source
        self.censor = censor
        self.input_index = input_index
        self.apply_to = apply_to
        self.discard_input = discard_input

    def __iter__(self) -> Iterator[Tuple[Sl, ...]]:
        """Pull value, generate mask, subset, and serve."""
        pull: Union[Sequence[Sl], Tuple[Sl, ...]]
        for pull in self.source:
            # get the boolean mask
            if self.input_index is None:
                mask = self.censor(pull)
            else:
                mask = self.censor(pull[self.input_index])
            if self.discard_input:
                # only entered if type is okay
                pull = tuple_remove(pull, self.input_index)  # type: ignore
            # this mask determines which items in the pull should be sampled.
            to_include_mask = indices_to_mask(self.apply_to, len(pull))
            # if we would be serving 0 arrays, then don't serve at all
            for s in _mask_to_slices(mask):
                derived = [
                    values[s] if included else values  # type: ignore
                    for values, included in safezip(pull, to_include_mask)
                ]
                yield tuple(derived)

    @property
    def remove_input(self) -> bool:
        """remove_input can only be true if input_index is an integer."""
        return self._remove_input

    @remove_input.setter
    def remove_input(self, value: bool) -> None:
        """remove_input can only be true if input_index is an integer."""
        if value is True and not isinstance(self.input_index, int):
            raise ValueError(
                "remove_input cannot be True if input_index is not an integer."
            )
        self._remove_input = value

    @remove_input.deleter
    def remove_input(self) -> None:
        del self._remove_input
