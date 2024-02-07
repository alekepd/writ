"""Basic tools used elsewhere in the package.

There are many modifiers here. Some are 
"""
from typing import (
    Any,
    Tuple,
    List,
    Sequence,
    Iterable,
    Iterator,
    Collection,
    Mapping,
    Hashable,
    Union,
    Optional,
    TypeVar,
)
from itertools import repeat, count
import re
import numpy as np

A = TypeVar("A")
A0 = TypeVar("A0")
T = TypeVar("T", bound=Hashable)
E = TypeVar("E", bound=Sequence)


def safezip(
    *args: Iterable[A], mask: Optional[Collection[bool]] = None
) -> Iterable[Tuple[A, ...]]:
    """Like zip, but raises error if iterators don't stop at the same time.

    There are standard alternatives to this in python 3.10, although they don't
    seem to support masks.

    Arguments:
    ---------
    *args:
        Iterables to treat like zip.
    mask:
        Mask determines which elements should be checked to obey these rules. Should
        have an boolean entry for each iterator in args (or be None). If the entry
        for an object is True, we check to see that it lasts as long as all of the other
        True-marked objects. If False, we do not. Note that if an object marked by
        False ends early, the iterator will also end early (like zip). None is
        considered to be a mask that is True for all objects.

    Returns:
    -------
    Iterator similar to that zip returns.

    Notes:
    -----
    It seems that the type hints here do not exactly match those of zip, but it's
    not clear how to fix this.

    """
    # holds the iterators from the iterables
    its = [iter(x) for x in args]
    if mask is None:
        mask = len(its) * [True]
    if len(mask) != len(its):
        raise ValueError(f"Invalid mask {mask}.")
    # we manually go through the iterators
    while True:
        # holds the results of the iteration
        results = []
        # holds whether an iterator was nexted or not
        successes = []
        # for each of the individual iterators
        for g in its:
            try:
                results.append(next(g))
                successes.append(True)
            except StopIteration:
                successes.append(False)
        # If we were able to take a value from all of the iterables, serve it
        if all(successes):
            yield tuple(results)
        # elif we were able to take a value from only some all the iterables,
        # get angry. Results from
        elif any(x if y else False for x, y in zip(successes, mask)):
            raise ValueError("Iterators did not end simultaneously.")
        else:
            # will give a stop iteration through the generator interface
            return


def tappend(tup: Iterable[A], value: A0) -> Tuple[Union[A, A0], ...]:
    """'Append' value to tuple.

    Tuples are immutable, so make a new tuple that adds the given value at the end.

    Arguments:
    ---------
    tup:
        Iterable (probably a tuple) to transform into a tuple with the same elements
        and "value" at the end.
    value:
        Value to append.

    Returns:
    -------
    Tuple containing the elements of tup and value as the last element.

    """
    l_form: List[Union[A, A0]] = list(tup)
    l_form.append(value)
    return tuple(l_form)


def chunker(
    data: Sequence[A],
    size: Optional[int],
    stride: int = 1,
) -> Iterator[Sequence[A]]:
    """Break a sequence into chunks via a generator, with striding.

    If you "concatenate" the output of this function, you obtain the data in
    source, with some caveats (see Notes). For example, in the case of
    ndarrays: g[::stride] == concatenate(chunker(g,size,stride=stride)) for
    _any_ positive size. This is possible as the slicing is not done blindly
    for each chunk individually.

    The specified size is the max size of returned chunks; smaller chunks
    may occur at the end of sequences.

    Arguments:
    ---------
    data:
        Sequence, likely a ndarray or h5.Dataset. Must be sliceable.
    size:
        Maximum size of chunks to return. None is an infinitely large max
        chunk size.
    stride:
        Amount to _effectively_ stride data by.

    Returns:
    -------
    A generator serving chunks of data, strided as appropriate.

    Notes:
    -----
    This function queries data via [a:b:c] slicing, and some objects (e.g.,
    h5py.Datasets) change types under this operation. The combined chunking
    and striding is in fact most useful for h5py.Datasets, as striding before
    chunking would cause the entire stride portion to be loaded.

    """
    length = len(data)
    # all the frame indices we will serve
    inds = range(0, length, stride)
    done = False
    if size is None:
        yield data[::stride]
        return
    else:
        # repeated try to serve chunks until we go out of bounds
        # on our indices.
        for chunk_index in count(start=0):
            start = inds[chunk_index * size]
            try:
                stop = inds[(chunk_index + 1) * size]
            except IndexError:
                stop = length
                done = True
            yield data[start:stop:stride]
            if done:
                return


def perframe(
    source: Iterable[Collection[Collection[A]]],
    mask: Optional[Collection[bool]] = None,
    limit: Optional[int] = None,
) -> Iterator[Tuple[A, ...]]:
    """Unchunked tuple iterator from an iterator returning collections of chunked data.

    This function returns a generator over an existing iterable that modifies
    how the data is iterated over. Consider the following target value and
    function call:
        target = [([1,2],['a','b']),
                  ([3,4],['c','d'])]
        list(perframe(target,mask=[True,True]))
        -> [(1, 'a'), (2, 'b'), (3, 'c'), (4, 'd')]
    each entry in target is a tuple. Each of these tuples has entries which are
    iterable. This function goes through and iterates over those iterable entries.

    Mask allows some of those entries to not be iterated over, but rather repeated
    for each served item:
        target = [([1,2],['a','b'],'##'),
                  ([3,4],['c','d'],'!!')]
        list(perframe(target,mask=[True,True,False]))
        ->[(1, 'a', '##'), (2, 'b', '##'), (3, 'c', '!!'), (4, 'd', '!!')]

    Arguments:
    ---------
    source:
        Iterable from which draw data.
    mask:
        An iterable of True and False values, which correspond to whether the
        corresponding entry in draws from source should be iterated over.
    limit:
        If not None, then maximum only this many values are served before
        iteration stops.

    Returns:
    -------
    Generator which returns tuples of values.

    Notes:
    -----
    This function can be used to change the default per-chunk iteration done in a
    SchemaH5 object into a per-frame iteration (perframe(SchemaH5(file,keys))).
    If include_id is specified, make sure the mask excludes the id variable.

    """
    count = 0
    for pull in source:
        if mask is None:
            iterables = pull
        else:
            # mypy is unhappy here, probably because it doesn't
            # understand _safezip types.
            iterables = [x if y else repeat(x) for x, y in safezip(pull, mask)]  # type: ignore
        for frame in safezip(*iterables, mask=mask):
            if limit is not None and count >= limit:
                return
            yield tuple(frame)
            count += 1


def tupleize(data: Mapping[T, A], keys: Iterable[T]) -> Tuple[A, ...]:
    """Transform mapping into tuple containing some of its values.

    The value and order are specified by listing them in keys. For example:
    data = {'a':1,'b':2,'c':3}, keys=['b','a'] returns (2,1).

    Arguments:
    ---------
    data:
        Mapping object we index via objects in keys.
    keys:
        Elements are used to index data.

    Returns:
    -------
    Tuple of values.

    """
    to_return = []
    for key in keys:
        to_return.append(data[key])
    return tuple(to_return)


def natural_sort(name: Union[Collection[str], str]) -> List[str]:
    """Perform natural sort on a string.

    Arguments:
    ---------
    name:
        string to sort

    Returns:
    -------
    A sorted version of the input, where order is determined by convert as applied to
    each chunk of name, chunks delimited via re ([0-9]+).

    """
    # it is unclear how this could ever encounter a digit based on the following
    # usage
    def convert(text: str) -> Union[int, str]:
        return int(text) if text.isdigit() else text.lower()

    def alphanum_key(key: str) -> List[Union[int, str]]:
        return [convert(c) for c in re.split("([0-9]+)", key)]

    return sorted(name, key=alphanum_key)


class TupleStrider:
    """Strides each object in an iterable and returns as tuple.

    When used in SchemaH5, this avoids reading more frames into memory than are needed.
    Note that the speed of this operation probably depends on the binary layout of the
    h5 analyzed.
    """

    def __init__(self, stride: int) -> None:
        """Set stride."""
        self.stride = stride

    def __call__(self, target: Iterable[E]) -> Tuple[Any, ...]:
        """Stride each entry in a iterable and return as tuple."""
        strided = []
        for value in target:
            strided.append(value[:: self.stride])
        return tuple(strided)


class TupleArraySampler:
    """Samples via np.array-indexing each array in an iterable.

    The sites specified by the random selection are shared between the
    different elements of the analyzed tuple.

    Sampling is done without replacement. If more samples are requested than are
    available, all samples are returned (this will likely be fewer than the requested
    number of samples).

    Unlike some other similar classes, this is only typed for np.ndarrays because of
    the indexing strategy.
    """

    def __init__(self, n_samples: int) -> None:
        """Set number of samples per iteration."""
        self.n_samples = n_samples
        self.rng = np.random.default_rng()

    def __call__(self, target: Iterable[np.ndarray]) -> Tuple[np.ndarray, ...]:
        """Sample each entry in a iterable and return as tuple."""
        strided = []
        # special behavior on first iteration.
        for order, value in enumerate(target):
            if order == 0:
                size = len(value)
                possibilities = np.arange(size)
                if size <= self.n_samples:
                    choices = possibilities
                else:
                    choices = np.sort(
                        self.rng.choice(possibilities, self.n_samples, replace=False)
                    )
            strided.append(value[choices])
        return tuple(strided)
