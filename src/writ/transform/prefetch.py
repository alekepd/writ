"""Transform for batching data from an iterable.

This module allows one to serve batches of of an source iterable. Suppose you
have the list [1,2,3,4,5,6,7] and you want to iterate over it in chunks of 
(max) three to get [1,2,3], [4,5,6], [7]. This is straightforward with a list,
but if one wants to serve generators that chunk a source generator, this is harder:
as the served generators are evaluated after being returned, we do not know
if they are empty. This module fixes this and the solution is implemented
in the function lazy_batch.

To accomplish this, we need  to be sure that a served iterable has a certain
number of elements without fully iterating through it (in this case, we need
to be sure that it has at least one element). This is done through the Prefetch
class.
"""
from typing import (
    Iterable,
    Iterator,
    Generic,
    TypeVar,
)
from itertools import chain, islice

A = TypeVar("A")


class Prefetch(Generic[A]):
    """Iterable that prefetches from a source upon initialization.

    This object creates an iterable over a source iterable. Upon initialization,
    it attempts to draw a certain number of items from the source. When iterated
    over, these prefetched items are first served, and then iteration over
    the source proceeds as normal. If sufficient items cannot be prefetched,
    a ValueError is raised.

    Note that prefetching depletes the derived iterator. As a result, if prefetch
    fails, the input may be partially consumed. This case is not relevant when
    fetch = 1.

    Attributes:
    ----------
    source:
        Iterator to pull from.
    fetch:
        Number of items to cache.
    used:
        Whether the object has been iterated over yet.

    Notes:
    -----
    One very useful application of this class is batching (see lazy_batch). It is
    straightforward to break a source iterable into an iterable of lists of lengths `n`,
    but if we want to do the same task serving batches as Generators instead of lists,
    we do not know if we will serve empty Generators (this is not evident until the
    served generator is used). Using this class with fetch=1 will ensure that the served
    generator has at least one value.

    """

    def __init__(self, source: Iterable[A], fetch: int = 1) -> None:
        """Initialize.

        Arguments:
        ---------
        source:
            Iterable to draw from.
        fetch:
            Number of samples to prefetch. If the iterable does not have this many
            samples in it, a ValueError is raised.

        """
        self.source = iter(source)
        self.saved = []
        self.used = False
        if fetch < 1:
            raise ValueError("fetch must be at least 1.")
        for _ in range(fetch):
            try:
                self.saved.append(next(self.source))
            except StopIteration as e:
                raise ValueError("Prefetch failed.") from e

    def __iter__(self) -> Iterator[A]:
        """Serve saved values, then iterate."""
        for pull in chain(self._get_and_discard(), self.source):
            yield pull
        self.used = True

    def _get_and_discard(self) -> Iterator[A]:
        """Take items from beginning saved list and remove them.

        This is implementing a fifo queue.
        """
        while True:
            if len(self.saved) > 0:
                yield self.saved.pop(0)
            else:
                return


def lazy_batched(
    target: Iterable[A],
    size: int,
) -> Iterator[Prefetch[A]]:
    """Return generator that returns iterables containing batches from an iterator.

    For example, [x for x in batched([1,2,3,4,5],2)] = [[1,2],[3,4],[5]], but with
    iterable objects.

    Warning:
    -------
    This function will not serve a batch iterator if the previous one in the iteration
    is not yet consumed. Doing so is rather ill posed, as the iterators themselves
    require the previous one in the sequence to have completed. Attempting to do so
    will raise a ValueError.

    Arguments:
    ---------
    target:
        iterator we take batches from.
    size:
        size of batches. Note that the last batch may be smaller.

    Returns:
    -------
    Iterator returning batches.

    """
    if size < 1:
        raise ValueError("size must be at least one.")
    it = iter(target)
    first_iteration = True
    while batch := islice(it, size):
        if first_iteration:
            try:
                g = Prefetch(batch)
                yield g
            except ValueError:
                return
        else:
            if g.used is False:
                raise ValueError(
                    "Will not serve next batch until previous batch is done."
                )
            try:
                g = Prefetch(batch)
                yield g
            except ValueError:
                return
        first_iteration = False
