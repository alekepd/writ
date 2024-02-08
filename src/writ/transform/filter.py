"""Provides a transform that skips over iterates failing a test.

The Filter class provides this functionality.
"""
from typing import (
    Iterable,
    Iterator,
    Callable,
    TypeVar,
)

A = TypeVar("A")


class Filter(Iterable[A]):
    """Provides an iterable that filters out source iterates that fail a test.

    Suppose you have the list [1,5,3,7,9] and you wish to produce an iterable that
    only serves the numbers below 4; that is, iterating over the result would give
    [1,3]. This can be done with the Filter class via Filter(<source>,test=lambda
    x: x<4). More generally, Filter allows an arbitrary test to be evaluated.
    """

    def __init__(
        self,
        source: Iterable[A],
        test: Callable[[A], bool],
        **kwargs,
    ) -> None:
        """Initialize.

        Arguments:
        ---------
        source:
            Iterable of objects to iterate over
        test:
            Function which we apply to each item drawn from source. If it
            returns a truthy value, we yield that item; if not, we skip it.
        **kwargs:
            Stored and passed to test at each call.

        """
        self.source = source
        self.test = test
        self.aux_args = kwargs

    def __iter__(self) -> Iterator[A]:
        """Iterate over input, only returning items which pass the test."""
        for pull in self.source:
            if self.test(pull, **self.aux_args):
                yield pull
