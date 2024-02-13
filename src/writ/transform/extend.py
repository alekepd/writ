"""Provides a way to add values to iterates from a given source.

New values are produced via callable or a dictionary, where the new value is
produced by supplying information from the current iterate to the callable or
mapping.

As a example, suppose your source iterate (`s`) returns pairs as so: (`data`,
`name`), where `name` is some unique identifier for that particular array.
Now suppose that you have analyzed this dataset before, and you have a dictionary
(`d`) that is of the form `{name:new_value}`, where `new_value` represents some 
pre-computed quantity. If you call extend `s` as so:
    `extended = Extend(s,extender=d,input_index=-1,remove_input=True)`
then iterating over extended will return pairs of (`data`,`new_value`), where 
`new_value` is chosen to match `name` for that particular iterate.

This module has more capabilities. You are not limited to mappings, but also may
use callables, which allows any form of computation to be used. The exact information
passed to the processing function may also be tuned.

Functionality is provided via Extend.
"""
from typing import (
    Iterable,
    Iterator,
    Generic,
    TypeVar,
    Callable,
    Union,
    Tuple,
    Sequence,
    Hashable,
)
from collections import Mapping
from ..util import tuple_remove, tuple_insert

A = TypeVar("A")
A0 = TypeVar("A0")


def _callable_mapping(mapping: Mapping[A, A0]) -> Callable[[A], A0]:
    """Make a function which looks up values in a Mapping.

    This allows Mappings to be called as functions.

    Arguments:
    ---------
    mapping:
        Mapping object to wrap.

    Returns:
    -------
    A function that evaluates the mapping via indexing.

    """

    def _eval(argument: A, /) -> A0:
        return mapping[argument]

    return _eval


class Extender(Generic[A, A0]):
    """Evaluates and adds a value to an source iterable via a callable or mapping.

    Attributes can be changed during runtime to alter behavior.

    Attributes:
    ----------
    source:
        Iterator to pull from.
    extender:
        Callable that is used to generate new values. If a mapping is passed at
        initialization, it is turned into a callable.
    input_index:
        The index (slice or integer) of the value taken from the iterate to be served
        to extender. May also be None, in which case the entire iterate is sent.
    remove_input:
        Whether the value used for input should be removed when serving the new
        iterate. Note that for this to be true, input_index must be an integer.
        If this condition is violated, a ValueError is raised.
    placement_index:
        Indexing the tuple served via iteration by this value will return the added
        content.

    """

    def __init__(
        self,
        source: Iterable[Sequence[A]],
        extender: Union[
            Mapping[Hashable, A0],
            Callable[..., A0],
        ],
        input_index: Union[None, int, slice] = None,
        remove_input: bool = False,
        placement_index: int = -1,
    ) -> None:
        """Initialize.

        Arguments:
        ---------
        source:
            Iterable to draw values from.
        extender:
            Used to produce new information added to the input iterates. See class
            documentation.
        input_index:
            Specifies what part of the input tuple should be give as input to
            extender. If None, the entire iterate is served. If an integer,
            see remove_input. May also be a slice.
        remove_input:
            If input index is an integer, setting this to True will remove the item
            indexed by that integer after it is used as input (i.e., it makes it as
            though it is consumed by this transformation). This cannot be true
            if input_index is not an integer.
        placement_index:
            Where to place the generated value in the tuple: the value is found
            where this index points on the produced tuple.

        """
        self.source = source
        self.input_index = input_index
        self.remove_input = remove_input
        self.placement_index = placement_index

        if isinstance(extender, Mapping):
            self.extender = _callable_mapping(extender)
        else:
            self.extender = extender

    def __iter__(self) -> Iterator[Tuple[Union[A, A0], ...]]:
        """Pull value, generate new value, edit, and add."""
        for pull in self.source:
            if self.input_index is None:
                to_pass: Union[A, Sequence[A]] = pull
            else:
                to_pass = pull[self.input_index]
            new = self.extender(to_pass)
            if self.remove_input:
                # we only enter here is the type is okay.
                pull = tuple_remove(pull, self.input_index)  # type: ignore
            yield tuple_insert(pull, new, self.placement_index)

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
