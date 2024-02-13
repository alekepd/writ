"""Performs rejection sampling on iterates drawn from a source.

Look at `en.wikipedia.org/wiki/Rejection_sampling` for more information.

Functionality is provided via RSampler; note that this object performs
sampling on the slices of the input objects (i.e., it does not keep or drop
each iterate as a single item).
"""
from typing import (
    Iterable,
    Generic,
    TypeVar,
    Union,
    List,
    Collection,
    Optional,
    Sequence,
    Final,
    Hashable,
    Mapping,
    Dict,
    Callable,
    Protocol,
    Tuple,
)
from numpy.random import default_rng, Generator
from numpy import ndarray, finfo
from numbers import Real
from ..util import tuple_remove, safezip


def _indices_to_mask(indices: Optional[Collection[int]], max_size: int) -> List[bool]:
    """Create a list of booleans that indicate presence in given indices.

    For example, _indices_to_mask([1,2],5) gives [False, True, True, False, False].
    The length of the list is 5 due to the max_size argument.

    Arguments:
    ---------
    indices:
        Collection of integers denoting which entries should be "preserved"; may also be
        None, in which case all indices will be preserved.
    max_size:
        Maximum index to consider when making map. If smaller than values in indices,
        those values are effectively discarded.

    Returns:
    -------
    List of booleans, True if the corresponding
    """
    if indices is None:
        return [True for _ in range(max_size)]
    else:
        return [True if x in indices else False for x in range(max_size)]


A = TypeVar("A")


class _DivMax(Protocol):
    """Class that support division and .max."""

    def __truediv__(self: A, d: Real) -> A:
        """Division that preserves type."""
        ...

    def max(self: A) -> Real:
        """Maximum."""
        ...


Dv = TypeVar("Dv", bound=_DivMax)
H = TypeVar("H", bound=Hashable)


def mapping_scale(
    content: Mapping[H, Dv],
    local_max: Callable[[Dv], Real] = lambda x: x.max(),
) -> Dict[H, Dv]:
    """Multiplicatively scale the values of a mapping by their global maximum.

    The values of content can be arrays.  If they are positive, this scales
    them to have values in the range 0 to 1. If the maximum value is zero, this
    function call will fail.

    Arguments:
    ---------
    content:
        A mapping where the items are compatible with local_max and can be divided by
        a number (probably arrays).
    local_max:
        Callable that is used to evaluate max on each individual array. Should take
        a value in content and return a Real (float).

    Returns:
    -------
    A dictionary where we have scaled the underlying arrays to have a max of 1.

    Notes:
    -----
    This function is type hinted in a restrictive way--- as long as local_max
    works on the elements of content and they can be divided, it will work.
    """
    keys, values = zip(*content.items())
    maximum = max([local_max(x) for x in values])
    return dict(zip(keys, (v / maximum for v in values)))


S = TypeVar("S", bound=Sequence)


class RSampler(Generic[S]):
    """Performs rejection sampling on items drawn from an iterable.

    Rejection sampling draws from a distribution g and accepts a subsample of these
    draws to produce a sample from distribution f. This subsample is stochastically
    determined by comparing random numbers to a scaled ration of the densities of
    the distributions.

    For example, if our source (`s`) iterate serves content of the form:
        `(data0, data1, ratios)`
    where the `data0` is drawn from distribution `g` and we wish to obtain samples from
    distribution `f`, and ratios contains scaled `g(x)/f(x)` for each frame `x
    in `data0`, then
        `RSampler(source=s,weights_index=-1,apply_to=(0,1),discard_weights=True)
    will return variates of the form
        `(data0, data1)`
    that reflect distribution `f`.

    If the objects that are drawn from the source iterable do not represent
    independent and identical variates, you may need to think about whether
    this procedure makes sense. In many cases, only after you have completed a
    complete round of samples to the produced samples _in aggregate_ reflect
    the target distribution.

    Note:
    ----
    Rejection sampling is performed on the individual pieces ("frames" of each
    iterable. For example, if the iterable serves arrays, this object will
    isolate entries along the leading index of each iterate.

    Warning:
    -------
    Sampled entries need to support boolean mask indexing. For example:
        g[ np.ndarray(True, False, False) ]

    The listed attributes can be modified during runtime to change behavior.

    Attributes:
    ----------
    source:
        Iterable to draw source values from. These iterates must provide the
        scaled probability ratios for subsequent sampling.
    weights_index:
        Index of the weights array in each pull from source.
    apply_to:
        Specifies which items in each iterate should be sampled. If None,
        all items are sampled.
    discard_weights:
        If true, we remove the weights from the served iterable.
    check_bound:
        If true, we check to make sure that no weights are larger than one.
        Larger than 1 weights are forbidden for the rejection sampling setup.
    drop_empty:
        If true, if we will serve only empty items in a iterate, we will skip serving
        that iterate.
    """

    # numerical tolerance for checking to see if number exceeds maximum
    _tol: Final = finfo(float).eps

    def __init__(
        self,
        source: Iterable[S],
        weights_index: int = -1,
        apply_to: Union[None, Collection[int], Collection[bool]] = None,
        discard_weights: bool = True,
        rng: Optional[Generator] = None,
        check_bound: bool = True,
        drop_empty: bool = True,
    ) -> None:
        """Initialize.

        Arguments:
        ---------
        source:
            Iterable to draw source values from. These iterates must provide the
            weights for subsequent sampling.
        weights_index:
            Index of the weights array in each pull from source.
        apply_to:
            Specifies which items in each iterate should be sampled. If None,
            all items are sampled.
        discard_weights:
            If true, we remove the weights from the served iterable.
        check_bound:
            If true, we check to make sure that no weights are larger than one.
            Larger than 1 weights are forbidden for the rejection sampling setup.
        drop_empty:
            If true, if we will serve empty iterables, we will skip serving that
            iterate.
        rng:
            numpy Generator instance or None. Used for subsequent sampling. If None,
            the default_rng from numpy is used

        """
        self.source = source
        self.weights_index = weights_index
        self.discard_weights = discard_weights
        self.check_bound = check_bound
        self.drop_empty = drop_empty
        self.apply_to = apply_to
        if rng is None:
            self._rng = default_rng()
        else:
            self._rng = rng

    def __iter__(self) -> Iterable[Tuple]:
        """Pull iterate from source, calculate mask, index, and filter."""
        pull: Union[S, Tuple]
        for pull in self.source:
            # get the boolean mask
            mask = self._sample_mask(pull[self.weights_index])
            if self.discard_weights:
                pull = tuple_remove(pull, self.weights_index)
            # this mask determines which items in the pull should be sampled.
            to_include_mask = _indices_to_mask(self.apply_to, len(pull))
            derived = [
                values[mask] if included else values
                for values, included in safezip(pull, to_include_mask)
            ]
            # if we would be serving 0 arrays, then don't serve at all
            if self.drop_empty and (0 == max(len(x) for x in derived)):
                pass
            else:
                yield tuple(derived)

    def _random_numbers(self, length: int) -> ndarray:
        """Generate a random array, variates in [0,1) ."""
        return self._rng.random(size=length)

    def _sample_mask(self, ratios: ndarray) -> ndarray:
        """Create Boolean array for subsampling from density ratios.

        Arguments:
        ---------
        ratios:
            Scaled ratio of densities: g/f, where g is the distribution of the sample
            we are presented with and f is the desired distribution. These values
            must be bounded by 1

        Returns:
        -------
        boolean ndarray reflecting which samples to retain or drop.

        Notes:
        -----
        If self.check_bound is True, this function will raise a ValueError if it finds
        a weight bigger than 1.
        """
        if self.check_bound:
            if ratios.max() > 1 + self._tol:
                raise ValueError("Found rejection sampling weight above one.")
        variates = self._random_numbers(len(ratios))
        return variates < ratios
