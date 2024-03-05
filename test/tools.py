"""Basic tools used by tests."""
from typing import Final, Iterable, Callable, TypeVar, Union, Tuple, Optional
import numpy as np
import xxhash

INT_MAX_BYTES: Final = 64
INT_BYTE_ORDER: Final = "big"


def arrhash(array: np.ndarray, /, include_shape: bool = True) -> int:
    """Hash a numpy array.

    Deterministic between processes.

    Arguments:
    ---------
    array:
        Array to hash. Positional argument only.
    include_shape:
        If true, we also add each integer in the shape tuple to the hash.

    Returns:
    -------
    int representing the hash.

    Notes:
    -----
    This function probably some size limits, but it's not clear. Should be
    stable across different machines. Probably cannot tell the difference in
    arrays that have different classes but identical data, which may be good or
    bad.

    """
    h = xxhash.xxh64()
    h.update(array.data.tobytes())
    if include_shape:
        for entry in array.shape:
            h.update(entry.to_bytes(INT_MAX_BYTES, INT_BYTE_ORDER))
    return h.intdigest()


def strhash(s: str, /) -> int:
    """Hash string.

    Arguments:
    ---------
    s:
        string to hash. Positional argument only.

    Returns:
    -------
    integer representing hash.

    """
    h = xxhash.xxh64()
    h.update(s)
    return h.intdigest()


def ghash(data: Union[np.ndarray, str, Tuple[str]], /) -> int:
    """Hash data entry that may be of different types.

    Wraps other hash methods.  Deterministic between processes.

    Arguments:
    ---------
    data:
        str or np.ndarray or tuple of str/np.ndaray. If np.ndarray, arrhash is called;
        if str, strhash is called; if a tuple, ihash with ghash is called.

    Returns:
    -------
    Int representing the hash.

    Notes:
    -----
    This is a convenience function when it is unknown whether the argument of
    an unspecified type. Currently only supports strings, tuples of strings/ndarrays,
    and np.ndarrays.

    """
    if isinstance(data, np.ndarray):
        return arrhash(data)
    elif isinstance(data, str):
        return strhash(data)
    elif isinstance(data, tuple):
        return ihash(data, hasher=ghash)
    else:
        raise NotImplementedError(f"Unknown how to hash {data}.")


T = TypeVar("T")


def ihash(
    data: Iterable[T],
    /,
    hasher: Callable[[T], int],
    order: bool = True,
    seed: Optional[str] = 'ahakwvota2',
    **kwargs,
) -> int:
    """Hash an iterable of objects given a underlying hash function.

    Deterministic between processes.

    Warning:
    -------
    Hash does not change if the type of iterable changes.

    Arguments:
    ---------
    data:
        Iterable arrays to hash. Position only argument.
    hasher:
        Callable which is used to hash each entry in data. These produced hashes
        are then hashed. If data is already hashed, set to lambda x: x. Should
        return an int for each item.
    order:
        If true, the hash will change if the order of arrays changes. If false,
        it should not. False behavior is imposed via sorted acting on the hashes.
    seed:
        Value to feed into the hasher prior to hashing iterates. Allows the produced
        hash of a single entry iterable to not be identical to a similar hash function
        applied applied to the iterable content. If None, no value is used.
    **kwargs:
        Passed to hasher.

    Returns:
    -------
    Int representing the hash.

    """
    hashes = [hasher(x, **kwargs) for x in data]
    h = xxhash.xxh64()
    if seed is not None:
        h.update(seed)
    if order:
        target = hashes
    else:
        target = sorted(hashes)
    for entry in target:
        h.update(entry.to_bytes(INT_MAX_BYTES, INT_BYTE_ORDER))

    return h.intdigest()
