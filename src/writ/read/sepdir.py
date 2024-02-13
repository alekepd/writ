"""Iterable interface for reading files from multiple directories simultaneously.

Each directory contains a different type of data; files are simultaneously read
from each directory and served together. Association of files is done via
their filename.

Functionality is provided via SepDirChunks.
"""

from typing import (
    List,
    Tuple,
    Final,
    Callable,
    Iterator,
    Optional,
    Union,
    Collection,
    cast,
)

from os.path import join
import re
import glob
import numpy as np
from .. import util

GLOB_PATTERN: Final = r"*"
REGEX_PATTERN: Final = r"(.*)"


def filename_tags(
    pattern: str,
    sort: Union[bool, str] = True,
    wc_pattern: str = r"{}",
) -> List[str]:
    """Search for possible ids in filenames matching a template string.

    For example, if the following files are present in a directory
        h_5_tag.npy h_9_tag.npu h_10_tag.npy
    and we set pattern to "h_{}_tag.npy", this function will return [5,9,10] (or some
    permutation of that list).

    Arguments:
    ---------
    pattern:
        Stride to use as template for looking for matching files. Should have a wildcard
        pattern (see wc_pattern, usually {}) present exactly once.
    sort:
        Whether to sort the tags, or if 'natural', to use natural_sort.
    wc_pattern:
        Pattern to replace with real wildcards. Usually "{}" so that the string used for
        formatted printing can also be the string.

    Returns:
    -------
    List of labels from files in the current directory fitting the criteria. May be a
    list of size zero.

    """
    # Check to see if exactly one {} is present
    pieces = pattern.split(wc_pattern)
    if len(pieces) == 1:
        raise ValueError(f"No wildcard ({wc_pattern}) present.")
    elif len(pieces) > 2:
        raise ValueError(f"Too many wildcards ({wc_pattern}) present.")

    # derive wildcard version
    glob_pattern = GLOB_PATTERN.join([glob.escape(x) for x in pieces])
    re_pattern = REGEX_PATTERN.join([re.escape(x) for x in pieces])

    ids = []
    # iterate over the matching files
    for filename in glob.iglob(glob_pattern):
        # extract chunk ids from those files
        match = re.search(re_pattern, filename)
        if match is None:
            raise ValueError("Internal error in searching for patterns.")
        else:
            ids.append(match.group(1))

    if sort is True:
        return sorted(ids)
    elif sort == "natural":
        return util.natural_sort(ids)
    else:
        return ids


class SepDirChunks:
    """Iterable interface to numpy arrays with programmatic names.

    Suppose we have two directories, "a/" and "b/", which contain files as follows:
        a/h_5_tag.npy
        a/h_9_tag.npu
        b/g_5_tag.npu
        b/g_9_tag.npu
    In certain cases, pairs of files from these directories are paired by the content in
    their names. For example, we may know that a/h_5_tag.npy and b/g_5_tag.npy are pairs
    because of the common placement of 5 in their name.

    This class provides an iterable interface over these kinds of files, where
    at each iteration groups of files are read by a supplied function and served
    as a tuple. In the above example, the first iteration could serve
    (a/h_5_tag.npy, b/g_5_tag.npu) and the second could serve (a/h_9_tag.npu,
    b/g_9_tag.npu). Note that we have here used filenames in the served tuples,
    but in practice the read content from each file would be returned at each iteration.

    To make this happen, we would instantiate the class as so:
        skeletons = ['a/h_{}_tag.npy','b/g_{}_tag.npy']
        SepDirChunks(skeletons)
    Each entry in skeletons is matched against files using {} as a wildcard, resulting
    in n (here, 2) different sequences of files. These files are then read and
    served as pairs during iteration. The label that is found via {} matches from
    any two files that are served is the same (in the above example, 5 and 9).
    """

    def __init__(
        self,
        patterns: Collection[str],
        parent: Optional[str] = None,
        transform: Callable = lambda x: x,
        singleton: bool = True,
        loader: Callable[[str], np.ndarray] = lambda x: cast(np.ndarray, np.load(x)),
        include_id: bool = False,
    ) -> None:
        """Initialize instance.

        Arguments:
        ---------
        patterns:
            Strings which specify which files we should look for. Each string
            must have a '{}', which we treat as a glob wildcard (do _not_ use "*" or
            ".*" as wildcards). Each string specifies one of the categories of files
            that will be drawn from to create pairs.
        parent:
            Prepended to all entries in patterns; typically a parent directory.
        transform:
            Callable which takes the tuple of produced data objects and can perform
            operations on it; the output of this function is served during iteration.
            Provided for compatibility of SchemaH5, but less important. Defaults to
            the identity.
        singleton:
            If true, then during iteration, if a size-1 tuple is to be returned, we
            instead simply return the sole element of the tuple.
        loader:
            Function which takes a filename and returns an object. Applied to the
            filenames matching patterns during iteration; typically something like
            np.load.
        include_id:
            If true, then instead of only serving file contents for each entry
            in patterns, we append a tuple identifying the source of the served files
            at each iteration. This tuple has the filename which was read for each file.

        Returns:
        -------
        None

        """
        if len(patterns) == 0:
            raise ValueError("Must specify at least 1 pattern.")
        if parent is None:
            self.patterns = patterns
        else:
            self.patterns = [join(parent, x) for x in patterns]

        tags = [filename_tags(x, sort=True) for x in self.patterns]
        # check to see if all the categories of files have the same tags.
        mask = [tags[0] == x for x in tags[1:]]
        if not all(mask):
            raise ValueError("The tags implied by the various patterns differ.")
        self.tags = tags[0]

        self.transform = transform
        self.loader = loader
        self.singleton = singleton
        self.include_id = include_id

    def __iter__(
        self,
    ) -> Iterator[Union[np.ndarray, Tuple[Union[np.ndarray, str], ...]]]:
        """Iterate over groups of read files.

        If singleton was set to True during initialization, if reading (and optionally
        labeling) provides a single element tuple, then the only value in this tuple
        is returned instead of the tuple. In all other cases the tuple is returned.
        """
        for tag in self.tags:
            filenames = tuple(x.format(tag) for x in self.patterns)
            datas = tuple(self.loader(x) for x in filenames)
            transformed = self.transform(datas)
            if self.include_id:
                ready = util.tuple_append(transformed, filenames)
            else:
                ready = transformed
            if self.singleton and len(ready) == 1:
                yield ready[0]
            else:
                yield cast(np.ndarray, ready)
