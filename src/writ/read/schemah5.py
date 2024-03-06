"""Iterable interface for reading h5py files of a specific style.

WARNING: These classes will not work properly if your h5 file has slashes in
descriptors. This seems to violate the h5 spec in itself, but is possible. h5py uses
slashes as delimiters when describing hierarchical locations in the file--- this is
perfectly fine.

Behavior is provided via the SchemaH5 class. See that class for more information.
"""

from typing import (
    Any,
    Dict,
    List,
    Tuple,
    Set,
    FrozenSet,
    Final,
    Mapping,
    Iterable,
    Hashable,
    Callable,
    Iterator,
    Optional,
    Union,
    Sequence,
    TypeVar,
)
from functools import cached_property
import h5py  # type: ignore [import-untyped]
from .. import util

T = TypeVar("T", bound=Hashable)


def data_anchors(dataset: h5py.Group, separator: str = "/") -> Set[Tuple[str, ...]]:
    """Extract unique lowest-level groups of data in h5py.Group.

    This function gives the unique group locations hat index groups which hold
    h5py.Datasets. The results are the strings that index these groups.

    Arguments:
    ---------
    dataset:
        h5py.Group (including h5py.File) to query.
    separator:
        Separator used to break down h5py-produced string descriptors. This is almost
        surely '/'.

    Returns:
    -------
    A set of all the locations (str) of groups of data.

    """
    record = set()

    # used to recurse through the h5.
    def process(name: str, object: Any) -> None:
        if isinstance(object, h5py.Dataset):
            entry: Tuple[str, ...] = tuple(name.split(separator))
            record.add(entry[: len(entry) - 1])

    dataset.visititems(process)
    return record


def _flush_items(data: Mapping[T, Any]) -> Dict[T, Any]:
    """Apply *[:] to each member of a mapping (e.g., dictionary).

    For variable T, the applied operation is T[:].  This causes the array-like member to
    be pulled into memory in most cases.

    Arguments:
    ---------
    data:
        A dictionary that has [:] compatible items.

    Returns:
    -------
    A dictionary with [:] applied to each member.

    """
    transformed = {}
    for key in data:
        transformed.update({key: data[key][:]})
    return transformed


E = TypeVar("E", bound=Sequence)

# Can't see any easy way to make this fully type annotated. We don't even really know
# what slicing does type-wise
def _flush_tuple(data: Iterable[E]) -> Tuple[Any, ...]:
    """Apply *[:] to each member of a iterable.

    For item T, the applied operation is T[:].  This causes the array-like member to
    be pulled into memory in most cases.

    Arguments:
    ---------
    data:
        Indexable (ideally, tuple) that we will draw elements from for transformation.

    Returns:
    -------
    A tuple with [:] applied to each member.

    """
    flushed = []
    for value in data:
        flushed.append(value[:])
    return tuple(flushed)


class SchemaH5(Iterable):
    """Provides an iterable interface to certain h5 files.

    This class provides an iterable interface to h5 files which contain groups of
    Datasets. Iteration iterates over these groups. Options allow only certain groups to
    be served during iteration. No interface for writing to h5 files is provided.

    Important attributes:
    --------------------
    self.schema
        A list of strings which must be present as keys of Datasets in order for a group
        to be served during iteration.

        self.schema may be overridden by assignment after instantiating, which may be
        useful in practice. See Notes.
    self.kinds (property)
        Provides a set containing all possible frozensets of keys of groups of data. If
        it only has one element, there is only one "kind" of data (group) in your
        dataset.


    Warning:
    -------
    This class will not work properly if your h5 file has slashes in descriptors. This
    seems to violate the h5 spec in itself, but is possible. h5py uses slashes as
    delimiters when describing hierarchical locations in the file--- this is perfectly
    fine. If you do not know what this warning is saying, you are probably okay.

    Notes:
    -----
    h5py files are convenient for storing data, but it can difficult to query them
    without knowledge of how they are organized in a programmatic way. This class
    provides a way to iterate over the many contained arrays.

    This can be best seen via an example. Suppose we have a h5 with the following
    approximate structure:

    file.h5py:
        a/b1/c/data1.array
        a/b1/c/data2.array

        a/b2/c/data1.array
        a/b2/c/data2.array

        a1/b1/c/data1.array
        a1/b1/c/data2.array

    We can view this file as containing 3 "collections" of data, each with an instance
    of data1.array and data2.array. Assuming that data1 and data2 are related to each
    other, we may have analysis where we would like to iterate over the pairs of
    (data1,data2) in each unique set of a/b/c. This may be done as follows:

    important_keys = ['data1.array','data2.array']
    for collection in SchemaH5("file.h5py",schema=important_keys):
        # collection is a tuple with an entry for each key in important_keys (in
        # the same order)
        < do analysis on collection[0] and collection[1] >

    To do this, SchemaH5 recursively first found all the datasets in in 'file.h5py'. It
    then organized those datasets based on their location in the group hierarchy,
    forming collections. Finally, it looked at which of those collections had items
    indexable by the strings in `important_keys`, and served those up via an iterable
    interface. Each value returned during iteration is a tuple ordered by the given
    keys.

    If, for example, important_keys=['data1.array'], during iteration the served
    tuple would only include an entry for 'data1.array' (and not 'data2.array').
    Furthermore, supposed we add the following dataset to our hypothetical file:

    a1/b1/c/unique.array

    As our keys do not specify 'unique.array', it will not be served during
    iteration.  Default behavior will still serve a1/b1/c/data1.array and
    a1/b1/c/data2.array (if both keys for both data1.array and data2.array are
    given). If strict=True, then this collection is skipped. If important_keys
    = ['data1.array','data2.array','unique.array'], then _only_ the data from
    a1/b1/c is served, as it is the only collection with all the requested
    keys.

    Furthermore, a group with a different hierarchy level is also fine. That is, if
    elements such as

    d/g/data1.array
    d/g/data2.array

    are present, they will also be included in iteration. This is the key feature of
    this class: it does not assume a fixed hierarchy. Instead, it finds all possible
    groups and iterates over them.

    Note that if you specify an invalid set of keys, then the iterator will be empty as
    there will be no compatible collections.  To see which collections of keys are
    present in your dataset (for use in important_keys above) use the `kinds` attribute.

    This may be used in combination with setting the schema attribute to work with files
    interactively. For example:
    g = SchemaH5("file.h5py",schema=[])

    Additional init options may be used to tweak behavior.

    """

    SEPARATOR: Final = "/"
    DEFAULT_ID_KEY: Final = "id"

    def __init__(
        self,
        target: Union[h5py.Dataset, str],
        schema: Optional[Iterable[str]],
        transform: Callable[[Tuple], Tuple] = _flush_tuple,
        strict: bool = False,
        include_id: bool = False,
        singleton: bool = True,
    ) -> None:
        """Initialize object.

        Arguments:
        ---------
        target:
            filename or h5py dataset to iterate over. If a filename, the file is opened
            read only and closed upon object deletion. When operating on an
            already-opened dataset, we assume that the file does not change its
            contents.
        schema:
            Iterable of strings which specify which strings to look for when iterating
            over groups.
        transform:
            Callable which is fed the tuple of h5py.Dataset objects prior to
            serving during iteration. For example, can stride or turn data into numpy
            arrays. The default turns all h5py datasets into numpy arrays via [:].

            Note that transform _CAN_ change the elements that are served during
            iteration to differ from those given by "schema". This includes order
            and content. The default choices do not do this (they just turn the h5
            arrays into numpy arrays).
        strict:
            If True, then only groups which have the schema and keys no other keys are
            presented during iteration. If False, groups having more than the specified
            keys are served. Note that the fields given by these additional keys are
            _not_ served--- only data related to schema is.
        include_id:
            The tuples served during iteration can have an additional final entry which
            contains a tuple of the hierarchical group location of the data. If
            include_ids is a string, it is used as a key for this value. If False,
            no such value is included.
        singleton:
            If true, then if only one field is to be returned at each iteration, it is
            not wrapped in a tuple. This enables simpler loop syntax. If false, it is
            still wrapped in a size 1 tuple.

        """
        self.strict = strict
        self.transform = transform
        self.include_id = include_id
        self.singleton = singleton

        if schema is None:
            self.schema: List[str] = []
        else:
            self.schema = list(schema)
        if len(set(self.schema)) != len(self.schema):
            raise ValueError("schema should not have duplicate values.")

        if isinstance(target, str):
            self.dataset = h5py.File(target, "r")
            self.responsible = True
        else:
            self.dataset = target
            self.responsible = False

        self.anchors = sorted(data_anchors(self.dataset, separator=self.SEPARATOR))

    @cached_property
    def kinds(self) -> Set[FrozenSet]:
        """The types of collections present in the h5py.

        Returns a set which containts frozensets of all the possible combinations of
        keys used in the various dataset collections in the underlying h5 file.
        """
        results = set()
        for index in self.anchors:
            results.add(frozenset(self._get_collection(index).keys()))
        return results

    def __iter__(self) -> Iterator[Union[Tuple[Any, ...], Any]]:
        """Iterate over groups of datasets.

        Data is served as tuples with values indexed by the items in self.schema.
        self.transform is applied to data before each iteration.

        Note that if singleton was set to True and only one value is to be returned,
        this does not return a tuple. In all other cases it does.
        """
        for anchor in self.anchors:
            data = self._get_collection(anchor)
            if not self._valid_schema(data):
                continue
            new_data = self.transform(util.tupleize(data, self.schema))
            if self.include_id:
                new_data = util.tuple_append(new_data, anchor)
            if len(new_data) == 1 and self.singleton:
                yield new_data[0]
            else:
                yield new_data

    def _get_collection(self, id: Tuple[str, ...]) -> h5py.Group:
        """Return the Group indexed by a index.

        This function is needed because we store the locations as tuples instead of the
        strings used for indexing h5py objections.
        """
        return self.dataset[self.SEPARATOR.join(id)]

    def _valid_schema(self, data: Mapping[str, object]) -> bool:
        """Check whether a data mapping is compatible with the stored schema."""
        keys = set(data.keys())
        if self.strict:
            return set(self.schema) == keys
        else:
            return set(self.schema) <= keys

    def __del__(self) -> None:
        """Optionally close opened files."""
        if self.responsible:
            self.dataset.close()
