"""Provides tools to write iterables of arrays to h5 files.

Currently, this only includes a function which allows one to map a iterable of
collections of arrays to saved h5py datasets.
"""
from typing import (
    Iterable,
    Collection,
    Union,
    Tuple,
    List,
)
import numpy as np
import h5py  # type: ignore [import-untyped]


def batched_h5_save(
    group: h5py.Group,
    names: Iterable[str],
    source: Iterable[Collection[np.ndarray]],
    chunked: bool = True,
    **kwargs: Union[None, Tuple[int, ...]],
) -> None:
    """Save data into h5py.Group object under given names.

    Arguments:
    ---------
    group:
        h5py.Group we will create Datasets in.
    names:
        Names under which to store the data. See source.
    source:
        Iterable of iterable of arrays (probably coord,force tuples). Each
        entry must have the same number of members as names.
    chunked:
        Whether to write data to h5 in chunks. Recommended.
    kwargs:
        Passed to create_dataset.

    Returns:
    -------
    None

    """
    if chunked:
        first_iteration = True
        for pull in source:
            # in the first iteration we need to set up the dataset
            if first_iteration:
                # we need extract the shapes, but we ignore the first dimension as
                # we will grow along it.
                shapes = [(None,) + sub.shape[1:] for sub in pull]
                # create the datasets we will subsequently write into.
                # we initialize them using the first chunks.
                dsets = [
                    group.create_dataset(name, chunks=True, maxshape=shape, data=sub)
                    for shape, name, sub in zip(shapes, names, pull)
                ]
                # marks contains size along the first axis of what has been written.
                marks = [sub.shape[0] for sub in pull]
            else:
                new_marks: List[int] = []
                for sub, mark, dset in zip(pull, marks, dsets):
                    size = sub.shape[0]
                    # we need to regrow the array to be new_total_size in order to
                    # incorporate the new chunk
                    new_total_size = mark + size
                    dset.resize((new_total_size,) + dset.shape[1:])
                    dset[mark:new_total_size] = sub
                    new_marks.append(new_total_size)
                # update the marks to reflect the grown arrays
                marks = new_marks
            first_iteration = False
    else:
        combined = (np.concatenate(x) for x in zip(*source))
        for name, data in zip(names, combined):
            group.create_dataset(name, data=data, **kwargs)
