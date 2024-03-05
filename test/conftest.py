"""Common test fixtures."""

from typing import Final
from pathlib import Path
import pytest

from writ.read import SchemaH5
from writ.util import TupleStrider

AAQAA_H5_PATH: Final = (
    Path(__file__).parent / Path("data") / Path("aaqaa_stride_10000_9ramp.h5py")
)
AAQAA_COORD_KEY: Final = "coords"
AAQAA_FORCE_KEY: Final = "Fs"


@pytest.fixture
def aaqaa_coord_h5() -> SchemaH5:
    """Produce SchemaH5 of aaqaa.

    Returns coord arrays in tuples, no ids.
    """
    return SchemaH5(
        target=str(AAQAA_H5_PATH),
        schema=[AAQAA_COORD_KEY],
        include_id=False,
        singleton=False,
    )


@pytest.fixture
def aaqaa_coord_force_h5() -> SchemaH5:
    """Produce SchemaH5 of aaqaa.

    Returns coord and force arrays in tuples, no ids.
    """
    return SchemaH5(
        target=str(AAQAA_H5_PATH),
        schema=[AAQAA_COORD_KEY, AAQAA_FORCE_KEY],
        include_id=False,
        singleton=False,
    )


@pytest.fixture
def aaqaa_coord_h5_singleton() -> SchemaH5:
    """Produce SchemaH5 of aaqaa.

    Returns coord arrays (no tuples), no ids.
    """
    return SchemaH5(
        target=str(AAQAA_H5_PATH),
        schema=[AAQAA_COORD_KEY],
        include_id=False,
        singleton=True,
    )


@pytest.fixture
def aaqaa_coord_h5_with_ids() -> SchemaH5:
    """Produce SchemaH5 of aaqaa.

    Returns coord arrays with ids in tuples.
    """
    return SchemaH5(
        target=str(AAQAA_H5_PATH),
        schema=[AAQAA_COORD_KEY],
        include_id=True,
        singleton=False,
    )


@pytest.fixture
def aaqaa_coord_h5_stride_2() -> SchemaH5:
    """Produce SchemaH5 of aaqaa.

    Returns coord arrays (strided by 2) in tuples, no ids.
    """
    return SchemaH5(
        target=str(AAQAA_H5_PATH),
        schema=[AAQAA_COORD_KEY],
        include_id=False,
        transform=TupleStrider(stride=2),
        singleton=False,
    )
