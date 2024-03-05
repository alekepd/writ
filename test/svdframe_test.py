"""Test RotateToSVDFrame"""

from typing import Final
from pathlib import Path
import pytest
import numpy as np

from writ.read import SchemaH5
from writ.transform.svdframe import RotateToSVDFrame

AAQAA_H5_PATH: Final = (
    Path(__file__).parent / Path("data") / Path("aaqaa_stride_10000_9ramp.h5py")
)
AAQAA_COORD_KEY: Final = "coords"
AAQAA_FORCE_KEY: Final = "Fs"


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


@pytest.mark.h5py
@pytest.mark.parametrize(
    "stride", [1, 5, 10], ids=["stride=1", "stride=5", "stride=10"]
)
def test_svdframe_rotation_force_magnitude(
    aaqaa_coord_force_h5: SchemaH5, stride: int
) -> None:
    """Test that rotation to SVD frame does not change the magnitude of forces.

    Rotation must preserve the absolute force acting on every particle because
    it is a linear transformation.
    """
    rotate_window = RotateToSVDFrame(
        source=aaqaa_coord_force_h5,
        coords_idx=0,
        forces_idx=1,
        stride_for_svd=stride,
    )
    for rotated, original in zip(rotate_window, aaqaa_coord_force_h5):
        assert np.allclose(
            np.sum(rotated[1] ** 2, axis=-1), np.sum(original[1] ** 2, axis=-1)
        )
