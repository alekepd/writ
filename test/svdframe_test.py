"""Test RotateToSVDFrame."""

import pytest
import numpy as np

from writ.read import SchemaH5
from writ.transform.svdframe import RotateToSVDFrame


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
        coords_index=0,
        forces_index=1,
        stride_for_svd=stride,
    )
    for rotated, original in zip(rotate_window, aaqaa_coord_force_h5):
        assert np.allclose(
            np.sum(rotated[1] ** 2, axis=-1), np.sum(original[1] ** 2, axis=-1)
        )


@pytest.mark.h5py
@pytest.mark.parametrize(
    "stride", [1, 5, 10], ids=["stride=1", "stride=5", "stride=10"]
)
def test_svdframe_rotation_no_op(aaqaa_coord_force_h5: SchemaH5, stride: int) -> None:
    """Test that rotation to SVD frame does not change things that are already rotated.

    Something that already is in its SVD frame should not change (much) by reapplying the
    rotation.
    Note that we could get the mirror image, so we need to compare absolute values.
    """
    rotate_window = RotateToSVDFrame(
        source=aaqaa_coord_force_h5,
        coords_index=0,
        forces_index=1,
        stride_for_svd=stride,
    )
    double_rotate_window = RotateToSVDFrame(
        source=rotate_window,
        coords_index=0,
        forces_index=1,
        stride_for_svd=stride,
    )
    for rotated, double_rotated in zip(rotate_window, double_rotate_window):
        assert np.allclose(
            np.abs(rotated[0]), np.abs(double_rotated[0]), rtol=1e-2, atol=1e-2
        )
        assert np.allclose(
            np.abs(rotated[1]), np.abs(double_rotated[1]), rtol=1e-2, atol=1e-2
        )
