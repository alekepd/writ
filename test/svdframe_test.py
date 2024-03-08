"""Test RotateToSVDFrame."""

from typing import Final
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
            np.sum(original[1] ** 2, axis=-1), np.sum(rotated[1] ** 2, axis=-1)
        )


@pytest.mark.h5py
@pytest.mark.parametrize("stride", [1, 10], ids=["stride=1", "stride=10"])
def test_svdframe_rotation_distances_preserved(
    aaqaa_coord_force_h5: SchemaH5, stride: int
) -> None:
    """Test that rotation does not change distances.

    Rotation is a linear transformation and should preserve distances. We check for
    selfdistances, i.e. all internal distances in a molecule and also for the distance
    between each particle and the origin. Note that we center the molecule around the
    mean before the SVD roation, so we need to do that here for the original too.
    """
    rotate_window = RotateToSVDFrame(
        source=aaqaa_coord_force_h5,
        coords_index=0,
        forces_index=1,
        stride_for_svd=stride,
    )
    for rotated, original in zip(rotate_window, aaqaa_coord_force_h5):
        original_centered = original[0] - np.mean(original[0], axis=1, keepdims=True)
        assert np.allclose(
            np.sum(original_centered**2, axis=-1), np.sum(rotated[0] ** 2, axis=-1)
        )
        # NOTE: self-distances are slow, move to separate function and mark as "slow"?
        for mol_original, mol_rotated in zip(original[0], rotated[0]):
            self_dists_original = np.array(
                [np.linalg.norm(mol_original - particle) for particle in mol_original]
            )
            self_dists_rotated = np.array(
                [np.linalg.norm(mol_rotated - particle) for particle in mol_rotated]
            )
            assert np.allclose(self_dists_original, self_dists_rotated)


@pytest.mark.h5py
@pytest.mark.parametrize(
    "stride", [1, 5, 10], ids=["stride=1", "stride=5", "stride=10"]
)
def test_svdframe_rotation_no_op(aaqaa_coord_force_h5: SchemaH5, stride: int) -> None:
    """Test that rotation to SVD frame does not change things that are already rotated.

    Something that already is in its SVD frame should not change (much) by reapplying
    the rotation.
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


@pytest.mark.h5py
def test_svdframe_rotation_ground_truth(aaqaa_coord_force_h5: SchemaH5) -> None:
    """Test rotation output against stored ground truth.

    Note that this test relies on a stable order of chunks to iterate over.
    Note also that we check against the absolute values because we could get
    the mirror image from SVD rotation.
    """
    rotated_coords_ground_truth: Final = np.array(
        [
            [
                [-9.17095, 9.095874, 0.37031505],
                [-9.963326, -0.40187073, -2.3426414],
                [2.8266892, -5.3515, -1.9443684],
                [12.989976, 0.74750394, -3.20283],
            ],
            [
                [-1.2203199, 8.673344, -1.4466324],
                [-0.83509415, 6.291748, 1.5290384],
                [5.9666147, -2.4167411, -2.9010293],
                [-5.365338, -8.141892, 1.5626348],
            ],
            [
                [-4.366285, -8.013836, 3.1795745],
                [-3.7420335, -1.8893983, -4.2813396],
                [-2.4140232, 3.5535374, -0.35216323],
                [9.603801, -3.6231415, 1.1866709],
            ],
        ]
    )
    rotated_forces_ground_truth: Final = np.array(
        [
            [
                [1.6832643, 10.094311, 1.5481346],
                [3.117689, -0.8834776, -1.6683146],
                [-7.0720596, -31.079372, 16.454138],
                [37.788605, 25.54765, -15.543807],
            ],
            [
                [-10.065261, 21.60349, 1.728706],
                [-2.3037937, -0.5277178, -8.910826],
                [-31.543222, 3.005161, 64.25549],
                [26.939291, -31.927063, 8.625921],
            ],
            [
                [5.995188, -6.611331, -1.5032207],
                [-7.49294, -0.23148148, 11.045828],
                [-4.8400354, 12.913263, -18.104023],
                [11.613076, -1.8417236, -7.2058487],
            ],
        ]
    )

    rotate_window = RotateToSVDFrame(
        source=aaqaa_coord_force_h5,
        coords_index=0,
        forces_index=1,
        stride_for_svd=1,
    )
    first_chunk_rotated = next(iter(rotate_window))
    for i, (mol_ground_truth_c, mol_ground_truth_f, mol_rot_c, mol_rot_f) in enumerate(
        zip(
            rotated_coords_ground_truth,
            rotated_forces_ground_truth,
            first_chunk_rotated[0],
            first_chunk_rotated[1],
        )
    ):
        assert np.allclose(np.abs(mol_ground_truth_c), np.abs(mol_rot_c[::50]))
        assert np.allclose(np.abs(mol_ground_truth_f), np.abs(mol_rot_f[::50]))
        if i == 2:
            # we only check the first 3 molecules
            break
