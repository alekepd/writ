"""Provides a transform that rotates all molecules into their SVD frame.

The RotateToSVDFrame class provides this functionality.
"""

from typing import (
    Iterable,
    Iterator,
    TypeVar,
    Annotated,
    Sequence,
    Literal,
    Tuple,
    TypeGuard,
)

import numpy as np
import numpy.typing as npt

A = TypeVar("A")

DType = TypeVar("DType", bound=np.generic)

ArrayNx3 = Annotated[npt.NDArray[DType], Literal["N", 3]]
ArrayMxNx3 = Annotated[npt.NDArray[DType], Literal["M", "N", 3]]


# TODO: write tests!
class RotateToSVDFrame(Iterable[Sequence[A]]):
    """Provides an iterable that rotates a source configuration into its SVD-frame.

    This class returns rotated coords and forces by modifying the respective entries
    of the iterates in-place. The positions of the coordinates and forces in iterate
    can be specified at initialization.
    """

    def __init__(
        self,
        source: Iterable[Sequence[A]],
        coords_idx: int = 0,
        forces_idx: int = 1,
        atom_stride_for_svd: int = 1,
    ) -> None:
        """Initialize.

        Arguments:
        ---------
        source:
            Iterable of objects to iterate over
        coords_idx: int
            Index of the coordinates array in the tuples we iterate over,
            by default 0.
        forces_idx: int
            Index of the forces array in the tuples we iterate over,
            by default 1.
        atom_stride_for_svd: int
            Stride to use when choosing atoms for SVD fit,
            by default 1.
            Note that if the stride would result in using less than 3 atoms,
            we will fallback to a stride of 1.

        """
        self.source = source
        self.coords_idx = coords_idx
        self.forces_idx = forces_idx
        self.atom_stride_for_svd = atom_stride_for_svd

    def _rotate(
        self, coords_batch: ArrayMxNx3, forces_batch: ArrayMxNx3
    ) -> Tuple[ArrayMxNx3, ArrayMxNx3]:
        rotated_coords = []
        rotated_forces = []
        for coords, forces in zip(coords_batch, forces_batch):
            c_rot, f_rot = self._rotate_single_frame(coords=coords, forces=forces)
            rotated_coords += [c_rot]
            rotated_forces += [f_rot]
        return (
            np.asarray(rotated_coords),
            np.asarray(rotated_forces),
        )

    def _rotate_single_frame(
        self, coords: ArrayNx3, forces: ArrayNx3
    ) -> Tuple[ArrayNx3, ArrayNx3]:
        # need at least 3 atoms to compute SVD frame
        if coords.shape[0] // self.atom_stride_for_svd < 3:
            coords_subset = coords
        else:
            coords_subset = coords[:: self.atom_stride_for_svd]
        # (again) make sure we have enough atoms to calculate svd frame
        if coords_subset.shape[0] > 2:
            U, S, Vh = np.linalg.svd(
                coords_subset, full_matrices=False, compute_uv=True
            )
            V = Vh.T
            # NOTE: np.matmul is slower than mat1 @ mat2...
            rotated_coords = coords @ V  # np.matmul(coords, Vh.T)
            rotated_forces = forces @ V  # np.matmul(forces, Vh.T)
        else:
            rotated_coords = coords
            rotated_forces = forces
        return rotated_coords, rotated_forces

    def _check_coords_forces_shapes_match(
        self, coords_batch: np.ndarray, forces_batch: np.ndarray
    ) -> TypeGuard[ArrayMxNx3]:
        coords_shape = coords_batch.shape
        forces_shape = forces_batch.shape
        assert len(coords_shape) == 3
        assert coords_shape[-1] == 3
        for dim_c, dim_f in zip(coords_shape, forces_shape):
            assert dim_c == dim_f
        return True

    def __iter__(self) -> Iterator[Sequence[A]]:
        """Iterate over input, returning rotated coords and forces."""
        for pull in self.source:
            coords_batch = pull[self.coords_idx]
            forces_batch = pull[self.forces_idx]
            if (
                isinstance(coords_batch, np.ndarray)
                and isinstance(forces_batch, np.ndarray)
                and self._check_coords_forces_shapes_match(coords_batch, forces_batch)
            ):
                coords_batch_rot, forces_batch_rot = self._rotate(
                    coords_batch=coords_batch,
                    forces_batch=forces_batch,
                )
                # repack the pull
                # NOTE: coords_batch and forces_batch are references to the entries in
                #       pull,
                #       we use [:] to reassign to the underlying array directly because
                #       tuples cant be mutated
                coords_batch[:] = coords_batch_rot
                forces_batch[:] = forces_batch_rot
                yield pull
