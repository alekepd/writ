from typing import (
    Iterable,
    Iterator,
    TypeVar,
    Annotated,
    Literal,
    Tuple,
)

import numpy as np
import numpy.typing as npt

A = TypeVar("A")

DType = TypeVar("DType", bound=np.generic)

ArrayNx3 = Annotated[npt.NDArray[DType], Literal["N", 3]]
ArrayMxNx3 = Annotated[npt.NDArray[DType], Literal["M", "N", 3]]


# TODO: write tests!
# TODO: option to use a subset of the atoms to do SVD frame fit
class RotateToSVDFrame(Iterable[A]):
    """Provides an iterable that rotates a source configuration into its SVD-frame
    and returns rotated coords and forces.

    """

    def __init__(
        self,
        source: Iterable[A],
        coords_idx: int = 0,
        forces_idx: int = 1,
        **kwargs,
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
        **kwargs:
            Stored and passed to rotate at each call.

        """
        self.source = source
        self.coords_idx = coords_idx
        self.forces_idx = forces_idx
        self.aux_args = kwargs

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
            np.concatenate(rotated_coords, axis=0),
            np.concatenate(rotated_forces, axis=0),
        )

    def _rotate_single_frame(
        self, coords: ArrayNx3, forces: ArrayNx3
    ) -> Tuple[ArrayNx3, ArrayNx3]:
        # need at least 3 atoms to compute SVD frame
        if coords.shape[0] > 2:
            U, S, Vh = np.linalg.svd(coords, full_matrices=False, compute_uv=True)
            V = Vh.T
            # NOTE: np.matmul is slower than mat1 @ mat2...
            rotated_coords = coords @ V  # np.matmul(coords, Vh.T)
            rotated_forces = forces @ V  # np.matmul(forces, Vh.T)
        else:
            rotated_coords = coords
            rotated_forces = forces
        return rotated_coords, rotated_forces

    def __iter__(self) -> Iterator[A]:
        """Iterate over input, returning rotated coords and forces."""
        for pull in self.source:
            coords_batch = pull[self.coords_idx]
            forces_batch = pull[self.forces_idx]
            coords_batch_rot, forces_batch_rot = self._rotate(
                coords_batch=coords_batch,
                forces_batch=forces_batch,
            )
            # repack the pull
            # NOTE: we use [:] to reassign to the underlying array directly because
            #       tuples cant be mutated
            pull[self.coords_idx][:] = coords_batch_rot
            pull[self.forces_idx][:] = forces_batch_rot
            yield pull
