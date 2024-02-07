"""Iterable interface for transforming a trajectory using TICA.

Note that during training, all read data is stored in its featurized form. This may
require large amounts of memory. After training, 
"""

from typing import Iterable, Iterator, Optional, Callable, Any
from typing_extensions import Self
from deeptime.util.data import TrajectoriesDataset, TrajectoryDataset  # type: ignore  [import-untyped]
from deeptime.decomposition import TICA  # type: ignore  [import-untyped]
import numpy as np


class TICWindow:
    """Creates a TICA transform from an iterable of molecular coordinate arrays.

    Arrays are assumed to be of shape (n_frames,n_sites,n_dims).

    Example:
    -------
    it = < iterable that returns chunks of coordinates >
    t = TICAWindow(source=it)
    t.fit() # this reads it onces and trains the tica transform
    for x in t:
        # x contains arrays of tica coordinates corresponding to it

    Note that .fit() can be called with an argument to train on a difference
    source of data.

    """

    def __init__(
        self,
        source: Iterable,
        lagtime: int = 1,
        featurizer: Optional[Callable] = None,
        dim: int = 2,
    ) -> None:
        """Initialize.

        Arguments:
        ---------
        source:
            Iterable of position arrays (shape (n_steps,nsizes,n_dims)).
        lagtime:
            Lagtime of TICA transform. Note that this compounds with any striding
            performed on the input data.
        featurizer:
            Featurizer to use prior to TICA projection. If not provided,
            defaults to all pairwise distances.
        dim:
            Number of dimensions of TICA transform to return.

        """
        if featurizer is None:

            def _feat(x: np.ndarray) -> np.ndarray:
                return distances(x, return_matrix=False)

            self.featurizer = _feat
        else:
            self.featurizer = featurizer

        self.lagtime = lagtime
        self.source = source
        self.tica: Any = None
        self.dim = dim

    def fit(self, fit_source: Optional[Iterable[np.ndarray]] = None) -> Self:
        """Fit TICA transform."""
        if fit_source is None:
            target = self.source
        else:
            target = fit_source

        feats = []
        for pull in target:
            feated = self.featurizer(pull)
            feats.append(TrajectoryDataset(lagtime=self.lagtime, trajectory=feated))

        self.tica = TICA(dim=self.dim)
        self.tica.fit(TrajectoriesDataset(feats))
        return self

    def __iter__(self) -> Iterator[np.ndarray]:
        """Read data and map using trained TICA."""
        if self.fit is None:
            raise ValueError("Object is not yet fit, so cannot transform data.")
        for pull in self.source:
            yield self.tica.transform(self.featurizer(pull))


def distances(
    xyz: np.ndarray,
    cross_xyz: Optional[np.ndarray] = None,
    return_matrix: bool = True,
    return_displacements: bool = False,
) -> np.ndarray:
    """Calculate the distances for each frame in a trajectory.

    Returns an array where each slice is the distance matrix of a single frame
    of an argument.

    Arguments:
    ---------
    xyz (np.ndarray):
        An array describing the cartesian coordinates of a system over time;
        assumed to be of shape (n_steps,n_sites,n_dim).
    cross_xyz (np.ndarray or None):
        An array describing the Cartesian coordinates of a different system over
        time or None; assumed to be of shape (n_steps,other_n_sites,n_dim). If
        present, then the returned distances are those between xyz and cross_xyz
        at each frame.  If present, return_matrix must be truthy.
    return_matrix (boolean):
        If true, then complete (symmetric) distance matrices are returned; if
        false, the upper half of each distance matrix is extracted, flattened,
        and then returned.
    return_displacements (boolean):
        If true, then instead of a distance array, an array of displacements is
        returned.

    Returns:
    -------
    Returns numpy.ndarrays, where the number of dimensions and size depend on
    the arguments.

    If return_displacements is False:
        If return_matrix and cross_xyz is None, returns a 3-dim numpy.ndarray of
        shape (n_steps,n_sites,n_sites), where the first index is the time step
        index and the second two are site indices. If return_matrix and
        cross_xyz is not None, then an array of shape
        (n_steps,other_n_sites,n_sites) is returned. If not return_matrix,
        return a 2-dim array (n_steps,n_distances), where n_distances indexes
        unique distances.
    else:
        return_matrix must be true, and a 4 dimensional array is returned,
        similar to the shapes above but with an additional terminal axis for
        dimension.

    """
    if cross_xyz is not None and not return_matrix:
        raise ValueError("Cross distances only supported when return_matrix is truthy.")
    if return_displacements and not return_matrix:
        raise ValueError("Displacements only supported when return_matrix is truthy.")

    if cross_xyz is None:
        displacement_matrix = xyz[:, None, :, :] - xyz[:, :, None, :]
    else:
        displacement_matrix = xyz[:, None, :, :] - cross_xyz[:, :, None, :]
    if return_displacements:
        return displacement_matrix
    distance_matrix = np.linalg.norm(displacement_matrix, axis=-1)
    if return_matrix:
        return distance_matrix
    n_sites = distance_matrix.shape[-1]
    indices0, indices1 = np.triu_indices(n_sites, k=1)
    subsetted_distances = distance_matrix[:, indices0, indices1]
    return subsetted_distances
