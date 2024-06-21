import numpy as np
import scipy

DEFAULT_NUM_EIGENPAIRS: int = 1 + 1


def diffusion_maps(points: np.ndarray, epsilon2: float,
                   num_eigenpairs: int = DEFAULT_NUM_EIGENPAIRS,
                   metric: str = 'sqeuclidean') \
        -> Tuple[np.ndarray, np.ndarray]:
    """Compute diffusion maps.
    Simple implementation of diffusion maps.
    Parameters
    ----------
    points : np.ndarray
        Array of points in the data set. The first index addresses each
        individual point.
    epsilon2 : float
        Squared bandwidth squared of the Gaussian kernel.
    num_eigenpairs : int
        Number of eigenpairs to obtain. The diffusion map coordinates will be
        one less than that value because the first eigenvector is constant.
    metric : str
        Metric to use for the construction of the kernel. The default is the
        Euclidean distance but since we use it in its squared form, we save
        some operations by directly specifying 'sqeuclidean' as default. See
        documentation for scipy.spatial.pdist for more.
    Returns
    -------
    ew : np.ndarray
        Array of all eigenvalues, except the first, sorted in descending
        absolute value.
    ev : np.ndarray
        Array of eigenvectors, except the first, of the random walk Laplacian
        addressed by second index.
    """
    distances2 = scipy.spatial.distance.pdist(points, metric=metric)

    kernel_matrix = scipy.spatial.distance.squareform(
        np.exp(-distances2 / (2.0 * epsilon2)))
    kernel_matrix[np.diag_indices_from(kernel_matrix)] = 1.0

    inv_sqrt_diag_vector = 1.0 / np.sqrt(np.sum(kernel_matrix, axis=0))
    normalized_kernel_matrix = ((kernel_matrix * inv_sqrt_diag_vector).T
                                * inv_sqrt_diag_vector).T

    ew, ev = scipy.sparse.linalg.eigsh(normalized_kernel_matrix,
                                       k=num_eigenpairs,
                                       v0=np.ones(kernel_matrix.shape[0]))
    indices = np.argsort(np.abs(ew))[::-1]
    ew = ew[indices]
    ev = ev[:, indices] * inv_sqrt_diag_vector[:, np.newaxis]

    # assert np.allclose((np.diag(1.0 / kernel_matrix.sum(axis=1))
    #                     @ kernel_matrix @ ev),
    #                    ev @ np.diag(ew))  # Assertion only for testing.
    return ew[1:], ev[:, 1:]