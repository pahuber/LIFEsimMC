import numpy as np


def cov_inv_sqrt(cov_matrix):
    # Eigen decomposition
    eigvals, eigvecs = np.linalg.eigh(cov_matrix)

    # Compute inverse square root of the eigenvalues
    inv_sqrt_eigvals = np.diag(1.0 / np.sqrt(eigvals))

    # Reconstruct the inverse square root matrix
    cov_inv_sqrt_matrix = eigvecs @ inv_sqrt_eigvals @ eigvecs.T

    return cov_inv_sqrt_matrix
