import numpy as np
from numba import njit
from typing import Tuple


@njit
def svd_optim(
    R: np.ndarray, F: np.ndarray, x0: np.ndarray
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    r"""
    Perform a singular value decomposition (SVD) of matrix R
    and transform F and x0 into the SVD coordinate system.

    Args:
        R:
            A 2D square matrix to decompose (shape: (n, n)).
        F:
            A 2D array to be transformed using the left singular vectors of R (shape: (n, m)).
        x0:
            A 1D array to be transformed using the right singular vectors of R (shape: (n,)).

    Returns
    -------
        S: (np.ndarray)
            A diagonal matrix of singular values (shape: (n, n)).
        F_prime: (np.ndarray)
            The transformed version of F in the SVD coordinate system (shape: (n, m)).
        x0_prime: (np.ndarray)
            The transformed version of x0 in the SVD coordinate system (shape: (n,)).
        V: (np.ndarray)
            The right singular vector matrix (shape: (n, n)).

    Notes:
        - This function uses the decomposition :math:`R = U S V^\top` where :math:`U` and :math:`V` are orthogonal matrices, and S is diagonal with singular values.
    """
    U, S_diag, VT = np.linalg.svd(R)
    S = np.zeros_like(R)
    np.fill_diagonal(S, S_diag)
    V, UT = VT.T, U.T
    F_prime = np.copy(UT @ F)
    x0_prime = VT @ x0
    return S, F_prime, x0_prime, V


@njit
def argmax(array: np.ndarray) -> int:
    """
    Find the index of the maximum value in a 1D array.

    Args:
        array:
            A 1D array of numeric values.

    Returns:
        The index of the maximum value in the array.

    Notes:
        - Equivalent to `np.argmax(array)`, but implemented manually
          for use in Numba-compiled functions.
        - Assumes the array has at least one element.
    """
    max_value = array[0]
    max_index = 0
    for i in range(1, len(array)):
        if array[i] > max_value:
            max_value = array[i]
            max_index = i
    return max_index
