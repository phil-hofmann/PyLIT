import numpy as np
from numba import njit


@njit
def jit_sub_mat_by_index_set(mat: np.ndarray, I: np.ndarray) -> np.ndarray:
    n = len(I)
    sub = np.zeros((n, n), dtype=mat.dtype)

    if len(mat.shape) == 2:
        for i_, i in enumerate(I):
            for j_, j in enumerate(I):
                sub[i_][j_] = mat[i][j]

    else:
        raise NotImplementedError("Unsupported number of dimensions for mat.")

    return sub


@njit
def jit_sub_vec_by_index_set(vec: np.ndarray, I: np.ndarray) -> np.ndarray:
    n = len(I)
    sub = np.zeros(n, dtype=vec.dtype)

    if len(vec.shape) == 1:
        for i_, i in enumerate(I):
            sub[i_] = vec[i]

    else:
        raise NotImplementedError("Unsupported number of dimensions for vec.")

    return sub


@njit
def svd_optim(R: np.ndarray, F: np.ndarray, x0: np.ndarray):
    U, S_diag, VT = np.linalg.svd(R)
    S = np.zeros_like(R)
    np.fill_diagonal(S, S_diag)
    V, UT = VT.T, U.T
    F_prime = np.copy(UT @ F)
    x0_prime = VT @ x0
    return S, F_prime, x0_prime, V


@njit
def argmax(array):
    max_value = array[0]
    max_index = 0
    for i in range(1, len(array)):
        if array[i] > max_value:
            max_value = array[i]
            max_index = i
    return max_index
