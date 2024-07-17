import numpy as np

from numba import njit
from pylit.backend.core import Method
from pylit.global_settings import (
    ARRAY,
    FLOAT_DTYPE,
    INT_DTYPE,
    PARALLEL,
    FASTMATH,
)
from pylit.backend.utils import jit_sub_mat_by_index_set, jit_sub_vec_by_index_set


def get(S: ARRAY, E: ARRAY, lambd: FLOAT_DTYPE = 1.0) -> Method:
    # Type check
    if not isinstance(S, ARRAY):
        raise TypeError("S must be an array.")

    if not isinstance(E, ARRAY):
        raise TypeError("E must be an array.")

    if not isinstance(lambd, FLOAT_DTYPE) and not isinstance(lambd, float):
        raise TypeError("lambd must be a float.")

    # Type Conversion
    E = E.astype(FLOAT_DTYPE)
    lambd = FLOAT_DTYPE(lambd)

    # Compute Total Variation operator
    n = E.shape[1]

    # Get method
    method = _standard(S, E, lambd)

    # Compile
    x_, R_, F_, P_ = (
        np.zeros((n), dtype=FLOAT_DTYPE),
        np.eye(n, dtype=FLOAT_DTYPE),
        np.zeros((n), dtype=FLOAT_DTYPE),
        np.array([0], dtype=INT_DTYPE),
    )

    _ = method.f(x_, R_, F_)
    _ = method.grad_f(x_, R_, F_)
    _ = method.solution(R_, F_, P_)
    _ = method.lr(R_)

    return method


def _standard(S, E, lambd) -> Method:
    """Least Squares with L1 Fitness."""

    @njit(cache=False, parallel=PARALLEL, fastmath=FASTMATH) # NOTE cache won't work
    def f(x, R, F) -> FLOAT_DTYPE:
        x = x.astype(FLOAT_DTYPE)
        R = R.astype(FLOAT_DTYPE)
        F = F.astype(FLOAT_DTYPE)

        return 0.5 * np.mean((R @ x - F) ** 2) + lambd * 0.5 * np.mean((E @ x - S) ** 2)

    @njit(cache=False, parallel=PARALLEL, fastmath=FASTMATH) # NOTE cache won't work
    def grad_f(x, R, F) -> ARRAY:
        x = x.astype(FLOAT_DTYPE)
        R = R.astype(FLOAT_DTYPE)
        F = F.astype(FLOAT_DTYPE)
        n = R.shape[0]
        return (R.T @ (R @ x - F) + lambd * E.T @ (E @ x - S)) / n

    @njit(cache=False, parallel=PARALLEL, fastmath=FASTMATH) # NOTE cache won't work
    def solution(R, F, P):
        R = R.astype(FLOAT_DTYPE)
        F = F.astype(FLOAT_DTYPE)
        P = P.astype(INT_DTYPE)

        # np.ix_ unsupported in numba

        A = R.T @ R + lambd * E.T @ E
        A = jit_sub_mat_by_index_set(A, P)

        b = R.T @ F
        b = jit_sub_vec_by_index_set(b, P)

        return np.linalg.solve(A, b)

    @njit(cache=False, parallel=PARALLEL, fastmath=FASTMATH) # NOTE cache won't work
    def lr(R) -> FLOAT_DTYPE:
        R = R.astype(FLOAT_DTYPE)
        n = R.shape[0]
        return n / (np.linalg.norm(R.T @ R) + lambd * np.linalg.norm(E.T @ E))

    return Method("lsq_l2_fit", f, grad_f, solution, lr)