import numpy as np

from numba import njit
from pylit.backend.core import Method
from pylit.global_settings import (
    ARRAY,
    FLOAT_DTYPE,
    INT_DTYPE,
    CACHE,
    PARALLEL,
    FASTMATH,
)
from pylit.backend.utils import jit_sub_mat_by_index_set, jit_sub_vec_by_index_set


def get(lambd: FLOAT_DTYPE = 1.0) -> Method:

    # Type check
    if not isinstance(lambd, FLOAT_DTYPE) and not isinstance(lambd, float):
        raise TypeError("lambd must be a float.")

    # Type Conversion
    lambd = FLOAT_DTYPE(lambd)

    # Get method
    method = _standard(lambd)

    # Compile
    x_, R_, F_, P_ = (
        np.zeros((2), dtype=FLOAT_DTYPE),
        np.eye(2, dtype=FLOAT_DTYPE),
        np.zeros((2), dtype=FLOAT_DTYPE),
        np.array([0], dtype=INT_DTYPE),
    )

    _ = method.f(x_, R_, F_)
    _ = method.grad_f(x_, R_, F_)
    _ = method.solution(R_, F_, P_)
    _ = method.lr(R_)

    return method


def _standard(lambd) -> Method:
    """Least Squares with L1 Regularization."""

    @njit(cache=CACHE, parallel=PARALLEL, fastmath=FASTMATH)
    def f(x, R, F) -> FLOAT_DTYPE:
        x = x.astype(FLOAT_DTYPE)
        R = R.astype(FLOAT_DTYPE)
        F = F.astype(FLOAT_DTYPE)

        return 0.5 * np.mean((R @ x - F) ** 2) + lambd * 0.5 * np.mean(x**2)

    @njit(cache=CACHE, parallel=PARALLEL, fastmath=FASTMATH)
    def grad_f(x, R, F) -> ARRAY:
        x = x.astype(FLOAT_DTYPE)
        R = R.astype(FLOAT_DTYPE)
        F = F.astype(FLOAT_DTYPE)
        n = R.shape[0]
        return (R.T @ (R @ x - F) + lambd * x) / n

    @njit(cache=CACHE, parallel=PARALLEL, fastmath=FASTMATH)
    def solution(R, F, P):
        R = R.astype(FLOAT_DTYPE)
        F = F.astype(FLOAT_DTYPE)
        P = P.astype(INT_DTYPE)

        # np.ix_ unsupported in numba
        m = R.shape[1]

        A = R.T @ R + lambd * np.eye(m)
        A = jit_sub_mat_by_index_set(A, P)

        b = R.T @ F
        b = jit_sub_vec_by_index_set(b, P)

        return np.linalg.solve(A, b)

    @njit(cache=CACHE, parallel=PARALLEL, fastmath=FASTMATH)
    def lr(R) -> FLOAT_DTYPE:
        R = R.astype(FLOAT_DTYPE)
        n, m = R.shape
        return n / (np.linalg.norm(R.T @ R) + lambd * np.linalg.norm(np.eye(m)))

    return Method("lsq_l2_reg", f, grad_f, solution, lr)