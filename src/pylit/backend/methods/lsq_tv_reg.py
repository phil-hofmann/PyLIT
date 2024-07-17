import pylit
import numpy as np

from numba import njit
from pylit.backend.methods import Method
from pylit.global_settings import (
    ARRAY,
    FLOAT_DTYPE,
    INT_DTYPE,
    CACHE,
    PARALLEL,
    FASTMATH,
)
from pylit.backend.utils import jit_sub_mat_by_index_set, jit_sub_vec_by_index_set


def get(E: ARRAY, lambd: FLOAT_DTYPE = 1.0, svd: bool = False) -> Method:
    # Type check
    if not isinstance(E, ARRAY):
        raise TypeError("E must be an array.")

    if not isinstance(lambd, FLOAT_DTYPE) and not isinstance(lambd, float):
        raise TypeError("lambd must be a float.")

    if not isinstance(svd, bool):
        raise TypeError("svd must be a bool.")

    # Type Conversion
    E = E.astype(FLOAT_DTYPE)
    lambd = FLOAT_DTYPE(lambd)

    # Compute Total Variation operator
    n = E.shape[1]
    TV = (E[: len(E) - 1] - E[1 : len(E)]) / len(E)

    # Get method
    method = _standard(TV, lambd) if not svd else _svd(TV, lambd)

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
    _ = method.pr(R_) if method.pr is not None else None

    return method


def _standard(TV, lambd) -> Method:
    """Least Squares with Total Variation Regularization."""

    @njit(cache=False, parallel=PARALLEL, fastmath=FASTMATH) # NOTE cache won't work
    def f(x, R, F) -> FLOAT_DTYPE:
        x = x.astype(FLOAT_DTYPE)
        R = R.astype(FLOAT_DTYPE)
        F = F.astype(FLOAT_DTYPE)

        return 0.5 * np.mean((R @ x - F) ** 2) + lambd * 0.5 * np.mean((TV @ x) ** 2)

    @njit(cache=False, parallel=PARALLEL, fastmath=FASTMATH) # NOTE cache won't work
    def grad_f(x, R, F) -> ARRAY:
        x = x.astype(FLOAT_DTYPE)
        R = R.astype(FLOAT_DTYPE)
        F = F.astype(FLOAT_DTYPE)
        n = R.shape[0]
        return (R.T @ (R @ x - F) + lambd * TV.T @ (TV @ x)) / n

    @njit(cache=False, parallel=PARALLEL, fastmath=FASTMATH) # NOTE cache won't work
    def solution(R, F, P):
        R = R.astype(FLOAT_DTYPE)
        F = F.astype(FLOAT_DTYPE)
        P = P.astype(INT_DTYPE)
        # np.ix_ unsupported in numba

        A = R.T @ R + lambd * TV.T @ TV
        A = jit_sub_mat_by_index_set(A, P)

        b = R.T @ F
        b = jit_sub_vec_by_index_set(b, P)

        return np.linalg.solve(A, b)

    @njit(cache=False, parallel=PARALLEL, fastmath=FASTMATH) # NOTE cache won't work
    def lr(R) -> FLOAT_DTYPE:
        R = R.astype(FLOAT_DTYPE)
        n = R.shape[0]
        return n / (np.linalg.norm(R.T @ R) + lambd * np.linalg.norm(TV.T @ TV))

    return Method("lsq_tv_reg", f, grad_f, solution, lr, None)


def _svd(TV, lambd) -> Method:
    """Least Squares with Total Variation Regularization using SVD."""
    pass


if __name__ == "__main__":
    pass
