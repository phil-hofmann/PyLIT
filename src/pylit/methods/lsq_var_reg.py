import pylit
import numpy as np

from numba import njit
from pylit.methods import Method
from pylit.global_settings import ARRAY, FLOAT_DTYPE, INT_DTYPE, TOL_LOG, CACHE, PARALLEL, FASTMATH


def get(omegas: ARRAY, E: ARRAY, lambd: FLOAT_DTYPE=1.0, svd: bool = False) -> Method:
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

    # Get method
    method = _standard(omegas, E, lambd) if not svd else _svd(omegas, E, lambd)

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


def _standard(omegas, E, lambd) -> Method:
    """Least Squares with Cross Entropy Fitness."""

    n = E.shape[1]
    norm = n**2 * np.linalg.norm(E)** 2

    @njit(cache=CACHE, parallel=PARALLEL, fastmath=FASTMATH)
    def f(x, R, F) -> FLOAT_DTYPE:
        x = x.astype(FLOAT_DTYPE)
        R = R.astype(FLOAT_DTYPE)
        F = F.astype(FLOAT_DTYPE)

        m = E @ x # Evaluate
        m = np.clip(m, a_min=0, a_max=None) # Clip Non-negative
        p = m / np.sum(m)  # Normalize to one
        E_p_omegas = np.sum(omegas * p)
        V_p_omegas = np.sum((E_p_omegas-omegas)**2)

        return 0.5 * np.sum((R @ x - F) ** 2) + lambd * 0.5 * V_p_omegas

    @njit(cache=CACHE, parallel=PARALLEL, fastmath=FASTMATH)
    def grad_f(x, R, F) -> ARRAY:
        x = x.astype(FLOAT_DTYPE)
        R = R.astype(FLOAT_DTYPE)
        F = F.astype(FLOAT_DTYPE)

        m = E @ x # Evaluate
        m = np.clip(m, a_min=0, a_max=None) # Clip Non-negative
        p = m / np.sum(m)  # Normalize to one
        E_p_omegas = np.sum(omegas * p)
        J = (
            E.T * np.sum(m) - np.sum(E.T, axis=1).reshape(-1, 1) @ m.reshape(1, -1)
        ) / np.sum(m) ** 2

        return R.T @ (R @ x - F) + lambd * (E_p_omegas - omegas) @ J.T

    @njit(cache=CACHE, parallel=PARALLEL, fastmath=FASTMATH)
    def solution(R, F, P):
        # Solution is not available
        return None

    @njit(cache=CACHE, parallel=PARALLEL, fastmath=FASTMATH)
    def lr(R) -> FLOAT_DTYPE:
        R = R.astype(FLOAT_DTYPE)
        return 1 / (
            np.linalg.norm(R.T @ R) + lambd * norm
        )

    return Method("lsq_max_entropy_fit", f, grad_f, solution, lr, None)


def _svd(omegas, S, E, lambd) -> Method:
    """Least Squares with Cross Entropy Fitness using SVD."""
    pass


if __name__ == "__main__":
    
    pass