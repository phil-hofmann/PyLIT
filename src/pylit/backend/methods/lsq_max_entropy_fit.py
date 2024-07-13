import pylit
import numpy as np

from numba import njit
from pylit.backend.methods import Method
from pylit.global_settings import (
    ARRAY,
    FLOAT_DTYPE,
    INT_DTYPE,
    TOL_LOG,
    CACHE,
    PARALLEL,
    FASTMATH,
)


def get(S: ARRAY, E: ARRAY, lambd: FLOAT_DTYPE = 1.0, svd: bool = False) -> Method:
    # Type check
    if not isinstance(S, ARRAY):
        raise TypeError("S must be an array.")

    if not isinstance(E, ARRAY):
        raise TypeError("E must be an array.")

    if not isinstance(lambd, FLOAT_DTYPE) and not isinstance(lambd, float):
        raise TypeError("lambd must be a float.")

    if not isinstance(svd, bool):
        raise TypeError("svd must be a bool.")

    # Type Conversion
    E = E.astype(FLOAT_DTYPE)
    lambd = FLOAT_DTYPE(lambd)

    n = E.shape[1]

    # Get method
    method = _standard(S, E, lambd) if not svd else _svd(S, E, lambd)

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


def _standard(S, E, lambd) -> Method:
    """Least Squares with Cross Entropy Fitness."""

    S = np.clip(S, a_min=0, a_max=None)  # Clip Non-negative
    q = S / np.sum(S)  # Normalize
    q_hat = np.clip(q, a_min=TOL_LOG, a_max=None)  # Clip Log
    log_q = np.log(q_hat)  # Take Log

    n = E.shape[1]
    norm = n**2 * np.linalg.norm(E) ** 2

    @njit(cache=CACHE, parallel=PARALLEL, fastmath=FASTMATH)
    def f(x, R, F) -> FLOAT_DTYPE:
        x = x.astype(FLOAT_DTYPE)
        R = R.astype(FLOAT_DTYPE)
        F = F.astype(FLOAT_DTYPE)

        m = E @ x  # Evaluate
        m = np.clip(m, a_min=0, a_max=None)  # Clip Non-negative
        p = m / np.sum(m)  # Normalize to one
        p_hat = np.clip(p, a_min=TOL_LOG, a_max=None)  # Clip Log
        log_p = np.log(p_hat)  # Take Log

        return 0.5 * np.sum((R @ x - F) ** 2) + lambd * np.sum(p * (log_p - log_q))

    @njit(cache=CACHE, parallel=PARALLEL, fastmath=FASTMATH)
    def grad_f(x, R, F) -> ARRAY:
        x = x.astype(FLOAT_DTYPE)
        R = R.astype(FLOAT_DTYPE)
        F = F.astype(FLOAT_DTYPE)

        m = E @ x  # Evaluate
        m = np.clip(m, a_min=0, a_max=None)  # Clip Non-negative
        p = m / np.sum(m)  # Normalize to one
        p_hat = np.clip(p, a_min=TOL_LOG, a_max=None)  # Clip Log
        log_p = np.log(p_hat)  # Take Log
        J = (
            E.T * np.sum(m) - np.sum(E.T, axis=1).reshape(-1, 1) @ m.reshape(1, -1)
        ) / np.sum(m) ** 2

        return R.T @ (R @ x - F) + lambd * J @ (log_p - log_q + 1)

    @njit(cache=CACHE, parallel=PARALLEL, fastmath=FASTMATH)
    def solution(R, F, P):
        # Solution is not available
        return None

    @njit(cache=CACHE, parallel=PARALLEL, fastmath=FASTMATH)
    def lr(R) -> FLOAT_DTYPE:
        R = R.astype(FLOAT_DTYPE)
        return 1 / (np.linalg.norm(R.T @ R) + lambd * norm)

    return Method("lsq_max_entropy_fit", f, grad_f, solution, lr, None)


def _svd(S, E, lambd) -> Method:
    """Least Squares with Cross Entropy Fitness using SVD."""
    pass


if __name__ == "__main__":

    # Test

    # @njit
    # def func():

    #     E = np.random.rand(10, 10) + 1
    #     x = np.random.rand(10) + 1

    #     m = E @ x # Evaluate
    #     print(m)
    #     m = np.clip(m, a_min=0, a_max=None) # Clip Non-negative
    #     print(m)
    #     p = m / np.sum(m)  # Normalize to one
    #     p_hat = np.clip(p, a_min=TOL_LOG, a_max=None) # Clip Log
    #     log_p = np.log(p_hat) # Take Log

    #     print(log_p)

    # func()

    _, _, grad, _, _, _ = _standard(np.random.rand(10), np.random.rand(10, 10), 1.0)
    print(grad(np.random.rand(10), np.random.rand(10, 10), np.random.rand(10)))
