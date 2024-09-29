import numpy as np

from numba import njit
from pylit.backend.core import Method
from pylit.global_settings import (
    ARRAY,
    FLOAT_DTYPE,
    INT_DTYPE,
    TOL_LOG,
    PARALLEL,
    FASTMATH,
)


def get(D: ARRAY, E: ARRAY, lambd: FLOAT_DTYPE = 1.0) -> Method:
    # Type Conversion
    D = np.asarray(D).astype(FLOAT_DTYPE)
    E = np.asarray(E).astype(FLOAT_DTYPE)
    lambd = FLOAT_DTYPE(lambd)

    # Get method
    method = _standard(D, E, lambd)

    # Compile
    n = E.shape[1]
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


def _standard(D, E, lambd) -> Method:
    """Least Squares with Cross Entropy Fitness."""

    D = np.clip(D, a_min=0, a_max=None)  # Clip Non-negative
    q = np.copy(D)
    q = np.clip(q, a_min=TOL_LOG, a_max=None)  # Clip Log
    log_q = np.log(q)  # Take Log
    norm = np.linalg.norm(E) ** 2

    @njit(cache=False, parallel=PARALLEL, fastmath=FASTMATH)  # NOTE cache won't work
    def f(x, R, F) -> FLOAT_DTYPE:
        x = x.astype(FLOAT_DTYPE)
        R = R.astype(FLOAT_DTYPE)
        F = F.astype(FLOAT_DTYPE)

        p = E @ x  # Evaluate
        p = np.clip(p, a_min=0, a_max=None)  # Clip Non-negative
        log_p = np.log(p)  # Take Log

        return 0.5 * np.mean((R @ x - F) ** 2) + lambd * np.mean(p * (log_p - log_q))

    @njit(cache=False, parallel=PARALLEL, fastmath=FASTMATH)  # NOTE cache won't work
    def grad_f(x, R, F) -> ARRAY:
        x = x.astype(np.float64)
        R = R.astype(np.float64)
        F = F.astype(np.float64)

        # Number of samples
        n = len(F)

        # Compute p and log_p
        p = E @ x
        p = np.clip(p, a_min=0, a_max=None)
        p_hat = np.clip(p, a_min=TOL_LOG, a_max=None)
        log_p = np.log(p_hat)

        # Gradient of the first term
        residual = R @ x - F
        grad_L1 = (R.T @ residual) / n

        # Gradient of the second term
        grad_L2 = lambd * (E.T @ (log_p - log_q + 1)) / n

        # Total gradient
        grad = grad_L1 + grad_L2
        return grad

    @njit(cache=False, parallel=PARALLEL, fastmath=FASTMATH)  # NOTE cache won't work
    def solution(R, F, P):
        # Solution is not available
        return None

    @njit(cache=False, parallel=PARALLEL, fastmath=FASTMATH)  # NOTE cache won't work
    def lr(R) -> FLOAT_DTYPE:
        R = R.astype(FLOAT_DTYPE)
        return 1 / (np.linalg.norm(R.T @ R) + lambd * norm)

    return Method("lsq_max_entropy_fit", f, grad_f, solution, lr)
