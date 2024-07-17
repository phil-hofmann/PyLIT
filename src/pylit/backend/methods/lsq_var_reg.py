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


def get(omegas: ARRAY, E: ARRAY, lambd: FLOAT_DTYPE = 1.0) -> Method:
    # Type check
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
    method = _standard(omegas, E, lambd)

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


def _standard(omegas, E, lambd) -> Method:
    """Least Squares with Cross Entropy Fitness."""

    n = E.shape[1]
    norm = np.linalg.norm(E) ** 2

    @njit(cache=False, parallel=PARALLEL, fastmath=FASTMATH) # NOTE cache won't work
    def f(x, R, F) -> FLOAT_DTYPE:
        x = x.astype(FLOAT_DTYPE)
        R = R.astype(FLOAT_DTYPE)
        F = F.astype(FLOAT_DTYPE)

        m = E @ x  # Evaluate
        E_omegas = np.mean(omegas * m) # compute expected value with Riemann sum
        V_omegas = np.mean((E_omegas - omegas) ** 2) # compute variance with Riemann sum

        return 0.5 * np.mean((R @ x - F) ** 2) + lambd * 0.5 * V_omegas

    @njit(cache=False, parallel=PARALLEL, fastmath=FASTMATH) # NOTE cache won't work
    def grad_f(x, R, F) -> ARRAY:
        x = x.astype(FLOAT_DTYPE)
        R = R.astype(FLOAT_DTYPE)
        F = F.astype(FLOAT_DTYPE)

        m = E @ x  # Evaluate
        E_omegas = np.mean(omegas * m) # compute expected value with Riemann sum

        # Gradient of the first term
        residual = R @ x - F
        grad_L1 = (R.T @ residual) / len(F)
        
        # Gradient of the second term
        grad_V_omegas = lambd * (E_omegas - omegas) / len(omegas)
        grad_E_omegas = np.mean(omegas * grad_V_omegas)
        grad_m = grad_E_omegas * omegas / len(omegas)
        grad_L2 = E.T @ grad_m
        
        # Total gradient
        grad = grad_L1 + grad_L2
        return grad

    @njit(cache=False, parallel=PARALLEL, fastmath=FASTMATH) # NOTE cache won't work
    def solution(R, F, P):
        # Solution is not available
        return None

    @njit(cache=False, parallel=PARALLEL, fastmath=FASTMATH) # NOTE cache won't work
    def lr(R) -> FLOAT_DTYPE:
        R = R.astype(FLOAT_DTYPE)
        return 1 / (np.linalg.norm(R.T @ R) + lambd * norm)

    return Method("lsq_var_reg_fit", f, grad_f, solution, lr)