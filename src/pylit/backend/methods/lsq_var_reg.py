import warnings
import numpy as np

from numba import njit
from numba.core.errors import NumbaPerformanceWarning
from pylit.backend.core import Method
from pylit.global_settings import (
    ARRAY,
    FLOAT_DTYPE,
    INT_DTYPE,
    PARALLEL,
    FASTMATH,
)

# Filter out NumbaPerformanceWarning
warnings.simplefilter("ignore", category=NumbaPerformanceWarning)


def get(omegas: ARRAY, E: ARRAY, lambd: FLOAT_DTYPE) -> Method:
    # Type Conversion
    omegas = np.asarray(omegas).astype(FLOAT_DTYPE)
    E = np.asarray(E).astype(FLOAT_DTYPE)
    lambd = FLOAT_DTYPE(lambd)

    # Get method
    method = _standard(omegas, E, lambd)

    # Compile
    _, m = E.shape
    x_, R_, F_, P_ = (
        np.zeros((m), dtype=FLOAT_DTYPE),
        np.eye(m, dtype=FLOAT_DTYPE),
        np.zeros((m), dtype=FLOAT_DTYPE),
        np.array([0], dtype=INT_DTYPE),
    )

    _ = method.f(x_, R_, F_)
    _ = method.grad_f(x_, R_, F_)
    _ = method.solution(R_, F_, P_)
    _ = method.lr(R_)

    return method


def _standard(omegas, E, lambd) -> Method:
    """Least Squares with Cross Entropy Fitness."""

    @njit(cache=False, parallel=PARALLEL, fastmath=FASTMATH)  # NOTE cache won't work
    def f(x, R, F) -> FLOAT_DTYPE:
        x = np.asarray(x).astype(FLOAT_DTYPE)
        R = np.asarray(R).astype(FLOAT_DTYPE)
        F = np.asarray(F).astype(FLOAT_DTYPE)
        _, m = R.shape
        E_ = E[:, :m]

        p = E_ @ x
        E_p = np.mean(p * omegas)
        V_p = np.mean((E_p - omegas) ** 2)

        return FLOAT_DTYPE(0.5 * np.mean((R @ x - F) ** 2) + lambd * 0.5 * V_p)

    @njit(cache=False, parallel=PARALLEL, fastmath=FASTMATH)  # NOTE cache won't work
    def grad_f(x, R, F) -> ARRAY:
        x = np.asarray(x).astype(FLOAT_DTYPE)
        R = np.asarray(R).astype(FLOAT_DTYPE)
        F = np.asarray(F).astype(FLOAT_DTYPE)
        n, m = R.shape
        E_ = E[:, :m]

        p = E_ @ x
        E_p = np.mean(p * omegas)
        omegas_mean = np.mean(omegas)

        return np.asarray(
            R.T @ (R @ x - F) / n + lambd * (E_p - omegas_mean) * E_.T @ omegas
        ).astype(FLOAT_DTYPE)

    @njit(cache=False, parallel=PARALLEL, fastmath=FASTMATH)  # NOTE cache won't work
    def solution(R, F, P) -> ARRAY:
        # TODO Solution is not available >yet<.
        return None

    @njit(cache=False, parallel=PARALLEL, fastmath=FASTMATH)  # NOTE cache won't work
    def lr(R) -> FLOAT_DTYPE:
        R = np.asarray(R).astype(FLOAT_DTYPE)
        n, m = R.shape
        k = len(omegas)
        E_ = E[:, :m]

        norm = (
            np.linalg.norm(E_.T @ omegas)
            * np.linalg.norm(E_)
            * np.linalg.norm(omegas)
            / k**2
        )

        return FLOAT_DTYPE(n / (np.linalg.norm(R.T @ R) + lambd * n * norm))

    return Method("lsq_var_reg_fit", f, grad_f, solution, lr)
