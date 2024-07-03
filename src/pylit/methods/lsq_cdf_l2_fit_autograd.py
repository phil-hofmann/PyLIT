import pylit
# import numpy as np

import autograd.numpy as np  # Thinly-wrapped numpy
from autograd import grad
# from numba import njit
from pylit.methods import Method
from pylit.global_settings import ARRAY, FLOAT_DTYPE, INT_DTYPE, CACHE, PARALLEL, FASTMATH


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

    # Compute Total Variation operator
    # än = E.shape[1]

    # Get method
    method = _standard(S, E, lambd) if not svd else _svd(S, E, lambd)

    return method


def _standard(S, E, lambd) -> Method:
    """Least Squares with L1 Fitness."""

    # @njit(cache=CACHE, parallel=PARALLEL, fastmath=FASTMATH)
    def f(x, R, F) -> FLOAT_DTYPE:
        x = x.astype(FLOAT_DTYPE)
        R = R.astype(FLOAT_DTYPE)
        F = F.astype(FLOAT_DTYPE)

        Ex = E @ x

        return 0.5 * np.sum((R @ x - F) ** 2) + lambd * 0.5 * np.sum(np.cumsum((Ex - S)**2))

    # @njit(cache=CACHE, parallel=PARALLEL, fastmath=FASTMATH)
    def grad_f(x, R, F) -> ARRAY:
        x = x.astype(FLOAT_DTYPE)
        R = R.astype(FLOAT_DTYPE)
        F = F.astype(FLOAT_DTYPE)

        return grad(f, argnum=0)(x, R, F)

    # @njit(cache=CACHE, parallel=PARALLEL, fastmath=FASTMATH)
    def solution(R, F, P):
        # Solution is not available
        return None

    # @njit(cache=CACHE, parallel=PARALLEL, fastmath=FASTMATH)
    def lr(R) -> FLOAT_DTYPE:
        R = R.astype(FLOAT_DTYPE)
        n = R.shape[0]
        return 1 / (
            np.linalg.norm(R.T @ R) + lambd * n**2 * np.linalg.norm(E)**2
        )

    return Method("lsq_l2_fit", f, grad_f, solution, lr, None)


def _svd(S, E, lambd) -> Method:
    """Least Squares with L1 Fitness using SVD."""
    pass


if __name__ == "__main__":

    # S = np.random.rand(10)
    # E = np.random.rand(10, 20) + 1

    # method = get(S, E)

    # method.grad_f(np.zeros(20), np.eye(20), np.zeros(20))

    pass
