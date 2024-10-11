import pylit
import numpy as np
import numba as nb

from pylit.backend.utils import argmax
from pylit.backend.core import Method, Solution
from pylit.global_settings import FLOAT_DTYPE, INT_DTYPE, ARRAY, TOL

# TODO FIX IMPLEMENTATION!


def nn_bro(
    R: ARRAY,
    F: ARRAY,
    x0: ARRAY,
    method: Method,
    maxiter: INT_DTYPE = None,
    tol: FLOAT_DTYPE = None,
    svd: bool = False,  # TODO add SVD
    protocol: bool = False,
) -> Solution:
    """Solves the optimization problem using the non-negative least square method of Bro."""
    # Integrity
    if not R.shape[0] == len(F):
        raise ValueError("The number of rows of R and the length of F must be equal")

    # Type Conversion
    R = R.astype(FLOAT_DTYPE)
    F = F.astype(FLOAT_DTYPE)
    x0 = x0.astype(FLOAT_DTYPE)

    # Prepare
    n, m = R.shape
    maxiter = 10 * n if maxiter is None else INT_DTYPE(maxiter)
    tol = 10 * max(m, n) * TOL if tol is None else FLOAT_DTYPE(tol)

    # Subroutine
    x = _nn_bro(
        R,
        F,
        method.f,
        method.grad_f,
        method.solution,
        x0,
        maxiter,
        tol,
        protocol,
    )
    fx = method.f(x, R, F)

    return Solution(x, fx, FLOAT_DTYPE(0.5 * np.sum((R @ x - F) ** 2)))


@nb.njit
def _nn_bro(R, F, f, grad_f, solution, x0, maxiter, tol, protocol) -> ARRAY:

    # Initialize variables:
    n = R.shape[1]
    x = np.copy(x0)
    P = np.zeros(n, dtype=INT_DTYPE)

    # Subroutine implementation:

    # Compute gradient
    grad_fx = -grad_f(x, R, F)
    fx = f(x, R, F)

    i = 0
    while (not (P > 0).all()) and (grad_fx[1 - P > 0] > tol).any():
        # Get the "most" active coeff index and move to inactive set
        grad_fx[P > 0] = -np.inf
        k = argmax(grad_fx)  # B.2
        P[k] = 1  # B.3

        # Iteration solution
        s = np.zeros(n, dtype=np.float64)
        s[P > 0] = solution(R, F, (P > 0).nonzero()[0])

        # Inner loop # TODO: Include the projection operator pr
        while (i < maxiter) and (s[P > 0].min() < 0):  # C.1 # NOTE  ... <= tol
            alpha_ind = (
                (s < 0) & (P > 0)  # NOTE s < tol
            ).nonzero()  # INTRODUCE pr AS PROJECTION OPERATOR !?!
            alpha = (x[alpha_ind] / (x[alpha_ind] - s[alpha_ind])).min()  # C.2
            x *= 1 - alpha
            x += alpha * s
            P[x <= tol] = 0  # NOTE x < tol
            s[P > 0] = solution(R, F, (P > 0).nonzero()[0])
            s[1 - P > 0] = 0  # C.6
            i += 1

            fx1 = f(x, R, F)
            if np.abs(fx - fx1) < tol:
                i = np.inf
            fx = fx1

        x[:] = s[:]  # automatically copies ?!?
        grad_fx = -grad_f(x, R, F)

        if i == maxiter:
            break

    return x


if __name__ == "__main__":
    pass
