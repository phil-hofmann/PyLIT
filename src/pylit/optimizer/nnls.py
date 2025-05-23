import pylit
import numpy as np
import numba as nb

from pylit.njit_utils import argmax
from pylit.core.data_classes import Method, Solution
from pylit.settings import FLOAT_DTYPE, INT_DTYPE, TOL
from pylit.njit_utils import (
    jit_sub_mat_by_index_set,
    jit_sub_vec_by_index_set,
)


def nnls(
    R: np.ndarray,
    F: np.ndarray,
    x0: np.ndarray,
    method: Method,
    maxiter: INT_DTYPE = None,
    tol: FLOAT_DTYPE = None,
    svd: bool = False,  # NOTE No svd
    protocol: bool = False,
) -> Solution:
    """Solves the optimization problem using the non-negative least square method of Bro."""

    # Type Conversion
    R = R.astype(FLOAT_DTYPE)
    F = F.astype(FLOAT_DTYPE)
    x0 = x0.astype(FLOAT_DTYPE)

    # Initialize Variables
    n, m = R.shape
    maxiter = 10 * n if maxiter is None else INT_DTYPE(maxiter)
    tol = 10 * max(m, n) * TOL if tol is None else FLOAT_DTYPE(tol)

    # Subroutine
    x = _nnls(
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
def _nnls(R, F, f, grad_f, solution, x0, maxiter, tol, protocol) -> np.ndarray:
    # Print header for protocol
    if protocol:
        print("Step", "Error")

    # A. Initialization
    n = R.shape[1]
    P = np.zeros(n, dtype=INT_DTYPE)  # A.1, A.2
    x = np.copy(x0)  # A.3
    grad_fx = R.T @ (F - R @ x)  # A.4
    fx = f(x, R, F)

    # B. Main loop
    i = 0
    while (P == 0).any() and (grad_fx[P == 0] > tol).any():  # B.1
        grad_fx[P == 1] = -np.inf  # B.2
        k = argmax(grad_fx)  # B.2
        P[k] = 1  # B.3
        s = np.zeros(n, dtype=np.float64)  # B.4
        s[P == 1] = solution(R, F, P.nonzero()[0])  # B.4

        # C. Inner loop
        while (i < maxiter) and (s[P == 1].min() <= 0.0):  # C.1
            alpha = -(x[P == 1] / (x[P == 1] - s[P == 1])).min()  # C.2
            x = x + alpha * (s - x)  # C.3
            P[x <= tol] = 0  # C.4
            s[P == 1] = solution(R, F, P.nonzero()[0])  # C.5 # TODO CHECK SOLUTIONS
            s[P == 0] = 0  # C.6
            i += 1

            fx1 = f(x, R, F)
            if np.abs(fx - fx1) < tol:
                if protocol:
                    print("Converged by tolerance.")
                i = np.inf
            fx = fx1

        x[:] = s[:]  # B.5
        grad_fx = -grad_f(x, R, F)  # B.6

        if i == maxiter:
            break

        if protocol:
            print(int(i + 1), fx1)

    # x[x < 0.0] = 0.0 # NOTE Uncomment if needed
    return x


if __name__ == "__main__":
    pass
