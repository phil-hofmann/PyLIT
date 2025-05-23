import numpy as np
import numba as nb

from pylit.settings import FLOAT_DTYPE, INT_DTYPE, TOL
from pylit.njit_utils import svd_optim
from pylit.core.data_classes import Method, Solution


def nesterov(
    R: np.ndarray,
    F: np.ndarray,
    x0: np.ndarray,
    method: Method,
    maxiter: INT_DTYPE = None,
    tol: FLOAT_DTYPE = None,
    svd: bool = False,
    protocol: bool = False,
) -> Solution:
    """Solves the optimization problem using the Nesterov's accelerated gradient method.

    - The method converges if the objective function is convex and the projection is onto a linear subspace with the right choice for the learning rate.
    - The learning rate should be smaller than the inverse of the Lipschitz constant of the gradient of the objective function.

    Args:
        grad_f : callable
            Gradient of the objective function.
        x0 : np.ndarray
            Initial guess. A one-dimensional array.
        lr : FLOAT_DTYPE
            Learning rate. A positive float.
        maxiter : INT_DTYPE
            Maximum number of iterations.
        tol : FLOAT_DTYPE
            Tolerance. A value between 0 and 1.
        pr : np.ndarray
            Projection function. Ideally, the projection should be onto a linear subspace. Otherwise, the method is not guaranteed to converge.

    Returns:
        np.ndarray:
            Returns solution.
    """

    # Type Conversion
    R = R.astype(FLOAT_DTYPE)
    F = F.astype(FLOAT_DTYPE)
    x0 = x0.astype(FLOAT_DTYPE)

    # Variables
    n, m = R.shape
    maxiter = 10 * n if maxiter is None else INT_DTYPE(maxiter)
    tol = 10 * max(m, n) * TOL if tol is None else FLOAT_DTYPE(tol)  # TODO check this

    # Subroutine
    x = _nesterov(
        R,
        F,
        method.f,
        method.grad_f,
        x0,
        method.lr,
        maxiter,
        tol,
        protocol,
        svd,
    )
    fx = method.f(x, R, F)

    return Solution(x, fx, FLOAT_DTYPE(0.5 * np.sum((R @ x - F) ** 2)))


@nb.njit
def _nesterov(R, F, f, grad_f, x0, lr, maxiter, tol, protocol, svd) -> np.ndarray:

    # Variables
    n = x0.shape[0]
    y = np.zeros(n)
    x = np.copy(x0)
    v = np.copy(x0)
    theta = 1.0
    fy = 0.0
    lr_ = lr(R)
    V = None

    # Singular Value Decomposition
    if svd:
        R, F, x0, V = svd_optim(R, F, x0)

    # Print header for protocol
    if protocol:
        print("Step", "Error")

    # Subroutine:
    for k in range(maxiter):
        # Update momentum
        theta = 2 / (k + 2)

        # Update the solution
        y = (1 - theta) * x + theta * v

        # Non-negativity constraint
        if V is None:
            # Without projection
            y[y < 0.0] = 0.0

        else:
            # With projection
            y = V @ y
            y[y < 0.0] = 0.0
            y = V.T @ y

        # Check tolerance
        # (Checking for the gradient is not useful since the gradient could be still large, due to the projection onto ">=0",
        #  even if the solution is close to the minimum of the objective function.)
        fy1 = f(y, R, F)
        if k > 0 and np.abs(fy - fy1) < tol:
            if protocol:
                print("Converged by tolerance")
            break
        fy = fy1

        # Update x and v
        grad_fy = np.copy(grad_f(y, R, F)).astype(FLOAT_DTYPE)
        x1 = y - lr_ * grad_fy
        v = x + (1.0 / theta) * (x1 - x)
        x = x1

        # Print protocol
        if protocol:
            print(k + 1, fy1)

    return y if V is None else V @ y
