import numpy as np
import numba as nb
from pylit.backend.core import Solution
from pylit.global_settings import FLOAT_DTYPE, INT_DTYPE, ARRAY, TOL


def nn_nesterov(
    R: ARRAY,
    F: ARRAY,
    x0: ARRAY,
    method: callable,
    maxiter: INT_DTYPE = None,
    tol: FLOAT_DTYPE = None,
    protocol: bool = False,
) -> Solution:
    """Solves the optimization problem using the Nesterov's accelerated gradient method.

    - The method converges if the objective function is convex and the projection is onto a linear subspace with the right choice for the learning rate.
    - The learning rate should be smaller than the inverse of the Lipschitz constant of the gradient of the objective function.

    Parameters
    ----------
    grad_f : callable
        Gradient of the objective function.
    x0 : ARRAY
        Initial guess. A one-dimensional array.
    lr : FLOAT_DTYPE
        Learning rate. A positive float.
    maxiter : INT_DTYPE
        Maximum number of iterations.
    tol : FLOAT_DTYPE
        Tolerance. A value between 0 and 1.
    pr : ARRAY
        Projection function. Ideally, the projection should be onto a linear subspace. Otherwise, the method is not guaranteed to converge.

    Returns
    -------
    ARRAY:
        Returns solution."""

    # Type Conversion
    R = R.astype(FLOAT_DTYPE)
    F = F.astype(FLOAT_DTYPE)
    x0 = x0.astype(FLOAT_DTYPE)

    n, m = R.shape
    maxiter = 10 * n if maxiter is None else INT_DTYPE(maxiter)
    tol = 10 * max(m, n) * TOL if tol is None else FLOAT_DTYPE(tol)

    # Call subroutine
    x = _nn_nesterov_subroutine(
        R,
        F,
        method.f,
        method.grad_f,
        x0,
        method.lr,
        maxiter,
        tol,
        method.pr,
        protocol,
    )
    fx = method.f(x, R, F)

    # Return solution
    return Solution(x, fx, FLOAT_DTYPE(0.5 * np.sum((R @ x - F) ** 2)))


# @nb.njit # TODO uncomment
def _nn_nesterov_subroutine(
    R, F, f, grad_f, x0, lr, maxiter, tol, pr, protocol
) -> ARRAY:

    # Initialize variables:
    n = x0.shape[0]
    y = np.zeros(n)
    x = np.copy(x0)
    v = np.copy(x0)

    theta = 1.0
    fy = 0.0
    lr_ = lr(R)

    # Subroutine implementation:
    for k in range(maxiter):
        # Update momentum
        theta = 2 / (k + 2)

        # Update the solution
        y = (1 - theta) * x + theta * v

        # Non-negativity constraint
        if pr is None:
            # Without projection
            y[y < 0.0] = 0.0

        else:
            # With projection
            y_pr = pr(y)
            y[y_pr < 0.0] = 0.0

        # Check tolerance
        # (Checking for the gradient is not useful since the gradient could be still large, due to the projection onto ">=0",
        #  even if the solution is close to the minimum of the objective function.)
        print(y.shape, R.shape, F.shape)
        fy1 = f(y, R, F)
        if k > 0 and np.abs(fy - fy1) < tol:
            break
        fy = fy1

        # Update x and v
        grad_fy = np.copy(grad_f(y, R, F)).astype(FLOAT_DTYPE)
        x1 = y - lr_ * grad_fy
        v = x + (1.0 / theta) * (x1 - x)
        x = x1

        # Print protocol
        if protocol:
            print("step: " + str(k + 1) + " of " + str(maxiter))

    return y if pr is None else pr(y)


if __name__ == "__main__":
    pass
