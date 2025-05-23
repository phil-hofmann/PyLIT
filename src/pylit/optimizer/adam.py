import numpy as np
import numba as nb

from pylit.njit_utils import svd_optim
from pylit.core.data_classes import Method, Solution
from pylit.settings import FLOAT_DTYPE, INT_DTYPE, TOL


def adam(
    R: np.ndarray,
    F: np.ndarray,
    x0: np.ndarray,
    method: Method,
    maxiter: INT_DTYPE = None,
    tol: FLOAT_DTYPE = None,
    svd: bool = False,
    protocol: bool = False,
) -> Solution:
    """Solves the optimization problem using the ADAM gradient method."""

    # Type Conversion
    R = R.astype(FLOAT_DTYPE)
    F = F.astype(FLOAT_DTYPE)
    x0 = x0.astype(FLOAT_DTYPE)

    n, m = R.shape
    maxiter = 10 * n if maxiter is None else INT_DTYPE(maxiter)
    tol = 10 * max(m, n) * TOL if tol is None else FLOAT_DTYPE(tol)

    # Subroutine
    x = _nn_adam(
        R,
        F,
        method.f,
        method.grad_f,
        method.lr,
        x0,
        maxiter,
        tol,
        protocol,
        svd,
    )
    fx = method.f(x, R, F)

    return Solution(x, fx, FLOAT_DTYPE(0.5 * np.sum((R @ x - F) ** 2)))


@nb.njit
def _nn_adam(R, F, f, grad_f, lr, x0, maxiter, tol, protocol, svd) -> np.ndarray:

    # Initialize variables
    n = len(x0)
    beta1 = 0.9
    beta2 = 0.999
    eps = 1e-8
    m = np.zeros(n)
    v = np.zeros(n)
    x = np.copy(x0)
    fx = 0.0
    lr_ = lr(R)
    V = None
    eps_tol = 0.0

    # Singular Value Decomposition
    if svd:
        R, F, x0, V = svd_optim(R, F, x0)

    # Print header for protocol
    if protocol:
        print("Step", "Error")

    # Subroutine
    for k in range(maxiter):
        grad = grad_f(x, R, F)
        m = beta1 * m + (1 - beta1) * grad
        v = beta2 * v + (1 - beta2) * (grad**2)
        m_hat = m / (1 - beta1 ** (k + 1))
        v_hat = v / (1 - beta2 ** (k + 1))
        x -= lr_ * m_hat / (np.sqrt(v_hat) + eps)

        # Non-negativity constraint
        if V is None:
            # Without projection
            x[x < 0.0] = 0.0

        else:
            # With projection
            x = V @ x
            x[x < 0.0] = 0.0
            x = V.T @ x

        # Check tolerance
        fx1 = f(x, R, F)
        if k > 0 and np.abs(fx - fx1) < tol:
            if protocol:
                print("Converged by tolerance")
            break
        fx = fx1

        # Print protocol
        if protocol:
            print(k + 1, fx1)
            
    return x if V is None else V @ x
