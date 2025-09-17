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
    r"""
    This is the ADAM optimization method. 
    `The interface is described in` :ref:`Optimizer <optimizer>`.

    Description
    -----------
    Solves the optimization problem :eq:`(*) <lsq-problem>` using the ADAM 
    (A Method for Stochastic Optimization) gradient method with non-negativity constraint.
    ADAM is a first-order stochastic optimization algorithm that adapts the learning
    rates for each variable individually by maintaining exponentially decaying averages
    of past gradients (first moment) and squared gradients (second moment). This 
    stabilizes convergence and is well-suited for problems with noisy, sparse, or 
    non-stationary gradients.

    Algorithm
    ---------
    Let :math:`\beta_1` and :math:`\beta_2` be the decay rates for the first and
    second moment estimates, and :math:`\epsilon` a small constant for numerical
    stability. At iteration :math:`k`, the updates are:

    .. math::

        m_k &= \beta_1 m_{k-1} + (1-\beta_1) \nabla f(x_{k-1}) \\
        v_k &= \beta_2 v_{k-1} + (1-\beta_2) (\nabla f(x_{k-1}))^2 \\
        \hat{m}_k &= \frac{m_k}{1 - \beta_1^k} \\
        \hat{v}_k &= \frac{v_k}{1 - \beta_2^k} \\
        x_k &= x_{k-1} - \eta \frac{\hat{m}_k}{\sqrt{\hat{v}_k} + \epsilon}
    
    The algorithm terminates when the change in the objective function between
    successive iterates falls below the tolerance ``tol`` or when the maximum
    number of iterations ``maxiter`` is reached.

    Returns:
        A Solution object containing the final iterate.

    References:
        - D. P. Kingma and J. Ba. *Adam: A Method for Stochastic Optimization*. International Conference on Learning Representations (ICLR), 2015.
    """

    # Convert types
    R = R.astype(FLOAT_DTYPE)
    F = F.astype(FLOAT_DTYPE)
    x0 = x0.astype(FLOAT_DTYPE)

    # Handle defaults
    n, m = R.shape
    maxiter = 10 * n if maxiter is None else INT_DTYPE(maxiter)
    tol = 10 * max(m, n) * TOL if tol is None else FLOAT_DTYPE(tol)

    # Call Subroutine
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

    # Compute objective
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

    # Perform svd
    if svd:
        R, F, x0, V = svd_optim(R, F, x0)

    # Print header
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
