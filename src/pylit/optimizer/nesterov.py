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
    r"""
    This is the Nesterov optimization method. 
    `The interface is described in` :ref:`Optimizer <optimizer>`.

    Description
    -----------
    Solves the optimization problem :eq:`(*) <lsq-problem>` using the Nesterov
    accelerated gradient method with a non-negativity constraint. The algorithm 
    maintains two sequences of iterates, a main sequence :math:`x_k` and an auxiliary 
    lookahead sequence :math:`y_k`. At each iteration, the lookahead point 
    is computed by combining the current and previous iterates with a 
    momentum parameter. A projected gradient descent step is then performed 
    from this lookahead point. Projection enforces feasibility with respect 
    to the non-negativity constraint. The lookahead mechanism reduces 
    oscillations and accelerates convergence, achieving the optimal 
    :math:`\mathcal{O}(1/k^2)` rate for first-order methods.

    Algorithm
    ---------
    Given a momentum parameter :math:`0 < \theta_k < 1` and learning rate 
    :math:`\eta > 0`, the updates are:

    .. math::

        \begin{align*}
            \boldsymbol{\beta}_0 &= \boldsymbol{\alpha}_0 \in \mathbb{R}^n, & & \text{(initial guess)} \\
            \boldsymbol{\beta}_k &= \max(0, \boldsymbol{\alpha}_k + \theta_k (\boldsymbol{\alpha}_k - \boldsymbol{\alpha}_{k-1})), 
                & & \text{(non-negative momentum step)} \\
            \boldsymbol{\alpha}_{k+1} &= \boldsymbol{\beta}_k - \eta \nabla f(\boldsymbol{\beta}_k), & & \text{(lookahead gradient step)}
        \end{align*}

    The algorithm terminates when the change in the objective function between
    successive iterates falls below the tolerance ``tol`` or when the maximum
    number of iterations ``maxiter`` is reached.
    
    Returns
    -------
    Solution
        A Solution object containing the final iterate.

    References
    ----------
        - Y. Nesterov. *Gradient methods for minimizing composite functions*. Mathematical Programming, 140(1), 125–161, 2012.  
    """

    # Convert types
    R = R.astype(FLOAT_DTYPE)
    F = F.astype(FLOAT_DTYPE)
    x0 = x0.astype(FLOAT_DTYPE)

    # Handle defaults
    n, m = R.shape
    maxiter = 10 * n if maxiter is None else INT_DTYPE(maxiter)
    tol = 10 * max(m, n) * TOL if tol is None else FLOAT_DTYPE(tol)

    # Call subroutine
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

    # Compute objective
    fx = method.f(x, R, F)

    return Solution(x, fx, FLOAT_DTYPE(0.5 * np.sum((R @ x - F) ** 2)))


@nb.njit
def _nesterov(R, F, f, grad_f, x0, lr, maxiter, tol, protocol, svd) -> np.ndarray:
    # Initialize variables
    n = x0.shape[0]
    y = np.zeros(n)
    x = np.copy(x0)
    # v = np.copy(x0)
    x_last = np.copy(x0)
    # theta = 1.0
    theta = 0.0
    fy = 0.0
    lr_ = lr(R)
    V = None

    # Perform svd
    if svd:
        R, F, x0, V = svd_optim(R, F, x0)

    # Print header
    if protocol:
        print("Step", "Error")

    # Subroutine:
    for k in range(maxiter):
        # Update momentum
        theta = k / (k + 3)

        # Update the solution
        y = x + theta * (x - x_last)

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

        x_last = x.copy()
        x = y - lr_ * grad_fy

        # Print protocol
        if protocol:
            print(k + 1, fy1)

    return y if V is None else V @ y
