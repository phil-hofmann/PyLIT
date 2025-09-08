import numpy as np
import numba as nb

from pylit.njit_utils import argmax
from pylit.core.data_classes import Method, Solution
from pylit.settings import FLOAT_DTYPE, INT_DTYPE, TOL


def nnls(
    R: np.ndarray,
    F: np.ndarray,
    x0: np.ndarray,
    method: Method,
    maxiter: INT_DTYPE = None,
    tol: FLOAT_DTYPE = None,
    svd: bool = False,  # NOTE No svd available
    protocol: bool = False,
) -> Solution:
    r"""
    This is the NNLS optimization method. 
    `The interface is described in` :ref:`Optimizer <optimizer>`.

    Description
    -----------
    Solves the optimization problem :eq:`(*) <lsq-problem>` using the active-set
    popularized by Bro et al. The algorithm maintains a partition of indices into an
    *active set* (variables fixed at zero) and a *passive set* (variables allowed to vary).
    At each step, the active set is updated according to the gradient, and a least-squares
    problem is solved on the passive set. This ensures feasibility with respect to the
    non-negativity constraint throughout the iterations.

    Algorithm
    ---------
    The procedure follows these labeled steps (corresponding to the implementation):

    **A. Initialization**
        - **A.1**   Initialize active set :math:`P = \mathbf{0}`.
        - **A.2**   Passive set empty.
        - **A.3**   Set initial guess :math:`x = x_0`.
        - **A.4**   Compute gradient :math:`\nabla f(x) = R^T (F - R x)`.

    **B. Main Loop**
        - **B.1**   While there exists an inactive variable with positive gradient,
        - **B.2**   Compute the index with maximum gradient.
        - **B.3**   Move this index to the active set.
        - **B.4**   Solve the least-squares subproblem on the active set.

    **C. Inner Loop**
        - **C.1**   If the solution has negative components,
        - **C.2**   Compute step length :math:`\alpha` to the nearest feasible point.
        - **C.3**   Update iterate :math:`x \leftarrow x + \alpha (s - x)`.
        - **C.4**   Remove variables with absolute value below the tolerance from the active set.
        - **C.5**   Recompute least-squares solution on active set.
        - **C.6**   Set inactive variables to zero.

    The algorithm terminates when all free variables satisfy the Karush–Kuhn–Tucker
    (KKT) conditions or when the maximum number of iterations ``maxiter`` is reached.

    Returns
    -------
    Solution
        A Solution object containing the final iterate.

    Notes
    -----
        - This implementation does **not** support SVD preconditioning.

    References
    ----------
        - Rasmus Bro, Sijmen De Jong. *A Fast Non-Negativity-Constrained Least Squares Algorithm*. 
        Journal of Chemometrics, Vol. 11, pp. 393–401, 1997.
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

    # Compute objective
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
            s[P == 1] = solution(R, F, P.nonzero()[0])  # C.5
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
