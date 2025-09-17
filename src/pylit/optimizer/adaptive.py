import numpy as np

from pylit.core.data_classes import Solution
from pylit.settings import INT_DTYPE, FLOAT_DTYPE


def adaptiveRF(
    R: np.ndarray,
    F: np.ndarray,
    x0: np.ndarray,
    steps: INT_DTYPE,
    optim_RFx0: callable,
    residuum_mode: bool = False,
) -> Solution:
    r"""
    This is a wrapper for optimization methods. 

    Description
    -----------
    Solves the optimization problem :eq:`(*) <lsq-problem>` using an adaptive
    incremental wrapper around a given optimization method. This method partitions the 
    parameter space into blocks of size ``steps`` and applies the provided optimizer
    ``optim_RFx0`` sequentially to increasingly larger subsets of features.
    After each block, the solution is updated based on either the residuum or 
    the objective error (epsilon). This ensures that the optimization focuses
    on the most relevant features first and improves convergence second.

    Algorithm
    ---------
    Let :math:`n` be the number of columns of :math:`R` and ``steps`` the block
    size. The algorithm proceeds as follows:

    **A. Initialization**
        - Verify that ``steps`` divides the number of columns of :math:`R`.
        - Initialize empty feature set and solution placeholders.

    **B. Iterative Block Optimization**
        - For each block of ``steps`` features:
            - Append the new block to the current feature set.
            - Run the optimizer ``optim_RFx0`` on the reduced regression problem.
            - Evaluate the solution using either the residuum or epsilon.
            - If the new solution improves the selected metric, update the
              current solution and feature set.

    **C. Final Assembly**
        - Construct the final solution vector by inserting optimized values at
          their corresponding indices.

    The procedure terminates when all feature blocks have been processed. The
    final solution corresponds to the block configuration yielding the minimal
    residuum or epsilon, depending on ``residuum_mode``.

    Args:
        R:
            Regression matrix of shape ``(m, n)``.
        F:
            Target vector of shape ``(m,)``.
        x0:
            Initial guess for the solution, shape ``(n,)``.
        steps:
            Block size for adaptive optimization. Must be positive and divide ``n``.
        optim_RFx0:
            Optimization function with signature ``optim_RFx0(R, F, x0)`` returning 
            a ``Solution`` object.
        residuum_mode:
            If ``True``, the selection of the best solution is based on residuum.
            Otherwise, the epsilon value is used. Default is ``False``.

    Returns:
        A ``Solution`` object containing the final iterate, epsilon, and residuum.

    Notes:
        - This adaptive wrapper is particularly useful for high-dimensional
          regression problems where a full optimization may be inefficient.
        - The truncation of ``x0`` as initial guess for blocks may not be optimal.
    """

    # Type Conversion
    R = R.astype(FLOAT_DTYPE)
    F = F.astype(FLOAT_DTYPE)
    x0 = x0.astype(FLOAT_DTYPE)

    # Subroutine
    x, eps, residuum = _adaptiveRF(
        R,
        F,
        x0,
        steps,
        optim_RFx0,
        residuum_mode,
    )

    return Solution(x, eps, residuum)


# NOTE The truncation of x0 as initial guess is not optimal ...
def _adaptiveRF(
    R,
    F,
    x0,
    steps,
    optim_RFx0,
    residuum_mode,
):
    if not steps > 0:
        raise ValueError("The number must be positive.")

    n = R.shape[1]

    if n % steps > 0:
        raise ValueError("The number of columns of R must be divisible by steps.")

    if steps >= n:
        return optim_RFx0(R, F, x0)

    N = INT_DTYPE(np.ceil(n / steps))
    x, eps, residuum, features = None, np.inf, np.inf, np.array([], dtype=INT_DTYPE)

    for i in range(N):
        lower = i * steps
        upper = np.min([(i + 1) * steps, n])
        add_features = np.arange(lower, upper)
        next_features = np.concatenate([features, add_features])
        R_ = R[:, next_features]
        x0_ = x0[next_features]
        res = optim_RFx0(_R=R_, _F=F, _x0=x0_)  # is of type solution
        x_, eps_, residuum_ = res.x, res.eps, res.residuum

        adapt = False

        if residuum_mode:
            adapt = residuum_ < residuum
        else:
            adapt = eps_ < eps

        if adapt:
            x = np.copy(x_)
            eps = eps_
            features = np.copy(next_features)

    # Create the final solution
    x_final = np.zeros(n)

    # Insert the values of c_brute at the correct indices
    x_final[features] = x

    # Return the solution
    return x_final, eps, residuum
