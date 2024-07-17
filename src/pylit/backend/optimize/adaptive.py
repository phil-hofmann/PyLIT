import numpy as np
import numba as nb

from pylit.backend.core import Solution
from pylit.global_settings import ARRAY, INT_DTYPE, FLOAT_DTYPE


def adaptive_RF(
    R: ARRAY,
    F: ARRAY,
    x0: ARRAY,
    steps: INT_DTYPE,
    optim_RFx0: callable,
    residuum_mode: bool = False,
) -> Solution:
    """Solves the optimization problem adaptively."""

    # Type Conversion
    R = R.astype(FLOAT_DTYPE)
    F = F.astype(FLOAT_DTYPE)
    x0 = x0.astype(FLOAT_DTYPE)

    # Subroutine
    x, eps, residuum = _adaptive_RF(
        R,
        F,
        x0,
        steps,
        optim_RFx0,
        residuum_mode,
    )

    return Solution(x, eps, residuum)


# NOTE The truncation of x0 as initial guess is not optimal ...
@nb.njit
def _adaptive_RF(
    R,
    F,
    x0,
    steps,
    optim_RFx0,
    residuum_mode,
):
    """Solves the optimization problem using an adaptive algorithm.

    Parameters
    ----------
    R : ARRAY
        Regression matrix.
    F: ARRAY
        Target values.
    steps : INT_DTYPE
        The cardinality of the first parameter set.
    optim_RFx0 : callable
        The optimization method.
    residuum_mode : bool
        If True, the residuum is used to determine the best solution. Otherwise, the epsilon is used.

    Returns
    -------
    ARRAY:
        Returns the solution."""

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
        res = optim_RFx0(R=R_, F=F, x0=x0)  # is of type solution
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
    return x, eps, residuum
