import numpy as np
from pylit.global_settings import ARRAY, INT_DTYPE

# @nb.njit
def adaptive_RF(
    R: ARRAY, F: ARRAY, n1: INT_DTYPE, optim_RF: callable
) -> ARRAY:
    """Solves the optimization problem using an adaptive algorithm.
    
    Parameters
    ----------
    R : ARRAY
        Regression matrix.
    F: ARRAY
        Target values.
    n1 : INT_DTYPE
        The cardinality of the first parameter set.
    R_method : callable
        The optimization method which is used to solve the subproblems and only takes the regression matrix as input.

    Returns
    -------
    ARRAY:
        Returns the solution."""

    if not n1 > 0:
        raise ValueError("The number must be positive.")

    n = R.shape[1]

    if n % n1 > 0:
        raise ValueError("The number of columns of R must be divisible by n1.")

    N = INT_DTYPE(n / n1)
    c, eps, features = None, np.inf, np.array([], dtype=INT_DTYPE)

    for i in range(N):
        features_ = np.concatenate([features, np.arange(i * n1, (i + 1) * n1)])
        R_ = R[:, features_]
        res_ = optim_RF(R_, F)
        c_, eps_ = res_["c"], res_["eps"]

        if eps_ < eps:
            c = np.copy(c_)
            eps = eps_
            features = np.copy(features_)

    # Create the final solution
    c_final = np.zeros(n)

    # Insert the values of c_brute at the correct indices
    c_final[features] = c

    # Return the solution
    return {"c": c_final, "eps": eps, "features": features}

if __name__ == "__main__":
    pass