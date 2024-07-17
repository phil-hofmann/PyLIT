import numpy as np

from numba import njit
from pylit.backend.core import Method
from pylit.global_settings import (
    ARRAY,
    FLOAT_DTYPE,
    INT_DTYPE,
    PARALLEL,
    FASTMATH,
)


def get(S: ARRAY, E: ARRAY, lambd: FLOAT_DTYPE = 1.0) -> Method:
    """
    Implements the wasserstein fitness. With the objectiv function

    .. math::
        f_{\psi}(u,w,\lambda) = \\frac{1}{2} \| \widehat u- \widehat w\|^2_{L^2(\mathbb{R})}
        + \\frac{\\lambda}{2} \cdot \| \mathrm{CDF}[u - w] \|_{L^2(\mathbb{R})}^2.

    which is here implemented as

    .. math::
        f(\\boldsymbol{\\alpha}) = \\frac{1}{2} \| \\boldsymbol{R} \\boldsymbol{\\alpha} - \\boldsymbol{F} \|^2_2
        + \lambda \cdot \\left( \sum_{\omega_k \in \Omega} \\frac{1}{2}
                               \\left(
                                   \sum_{i \in I} (\\boldsymbol{E} \\boldsymbol{\\alpha})_i  - \\boldsymbol{D}_i
                                   \\right)^2
                               \\right),

    with the gradient

    .. math::
        \\nabla f(\\boldsymbol{\\alpha}) = \\boldsymbol{G}^\\top(\\boldsymbol{G} \\boldsymbol{\\alpha} - \\boldsymbol{F})
        + \lambda \cdot \\left( \sum_{\omega_k \in \Omega} \\frac{1}{2}
                               \\left(
                                   \sum_{i \in I} (\\boldsymbol{E} \\boldsymbol{\\alpha})_i  - \\boldsymbol{D}_i
                                   \\right) \\boldsymbol{E}^\\top \\boldsymbol{1}
                               \\right),

    where

    * :math:`\\boldsymbol{R}`: regression matrix
    * :math:`\\boldsymbol{E}`: evaluation matrix
    * :math:`\\boldsymbol{D}`: default model
    * :math:`\\boldsymbol{\\alpha}`: desired coefficients

    Parameters
    ----------
    S : ARRAY
        Default Model.
    E : ARRAY
        Evaluation Matrix.
    lambd : FLOAT_DTYPE, optional
        Parameter. The default is 1.0.

    Raises
    ------
    TypeError
        If input is incorrect.

    Returns
    -------
    Method
        Implemented formulation for wasserstein fitness.

    """
    # Type check
    if not isinstance(S, ARRAY):
        raise TypeError("S must be an array.")

    if not isinstance(E, ARRAY):
        raise TypeError("E must be an array.")

    if not isinstance(lambd, FLOAT_DTYPE) and not isinstance(lambd, float):
        raise TypeError("lambd must be a float.")

    # Type Conversion
    E = E.astype(FLOAT_DTYPE)
    lambd = FLOAT_DTYPE(lambd)

    # Compute Total Variation operator
    n = E.shape[1]

    # Get method
    method = _standard(S, E, lambd)

    # Compile
    alpha_, R_, F_, P_ = (
        np.zeros((n), dtype=FLOAT_DTYPE),
        np.eye(n, dtype=FLOAT_DTYPE),
        np.zeros((n), dtype=FLOAT_DTYPE),
        np.array([0], dtype=INT_DTYPE),
    )

    _ = method.f(alpha_, R_, F_)
    _ = method.grad_f(alpha_, R_, F_)
    _ = method.solution(R_, F_, P_)
    _ = method.lr(R_)

    return method


def _standard(S, E, lambd) -> Method:
    """Implements the wasserstein fitness"""

    @njit(cache=False, parallel=PARALLEL, fastmath=FASTMATH) # NOTE cache won't work
    def f(x, R, F) -> FLOAT_DTYPE:
        x = x.astype(FLOAT_DTYPE)
        R = R.astype(FLOAT_DTYPE)
        F = F.astype(FLOAT_DTYPE)

        p = E @ x
        n_vec = np.arange(1, len(p) + 1)
        
        return 0.5 * np.mean((R @ x - F) ** 2) + lambd * 0.5 * np.mean(
            np.cumsum((p - S) ** 2) / n_vec
        )

    @njit(cache=False, parallel=PARALLEL, fastmath=FASTMATH) # NOTE cache won't work
    def grad_f(x, R, F) -> ARRAY:
        x = x.astype(FLOAT_DTYPE)
        R = R.astype(FLOAT_DTYPE)
        F = F.astype(FLOAT_DTYPE)

        p = E @ x
        n_vec = np.arange(1, len(p) + 1)

        # Gradient of the first term
        residual = R @ x - F
        grad_L1 = (R.T @ residual) / len(F)

        # Gradient of the second term
        cumsum_diff = np.cumsum(p - S)
        grad_L2 = lambd * (E.T @ (cumsum_diff / n_vec)) / len(F)

        # Total gradient
        grad = grad_L1 + grad_L2
        return grad

    @njit(cache=False, parallel=PARALLEL, fastmath=FASTMATH) # NOTE cache won't work
    def solution(R, F, P):
        # Solution is not available
        return None

    @njit(cache=False, parallel=PARALLEL, fastmath=FASTMATH) # NOTE cache won't work
    def lr(R) -> FLOAT_DTYPE:
        R = R.astype(FLOAT_DTYPE)
        n = R.shape[0] # TODO put n below?? TEST !
        return 1 / (np.linalg.norm(R.T @ R) + lambd * np.linalg.norm(E) ** 2)

    return Method("lsq_cdf_l2_fit", f, grad_f, solution, lr)