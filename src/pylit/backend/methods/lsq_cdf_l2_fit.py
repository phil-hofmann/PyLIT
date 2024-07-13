import pylit
import numpy as np

from numba import njit
from pylit.backend.methods import Method
from pylit.global_settings import (
    ARRAY,
    FLOAT_DTYPE,
    INT_DTYPE,
    CACHE,
    PARALLEL,
    FASTMATH,
)


def get(D: ARRAY, E: ARRAY, lambd: FLOAT_DTYPE = 1.0, svd: bool = False) -> Method:
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
    D : ARRAY
        Default Model.
    E : ARRAY
        Evaluation Matrix.
    lambd : FLOAT_DTYPE, optional
        Parameter. The default is 1.0.
    svd : bool, optional
        Wether SVD should be used or not. The default is False.

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
    if not isinstance(D, ARRAY):
        raise TypeError("S must be an array.")

    if not isinstance(E, ARRAY):
        raise TypeError("E must be an array.")

    if not isinstance(lambd, FLOAT_DTYPE) and not isinstance(lambd, float):
        raise TypeError("lambd must be a float.")

    if not isinstance(svd, bool):
        raise TypeError("svd must be a bool.")

    # Type Conversion
    E = E.astype(FLOAT_DTYPE)
    lambd = FLOAT_DTYPE(lambd)

    # Compute Total Variation operator
    n = E.shape[1]

    # Get method
    method = _standard(D, E, lambd) if not svd else _svd(D, E, lambd)

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
    _ = method.pr(R_) if method.pr is not None else None

    return method


def _standard(D, E, lambd) -> Method:
    """Implements the wasserstein fitness"""

    @njit(cache=CACHE, parallel=PARALLEL, fastmath=FASTMATH)
    def f(alpha, R, F) -> FLOAT_DTYPE:
        alpha = alpha.astype(FLOAT_DTYPE)
        R = R.astype(FLOAT_DTYPE)
        F = F.astype(FLOAT_DTYPE)

        E_alpha = E @ alpha

        return 0.5 * np.sum((R @ alpha - F) ** 2) + lambd * 0.5 * np.sum(
            np.cumsum((E_alpha - D) ** 2)
        )

    @njit(cache=CACHE, parallel=PARALLEL, fastmath=FASTMATH)
    def grad_f(alpha, R, F) -> ARRAY:
        alpha = alpha.astype(FLOAT_DTYPE)
        R = R.astype(FLOAT_DTYPE)
        F = F.astype(FLOAT_DTYPE)

        E_alpha = E @ alpha

        return R.T @ (R @ alpha - F) + lambd * np.sum(
            np.dot(np.cumsum(E_alpha - D), np.tril(np.sum(E, axis=1)))
        )

    @njit(cache=CACHE, parallel=PARALLEL, fastmath=FASTMATH)
    def solution(R, F, P):
        # Solution is not available
        return None

    @njit(cache=CACHE, parallel=PARALLEL, fastmath=FASTMATH)
    def lr(R) -> FLOAT_DTYPE:
        R = R.astype(FLOAT_DTYPE)
        n = R.shape[0]
        return 1 / (np.linalg.norm(R.T @ R) + lambd * n**2 * np.linalg.norm(E) ** 2)

    return Method("lsq_l2_fit", f, grad_f, solution, lr, None)


def _svd(S, E, lambd) -> Method:
    """Least Squares with L1 Fitness using SVD."""
    pass


if __name__ == "__main__":

    # S = np.random.rand(10)
    # E = np.random.rand(10, 20) + 1

    # method = get(S, E)

    # method.grad_f(np.zeros(20), np.eye(20), np.zeros(20))

    pass
