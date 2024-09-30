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


def get(D: ARRAY, E: ARRAY, lambd: FLOAT_DTYPE) -> Method:
    r"""
    # Least Squares Cumulative Distribution Function L2 Fit

    Implements the Wasserstein fitness with the objective function

    \\[
        f(u,w,\lambda) = 
        \frac{1}{2} \| \widehat u - \widehat w\|^2_{L^2(\mathbb{R})} + 
        \frac{1}{2} \lambda \| \mathrm{CDF}[u - w] \|_{L^2(\mathbb{R})}^2
    \\]

    which is here implemented as

    \\[
        f(\boldsymbol{\alpha}) = 
        \frac{1}{2} \frac{1}{n} \| \boldsymbol{R} \boldsymbol{\alpha} - \boldsymbol{F} \|^2_2 + 
        \frac{1}{2} \lambda \left( \frac{1}{n} \sum_{j=1}^n \frac{1}{j} \sum_{i=1}^j(\boldsymbol{E} \boldsymbol{\alpha} - \boldsymbol{D})_i^2 \right)
    \\]

    with the gradient

    \\[
        \nabla_{\boldsymbol{\alpha}} f(\boldsymbol{\alpha}) = 
        \frac{1}{n} \boldsymbol{R}^\top(\boldsymbol{R} \boldsymbol{\alpha} - \boldsymbol{F}) +
        \lambda \frac{1}{n} \boldsymbol{E}^\top \left( \frac{1}{j} \sum_{i=1}^j(\boldsymbol{E} \boldsymbol{\alpha} - \boldsymbol{D})_i \right)_j
    \\]
    
    where

    - **$\boldsymbol{R}$**: Regression matrix
    - **$\boldsymbol{E}$**: Evaluation matrix
    - **$\boldsymbol{D}$**: Default model
    - **$\boldsymbol{\alpha}$**: Desired coefficients
    - **$\lambda$**: Regularization parameter
    - **$n$**: Number of samples

    ### Parameters
    - **D** (np.ndarray): Default Model.
    - **E** (np.ndarray): Evaluation Matrix.
    - **lambd** (np.float64, optional): Regularization Parameter.

    ### Returns
    - **Method**: Implemented formulation for Wasserstein fitness.
    """

    # Type Conversion
    D = np.asarray(D).astype(FLOAT_DTYPE)
    E = np.asarray(E).astype(FLOAT_DTYPE)
    lambd = FLOAT_DTYPE(lambd)

    # Get Method
    method = _standard(D, E, lambd)

    # Compile
    n = E.shape[1]
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


def _standard(D, E, lambd) -> Method:

    @njit(cache=False, parallel=PARALLEL, fastmath=FASTMATH)  # NOTE cache won't work
    def f(x, R, F) -> FLOAT_DTYPE:
        x = np.asarray(x).astype(FLOAT_DTYPE)
        R = np.asarray(R).astype(FLOAT_DTYPE)
        F = np.asarray(F).astype(FLOAT_DTYPE)
        n = len(F)

        return FLOAT_DTYPE(0.5 * np.mean((R @ x - F) ** 2) + 0.5 * lambd * np.mean(
            np.cumsum((E @ x - D) ** 2) / n
        ))

    @njit(cache=False, parallel=PARALLEL, fastmath=FASTMATH)  # NOTE cache won't work
    def grad_f(x, R, F) -> ARRAY:
        x = np.asarray(x).astype(FLOAT_DTYPE)
        R = np.asarray(R).astype(FLOAT_DTYPE)
        F = np.asarray(F).astype(FLOAT_DTYPE)
        n, _ = R.shape
        k = len(D)

        # Gradient of the first term
        grad_1 = R.T @ (R @ x - F) / n

        # Gradient of the second term
        grad_2 = lambd * E.T @ ((np.arange(k, 0, -1) / n**2) * (E @ x - D))

        # Total gradient
        grad = grad_1 + grad_2

        return np.asarray(grad).astype(FLOAT_DTYPE)

    @njit(cache=False, parallel=PARALLEL, fastmath=FASTMATH)  # NOTE cache won't work
    def solution(R, F, P):
        # Solution is not available
        return None

    @njit(cache=False, parallel=PARALLEL, fastmath=FASTMATH)  # NOTE cache won't work
    def lr(R) -> FLOAT_DTYPE:
        R = np.asarray(R).astype(FLOAT_DTYPE)
        n, _ = R.shape
        k = len(D)
        return FLOAT_DTYPE(n / np.linalg.norm(R.T @ R + lambd * E.T @ np.diag(np.arange(k, 0, -1) / n) @ E))

    return Method("lsq_cdf_l2_fit", f, grad_f, solution, lr)
