import warnings
import numpy as np

from numba import njit
from numba.core.errors import NumbaPerformanceWarning
from pylit.core.data_classes import Method
from pylit.settings import (
    FLOAT_DTYPE,
    INT_DTYPE,
    FASTMATH,
)
from pylit.njit_utils import (
    jit_sub_mat_by_index_set,
    jit_sub_vec_by_index_set,
)

# Filter out NumbaPerformanceWarning
warnings.simplefilter("ignore", category=NumbaPerformanceWarning)


def l2_reg(lambd: FLOAT_DTYPE) -> Method:
    r"""
    # Least Squares with L2 Regularization

    Implements the Least Squares with L2 Regularization with the objective function

    \\[
        f(u, w, \lambda) =
        \frac{1}{2} \| \widehat u - \widehat w\|^2_{L^2(\mathbb{R})} +
        \frac{1}{2} \lambda \| u \|_{L^2(\mathbb{R})}^2
    \\]

    which is here implemented as

    \\[
        f(\boldsymbol{\alpha}) =
        \frac{1}{2} \frac{1}{n} \| \boldsymbol{R} \boldsymbol{\alpha} - \boldsymbol{F} \|^2_2 +
        \frac{1}{2} \lambda \frac{1}{n} \| \boldsymbol{\alpha} \|^2_2
    \\]

    with the gradient

    \\[
        \nabla_{\boldsymbol{\alpha}} f(\boldsymbol{\alpha}) =
        \frac{1}{n} \boldsymbol{R}^\top(\boldsymbol{R} \boldsymbol{\alpha} - \boldsymbol{F}) +
        \lambda \frac{1}{n} \boldsymbol{\alpha}
    \\]

    with the learning rate

    \\[
        \eta = \frac{n}{\| \boldsymbol{R}^\top \boldsymbol{R} + \lambda \boldsymbol{I} \|}
    \\]

    and the solution

    \\[
        \boldsymbol{\alpha}^* = (\boldsymbol{R}^\top \boldsymbol{R} + \lambda \boldsymbol{I})^{-1} \boldsymbol{R}^\top \boldsymbol{F}
    \\]

    where

    - **$\boldsymbol{R}$**: Regression matrix
    - **$\boldsymbol{F}$**: Target vector
    - **$\boldsymbol{I}$**: Identity matrix
    - **$\boldsymbol{\alpha}$**: Coefficient vector
    - **$\lambda$**: Regularization parameter
    - **$n$**: Number of samples

    # Arguments
    - **lambd**(np.float64): Regularization parameter.

    # Returns
    - **Method**(Method): Least Squares with L2 Regularization.
    """

    # Type Conversion
    lambd = FLOAT_DTYPE(lambd)

    # Get Method
    method = _l2_reg(lambd)

    # Compile
    x_, R_, F_, P_ = (
        np.zeros((2), dtype=FLOAT_DTYPE),
        np.eye(2, dtype=FLOAT_DTYPE),
        np.zeros((2), dtype=FLOAT_DTYPE),
        np.array([0], dtype=INT_DTYPE),
    )

    _ = method.f(x_, R_, F_)
    _ = method.grad_f(x_, R_, F_)
    _ = method.solution(R_, F_, P_)
    _ = method.lr(R_)

    return method


def _l2_reg(lambd) -> Method:

    @njit(fastmath=FASTMATH)
    def f(x, R, F) -> FLOAT_DTYPE:
        x = x.astype(FLOAT_DTYPE)
        R = R.astype(FLOAT_DTYPE)
        F = F.astype(FLOAT_DTYPE)

        return FLOAT_DTYPE(
            0.5 * np.mean((R @ x - F) ** 2) + lambd * 0.5 * np.mean(x**2)
        )

    @njit(fastmath=FASTMATH)
    def grad_f(x, R, F) -> np.ndarray:
        x = np.asarray(x).astype(FLOAT_DTYPE)
        R = np.asarray(R).astype(FLOAT_DTYPE)
        F = np.asarray(F).astype(FLOAT_DTYPE)
        n, _ = R.shape

        # Gradient of the first term
        grad_1 = R.T @ (R @ x - F) / n

        # Gradient of the second term
        grad_2 = lambd * x / n

        # Total gradient
        grad = grad_1 + grad_2
        return np.asarray(grad).astype(FLOAT_DTYPE)

    @njit(fastmath=FASTMATH)
    def solution(R, F, P) -> np.ndarray:
        R = np.asarray(R).astype(FLOAT_DTYPE)
        F = np.asarray(F).astype(FLOAT_DTYPE)
        P = np.asarray(P).astype(INT_DTYPE)
        _, m = R.shape

        # np.ix_ unsupported in numba

        A = R.T @ R + lambd * np.eye(m)
        A = jit_sub_mat_by_index_set(A, P)

        b = R.T @ F
        b = jit_sub_vec_by_index_set(b, P)

        return np.asarray(np.linalg.solve(A, b)).astype(FLOAT_DTYPE)

    @njit(fastmath=FASTMATH)
    def lr(R) -> FLOAT_DTYPE:
        R = np.asarray(R).astype(FLOAT_DTYPE)
        n, m = R.shape

        return FLOAT_DTYPE(
            n / (np.linalg.norm(R.T @ R) + lambd * np.linalg.norm(np.eye(m)))
        )

    return Method("l2_reg", f, grad_f, solution, lr)
