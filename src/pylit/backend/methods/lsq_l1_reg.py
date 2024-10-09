import warnings
import numpy as np

from numba import njit
from numba.core.errors import NumbaPerformanceWarning
from pylit.backend.core import Method
from pylit.global_settings import (
    ARRAY,
    FLOAT_DTYPE,
    INT_DTYPE,
    CACHE,
    PARALLEL,
    FASTMATH,
)
from pylit.backend.utils import jit_sub_mat_by_index_set, jit_sub_vec_by_index_set

# Filter out NumbaPerformanceWarning
warnings.simplefilter("ignore", category=NumbaPerformanceWarning)


def get(lambd: FLOAT_DTYPE) -> Method:
    r"""
    # Least Squares with L1 Regularization

    Implements the Least Squares with L1 Regularization with the objective function

    \\[
        f(u, w, \lambda) =
        \frac{1}{2} \| \widehat u - \widehat w\|^2_{L^2(\mathbb{R})} +
        \lambda \| u \|_{L^1(\mathbb{R})}
    \\]

    which is here implemented as

    \\[
        f(\boldsymbol{\alpha}) =
        \frac{1}{2} \frac{1}{n} \| \boldsymbol{R} \boldsymbol{\alpha} - \boldsymbol{F} \|^2_2 +
        \lambda \frac{1}{n} \| \boldsymbol{\alpha} \|_1
    \\]

    with the gradient

    \\[
        \nabla_{\boldsymbol{\alpha}} f(\boldsymbol{\alpha}) =
        \frac{1}{n} \boldsymbol{R}^\top(\boldsymbol{R} \boldsymbol{\alpha} - \boldsymbol{F}) \pm
        \lambda \frac{1}{n}, \quad \boldsymbol{\alpha} \neq 0
    \\]

    with the learning rate

    \\[
        \eta = \frac{n}{\| \boldsymbol{R}^\top \boldsymbol{R} \|}, \quad \boldsymbol{\alpha} \neq 0
    \\]

    and the solution

    \\[
        \boldsymbol{\alpha}^* = (\boldsymbol{R}^\top \boldsymbol{R})^{-1} (\boldsymbol{R}^\top \boldsymbol{F} \pm \lambda), \quad \boldsymbol{\alpha} \neq 0
    \\]

    where

    - **$\boldsymbol{R}$**: Regression matrix
    - **$\boldsymbol{F}$**: Target vector
    - **$\boldsymbol{\alpha}$**: Coefficient vector
    - **$\lambda$**: Regularization parameter
    - **$n$**: Number of samples

    ### Arguments
    - **lambd** (np.float64): Regularization Parameter.

    ### Returns
    - **Method**(Method): Implemented formulation for Least Squares with L1 Regularization.
    """
    # Type Conversion
    lambd = FLOAT_DTYPE(lambd)

    # Get Method
    method = _standard(lambd)

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


def _standard(lambd) -> Method:

    @njit(cache=CACHE, parallel=PARALLEL, fastmath=FASTMATH)
    def f(x, R, F) -> FLOAT_DTYPE:
        x = np.asarray(x).astype(FLOAT_DTYPE)
        R = np.asarray(R).astype(FLOAT_DTYPE)
        F = np.asarray(F).astype(FLOAT_DTYPE)

        return FLOAT_DTYPE(0.5 * np.mean((R @ x - F) ** 2) + lambd * np.mean(x))

    @njit(cache=CACHE, parallel=PARALLEL, fastmath=FASTMATH)
    def grad_f(x, R, F) -> ARRAY:
        x = np.asarray(x).astype(FLOAT_DTYPE)
        R = np.asarray(R).astype(FLOAT_DTYPE)
        F = np.asarray(F).astype(FLOAT_DTYPE)
        n, _ = R.shape

        # Gradient of the first term
        grad_1 = R.T @ (R @ x - F) / n

        # Gradient of the second term
        grad_2 = lambd * np.sign(x) / n

        # Total gradient
        grad = grad_1 + grad_2
        return np.asarray(grad).astype(FLOAT_DTYPE)

    @njit(cache=CACHE, parallel=PARALLEL, fastmath=FASTMATH)
    def solution(R, F, P) -> ARRAY:
        R = np.asarray(R).astype(FLOAT_DTYPE)
        F = np.asarray(F).astype(FLOAT_DTYPE)
        P = np.asarray(P).astype(INT_DTYPE)

        # np.ix_ unsupported in numba

        A = R.T @ R
        A = jit_sub_mat_by_index_set(A, P)

        b = R.T @ F - lambd
        b = jit_sub_vec_by_index_set(b, P)

        return np.asarray(np.linalg.solve(A, b)).astype(FLOAT_DTYPE)

    @njit(cache=CACHE, parallel=PARALLEL, fastmath=FASTMATH)
    def lr(R) -> FLOAT_DTYPE:
        R = np.asarray(R).astype(FLOAT_DTYPE)
        n, _ = R.shape

        return FLOAT_DTYPE(n / np.linalg.norm(R.T @ R))

    return Method("lsq_l1_reg", f, grad_f, solution, lr)
