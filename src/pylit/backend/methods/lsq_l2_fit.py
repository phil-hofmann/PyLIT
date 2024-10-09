import warnings
import numpy as np

from numba import njit
from numba.core.errors import NumbaPerformanceWarning
from pylit.backend.core import Method
from pylit.global_settings import (
    ARRAY,
    FLOAT_DTYPE,
    INT_DTYPE,
    PARALLEL,
    FASTMATH,
)
from pylit.backend.utils import jit_sub_mat_by_index_set, jit_sub_vec_by_index_set

# Filter out NumbaPerformanceWarning
warnings.simplefilter("ignore", category=NumbaPerformanceWarning)


def get(D: ARRAY, E: ARRAY, lambd: FLOAT_DTYPE) -> Method:
    r"""
    # Least Squares with L2 Fitness

    Implements the Least Squares with L2 Fitness with the objective function

    \\[
        f(u, w, \lambda) =
        \frac{1}{2} \| \widehat u - \widehat w\|^2_{L^2(\mathbb{R})} +
        \frac{1}{2} \lambda \| u - w \|_{L^2(\mathbb{R})}^2
    \\]

    which is here implemented as

    \\[
        f(\boldsymbol{\alpha}) =
        \frac{1}{2} \frac{1}{n} \| \boldsymbol{R} \boldsymbol{\alpha} - \boldsymbol{F} \|^2_2 +
        \frac{1}{2} \lambda \frac{1}{n} \| \boldsymbol{E} \boldsymbol{\alpha} - \boldsymbol{D} \|^2_2
    \\]

    with the gradient

    \\[
        \nabla_{\boldsymbol{\alpha}} f(\boldsymbol{\alpha}) =
        \frac{1}{n} \boldsymbol{R}^\top(\boldsymbol{R} \boldsymbol{\alpha} - \boldsymbol{F}) +
        \lambda \frac{1}{n} \boldsymbol{E}^\top(\boldsymbol{E} \boldsymbol{\alpha} - \boldsymbol{D})
    \\]

    with the learning rate

    \\[
        \eta = \frac{n}{\| \boldsymbol{R}^\top \boldsymbol{R} + \lambda \boldsymbol{E}^\top \boldsymbol{E} \|}
    \\]

    and the solution

    \\[
        \boldsymbol{\alpha}^* = (\boldsymbol{R}^\top \boldsymbol{R} + \lambda \boldsymbol{E}^\top \boldsymbol{E})^{-1} (\boldsymbol{R}^\top \boldsymbol{F} + \lambda \boldsymbol{E}^\top \boldsymbol{D})
    \\]

    where

    - **$\boldsymbol{R}$**: Regression matrix
    - **$\boldsymbol{F}$**: Target vector
    - **$\boldsymbol{E}$**: Evaluation matrix
    - **$\boldsymbol{D}$**: Default model vector
    - **$\boldsymbol{\alpha}$**: Coefficient vector
    - **$\lambda$**: Regularization parameter
    - **$n$**: Number of samples

    ### Arguments
    - **D** (np.ndarray): Default model vector.
    - **E** (np.ndarray): Evaluation matrix.
    - **lambd** (np.float64): Regularization parameter.

    ### Returns
    - **Method**(Method): Implemented formulation for Least Squares with L2 Fitness.
    """
    # Type Conversion
    D = np.asarray(D).astype(FLOAT_DTYPE)
    E = np.asarray(E).astype(FLOAT_DTYPE)
    lambd = FLOAT_DTYPE(lambd)

    # Get Method
    method = _standard(D, E, lambd)

    # Compile
    _, m = E.shape
    x_, R_, F_, P_ = (
        np.zeros((m), dtype=FLOAT_DTYPE),
        np.eye(m, dtype=FLOAT_DTYPE),
        np.zeros((m), dtype=FLOAT_DTYPE),
        np.array([0], dtype=INT_DTYPE),
    )

    _ = method.f(x_, R_, F_)
    _ = method.grad_f(x_, R_, F_)
    _ = method.solution(R_, F_, P_)
    _ = method.lr(R_)

    return method


def _standard(D, E, lambd) -> Method:

    @njit(cache=False, parallel=PARALLEL, fastmath=FASTMATH)  # NOTE cache won't work
    def f(x, R, F) -> FLOAT_DTYPE:
        x = x.astype(FLOAT_DTYPE)
        R = R.astype(FLOAT_DTYPE)
        F = F.astype(FLOAT_DTYPE)
        _, m = R.shape
        E_ = E[:, :m]

        return 0.5 * np.mean((R @ x - F) ** 2) + lambd * 0.5 * np.mean(
            (E_ @ x - D) ** 2
        )

    @njit(cache=False, parallel=PARALLEL, fastmath=FASTMATH)  # NOTE cache won't work
    def grad_f(x, R, F) -> ARRAY:
        x = x.astype(FLOAT_DTYPE)
        R = R.astype(FLOAT_DTYPE)
        F = F.astype(FLOAT_DTYPE)
        n, m = R.shape
        E_ = E[:, :m]

        # Gradient of the first term
        grad_1 = R.T @ (R @ x - F) / n

        # Gradient of the second term
        grad_2 = lambd * E_.T @ (E_ @ x - D) / n

        # Total gradient
        grad = grad_1 + grad_2
        return np.asarray(grad).astype(FLOAT_DTYPE)

    @njit(cache=False, parallel=PARALLEL, fastmath=FASTMATH)  # NOTE cache won't work
    def solution(R, F, P):
        R = R.astype(FLOAT_DTYPE)
        F = F.astype(FLOAT_DTYPE)
        P = P.astype(INT_DTYPE)
        _, m = R.shape
        E_ = E[:, m]

        # np.ix_ unsupported in numba

        A = R.T @ R + lambd * E_.T @ E_
        A = jit_sub_mat_by_index_set(A, P)

        b = R.T @ F + lambd * E_.T @ D
        b = jit_sub_vec_by_index_set(b, P)

        return np.linalg.solve(A, b)

    @njit(cache=False, parallel=PARALLEL, fastmath=FASTMATH)  # NOTE cache won't work
    def lr(R) -> FLOAT_DTYPE:
        R = R.astype(FLOAT_DTYPE)
        n, m = R.shape
        E_ = E[:, :m]

        return n / (np.linalg.norm(R.T @ R) + lambd * np.linalg.norm(E_.T @ E_))

    return Method("lsq_l2_fit", f, grad_f, solution, lr)
