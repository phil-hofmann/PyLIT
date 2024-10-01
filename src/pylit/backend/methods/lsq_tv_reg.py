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
from pylit.backend.utils import jit_sub_mat_by_index_set, jit_sub_vec_by_index_set


def get(E: ARRAY, lambd: FLOAT_DTYPE) -> Method:
    r"""
    # Least Squares with Total Variation Regularization

    Implements the Least Squares with Total Variation Regularization with the objective function
    (correct that)
    \\[
        f(u, w, \lambda) =
        \frac{1}{2} \| \widehat u - \widehat w\|^2_{L^2(\mathbb{R})} +
        \frac{1}{2} \lambda \left\| \frac{du}{d\omega} \right\|_{L^2(\mathbb{R})}^2
    \\]

    which is here implemented as

    \\[
        f(\boldsymbol{\alpha}) =
        \frac{1}{2} \frac{1}{n} \| \boldsymbol{R} \boldsymbol{\alpha} - \boldsymbol{F} \|^2_2 +
        \frac{1}{2} \lambda \left\| \boldsymbol{V}_\boldsymbol{E} \boldsymbol{\alpha} \right\|_{2}^2
    \\]

    with the gradient

    \\[
        \nabla_{\boldsymbol{\alpha}} f(\boldsymbol{\alpha}) =
        \frac{1}{n} \boldsymbol{R}^\top(\boldsymbol{R} \boldsymbol{\alpha} - \boldsymbol{F}) +
        \lambda \boldsymbol{V}_\boldsymbol{E}^\top \boldsymbol{V}_\boldsymbol{E} \boldsymbol{\alpha}
    \\]

    with the learning rate

    \\[
        \eta = \frac{n}{\| \boldsymbol{R}^\top \boldsymbol{R} \| + \lambda n \|\boldsymbol{V}_\boldsymbol{E}^\top \boldsymbol{V}_\boldsymbol{E}\|}
    \\]

    and the solution

    \\[
        \textit{no closed form solution available}
    \\]

    where

    - **$\boldsymbol{R}$**: Regression matrix
    - **$\boldsymbol{F}$**: Target vector
    - **$\boldsymbol{V}_\boldsymbol{E}$**: Variation matrix of the evaluation matrix
    - **$\boldsymbol{\alpha}$**: Coefficient vector
    - **$\lambda$**: Regularization parameter
    - **$n$**: Number of samples

    ### Arguments
    - **E** (np.ndarray): Evaluation matrix
    - **lambd** (np.float64): Regularization parameter.

    ### Returns
    - **Method**(Method): Least Squares with Total Variation Regularization Method.
    """

    # Type Conversion
    E = np.asarray(E).astype(FLOAT_DTYPE)
    lambd = FLOAT_DTYPE(lambd)

    # Compute Total Variation operator
    n = E.shape[1]
    V_E = (E[: len(E) - 1] - E[1 : len(E)])

    # Get method
    method = _standard(V_E, lambd)

    # Compile
    x_, R_, F_, P_ = (
        np.zeros((n), dtype=FLOAT_DTYPE),
        np.eye(n, dtype=FLOAT_DTYPE),
        np.zeros((n), dtype=FLOAT_DTYPE),
        np.array([0], dtype=INT_DTYPE),
    )

    _ = method.f(x_, R_, F_)
    _ = method.grad_f(x_, R_, F_)
    _ = method.solution(R_, F_, P_)
    _ = method.lr(R_)

    return method


def _standard(V_E, lambd) -> Method:

    @njit(cache=False, parallel=PARALLEL, fastmath=FASTMATH)  # NOTE cache won't work
    def f(x, R, F) -> FLOAT_DTYPE:
        x = np.asarray(x).astype(FLOAT_DTYPE)
        R = np.asarray(R).astype(FLOAT_DTYPE)
        F = np.asarray(F).astype(FLOAT_DTYPE)

        return FLOAT_DTYPE(
            0.5 * np.mean((R @ x - F) ** 2) + lambd * 0.5 * np.sum((V_E @ x) ** 2)
        )

    @njit(cache=False, parallel=PARALLEL, fastmath=FASTMATH)  # NOTE cache won't work
    def grad_f(x, R, F) -> ARRAY:
        x = np.asarray(x).astype(FLOAT_DTYPE)
        R = np.asarray(R).astype(FLOAT_DTYPE)
        F = np.asarray(F).astype(FLOAT_DTYPE)
        n, _ = R.shape
        return np.asarray(R.T @ (R @ x - F) / n + lambd * V_E.T @ (V_E @ x)).astype(
            FLOAT_DTYPE
        )

    @njit(cache=False, parallel=PARALLEL, fastmath=FASTMATH)  # NOTE cache won't work
    def solution(R, F, P) -> ARRAY:
        R = np.asarray(R).astype(FLOAT_DTYPE)
        F = np.asarray(F).astype(FLOAT_DTYPE)
        P = np.asarray(P).astype(INT_DTYPE)
        n, _ = R.shape

        # np.ix_ unsupported in numba

        A = R.T @ R / n + lambd * V_E.T @ V_E
        A = jit_sub_mat_by_index_set(A, P)

        b = R.T @ F
        b = jit_sub_vec_by_index_set(b, P)

        return np.asarray(np.linalg.solve(A, b)).astype(FLOAT_DTYPE)

    @njit(cache=False, parallel=PARALLEL, fastmath=FASTMATH)  # NOTE cache won't work
    def lr(R) -> FLOAT_DTYPE:
        R = np.asarray(R).astype(FLOAT_DTYPE)
        n, _ = R.shape
        return FLOAT_DTYPE(
            n / (np.linalg.norm(R.T @ R) + lambd * n * np.linalg.norm(V_E.T @ V_E))
        )

    return Method("lsq_tv_reg", f, grad_f, solution, lr)
