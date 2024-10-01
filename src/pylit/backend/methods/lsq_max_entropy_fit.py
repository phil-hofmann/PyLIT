import numpy as np

from numba import njit
from pylit.backend.core import Method
from pylit.global_settings import (
    ARRAY,
    FLOAT_DTYPE,
    INT_DTYPE,
    TOL_LOG,
    CACHE,
    PARALLEL,
    FASTMATH,
)


def get(D: ARRAY, E: ARRAY, lambd: FLOAT_DTYPE) -> Method:
    r"""
    # Least Squares Maximum Entropy Fit

    Implements the Least Squares Maximum Entropy Fit with the objective function

    \\[
        f(u, w, \lambda) =
        \frac{1}{2} \| \widehat u - \widehat w\|^2_{L^2(\mathbb{R})} -
        \lambda \int_{-\infty}^\infty u(\omega) \log \left( \frac{u(\omega)}{w(\omega)} \right) d\omega
    \\]

    which is here implemented as

    \\[
        f(\boldsymbol{\alpha}) =
        \frac{1}{2} \frac{1}{n} \| \boldsymbol{R} \boldsymbol{\alpha} - \boldsymbol{F} \|^2_2 +
        \lambda \frac{1}{n} \sum_{i=1}^n (\boldsymbol{E} \boldsymbol{\alpha})_i \log \frac{(\boldsymbol{E} \boldsymbol{\alpha})_i}{D_i}
    \\]

    with the gradient

    \\[
        \nabla_{\boldsymbol{\alpha}} f(\boldsymbol{\alpha}) =
        \frac{1}{n} \boldsymbol{R}^\top(\boldsymbol{R} \boldsymbol{\alpha} - \boldsymbol{F}) +
        \lambda \frac{1}{n} \boldsymbol{E}^\top(\log \boldsymbol{E} \boldsymbol{\alpha} - \log \boldsymbol{D} + 1)
    \\]

    with the learning rate

    \\[
        \eta = \frac{1}{\| \boldsymbol{R}^\top \boldsymbol{R} \| + \lambda \|\boldsymbol{E}\|^2}
    \\]

    and the solution

    \\[
        \textit{no closed form solution available}
    \\]

    where

    - **$\boldsymbol{R}$**: Regression matrix
    - **$\boldsymbol{F}$**: Target vector
    - **$\boldsymbol{D}$**: Default model vector
    - **$\boldsymbol{E}$**: Evaluation matrix
    - **$\boldsymbol{\alpha}$**: Coefficient vector
    - **$\lambda$**: Regularization parameter
    - **$n$**: Number of samples

    # Arguments
    - **D**(np.ndarray): Default model vector
    - **E**(np.ndarray): Evaluation matrix
    - **lambd**(np.float64): Regularization parameter.

    # Returns
    - **Method**(Method): Least Squares Maximum Entropy Fit.
    """

    # Type Conversion
    D = np.asarray(D).astype(FLOAT_DTYPE)
    E = np.asarray(E).astype(FLOAT_DTYPE)
    lambd = FLOAT_DTYPE(lambd)

    # Get method
    method = _standard(D, E, lambd)

    # Compile
    k = len(D)
    x_, R_, F_, P_ = (
        np.zeros((k), dtype=FLOAT_DTYPE),
        np.eye(k, dtype=FLOAT_DTYPE),
        np.zeros((k), dtype=FLOAT_DTYPE),
        np.array([0], dtype=INT_DTYPE),
    )

    _ = method.f(x_, R_, F_)
    _ = method.grad_f(x_, R_, F_)
    _ = method.solution(R_, F_, P_)
    _ = method.lr(R_)

    return method


def _standard(D, E, lambd) -> Method:

    D = np.clip(D, a_min=0, a_max=None)  # Clip Non-negative
    q = np.copy(D)
    q = np.clip(q, a_min=TOL_LOG, a_max=None)  # Clip Log
    log_q = np.log(q)  # Take Log
    norm_E = np.linalg.norm(E)

    @njit(cache=False, parallel=PARALLEL, fastmath=FASTMATH)  # NOTE cache won't work
    def f(x, R, F) -> FLOAT_DTYPE:
        x = np.asarray(x).astype(FLOAT_DTYPE)
        R = np.asarray(R).astype(FLOAT_DTYPE)
        F = np.asarray(F).astype(FLOAT_DTYPE)

        p = E @ x  # Evaluate
        p = np.clip(p, a_min=0, a_max=None)  # Clip Non-negative
        log_p = np.log(p)  # Take Log

        return FLOAT_DTYPE(
            0.5 * np.mean((R @ x - F) ** 2) + lambd * np.mean(p * (log_p - log_q))
        )

    @njit(cache=False, parallel=PARALLEL, fastmath=FASTMATH)  # NOTE cache won't work
    def grad_f(x, R, F) -> ARRAY:
        x = np.asarray(x).astype(np.float64)
        R = np.asarray(R).astype(np.float64)
        F = np.asarray(F).astype(np.float64)
        n, _ = R.shape

        # Compute p and log_p
        p = E @ x
        p = np.clip(p, a_min=0, a_max=None)
        p_hat = np.clip(p, a_min=TOL_LOG, a_max=None)
        log_p = np.log(p_hat)

        # Gradient of the first term
        residual = R @ x - F
        grad_L1 = (R.T @ residual) / n

        # Gradient of the second term
        grad_L2 = lambd * (E.T @ (log_p - log_q + 1)) / n

        # Total gradient
        grad = grad_L1 + grad_L2
        return np.asarray(grad).astype(FLOAT_DTYPE)

    @njit(cache=CACHE, parallel=PARALLEL, fastmath=FASTMATH)
    def solution(R, F, P):
        # No closed form solution available
        raise NotImplementedError("No closed form solution available")

    @njit(cache=False, parallel=PARALLEL, fastmath=FASTMATH)  # NOTE cache won't work
    def lr(R) -> FLOAT_DTYPE:
        R = np.asarray(R).astype(FLOAT_DTYPE)
        n, _ = R.shape

        return FLOAT_DTYPE(n / (np.linalg.norm(R.T @ R) + lambd * norm_E ** 2))

    return Method("lsq_max_entropy_fit", f, grad_f, solution, lr)
