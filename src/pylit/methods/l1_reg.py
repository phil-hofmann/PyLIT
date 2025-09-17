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

# Filter out NumbaPerformanceWarning
warnings.simplefilter("ignore", category=NumbaPerformanceWarning)


def l1_reg(lambd: FLOAT_DTYPE) -> Method:
    r"""
    This is the L1 regularization method. `The interface is described in` :ref:`Methods <methods>`.

    The objective function

    .. math::

        f(u, w, \lambda) =
        \frac{1}{2} \| \widehat u - \widehat w\|^2_{L^2(\mathbb{R})} +
        \lambda \| u \|_{L^1(\mathbb{R})},

    is implemented as

    .. math::

        f(\boldsymbol{\alpha}) =
        \frac{1}{2} \frac{1}{n} \| \boldsymbol{R} \boldsymbol{\alpha} - \boldsymbol{F} \|^2_2 +
        \lambda \frac{1}{n} \| \boldsymbol{\alpha} \|_1,


    with the gradient

    .. math::

        \nabla_{\boldsymbol{\alpha}} f(\boldsymbol{\alpha}) =
        \frac{1}{n} \boldsymbol{R}^\top(\boldsymbol{R} \boldsymbol{\alpha} - \boldsymbol{F}) \pm
        \lambda \frac{1}{n}, \quad \boldsymbol{\alpha} \neq 0,
    

    the learning rate

    .. math::

        \eta = \frac{n}{\| \boldsymbol{R}^\top \boldsymbol{R} \|}, \quad \boldsymbol{\alpha} \neq 0,
    

    and the solution

    .. math::

        \boldsymbol{\alpha}^* = (\boldsymbol{R}^\top \boldsymbol{R})^{-1} (\boldsymbol{R}^\top \boldsymbol{F} \pm \lambda), \quad \boldsymbol{\alpha} \neq 0,

    where

    - :math:`\boldsymbol{R}`: Regression matrix,
    - :math:`\boldsymbol{F}`: Target vector,
    - :math:`\boldsymbol{\alpha}`: Coefficient vector,
    - :math:`\lambda`: Regularization parameter,
    - :math:`n`: Number of samples.
    """

    # Type Conversion
    lambd = FLOAT_DTYPE(lambd)

    # Get Method
    method = _l1_reg(lambd)

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


def _l1_reg(lambd) -> Method:

    @njit(fastmath=FASTMATH)
    def f(x, R, F) -> FLOAT_DTYPE:
        x = np.asarray(x).astype(FLOAT_DTYPE)
        R = np.asarray(R).astype(FLOAT_DTYPE)
        F = np.asarray(F).astype(FLOAT_DTYPE)

        return FLOAT_DTYPE(0.5 * np.mean((R @ x - F) ** 2) + lambd * np.mean(x))

    @njit(fastmath=FASTMATH)
    def grad_f(x, R, F) -> np.ndarray:
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

    @njit(fastmath=FASTMATH)
    def solution(R, F, P) -> np.ndarray:
        R = np.asarray(R).astype(FLOAT_DTYPE)
        F = np.asarray(F).astype(FLOAT_DTYPE)
        P = np.asarray(P).astype(INT_DTYPE)

        # np.ix_ unsupported in numba
        R_P = R[:, P]
        A = R_P.T @ R_P
        b = R_P.T @ F - lambd

        return np.asarray(np.linalg.solve(A, b)).astype(FLOAT_DTYPE)

    @njit(fastmath=FASTMATH)
    def lr(R) -> FLOAT_DTYPE:
        R = np.asarray(R).astype(FLOAT_DTYPE)
        n, _ = R.shape

        return FLOAT_DTYPE(n / np.linalg.norm(R.T @ R))

    return Method("l1_reg", f, grad_f, solution, lr)
