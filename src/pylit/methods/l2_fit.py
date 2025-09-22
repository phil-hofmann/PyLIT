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


def l2_fit(D: np.ndarray, E: np.ndarray, lambd: FLOAT_DTYPE) -> Method:
    r"""
    This is the L2 fitting method. `The interface is described in` :ref:`Methods <methods>`.

    The objective function

    .. math::

        f(u, w, \lambda) =
        \frac{1}{2} \| \widehat u - \widehat w\|^2_{L^2(\mathbb{R})} +
        \frac{1}{2} \lambda \| u - w \|_{L^2(\mathbb{R})}^2,

    is implemented as

    .. math::

        f(\boldsymbol{\alpha}) =
        \frac{1}{2} \| \boldsymbol{R} \boldsymbol{\alpha} - \boldsymbol{F} \|^2_2 +
        \frac{1}{2} \lambda \| \boldsymbol{E} \boldsymbol{\alpha} - \boldsymbol{D} \|^2_2,

    with the gradient

    .. math::

        \nabla_{\boldsymbol{\alpha}} f(\boldsymbol{\alpha}) =
        \boldsymbol{R}^\top(\boldsymbol{R} \boldsymbol{\alpha} - \boldsymbol{F}) +
        \lambda \boldsymbol{E}^\top(\boldsymbol{E} \boldsymbol{\alpha} - \boldsymbol{D}),

    the learning rate

    .. math::

        \eta = \frac{1}{\| \boldsymbol{R}^\top \boldsymbol{R} + \lambda \boldsymbol{E}^\top \boldsymbol{E} \|},

    and the solution

    .. math::

        \boldsymbol{\alpha}^* = (\boldsymbol{R}^\top \boldsymbol{R} + \lambda \boldsymbol{E}^\top \boldsymbol{E})^{-1} (\boldsymbol{R}^\top \boldsymbol{F} + \lambda \boldsymbol{E}^\top \boldsymbol{D}),

    where

    - :math:`\boldsymbol{R}`: Regression matrix,
    - :math:`\boldsymbol{F}`: Target vector,
    - :math:`\boldsymbol{E}`: Evaluation matrix,
    - :math:`\boldsymbol{D}`: Default model vector,
    - :math:`\boldsymbol{\alpha}`: Coefficient vector,
    - :math:`\lambda`: Regularization parameter,
    - :math:`n`: Number of samples.
    """

    # Type Conversion
    D = np.asarray(D).astype(FLOAT_DTYPE)
    E = np.asarray(E).astype(FLOAT_DTYPE)
    lambd = FLOAT_DTYPE(lambd)

    # Get Method
    method = _l2_fit(D, E, lambd)

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


def _l2_fit(D, E, lambd) -> Method:

    @njit(fastmath=FASTMATH)
    def f(x, R, F) -> FLOAT_DTYPE:
        x = x.astype(FLOAT_DTYPE)
        R = R.astype(FLOAT_DTYPE)
        F = F.astype(FLOAT_DTYPE)
        _, m = R.shape
        E_ = E[:, :m]

        return 0.5 * np.sum((R @ x - F) ** 2) + lambd * 0.5 * np.sum((E_ @ x - D) ** 2)

    @njit(fastmath=FASTMATH)
    def grad_f(x, R, F) -> np.ndarray:
        x = x.astype(FLOAT_DTYPE)
        R = R.astype(FLOAT_DTYPE)
        F = F.astype(FLOAT_DTYPE)
        _, m = R.shape
        E_ = E[:, :m]

        # Gradient of the first term
        grad_1 = R.T @ (R @ x - F)

        # Gradient of the second term
        grad_2 = lambd * E_.T @ (E_ @ x - D)

        # Total gradient
        grad = grad_1 + grad_2
        return np.asarray(grad).astype(FLOAT_DTYPE)

    @njit(fastmath=FASTMATH)
    def solution(R, F, P):
        R = R.astype(FLOAT_DTYPE)
        F = F.astype(FLOAT_DTYPE)
        P = P.astype(INT_DTYPE)
        _, m = R.shape
        E_R = E[:, :m]

        # np.ix_ unsupported in numba
        R_P = R[:, P]
        E_P = E_R[:, P]
        A = R_P.T @ R_P + lambd * E_P.T @ E_P
        b = R_P.T @ F + lambd * E_P.T @ D

        return np.linalg.solve(A, b)

    @njit(fastmath=FASTMATH)
    def lr(R) -> FLOAT_DTYPE:
        R = R.astype(FLOAT_DTYPE)
        _, m = R.shape
        E_ = E[:, :m]

        return 1.0 / (np.linalg.norm(R.T @ R) + lambd * np.linalg.norm(E_.T @ E_))

    return Method("l2_fit", f, grad_f, solution, lr)
