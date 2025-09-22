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


def tv_reg(E: np.ndarray, lambd: FLOAT_DTYPE) -> Method:
    r"""
    This is the total variation regularization method. `The interface is described in` :ref:`Methods <methods>`.

    The objective function (correct that)

    .. math::

        f(u, w, \lambda) =
        \frac{1}{2} \| \widehat u - \widehat w\|^2_{L^2(\mathbb{R})} +
        \frac{1}{2} \lambda \left\| \frac{du}{d\omega} \right\|_{L^2(\mathbb{R})}^2

    is implemented as

    .. math::

        f(\boldsymbol{\alpha}) =
        \frac{1}{2} \frac{1}{n} \| \boldsymbol{R} \boldsymbol{\alpha} - \boldsymbol{F} \|^2_2 +
        \frac{1}{2} \lambda \left\| \boldsymbol{V}_\boldsymbol{E} \boldsymbol{\alpha} \right\|_{2}^2

    with the gradient

    .. math::

        \nabla_{\boldsymbol{\alpha}} f(\boldsymbol{\alpha}) =
        \frac{1}{n} \boldsymbol{R}^\top(\boldsymbol{R} \boldsymbol{\alpha} - \boldsymbol{F}) +
        \lambda \boldsymbol{V}_\boldsymbol{E}^\top \boldsymbol{V}_\boldsymbol{E} \boldsymbol{\alpha}

    the learning rate

    .. math::

        \eta = \frac{1}{\| \boldsymbol{R}^\top \boldsymbol{R} \| + \lambda n \|\boldsymbol{V}_\boldsymbol{E}^\top \boldsymbol{V}_\boldsymbol{E}\|}

    and the solution

    .. math::

        \boldsymbol{\alpha}^* = \left(\boldsymbol{R}^\top \boldsymbol{R} + \lambda \, \boldsymbol{V}_{\boldsymbol{E}}^\top \boldsymbol{V}_{\boldsymbol{E}} \right)^{-1} \boldsymbol{R}^\top \boldsymbol{F},

    where

    - :math:`\boldsymbol{R}`: Regression matrix,
    - :math:`\boldsymbol{F}`: Target vector,
    - :math:`\boldsymbol{V}_{\boldsymbol{E}}`: Variation matrix of the evaluation matrix.
    - :math:`\boldsymbol{\alpha}`: Coefficient vector,
    - :math:`\lambda`: Regularization parameter,
    - :math:`n`: Number of samples.
    """

    # Type Conversion
    E = np.asarray(E).astype(FLOAT_DTYPE)
    lambd = FLOAT_DTYPE(lambd)

    # Get method
    method = _tv_reg(E, lambd)

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


def _tv_reg(E, lambd) -> Method:

    # Compute Finite Difference Operator
    n, _ = E.shape
    FD = np.diag(-np.ones(n), 0) + np.diag(np.ones(n - 1), 1)
    FD[-1, :] = 0

    @njit(fastmath=FASTMATH)
    def f(x, R, F) -> FLOAT_DTYPE:
        x = np.asarray(x).astype(FLOAT_DTYPE)
        R = np.asarray(R).astype(FLOAT_DTYPE)
        F = np.asarray(F).astype(FLOAT_DTYPE)
        _, m = R.shape
        E_ = E[:, :m]
        V_E = FD @ E_

        return FLOAT_DTYPE(
            0.5 * np.sum((R @ x - F) ** 2) + lambd * 0.5 * np.sum((V_E @ x) ** 2)
        )

    @njit(fastmath=FASTMATH)
    def grad_f(x, R, F) -> np.ndarray:
        x = np.asarray(x).astype(FLOAT_DTYPE)
        R = np.asarray(R).astype(FLOAT_DTYPE)
        F = np.asarray(F).astype(FLOAT_DTYPE)
        _, m = R.shape
        E_ = E[:, :m]
        V_E = FD @ E_

        # Gradient of the first term
        grad_1 = R.T @ (R @ x - F)

        # Gradient of the second term
        grad_2 = lambd * V_E.T @ (V_E @ x)

        # Total gradient
        grad = grad_1 + grad_2
        return np.asarray(grad).astype(FLOAT_DTYPE)

    @njit(fastmath=FASTMATH)
    def solution(R, F, P) -> np.ndarray:
        # TODO Not sure if this is correct yet ...
        R = np.asarray(R).astype(FLOAT_DTYPE)
        F = np.asarray(F).astype(FLOAT_DTYPE)
        P = np.asarray(P).astype(INT_DTYPE)
        _, m = R.shape
        E_R = E[:, :m]
        V_E_R = FD @ E_R

        # np.ix_ unsupported in numba
        R_P = R[:, P]
        V_E_P = V_E_R[:, P]
        A = R_P.T @ R_P + lambd * V_E_P.T @ V_E_P
        b = R_P.T @ F

        return np.asarray(np.linalg.solve(A, b)).astype(FLOAT_DTYPE)

    @njit(fastmath=FASTMATH)
    def lr(R) -> FLOAT_DTYPE:
        R = np.asarray(R).astype(FLOAT_DTYPE)
        _, m = R.shape
        E_ = E[:, :m]
        V_E = FD @ E_

        norm = np.linalg.norm(V_E.T @ V_E)
        return FLOAT_DTYPE(1.0 / (np.linalg.norm(R.T @ R) + lambd * norm))

    return Method("tv_reg", f, grad_f, solution, lr)
