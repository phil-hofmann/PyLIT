import warnings
import numpy as np

from numba import njit
from numba.core.errors import NumbaPerformanceWarning
from pylit.core.data_classes import Method
from pylit.settings import (
    FLOAT_DTYPE,
    INT_DTYPE,
    FASTMATH,
    TOL_LOG,
)

# Filter out NumbaPerformanceWarning
warnings.simplefilter("ignore", category=NumbaPerformanceWarning)


def max_entropy_fit(D: np.ndarray, E: np.ndarray, lambd: FLOAT_DTYPE) -> Method:
    r"""
    This is the maximum entropy fitting method. `The interface is described in` :ref:`Methods <methods>`.

    The objective function

    .. math::

        f(u, w, \lambda) =
        \frac{1}{2} \| \widehat u - \widehat w\|^2_{L^2(\mathbb{R})} -
        \lambda \int_{-\infty}^\infty u(\omega) \log \left( \frac{u(\omega)}{w(\omega)} \right) d\omega,

    is implemented as

    .. math::

        f(\boldsymbol{\alpha}) =
        \frac{1}{2} \| \boldsymbol{R} \boldsymbol{\alpha} - \boldsymbol{F} \|^2_2 +
        \lambda \sum_{i=1}^n (\boldsymbol{E} \boldsymbol{\alpha})_i \log \frac{(\boldsymbol{E} \boldsymbol{\alpha})_i}{D_i},

    with the gradient

    .. math::

        \nabla_{\boldsymbol{\alpha}} f(\boldsymbol{\alpha}) =
        \boldsymbol{R}^\top(\boldsymbol{R} \boldsymbol{\alpha} - \boldsymbol{F}) +
        \lambda \boldsymbol{E}^\top(\log \boldsymbol{E} \boldsymbol{\alpha} - \log \boldsymbol{D} + 1),

    the learning rate

    .. math::

        \eta = \frac{1}{\| \boldsymbol{R}^\top \boldsymbol{R} \| + \lambda \|\boldsymbol{E}\|^2},

    and the solution

    .. math::

        \textit{No closed form solution available},

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

    # Get method
    method = _max_entropy_fit(D, E, lambd)

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


def _max_entropy_fit(D, E, lambd) -> Method:

    D = np.clip(D, a_min=0, a_max=None)
    q = np.copy(D)
    q = np.clip(q, a_min=TOL_LOG, a_max=None)
    log_q = np.log(q)

    @njit(fastmath=FASTMATH)
    def f(x, R, F) -> FLOAT_DTYPE:
        x = np.asarray(x).astype(FLOAT_DTYPE)
        R = np.asarray(R).astype(FLOAT_DTYPE)
        F = np.asarray(F).astype(FLOAT_DTYPE)
        _, m = R.shape
        E_ = E[:, :m]

        p = E_ @ x + 10e-10  # NOTE Default model should not take values close to zero.
        p = np.clip(p, a_min=0, a_max=None)
        log_p = np.log(p)

        return FLOAT_DTYPE(
            0.5 * np.sum((R @ x - F) ** 2) + lambd * np.sum(q - p + p * (log_p - log_q))
        )

    @njit(fastmath=FASTMATH)
    def grad_f(x, R, F) -> np.ndarray:
        x = np.asarray(x).astype(np.float64)
        R = np.asarray(R).astype(np.float64)
        F = np.asarray(F).astype(np.float64)
        _, m = R.shape
        E_ = E[:, :m]

        p = E_ @ x
        p = np.clip(p, a_min=0, a_max=None)
        p_hat = np.clip(p, a_min=TOL_LOG, a_max=None)
        log_p = np.log(p_hat)

        # Gradient of the first term
        grad_1 = R.T @ (R @ x - F)

        # Gradient of the second term
        grad_2 = lambd * (E_.T @ (log_p - log_q))

        # Total gradient
        grad = grad_1 + grad_2
        return np.asarray(grad).astype(FLOAT_DTYPE)

    @njit(fastmath=FASTMATH)
    def solution(R, F, P) -> np.ndarray:
        # No closed form solution available
        return None

    @njit(fastmath=FASTMATH)
    def lr(R) -> FLOAT_DTYPE:
        R = np.asarray(R).astype(FLOAT_DTYPE)
        _, m = R.shape
        E_ = E[:, :m]

        return FLOAT_DTYPE(
            1 / (np.linalg.norm(R.T @ R) + lambd * np.linalg.norm(E_) ** 2)
        )

    return Method("max_entropy_fit", f, grad_f, solution, lr)
