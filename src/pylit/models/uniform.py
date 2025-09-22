import numpy as np
from typing import List

from pylit.settings import FLOAT_DTYPE
from pylit.models.lrm import LinearRegressionModel


class Uniform(LinearRegressionModel):
    """This is the linear regression model with Uniform model functions."""

    def __init__(
        self,
        tau: np.ndarray[FLOAT_DTYPE],
        mu: np.ndarray[FLOAT_DTYPE],
    ) -> None:
        """Initialize the model."""
        # Type Conversion
        tau = np.asarray(tau).astype(FLOAT_DTYPE)
        mu = np.asarray(mu).astype(FLOAT_DTYPE)

        # Integrity
        if not all(np.diff(mu) > 0):
            raise ValueError("The support points must strictly increasing.")

        # Initialize the Parent Class
        super().__init__("UniformLRM", tau, [np.arange(len(mu) - 1, dtype=FLOAT_DTYPE)])
        self._mu = mu

    def kernel(
        self,
        omega: np.ndarray[FLOAT_DTYPE],
        param: List[float],
    ) -> np.ndarray:
        r"""Evaluate the uniform kernel function for a given set of parameters.

        This method overrides :meth:`~pylit.models.lrm.LinearRegressionModel.kernel`.

        Args:
            omega:
                Discrete frequency axis.
            param:
                Parameter tuple [k], where k indexes the interval
                [mu_k, mu_{k+1}] of the uniform kernel support.

        Returns:
            Values of the uniform kernel

            .. math::
                K(\omega; \mu_k, \mu_{k+1}) =
                \frac{\mathbf{1}_{[\mu_k, \mu_{k+1})}(\omega)}{\mu_{k+1} - \mu_k},
        """
        k = int(param[0])
        a, b = self._mu[k], self._mu[k + 1]
        return (np.heaviside(omega - a, 1.0) - np.heaviside(omega - b, 1.0)) / (b - a)

    def ltransform(
        self,
        tau: np.ndarray[FLOAT_DTYPE],
        param: List[float],
    ) -> np.ndarray[FLOAT_DTYPE]:
        r"""Evaluate the Laplace-transformed uniform kernel at the discrete time axis.

        This method overrides :meth:`~pylit.models.lrm.LinearRegressionModel.ltransform`.

        Args:
            tau: 
                Discrete time axis.
            param: 
                Parameter tuple [k], where k indexes the interval
                [mu_k, mu_{k+1}] of the uniform kernel support.

        Returns:
            Values of the Laplace-transformed uniform kernel

            .. math::
                \widehat{K}(\tau; \mu_k, \mu_{k+1}) =
                \begin{cases}
                    1, & \tau = 0, \\
                    \frac{e^{-\tau \mu_{k+1}} - e^{-\tau \mu_k}}{-\tau (\mu_{k+1}-\mu_k)}, & \tau \neq 0.
                \end{cases}
        """
        k = int(param[0])
        a, b = self._mu[k], self._mu[k + 1]
        return np.where(
            tau == 0,
            1,  # Limit as tau -> 0
            (np.exp(-tau * b) - np.exp(-tau * a)) / (-tau * (b - a)),
        )
