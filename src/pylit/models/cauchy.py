import numpy as np
from typing import List

from pylit.settings import FLOAT_DTYPE
from pylit.models.lrm import LinearRegressionModel


class Cauchy(LinearRegressionModel):
    """This is the linear regression model with Cauchy model functions."""

    def __init__(
        self,
        tau: np.ndarray[FLOAT_DTYPE],
        mu: np.ndarray[FLOAT_DTYPE],
        sigma: np.ndarray[FLOAT_DTYPE],
    ) -> None:
        """Initialize the model."""
        # Type Conversion
        tau = np.asarray(tau).astype(FLOAT_DTYPE)
        mu = np.asarray(mu).astype(FLOAT_DTYPE)
        sigma = np.asarray(sigma).astype(FLOAT_DTYPE)

        # Integrity
        if np.any(sigma <= 0.0):
            raise ValueError("The standard deviations must be positive.")

        # Initialize the Parent Class
        super().__init__("GaussLRM", tau, [mu, sigma])

    def kernel(
        self,
        omega: np.ndarray[FLOAT_DTYPE],
        param: List[float],
    ) -> np.ndarray:
        r"""Evaluate the Cauchy kernel function for a given set of parameters.

        This method overrides :meth:`~pylit.models.lrm.LinearRegressionModel.kernel`.

        Args:
            omega:
                Discrete frequency axis.
            param:
                Parameter tuple [mu, sigma].

        Returns:
            Values of the Cauchy kernel

            .. math::
                K(\omega; \mu, \sigma) = \frac{\sigma}{\pi ((\omega-\mu)^2 + \sigma^2)}.
        """

        mu, sigma = param
        return sigma / (np.pi * ((omega - mu) ** 2 + sigma**2))

    def ltransform(
        self,
        tau: np.ndarray[FLOAT_DTYPE],
        param: List[float],
    ) -> np.ndarray[FLOAT_DTYPE]:
        r"""Evaluate the Laplace-transformed Cauchy kernel at the discrete time axis.

        This method overrides :meth:`~pylit.models.lrm.LinearRegressionModel.ltransform`.

        Args:
            tau:
                Discrete time axis.
            param:
                Parameter tuple [mu, sigma].

        Returns:
            Laplace-transformed kernel

            .. math::
                \widehat K(\tau; \mu, \sigma) = \exp(-\mu \tau - \sigma |\tau|).
        """

        mu, sigma = param
        return np.exp(-mu * tau - sigma * np.abs(tau))
