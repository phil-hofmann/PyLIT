import numpy as np
from typing import List

from pylit.settings import FLOAT_DTYPE
from pylit.models.lrm import LinearRegressionModel


class Gauss(LinearRegressionModel):
    """This is the linear regression model with Gaussian model functions."""

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
        r"""Evaluate the Gaussian kernel function for a given set of parameters.

        This method overrides :meth:`~pylit.models.lrm.LinearRegressionModel.kernel`.

        Args:
            omega:
                Discrete frequency axis.
            param:
                Parameter tuple [mu, sigma].

        Returns:
            Values of the Gaussian kernel

            .. math::
                K(\omega; \mu, \sigma) =
                \frac{1}{\sigma \sqrt{2 \pi}} \exp\Big(-\frac{1}{2} \frac{(\omega-\mu)^2}{\sigma^2}\Big).
        """

        mu, sigma = param
        return (1 / (sigma * np.sqrt(2 * np.pi))) * np.exp(
            -0.5 * (omega - mu) ** 2 / sigma**2
        )

    def ltransform(
        self,
        tau: np.ndarray[FLOAT_DTYPE],
        param: List[float],
    ) -> np.ndarray[FLOAT_DTYPE]:
        r"""Evaluate the Laplace-transformed Gaussian kernel at the discrete time axis.

        This method overrides :meth:`~pylit.models.lrm.LinearRegressionModel.ltransform`.

        Args:
            tau:
                Discrete time axis.
            param:
                Parameter tuple [mu, sigma].

        Returns:
            Values of the Laplace-transformed Gaussian kernel

            .. math::
                \widehat{K}(\tau; \mu, \sigma) =
                \exp\Big(-\mu \tau + \frac{1}{2} \sigma^2 \tau^2\Big).
        """

        mu, sigma = param
        return np.exp(-mu * tau + 0.5 * sigma**2 * tau**2)
