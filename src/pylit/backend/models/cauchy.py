import numpy as np
from typing import List

from pylit.global_settings import FLOAT_DTYPE
from pylit.backend.models.lrm import LinearRegressionModel


class CauchyLRM(LinearRegressionModel):
    """This is the linear regression model with Cauchy model functions."""

    def __init__(
        self,
        # tau: np.ndarray[FLOAT_DTYPE],
        # mu: np.ndarray[FLOAT_DTYPE],
        # sigma: np.ndarray[FLOAT_DTYPE],
        tau: np.ndarray,
        mu: np.ndarray,
        sigma: np.ndarray,
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
        """The Gaussian kernel function."""
        mu, sigma = param
        return sigma / (np.pi * ((omega - mu) ** 2 + sigma**2))

    def ltransform(
        self,
        tau: np.ndarray[FLOAT_DTYPE],
        param: List[float],
    ) -> np.ndarray[FLOAT_DTYPE]:
        """The Laplace transform of the Gaussian kernel."""
        mu, sigma = param
        return np.exp(-mu * tau - sigma * np.abs(tau))
