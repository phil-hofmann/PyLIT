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
        """The Gaussian kernel function."""
        mu, sigma = param
        return (1 / (sigma * np.sqrt(2 * np.pi))) * np.exp(
            -0.5 * (omega - mu) ** 2 / sigma**2
        )

    def ltransform(
        self,
        tau: np.ndarray[FLOAT_DTYPE],
        param: List[float],
    ) -> np.ndarray[FLOAT_DTYPE]:
        """The Laplace transform of the Gaussian kernel."""
        mu, sigma = param
        return np.exp(-mu * tau + 0.5 * sigma**2 * tau**2)
