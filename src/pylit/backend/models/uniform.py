import numpy as np
from typing import List

from pylit.global_settings import FLOAT_DTYPE
from pylit.backend.models.lrm import LinearRegressionModel


class UniformLRM(LinearRegressionModel):
    """This is the linear regression model with Uniform model functions."""

    def __init__(
        self,
        # tau: np.ndarray[FLOAT_DTYPE],
        # mu: np.ndarray[FLOAT_DTYPE],
        tau: np.ndarray,
        mu: np.ndarray,
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
        """The uniform kernel function."""
        k = int(param[0])
        a, b = self._mu[k], self._mu[k + 1]
        return (np.heaviside(omega - a, 1.0) - np.heaviside(omega - b, 1.0)) / (b - a)

    def ltransform(
        self,
        tau: np.ndarray[FLOAT_DTYPE],
        param: List[float],
    ) -> np.ndarray[FLOAT_DTYPE]:
        """The Laplace transform of the uniform kernel."""
        k = int(param[0])
        a, b = self._mu[k], self._mu[k + 1]
        return np.where(
            tau == 0,
            1,  # Limit as tau -> 0
            (np.exp(-tau * b) - np.exp(-tau * a)) / (-tau * (b - a)),
        )
