import numpy as np
from typing import List

from pylit.settings import FLOAT_DTYPE
from pylit.models.lrm import LinearRegressionModel


class Laplace(LinearRegressionModel):
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
        if np.any(sigma <= 0.0) or np.any(sigma > 1.0):
            # TODO actually: 0 < sigma < 1/beta
            raise ValueError("The standard deviations must be in (0, 1).")

        # Initialize the Parent Class
        super().__init__("LaplaceLRM", tau, [mu, sigma])

    def kernel(
        self,
        omega: np.ndarray[FLOAT_DTYPE],
        param: List[float],
    ) -> np.ndarray:
        r"""Evaluate the Laplacian kernel function for a given set of parameters.

        This method overrides :meth:`~pylit.models.lrm.LinearRegressionModel.kernel`.

        Args:
            omega:
                Discrete frequency axis.
            param:
                Parameter tuple [mu, sigma].

        Returns:
            Values of the Laplacian kernel

            .. math::
                K(\omega; \mu, \sigma) =
                \frac{1}{2 \sigma} \exp\Big(-\frac{|\omega-\mu|}{\sigma}\Big).
        """

        mu, sigma = param
        return (1 / (2 * sigma)) * np.exp(-np.abs(omega - mu) / sigma)

    def ltransform(
        self,
        tau: np.ndarray[FLOAT_DTYPE],
        param: List[float],
    ) -> np.ndarray[FLOAT_DTYPE]:
        r"""Evaluate the Laplace-transformed Laplacian kernel at the discrete time axis.

        This method overrides :meth:`~pylit.models.lrm.LinearRegressionModel.ltransform`.

        Args:
            tau:
                Discrete time axis.
            param:
                Parameter tuple [mu, sigma].

        Returns:
            Values of the Laplace-transformed Laplacian kernel

            .. math::
                \widehat{K}(\tau; \mu, \sigma) =
                \frac{\exp(-\mu \tau)}{1 - \sigma^2 \tau^2}.
        """

        mu, sigma = param
        return np.exp(-mu * tau) / (1 - sigma**2 * tau**2)
