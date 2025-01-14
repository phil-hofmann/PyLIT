import numpy as np

from typing import List
from pylit.global_settings import FLOAT_DTYPE
from pylit.backend.models.lrm import LinearRegressionModel

"""Module for applying detailed balance to models."""


class DetailedBalance:

    def __init__(self, tau: np.ndarray[FLOAT_DTYPE]) -> None:
        """Initialize the DetailedBalance.

        This is a parent class which is used to compute the detailed balance.

        Parameters:
            beta : FLOAT_DTYPE
                The new parameter beta for the detailed balance.

        Raises:
            ValueError
                If the beta is not positive."""

        # Type Conversion
        tau = np.asarray(tau).astype(FLOAT_DTYPE)

        # Compute beta
        self._beta = np.max(tau)

        # Integrity
        if self._beta <= 0:
            raise ValueError("The beta must be strictly positive.")

    @property
    def beta(self) -> FLOAT_DTYPE:
        return self._beta

    @beta.setter
    def beta(self, beta: FLOAT_DTYPE):
        raise PermissionError("The beta parameter is read-only.")


class ForwardDetailedBalance(DetailedBalance):

    def __init__(self, tau: np.ndarray[FLOAT_DTYPE]) -> None:
        """Initialize the Forward Detailed Balance."""

        super().__init__(tau)

    def __call__(self, func):
        """Returns the scaled version of the given Laplace transform.

        Parameters:
            func : callable
                The Laplace transform.

        Returns:
            callable
                The scaled Laplace transform.
        """

        def wrapper(tau: np.ndarray[FLOAT_DTYPE], *args, **kwargs):
            return func(self.beta - tau, *args, **kwargs) + func(tau, *args, **kwargs)

        return wrapper


class InverseDetailedBalance(DetailedBalance):

    def __init__(self, tau: np.ndarray[FLOAT_DTYPE]) -> None:
        """Initialize the Inverse Detailed Balance."""

        super().__init__(tau)

    def __call__(self, func):
        """Returns the rescaled version of the given inverse Laplace transform.

        Parameters:
            func : callable
                The inverse Laplace transform.

        Returns:
            callable
                The rescaled inverse Laplace transform.
        """

        def wrapper(omega: np.ndarray[FLOAT_DTYPE], *args, **kwargs):
            return np.exp(self.beta * omega) * func(-omega, *args, **kwargs) + func(
                omega, *args, **kwargs
            )

        return wrapper


def detailed_balance(
    lrm: LinearRegressionModel, beta: float = 1.0
) -> LinearRegressionModel:
    """Decorator for the Linear Regression Model.

    This decorator is used to apply the detailed balance to the Linear Regression Model.

    Parameters:
        lrm : LinearRegressionModel
            The  linear regression model.
        beta : float
            The new parameter beta for the detailed balance.

    Returns:
        LinearRegressionModel:
            The detailed balanced linear regression model.
    """

    # Apply the scaling decorators to the regression matrix
    lrm.kernel = InverseDetailedBalance(lrm.tau)(lrm.kernel)

    # Apply the scaling decorators to the regression matrix
    lrm.ltransform = ForwardDetailedBalance(lrm.tau)(lrm.ltransform)

    return lrm
