import numpy as np
from pylit.settings import FLOAT_DTYPE

"""Module for applying detailed balance to models."""


class DetailedBalance:

    def __init__(self, tau: np.ndarray) -> None:
        """Initialize the DetailedBalance.

        This is a parent class which is used to compute the detailed balance.

        Parameters:
            tau : np.ndarray
                The time scale parameter.

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
    def beta(self) -> float:
        return self._beta

    @beta.setter
    def beta(self, beta: float):
        raise PermissionError("The beta parameter is read-only.")


class ForwardDetailedBalance(DetailedBalance):

    def __init__(self, tau: np.ndarray) -> None:
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

        def wrapper(tau: np.ndarray, *args, **kwargs):
            return func(self.beta - tau, *args, **kwargs) + func(tau, *args, **kwargs)

        return wrapper


class InverseDetailedBalance(DetailedBalance):

    def __init__(self, tau: np.ndarray) -> None:
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

        def wrapper(omega: np.ndarray, *args, **kwargs):
            return np.exp(self.beta * omega) * func(-omega, *args, **kwargs) + func(
                omega, *args, **kwargs
            )

        return wrapper
