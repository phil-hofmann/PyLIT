import numpy as np

from pylit.settings import FLOAT_DTYPE
from pylit.utils import diff_interval

"""Module for scaling and rescaling of Laplace transforms for (all kinds of) models."""


class LinearScaling:

    def __init__(self, tau: np.ndarray, beta: float) -> None:
        """Initialize the LinearScaling.

        This is a parent class which is used to compute the scaling for Laplace transforms.

        Parameters:
            tau : ARRAY
                The nodes tau.
            beta : FLOAT
                The new right end point of the nodes tau.

        Raises:
            ValueError
                If the grid points are not given.
                If the grid points are not one-dimensional.
                If the left end point is greater or equal to the right end point."""

        # Type Conversion
        tau = np.asarray(tau).astype(FLOAT_DTYPE)
        beta = float(beta)

        # Integrity
        if not tau.ndim == 1:
            raise ValueError("The nodes must be a one-dimensional array.")
        if beta <= 0:
            raise ValueError("The right end point must be strictly positive.")

        # Compute tau1 and tau0 - scaling endpoints
        self._tau1 = np.max(tau) # TODO change to beta ...
        self._tau0 = 0.0

        # Compute the diffeomorphism of the interval
        self._psy, _ = diff_interval(self._tau1, self._tau0)

    @property
    def tau1(self) -> float:
        return self._tau1

    @tau1.setter
    def tau1(self, tau1: float):
        raise PermissionError("The tau1 parameter is read-only.")

    @property
    def tau0(self) -> float:
        return self._tau0

    @tau0.setter
    def tau0(self, tau0: float):
        raise PermissionError("The tau0 parameter is read-only.")

    @property
    def psy(self) -> callable:
        """The diffeomorphism onto the interval [0, b]."""
        return self._psy

    @psy.setter
    def psy(self, psy: callable):
        raise PermissionError("The psy parameter is read-only.")


class ForwardLinearScaling(LinearScaling):

    def __init__(self, tau: np.ndarray, beta: float) -> None:
        """Initialize the Forward Scaling."""

        super().__init__(tau, beta)

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
            return func(self.psy(tau), *args, **kwargs)

        return wrapper


class InverseLinearRescaling(LinearScaling):

    def __init__(self, tau: np.ndarray, beta: float) -> None:
        """Initialize the Inverse Rescaling."""

        super().__init__(tau, beta)

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
            return (
                (self.tau1 - self.tau0)
                * np.exp(self.tau0 * omega)
                * func((self.tau1 - self.tau0) * omega, *args, **kwargs)
            )

        return wrapper
