import numpy as np
from pylit.settings import FLOAT_DTYPE


class DetailedBalance:

    r"""Base class for detailed balance computations.

    This class stores the inverse temperature parameter :math:`\beta`, given as the
    maximum of the provided time axis :math:`\tau`."""

    def __init__(self, tau: np.ndarray) -> None:
        """Initialize the DetailedBalance class.

        Args:
            tau:
                The time axis.

        Raises:
            ValueError:If the beta is not positive.
        """

        # Type Conversion
        tau = np.asarray(tau).astype(FLOAT_DTYPE)

        # Compute beta
        self._beta = np.max(tau)

        # Integrity
        if self._beta <= 0:
            raise ValueError("The beta must be strictly positive.")

    @property
    def beta(self) -> float:
        """Inverse temperature parameter."""
        return self._beta

    @beta.setter
    def beta(self, beta: float):
        raise PermissionError("The beta parameter is read-only.")


class TauDetailedBalance(DetailedBalance):

    def __init__(self, tau: np.ndarray) -> None:
        """Initialize the TauDetailedBalance by means of its superclass."""

        super().__init__(tau)

    def __call__(self, func):
        r"""
        Apply detailed balance to a Laplace transform.

        Given a Laplace transform :math:`F(\tau)`, this decorator returns a
        symmetrized version of the function:

        .. math::

            F_{DB}(\tau) = F(\beta - \tau) + F(\tau)

        where :math:`\beta = \max \tau`.

        Args:
            func:
                A Laplace transform :math:`F(\tau)`.

        Returns:
            The detailed balanced Laplace transform.

        Examples:
            >>> import numpy as np
            >>> def f(tau): return np.exp(-tau)
            >>> db = ForwardDetailedBalance(tau=np.array([2.0]))
            >>> balanced = db(f)
            >>> balanced(0.5)
            np.exp(-0.5) + np.exp(-(2.0 - 0.5))
        """

        def wrapper(tau: np.ndarray, *args, **kwargs):
            return func(self.beta - tau, *args, **kwargs) + func(tau, *args, **kwargs)

        return wrapper


class OmegaDetailedBalance(DetailedBalance):

    def __init__(self, tau: np.ndarray) -> None:
        """Initialize the OmegaDetailedBalance by means of its superclass."""

        super().__init__(tau)

    def __call__(self, func):
        r"""Apply detailed balance to a kernel function.

        Given a kernel function :math:`S(\omega)`, this decorator
        rescales it as:

        .. math::

            S_{DB}(\omega) = e^{\beta \omega} S(-\omega) + S(\omega)

        where :math:`\beta = \max \tau`.

        Args:
            func:
                An kernel function :math:`S(\omega)`.

        Returns:
            The kernel function fulfilling the detailed balance with respect to :math:`\beta`

        Examples:
            >>> import numpy as np
            >>> def S(omega): return np.exp(-omega**2)
            >>> db = InverseDetailedBalance(tau=np.array([1.0]))
            >>> balanced = db(S)
            >>> balanced(0.5)
            np.exp(1.0 * 0.5) * np.exp(-(-0.5)**2) + np.exp(-(0.5)**2)
        """

        def wrapper(omega: np.ndarray, *args, **kwargs):
            return np.exp(self.beta * omega) * func(-omega, *args, **kwargs) + func(
                omega, *args, **kwargs
            )

        return wrapper
