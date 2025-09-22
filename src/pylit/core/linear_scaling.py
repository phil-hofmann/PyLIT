import numpy as np

from pylit.settings import FLOAT_DTYPE
from pylit.utils import diff_interval


class LinearScaling:
    r"""Base class for linear scaling of Laplace/inverse Laplace transforms.

    This class sets up a diffeomorphism :math:`\psi`, which maps the time axis
    :math:`\tau` from the interval :math:`[\tau_0, \tau_1]` to :math:`[0, 1]`.
    """

    def __init__(self, tau: np.ndarray) -> None:
        r"""Initialize the LinearScaling class.

        Args:
            tau:
                The discretised time axis :math:`\tau`. Must be a
                one-dimensional array. The right endpoint ``tau1`` is
                taken as :math:`\max \tau`.

        Raises:
            ValueError: If the nodes are not one-dimensional.
            ValueError: If the right endpoint is not strictly positive.
        """

        # Type Conversion
        tau = np.asarray(tau).astype(FLOAT_DTYPE)
        beta = float(np.max(tau))

        # Integrity
        if not tau.ndim == 1:
            raise ValueError("The nodes must be a one-dimensional array.")
        if beta <= 0:
            raise ValueError("The right end point must be strictly positive.")

        # Scaling endpoints
        self._tau1 = beta
        self._tau0 = 0.0

        # Compute the diffeomorphism of the interval
        self._psy, _ = diff_interval(self._tau1, self._tau0)

    @property
    def tau1(self) -> float:
        """Right endpoint of the interval is always :math:`\max \tau`."""
        return self._tau1

    @tau1.setter
    def tau1(self, tau1: float):
        raise PermissionError("The tau1 parameter is read-only.")

    @property
    def tau0(self) -> float:
        """Left endpoint of the interval is always :math:`0.0`."""
        return self._tau0

    @tau0.setter
    def tau0(self, tau0: float):
        raise PermissionError("The tau0 parameter is read-only.")

    @property
    def psy(self) -> callable:
        """The diffeomorphism mapping the interval ``[tau0, tau1]`` onto ``[0, tau1]``."""
        return self._psy

    @psy.setter
    def psy(self, psy: callable):
        raise PermissionError("The psy parameter is read-only.")


class TauLinearScaling(LinearScaling):

    def __init__(self, tau: np.ndarray) -> None:
        """Initialize the TauLinearScaling by means of its superclass."""

        super().__init__(tau)

    def __call__(self, func):
        r"""Apply linear scaling to a Laplace transform.

        Given a Laplace transform :math:`F(\tau)`, this decorator
        scales the time axis using the diffeomorphism :math:`\psi`:

        .. math::

            F_{\text{scaled}}(\tau) = F(\psi(\tau))

        Args:
            func:
                A Laplace transformed function :math:`F(\tau)`.

        Returns:
            The scaled Laplace transform :math:`F_{\text{scaled}}(\tau)`.

        Examples:
            >>> import numpy as np
            >>> def f(tau): return np.exp(-tau)
            >>> scale = TauLinearScaling(tau=np.linspace(0, 8, 20))
            >>> scaled_f = scale(f)
            >>> scaled_f(0.5)
            np.exp(-scale.psy(0.5))
        """

        def wrapper(tau: np.ndarray, *args, **kwargs):
            return func(self.psy(tau), *args, **kwargs)

        return wrapper


class OmegaLinearScaling(LinearScaling):

    def __init__(self, tau: np.ndarray) -> None:
        """Initialize the OmegaLinearScaling by means of its superclass."""

        super().__init__(tau)

    def __call__(self, func):
        r"""Apply linear scaling to a model function.

        Given a model function :math:`f(\omega)`, this
        decorator scales both the frequency variable and amplitude:

        .. math::

            S_{\text{scaled}}(\omega)
            = (\tau_1 - \tau_0)\, e^{\tau_0 \omega}\,
              S\big((\tau_1 - \tau_0)\, \omega\big)

        where :math:`[\tau_0, \tau_1]` is the original time interval.


        Args:
            func:
                A model function :math:`S(\omega)`.

        Returns:
            The scaled model function :math:`S_{\text{scaled}}(\omega)`.

        Examples:
            >>> import numpy as np
            >>> def S(omega): return np.exp(-omega**2)
            >>> scale = OmegaLinearScaling(tau=np.linspace(0, 8, 20))
            >>> scaled_S = scale(S)
            >>> scaled_S(0.5)
            (scale.tau1 - scale.tau0) * np.exp(scale.tau0 * 0.5) * S((scale.tau1 - scale.tau0) * 0.5)
        """

        def wrapper(omega: np.ndarray, *args, **kwargs):
            return (
                (self.tau1 - self.tau0)
                * np.exp(self.tau0 * omega)
                * func((self.tau1 - self.tau0) * omega, *args, **kwargs)
            )

        return wrapper
