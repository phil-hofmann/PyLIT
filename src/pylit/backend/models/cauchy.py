import numpy as np
from typing import List

from pylit.backend.utils import str_to_int_array, int_array_to_str
from pylit.global_settings import ARRAY, FLOAT_DTYPE, INT_DTYPE
from pylit.backend.models.rlrm import RegularLinearRegressionModel


class CauchyRLRM(RegularLinearRegressionModel):
    """This is the regular linear regression model with Laplace model functions.

    TODO"""

    def __init__(
        self, omegas: ARRAY, gamma: ARRAY, beta: FLOAT_DTYPE = 1.0, order: str = "0,1"
    ):
        """Initialize the model."""
        # Integrity
        if not isinstance(omegas, ARRAY):
            raise TypeError(f"The support points must be of type {ARRAY}.")
        if not isinstance(gamma, ARRAY):
            raise TypeError(f"The parameter 'b' must be of type {ARRAY}.")
        if np.any(gamma <= 0.0):
            raise ValueError("The parameter 'gamma' must be positive.")
        # Type Conversion
        omegas = omegas.astype(FLOAT_DTYPE)
        gamma = gamma.astype(FLOAT_DTYPE)
        beta = FLOAT_DTYPE(beta)
        order = str(order)
        # Initialize the Parent Class
        if order == "0,1":
            super().__init__(name="LaplaceRLRM", params=[omegas, gamma])
        elif order == "1,0":
            super().__init__(name="LaplaceRLRM", params=[gamma, omegas])
        else:
            raise ValueError('The order must be "0,1" or "1,0".')
        # Set attributes
        self._beta = beta
        self._order = str_to_int_array(order)

    def _model_function(self, param: List[ARRAY], x: ARRAY) -> ARRAY:
        """The model function."""

        omega, gamma = param[self._order]
        scaling = np.exp(self._beta / 2 * x)

        # Using vectorization:
        return scaling / (np.pi * gamma * (1 + ((x - omega) / gamma)**2))

    def _compute_regression_matrix(self, grid_points: ARRAY) -> ARRAY:
        """Override."""

        # Alternative: return super()._compute_regression_matrix()

        # Initialise the regression matrix
        reg_mat = np.zeros((grid_points.shape[0], self._degree), dtype=FLOAT_DTYPE)

        for i in range(self._degree):
            mi = self._multi_index_set[i]
            omega, gamma = (
                self._params[self._order[0]][mi[0]],
                self._params[self._order[1]][mi[1]],
            )

            # Using vectorization:
            tau = grid_points - self._beta / 2
            reg_mat[:, i] = np.exp((- tau * omega) - gamma * np.abs(tau))
            # NOTE tau will default in (0, 1)

        return reg_mat


if __name__ == "__main__":

    pass
