import numpy as np
from typing import List

from pylit.backend.utils import str_to_int_array, int_array_to_str
from pylit.global_settings import ARRAY, FLOAT_DTYPE, INT_DTYPE
from pylit.backend.models.rlrm import RegularLinearRegressionModel

class GaussRLRM(RegularLinearRegressionModel):

    """This is the regular linear regression model with Gaussian model functions.

    Model:

        :math:`M(\omega) = \left(\\varphi_\\alpha(\omega) = \left(\sigma_{\\alpha_2} \cdot \sqrt{2 \cdot \pi}\\right)^{-1} \cdot \\text{exp}\left(-\\frac{1}{2} \cdot \\frac{(\omega - \omega_{\\alpha_1})^2}{\sigma_{\\alpha_2}^2}  \\right) : \\alpha \in A\\right).`

    Params:

        :math:`\Theta_1 = [\omega_1, ..., \omega_{n_1}]` ... Support points or rather expected values\n
        :math:`\Theta_2 = [\sigma_1, ..., \sigma_{n_2}]` ... Standard deviations

    Multi-Index Set:
        
        :math:`A = \{(\\alpha_1, \\alpha_2) : \\alpha_1 \in \{1, ..., n_1\}, \\alpha_2 \in \{1, ..., n_2\})\}`
    
    Double Sided Laplace Transform:

        :math:`B(M)[\\tau] = \left(B(\\varphi_\\alpha)[\\tau] = \\text{exp}(-\\tau \cdot \omega_{\\alpha_1} + 0.5 \cdot \\tau^2 \cdot \sigma_{\\alpha_2}^2) : \\alpha \in A\\right).`
    
    Example Simulation:

        .. image:: ../../../notebooks/GaussRLRM/simulation/GIF/Model.gif
           :align: center    

    .. only:: not(html)

        Notes:
        ------
            - TODO: Unittest and doctest."""

    def __init__(self, omegas: ARRAY, sigmas:ARRAY, beta: FLOAT_DTYPE = 1.0, order: str = "0,1"):

        """Initialize the model."""

        # Integrity 
        if not isinstance(omegas, ARRAY):
            raise TypeError(f'The support points must be of type {ARRAY}.')
        if not isinstance(sigmas, ARRAY):
            raise TypeError(f'The standard deviations must be of type {ARRAY}.')
        if np.any(sigmas < 0.0):
            raise ValueError('The standard deviations must be non-negative.')
        # Type Conversion
        omegas = omegas.astype(FLOAT_DTYPE)
        sigmas = sigmas.astype(FLOAT_DTYPE)
        beta = FLOAT_DTYPE(beta)
        order = str(order)
        # Initialize the Parent Class
        if order == "0,1":
            super().__init__(name="GaussRLRM", params=[omegas, sigmas])
        elif order == "1,0":
            super().__init__(name="GaussRLRM", params=[sigmas, omegas])
        else:
            raise ValueError('The order must be "0,1" or "1,0".')
        # Set attributes
        self._beta = beta
        self._order = str_to_int_array(order)

    def copy(self, amount: INT_DTYPE = 1, decorators=None) -> List['GaussRLRM']:
        # NOTE Deprecated! 
        stencils = [
            GaussRLRM(self._params[self._order[0]], self._params[self._order[1]], self._beta, str(self._order[0]) + "," + str(self._order[1]))
            for _ in range(amount)
            ]
        
        for i in range(amount):
            stencils[i]._coeffs = self._coeffs.copy()
            stencils[i]._reg_mat = self._reg_mat.copy()
            stencils[i]._grid_points = self._grid_points.copy()

        if decorators is not None:
            for i in range(amount):
                for decorator in decorators:
                    stencils[i] = decorator(stencils[i])

        return stencils


    def _model_function(self, param: List[ARRAY], x: ARRAY) -> ARRAY:
        
        """The model function."""
        
        omega, sigma = param[self._order]

        # Using vectorization:
        return np.exp(self._beta/2 * x) * (1 / (sigma * np.sqrt(2 * np.pi)) ) * np.exp(-0.5 * (x - omega)**2 / sigma**2)
    
    def _compute_regression_matrix(self, grid_points: ARRAY) -> ARRAY:

        """Override."""

        # Alternative: return super()._compute_regression_matrix()

        # Initialise the regression matrix
        reg_mat = np.zeros((grid_points.shape[0], self._degree), dtype=FLOAT_DTYPE)

        for i in range(self._degree):
            mi = self._multi_index_set[i]
            omega, sigma = self._params[self._order[0]][mi[0]], self._params[self._order[1]][mi[1]]
        
            # Using vectorization:
            reg_mat[:, i] = np.exp(-(grid_points-self._beta/2) * omega + 0.5 * (grid_points-self._beta/2)**2 * sigma**2)

        return reg_mat


if __name__ == '__main__':
    
    pass