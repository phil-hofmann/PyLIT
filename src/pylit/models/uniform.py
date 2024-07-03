from typing import List
import numpy as np
from pylit.models.rlrm import RegularLinearRegressionModel
from pylit.global_settings import ARRAY, FLOAT_DTYPE, INT_DTYPE


class UniformRLRM(RegularLinearRegressionModel):

    """This is the regular linear regression model with uniform model functions.

    Model:

        :math:`M(\omega) = \left(\\varphi_\\alpha(\omega) = \\frac{1}{\omega_{\\alpha_2} - \omega_{\\alpha_1}} \cdot 1_{[\omega_{\\alpha_1}, \omega_{\\alpha_2}]}(\omega) : \\alpha \in A\\right).`

    Params:

        :math:`\Theta_1 = [\omega_1, ..., \omega_{n_1}]` ... Left end points\n
        :math:`\Theta_2 = [\omega_2, ..., \omega_{n_2}]` ... Right end points

    Multi-Index Set:

        :math:`A = \{(0,0), (1,1), ..., (n_1, n_2)\}`
        
    Double Sided Laplace Transform:

        :math:`B(M)[\\tau] = \left(B(\\varphi_\\alpha)[\\tau] = \\frac{1}{\\tau \cdot (\omega_{\\alpha_2} - \omega_{\\alpha_1})} \cdot (\\text{exp}(\omega_{\\alpha_2}\cdot\\tau)-\\text{exp}(\omega_{\\alpha_1}\cdot\\tau)) : \\alpha \in A\\right).`

    .. only:: not(html)

        Notes
        -----
        TODO: Unittest and doctest."""

    def __init__(self, omegas: ARRAY, beta: FLOAT_DTYPE = 1.0):

        """Initialize the model."""

        # Integrity
        if not all(np.diff(omegas) > 0):
            raise ValueError("The support points must be non-increasing.")
        
        # Type Conversion
        omegas = omegas.astype(FLOAT_DTYPE)
        
        # Initialize the Parent Class
        super().__init__(name="UniformRLRM", params=[omegas[:-1], omegas[1:]])
        # Set attributes
        self._beta = beta

    def _generate_multi_index_set(self) -> ARRAY:
    
        """Method for generating the multi index set."""

        return np.array([[i, i] for i in range(len(self.params[0]))], dtype=INT_DTYPE)
    
    def _model_function(self, param: List[ARRAY], x: ARRAY) -> ARRAY:
        
        """The model function."""
        
        a, b = param

        # Using vectorization:
        return np.exp(self._beta/2 * x)  * (np.heaviside(x - a, 1.) - np.heaviside(x - b, 1.))/(b-a)
    
    def _compute_regression_matrix(self, grid_points: ARRAY) -> ARRAY:

        """Override."""

        if any(grid_points == 0.0):
            raise ValueError('The grid points are not allowed to include the point 0.0 due to the division by zero in the Laplace transform.')

        # Alternative: return super()._compute_regression_matrix()
    
        reg_mat = np.zeros((grid_points.shape[0], self._degree), dtype=FLOAT_DTYPE)

        for i in range(self._degree):
            
            mi = self._multi_index_set[i]
            a, b = self._params[0][mi[0]], self._params[1][mi[1]]
        
            # Using vectorization:
            reg_mat[:, i] = (np.exp(b * (grid_points-self._beta/2)) - np.exp(a * (grid_points-self._beta/2)))/((b-a)*(grid_points-self._beta/2))

        return reg_mat
    
        
if __name__ == '__main__':
    
    pass