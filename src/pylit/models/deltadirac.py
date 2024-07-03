import numpy as np
from typing import List

from pylit.global_settings import ARRAY, FLOAT_DTYPE, INT_DTYPE
from pylit.models.rlrm import RegularLinearRegressionModel


class DeltaDiracRLRM(RegularLinearRegressionModel):

    """This is the regular linear regression model with delta Dirac model functions.

    Model:

        :math:`M(\omega) = \left(\\varphi_\\alpha(\omega) = ??? : \\alpha \in A\\right).`

    Params:

        :math:`\Theta_1 = [\omega_1, ..., \omega_{n_1}]` ... Left end points\n
        :math:`\Theta_2 = [\omega_2, ..., \omega_{n_2}]` ... Right end points

    Multi-Index Set:
        
        :math:`A = \{(\\alpha_1) : \\alpha_1 \in \{1, ..., n_1\})\}`
    
    Double Sided Laplace Transform:

        :math:`B(M)[\\tau] = \left(B(\\varphi_\\alpha)[\\tau] = \\text{exp}(-\\tau \cdot \omega_{\\alpha_1}) : \\alpha \in A\\right).`
        
    .. only:: not(html)

        Notes:
        ------
            - TODO: Unittest and doctest."""

    def __init__(self, omegas: ARRAY):

        """Initialize the regular linear regression model with delta Dirac model functions.
        
        Parameters
        ----------
        omegas : ARRAY
            The support points of the delta Dirac model functions.

        Raises
        ------
        ValueError
            If the support points are non-increasing."""

        # Integrity
        if not all(np.diff(omegas) > 0):
            raise ValueError("The support points must be non-increasing.")
        
        # Type Conversion
        omegas = omegas.astype(FLOAT_DTYPE)
        
        # Initialize the Parent Class
        super().__init__(params=[omegas[:-1], omegas[1:]])

    def _generate_multi_index_set(self) -> ARRAY:
    
        """Method for generating the multi index set.

        Returns
        -------
        ARRAY
            Returns the multi index set of the regular linear regression model.

        .. only:: not(html)
        
            Notes
            -----
            - NOTE This method can be overridden by the concrete child class to work with a different type of multi index set."""

        return np.array([[i, i] for i in range(len(self.params[0]))], dtype=INT_DTYPE)

    def _model_function(self, param: List[ARRAY], x: ARRAY) -> ARRAY:
        
        """The delta Dirac model function.
        
        Parameters
        ----------
        param : List[ARRAY]
            A parameter specification of the delta Dirac model function that is a pair [a, b] with a < b.

        x : ARRAY
            The input of the delta Dirac model function.
        
        Returns
        -------
        ARRAY
            Returns the evaluation of the corresponding delta Dirac model function."""
        
        a, b = param

        # Using vectorization:
        return np.heaviside(x - a, 1.) - np.heaviside(x - b, 1.)
    
    def _compute_regression_matrix(self, grid_points: ARRAY) -> ARRAY:

        """Override."""

        # Alternative: return super()._compute_regression_matrix()

        # Initialise the regression matrix
        reg_mat = np.zeros((grid_points.shape[0], self._degree), dtype=FLOAT_DTYPE)

        for i in range(self._degree):
            mi = self._multi_index_set[i]
            a, b = self._params[0][mi[0]], self._params[1][mi[1]]

            # Use midpoints for the delta Dirac double sided Laplace transform
            midpoint = (a + b) / 2
            print(midpoint)
        
            # Using vectorization:
            reg_mat[:, i] = np.exp(-grid_points * midpoint)

        return reg_mat
    
        
if __name__ == '__main__':
    
    pass