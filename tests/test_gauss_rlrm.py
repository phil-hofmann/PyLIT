import pylit
import unittest
import numpy as np
from pylit.global_settings import FLOAT_DTYPE

class TestRLRMGauss(unittest.TestCase):

    """ Test the Regular Linear Regression Model with Gaussian Basis Functions. """

    def test_init(self):
        """ Test the __init__ method of the GaussianRLRM class for spatial dimension one. """
        
        n, m = np.random.randint(1, 10, 2)
        omegas = np.random.rand(n).astype(FLOAT_DTYPE).reshape(-1, 1)
        sigmas = np.random.rand(m).astype(FLOAT_DTYPE).reshape(-1, 1)
        gauss_rlr = pylit.models.GaussianRLRM(omegas, sigmas)

        self.assertTrue(
            all(gauss_rlr.params[1] == sigmas) and all(gauss_rlr.params[0] == omegas)
        )
            
    def test_model(self):
        """ Test the _model method of the GaussianRLRM class for spatial dimension one. """
        
        n, m = 1, 1
        omegas = np.random.rand(n).astype(FLOAT_DTYPE).reshape(-1, 1)
        sigmas = np.random.rand(m).astype(FLOAT_DTYPE).reshape(-1, 1)
        gauss_rlr = pylit.models.GaussianRLRM(omegas, sigmas)
        
        gauss_basis = gauss_rlr._model([omegas, sigmas])
        
        self.assertEqual(gauss_basis(0.0), 1 / (np.sqrt(2 * np.pi)*sigmas[0] ) * np.exp(-0.5 * (0.0 - omegas[0])**2 / sigmas[0]**2))
        
    def test_compute_regression_matrix(self):
        """ Test the _compute_regression_matrix method of the GaussianRLRM class. """
        
        # TODO
        
    def test_compute_regression_matrix(self):
        """ Test the _compute_regression_matrix method of the GaussianRLRM class. """
        
        # TODO

if __name__ == '__main__':
    
    unittest.main()