import unittest
import numpy as np
from pylit.data_loader import DataLoader
from pylit.ui.settings import PATH_DATA

""" Test the DataLoader class. """


class TestDataLoader(unittest.TestCase):

    def test_load_data(self):
        
        file_name = '/raw_F/F10.dat'
        file_path = PATH_DATA + file_name
        header = ["q", "tau", "F_r", "F_s", "F_i"]

        # test data loader class
        dl = DataLoader(file_name, header)
        dl.fetch()
        q, tau, F_r = dl("q", "tau", "F_r")

        og_data = np.genfromtxt(file_path, names=header)

        np.testing.assert_array_equal(tau, og_data['tau'])

if __name__ == '__main__':

    unittest.main()