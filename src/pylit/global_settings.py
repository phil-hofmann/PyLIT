import numpy as np

""" This module contains the global settings of the Pylit package. """

# Typing
INT_DTYPE = np.int64
FLOAT_DTYPE = np.float64
ARRAY = np.ndarray

# Optimisation
TOL = np.spacing(1.0)
MAX_ITER = 100
TOL_LOG = 1e-3

# Numba
CACHE = True
PARALLEL = False
FASTMATH = True
