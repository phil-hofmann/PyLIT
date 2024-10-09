import numpy as np

""" This module contains the global settings of the Pylit package. """

MOMENT_ORDERS = [-1, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

# Typing
INT_DTYPE = np.int64
FLOAT_DTYPE = np.float64
ARRAY = np.ndarray

# Optimisation
TOL = FLOAT_DTYPE(f"{np.spacing(1.0):.1e}")
MAX_ITER = 100
TOL_LOG = 1e-3

# Numba
CACHE = False
PARALLEL = False
FASTMATH = True

# Plotting
FIGSIZE = (12, 6)
DPI = 300
COLOR_F = "#0068c9"
COLOR_F_SHADED = "rgba(0, 104, 201, 0.2)"
COLOR_S = "#50C878"
COLOR_S_SHADED = "rgba(80, 200, 120, 0.2)"
COLOR_D = "#FF6347"
COLOR_D_SHADED = "rgba(255, 99, 71, 0.2)"