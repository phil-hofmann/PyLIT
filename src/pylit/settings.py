import numpy as np

INT_DTYPE = np.int64
FLOAT_DTYPE = np.float64
FASTMATH = False
TOL = FLOAT_DTYPE(f"{np.spacing(1.0):.1e}")
MAX_ITER = 100
TOL_LOG = 1e-3
MOMENT_ORDERS = [-1, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
