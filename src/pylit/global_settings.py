import numpy as np

""" This module contains the global settings of the NexPy package. """


# Typing

INT_DTYPE = np.int64

FLOAT_DTYPE = np.float64

ARRAY = np.ndarray

# Testing


def EPS(x, y):
    return np.max(np.abs(x - y))


EPS_HIGH = 10**-18

EPS_MID = 10**-14

EPS_LOW = 10**-10

# Styling

# ...

# Optimisation

TOL = np.spacing(1.0)
MAX_ITER = 100
TOL_LOG = 1e-3

# Numba

CACHE, PARALLEL, FASTMATH = True, False, True

if __name__ == "__main__":
    pass
