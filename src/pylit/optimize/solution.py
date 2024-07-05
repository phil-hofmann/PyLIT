import pylit
import numpy as np
from pylit.global_settings import FLOAT_DTYPE, ARRAY

class Solution:
    """Represents a solution to an optimization problem."""

    es = { # Expected Structure
        "_x": (ARRAY, FLOAT_DTYPE),
        "_abs_norm_x": FLOAT_DTYPE,
        "_eps": FLOAT_DTYPE,
        "_residuum": FLOAT_DTYPE,
    }

    def __init__(self, x: ARRAY, eps: FLOAT_DTYPE, residuum: FLOAT_DTYPE):
        self.x = x
        self.abs_norm_x = np.linalg.norm(x, ord=1) # L1 Norm
        self.eps = eps
        self.residuum = residuum

    @property
    def x(self) -> ARRAY:
        """Solution."""
        return self._x

    @x.setter
    def x(self, value: ARRAY):
        if not isinstance(value, ARRAY):
            raise TypeError(f"x must be a of type {ARRAY}.")
        if not isinstance(value[0], FLOAT_DTYPE):
            raise TypeError(f"x must be of type {FLOAT_DTYPE}.")
        self._x = value

    @property
    def abs_norm_x(self) -> FLOAT_DTYPE:
        """Absolute norm of the solution."""
        return self._abs_norm_x

    @abs_norm_x.setter
    def abs_norm_x(self, value: FLOAT_DTYPE):
        if not isinstance(value, FLOAT_DTYPE):
            raise TypeError(f"abs_norm_x must be of type {FLOAT_DTYPE}.")
        self._abs_norm_x = value

    @property
    def eps(self) -> FLOAT_DTYPE:
        """Epsilon."""
        return self._eps
    
    @eps.setter
    def eps(self, value: FLOAT_DTYPE):
        # if not isinstance(value, FLOAT_DTYPE):
        #     raise TypeError(f"eps must be of type {FLOAT_DTYPE}.")
        self._eps = value
    
    @property
    def residuum(self) -> FLOAT_DTYPE:
        """Residuum."""
        return self._residuum

    @residuum.setter
    def residuum(self, value: FLOAT_DTYPE):
        if not isinstance(value, FLOAT_DTYPE):
            raise TypeError(f"residuum must be of type {FLOAT_DTYPE}.")
        self._residuum = value

    def __str__(self) -> str:
        """String method for the Solution

        Returns
        -------
        str
            The string representation of Solution."""

        return pylit.utils.to_string(self)
    
    def to_dict(self) -> dict:
        """Converts the Solution to a dictionary.

        Returns
        -------
        dict
            The Solution as a dictionary."""
        
        mydict = pylit.utils.to_dict(self)
        del mydict["class_name"]

        return mydict