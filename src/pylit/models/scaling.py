import numpy as np
from typing import List
from pylit.utils import diff_interval
from pylit.models.ABC import LinearRegressionModelABC
from pylit.global_settings import FLOAT_DTYPE, ARRAY

"""Module for scaling and rescaling of Laplace transforms for (all kinds of) models."""


class LinearScaling:

    def __init__(self, grid_points: ARRAY, right_end_point: FLOAT_DTYPE = 1.0):
        """Initialize the LinearScaling.

        This is a parent class which is used to compute the scaling for Laplace transforms.

        Parameters
        ----------
        grid_points : ARRAY
            The grid points.

        right_end_point : FLOAT_DTYPE
            The right end point of the scaling.

        Raises
        ------
        ValueError
            If the grid points are not given.
            If the grid points are not one-dimensional.
            If the left end point is greater or equal to the right end point."""

        # Integrity

        if grid_points is None:
            raise ValueError("The grid points must be given.")

        if not len(grid_points.shape) == 1:
            raise ValueError("The grid points must be a one-dimensional array.")

        if right_end_point < 0.0:
            raise ValueError("The right end point must be greater or equal to zero.")

        # Type Conversion
        grid_points = grid_points.astype(FLOAT_DTYPE)
        right_end_point = FLOAT_DTYPE(right_end_point)

        # Compute tau1 and tau0 - scaling endpoints
        self._tau1 = np.max(grid_points) / right_end_point
        self._tau0 = np.min(grid_points) / right_end_point

        # Compute the diffeomorphism of the interval
        self._psy, _ = diff_interval(self._tau1, self._tau0)

    @property
    def tau1(self) -> FLOAT_DTYPE:
        return self._tau1

    @property
    def tau0(self) -> FLOAT_DTYPE:
        return self._tau0

    @property
    def psy(self) -> callable:
        """The diffeomorphism onto the interval [0, b]."""
        return self._psy


class ForwardLinearScaling(LinearScaling):

    def __init__(self, grid_points: ARRAY, right_end_point: FLOAT_DTYPE = 1.0):
        """Initialize the Forward Scaling."""

        super().__init__(grid_points, right_end_point)

    def __call__(self, func):
        """Returns the scaled version of the given Laplace transform.

        Parameters
        ----------
        func : callable
            The Laplace transform.

        Returns
        -------
        callable
            The scaled Laplace transform."""

        def wrapper(input):
            return func(self._psy(input))

        return wrapper


class InverseLinearRescaling(LinearScaling):

    def __init__(self, grid_points: ARRAY, right_end_point: FLOAT_DTYPE = 1.0):
        """Initialize the Inverse Rescaling."""

        super().__init__(grid_points, right_end_point)

    def __call__(self, func):
        """Returns the rescaled version of the given inverse Laplace transform.

        Parameters
        ----------
        func : callable
            The inverse Laplace transform.

        Returns
        -------
        callable
            The rescaled inverse Laplace transform."""

        def wrapper(params: List[ARRAY], input: ARRAY):
            return (
                (self._tau1 - self._tau0)
                * np.exp(self._tau0 * input)
                * func(params, (self._tau1 - self._tau0) * input)
            )

        return wrapper


def linear(
    lrm: LinearRegressionModelABC, right_end_point: FLOAT_DTYPE = 1.0
) -> LinearRegressionModelABC:
    """Decorator for the Linear Regression Model.

    This decorator is used to scale the Linear Regression Model with given right end point.

    Parameters:
    -----------
    lrm : LinearRegressionModel
        The  linear regression model.

    right_end_point : FLOAT_DTYPE
        The right end point ``r`` of the interval. Default is 1.0.

    Returns:
    --------
    LinearRegressionModel:
        The scaled linear regression model."""

    # Apply the scaling decorators to the regression matrix
    lrm._model_function = InverseLinearRescaling(
        grid_points=lrm._grid_points, right_end_point=right_end_point
    )(lrm._model_function)

    # Apply the scaling decorators to the regression matrix
    lrm._compute_regression_matrix = ForwardLinearScaling(
        grid_points=lrm._grid_points, right_end_point=right_end_point
    )(lrm._compute_regression_matrix)

    return lrm
