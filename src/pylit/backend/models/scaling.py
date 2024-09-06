import numpy as np
from typing import List
from pylit.backend.utils import diff_interval
from pylit.backend.models.ABC import LinearRegressionModelABC
from pylit.global_settings import FLOAT_DTYPE, ARRAY

"""Module for scaling and rescaling of Laplace transforms for (all kinds of) models."""


class LinearScaling:

    def __init__(self, grid_points: ARRAY):
        """Initialize the LinearScaling.

        This is a parent class which is used to compute the scaling for Laplace transforms.

        Parameters
        ----------
        grid_points : ARRAY
            The grid points.

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

        # Type Conversion
        grid_points = grid_points.astype(FLOAT_DTYPE)

        # Compute tau1 and tau0 - scaling endpoints
        self._tau1 = np.max(grid_points)
        self._tau0 = np.min(grid_points)

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
        """The diffeomorphism onto the interval [0, 1]."""
        return self._psy


class ForwardLinearScaling(LinearScaling):

    def __init__(self, grid_points: ARRAY):
        """Initialize the Forward Scaling."""

        super().__init__(grid_points)

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

    def __init__(self, grid_points: ARRAY):
        """Initialize the Inverse Rescaling."""

        super().__init__(grid_points)

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
    lrm: LinearRegressionModelABC
) -> LinearRegressionModelABC:
    """Decorator for the Linear Regression Model.

    This decorator is used to scale the Linear Regression Model with given right end point.

    Parameters:
    -----------
    lrm : LinearRegressionModel
        The  linear regression model.

    Returns:
    --------
    LinearRegressionModel:
        The scaled linear regression model."""

    # Apply the scaling decorators to the regression matrix
    lrm._model_function = InverseLinearRescaling(
        grid_points=lrm._grid_points
    )(lrm._model_function)

    # Apply the scaling decorators to the regression matrix
    lrm._compute_regression_matrix = ForwardLinearScaling(
        grid_points=lrm._grid_points
    )(lrm._compute_regression_matrix)

    return lrm
