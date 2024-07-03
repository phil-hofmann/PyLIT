import warnings
import numpy as np

from typing import List
from scipy.integrate import quad
from pylit.utils import generate_multi_index_set, to_string, to_dict
from pylit.models.ABC import RegularLinearRegressionModelABC
from pylit.global_settings import ARRAY, FLOAT_DTYPE, INT_DTYPE


class RegularLinearRegressionModel(RegularLinearRegressionModelABC):
    """This class represents the PARENT CLASS of the regular linear regression model.

    Notes:
    ------
    - NOTE This class is a parent class which should not be instantiated.
    - NOTE Please inherit from this class and override the following methods:
        - _model_function [REQUIRED]
        - _compute_regression_matrix [OPTIONAL]: e.g. to compute it with analytic expression
        - _generate_multi_index_set [OPTIONAL]: e.g. to work with a different type of multi index set
    """

    def __init__(self, name: str, params: List[ARRAY]):
        """Initializes the regular linear regression model.

        Parameters:
        -----------
        params : List[ARRAY]
            The parameters of the regular linear regression model."""
        super().__init__()
        self._init_(params)

    def _init_(self, params: List[ARRAY]) -> None:
        """Initializes the regular linear regression model.

        Raises
        ------
        TypeError
            If the parameters are not a list.
        TypeError
            If any parameter is not of type ARRAY.
        TypeError
            If the data type of the parameters is not a floating point data type.
        ValueError
            If the amount of parameters is less than one.
        ValueError
            If any parameter array not one-dimensional.
        ValueError
            If any parameter has less than one configuration.

        Notes:
        ------
        - NOTE This method is called by the constructor of the parent class.
        - NOTE The coefficients of the polynomial are initialized with zeros.
        - NOTE The degree of the polynomial is equal to the cardinality of the multi index set.
        """
        # Typing
        if not isinstance(params, List):
            raise TypeError("The parameters must be a list.")
        if not all(isinstance(itr_p, ARRAY) for itr_p in params):
            raise TypeError(f"Every parameter must be of type {ARRAY}.")
        if not all(itr_p.dtype == FLOAT_DTYPE for itr_p in params):
            raise TypeError(
                f"The data type of the parameters must be a floating point data type. ({FLOAT_DTYPE})"
            )
        # Integrity
        if len(params) == 0:
            raise ValueError("The amount of parameters must be at least one.")
        if any(len(itr_p.shape) != 1 for itr_p in params):
            raise ValueError("Every parameter array must be one dimensional.")
        if any(itr_p.shape[0] == 0 for itr_p in params):
            raise ValueError(
                "There need to be at least one configuration for every parameter."
            )
        # Determine
        spatial_dimension = 1
        params_dimension = len(params)
        params = [
            itr_p.astype(FLOAT_DTYPE)
            if isinstance(itr_p, ARRAY)
            else np.array(itr_p, dtype=FLOAT_DTYPE)
            for itr_p in params
        ]
        # Assign
        self._spatial_dimension = spatial_dimension
        self._params_dimension = params_dimension
        self._params = params
        # Determine
        multi_index_set = self._generate_multi_index_set()
        degree = multi_index_set.shape[0]
        coeffs = np.zeros(degree, dtype=FLOAT_DTYPE)
        model = [
            lambda x, mi=mi: self._model_function(
                np.array(
                    [params[j][mi[j]] for j in range(params_dimension)],
                    dtype=FLOAT_DTYPE,
                ),
                x,
            )
            for mi in multi_index_set
        ]
        # Assign
        self._multi_index_set = multi_index_set
        self._degree = degree
        self._coeffs = coeffs
        self._model = model

    def __str__(self) -> str:
        """Class string method."""
        return to_string(self)

    def to_dict(self) -> dict:
        """Class to dictionary method."""
        attr = to_dict(self)
        del attr["_model"]
        del attr["_model_function"]
        del attr["_compute_regression_matrix"]
        return attr

    @property
    def coeffs(self) -> ARRAY:
        """Coefficients."""
        return self._coeffs

    @coeffs.setter
    def coeffs(self, coeffs: ARRAY) -> None:
        # Typing
        if not isinstance(coeffs, ARRAY):
            raise TypeError(f"The coefficients must be of type {ARRAY}.")
        # TODO Coeffs can be two dimensional
        # if not coeffs.dtype == FLOAT_DTYPE:
        #     raise TypeError(
        #         "The data type of the coefficients must be a floating point data type."
        #     )
        # # Integrity
        # if not len(coeffs) == self._degree:
        #     raise ValueError(
        #         "The amount of coefficients must equal the degree of the model."
        #     )
        self._coeffs = coeffs

    @property
    def spatial_dimension(self) -> INT_DTYPE:
        """Spatial dimension."""
        return self._dimension

    @spatial_dimension.setter
    def spatial_dimension(self, spatial_dimension: INT_DTYPE) -> None:
        raise PermissionError(
            "The spatial dimension of the model cannot be changed. It is determined by the parameters."
        )

    @property
    def degree(self) -> INT_DTYPE:
        """Degree."""
        return self._degree

    @degree.setter
    def degree(self, degree: INT_DTYPE) -> None:
        raise PermissionError(
            "The degree of the model cannot be changed. It is determined by the parameters."
        )

    @property
    def grid_points(self) -> ARRAY:
        """Grid points."""
        return self._grid_points

    @grid_points.setter
    def grid_points(self, grid_points: ARRAY) -> None:
        """Grid points setter.

        Parameters
        ----------
        grid : ARRAY
            The grid of the regular linear regression model.

        Raises:
        -------
        TypeError
            If the grid points are not of type ARRAY.
        TypeError
            If the data type of the grid points is not a floating point data type.
        ValueError
            If the spatial dimension of the grid points is not one.

        Notes
        -----
        - NOTE Automatically triggers the update of the regression matrix, once it has been computed.
        - NOTE The spatial dimension of the grid is not allowed to change."""
        # Typing
        if not isinstance(grid_points, ARRAY):
            raise TypeError(f"The grid points must be of type {ARRAY}.")
        if not grid_points.dtype == FLOAT_DTYPE:
            raise TypeError(
                f"The data type of the grid points must be a floating point data type. ({FLOAT_DTYPE})"
            )
        # Integrity
        if not len(grid_points.shape) == 1:
            raise ValueError("The spatial dimension of the grid must be one.")
        # Assign
        self._grid_points = grid_points
        # Update the regression matrix if it has already been computed
        if self._reg_mat is not None:
            self._reg_mat = self._compute_regression_matrix(grid_points)

    @property
    def regression_matrix(self) -> ARRAY:
        """Regression matrix."""
        if self._reg_mat is None:
            # Integrity
            if self._grid_points is None:
                raise ValueError(
                    "The grid must be set before computing the regression matrix."
                )
            # Compute and assign
            self._reg_mat = self._compute_regression_matrix(self._grid_points)
        return self._reg_mat

    @regression_matrix.setter
    def regression_matrix(self, reg_mat: ARRAY) -> None:
        """Setter method for the regression matrix of the regular linear regression model.

        Parameters
        ----------
        reg_mat : ARRAY
            The regression matrix of the regular linear regression model."""
        # Warning
        warnings.warn(
            "WARNING: The regression matrix of the regular linear regression model has been set manually."
        )
        # Typing
        if not isinstance(reg_mat, ARRAY):
            raise TypeError(f"The regression matrix must be of type {ARRAY}.")
        if not reg_mat.dtype == FLOAT_DTYPE:
            raise TypeError(
                f"The data type of the regression matrix must be a floating point data type. ({FLOAT_DTYPE})"
            )
        # Integrity
        if not len(reg_mat.shape) == 2:
            raise ValueError(
                "The regression matrix must be a two dimensional NumPy array."
            )
        if not reg_mat.shape[0] == self._grid_points.shape[0]:
            raise ValueError(
                "The amount of rows of the regression matrix must equal the amount of grid points."
            )
        if not reg_mat.shape[1] == self._degree:
            raise ValueError(
                "The amount of columns of the regression matrix must equal the degree of the model."
            )
        # Assign
        self._reg_mat = reg_mat

    @property
    def params(self) -> List[ARRAY]:
        """Parameters."""
        return self._params

    @params.setter
    def params(self, params: List[ARRAY]) -> None:
        """Parameters setter.

        Parameters:
        -----------
        params : List[ARRAY]
            The parameters.

        Notes
        -----
        - NOTE Automatically triggers the update of all properties and the regression matrix, once it has been computed.
        - TODO Deprecated or not?
        - TODO Mapping for the coefficients.
        - TODO Adaptive method instead of re-initializing the whole model."""
        # Warning
        warnings.warn(
            "WARNING: The parameters of the regular linear regression model have been set manually. The whole model will be re-initialized."
        )
        warnings.warn(
            "WARNING: You must map the old coefficients to the new coefficients."
        )
        # Re-initialize the model
        self._init(params)
        # Mapping the old coefficients to the new coefficients
        # ... TODO ...
        # Update the regression matrix if it has already been computed
        if self._reg_mat is not None:
            self._reg_mat = self._compute_regression_matrix(self._grid_points)

    @property
    def params_dimension(self) -> INT_DTYPE:
        """Dimension of the parameter space."""
        return self._params_dimension

    @params_dimension.setter
    def params_dimension(self, params_dimension: INT_DTYPE) -> None:
        raise PermissionError(
            "The dimension of the model cannot be changed directly. It is determined by the parameters."
        )

    @property
    def multi_index_set(self) -> ARRAY:
        """Multi-Index Set."""
        return self._multi_index_set

    @multi_index_set.setter
    def multi_index_set(self, multi_index_set: ARRAY) -> None:
        raise PermissionError(
            "The multi index set of the model cannot be changed directly. It is determined by the _generate_multi_index_set method."
        )

    @property
    def model(self) -> List[callable]:
        """Model functions in the form of a list of callables."""
        return self._model

    @model.setter
    def model(self, model: List[callable]) -> None:
        raise PermissionError(
            "The model functions of the model cannot be changed directly. They are determined by the parameters."
        )

    def forward(self) -> ARRAY:
        """Method for evaluating the regular linear regression model at the given grid points.

        Returns
        -------
        ARRAY
            The values of the regular linear regression model at the given grid points.
        """

        if len(self._coeffs.shape) == 1:
            return self.regression_matrix.dot(self._coeffs)
        elif len(self._coeffs.shape) == 2:
            return (self.regression_matrix @ self._coeffs.T).T
        else:
            raise ValueError("The coefficients must be either one or two dimensional.")

    def __call__(self, input: ARRAY, matrix: bool = False) -> ARRAY:
        """Method for evaluating the regular linear regression model at the given input.

        Parameters
        ----------
        input : ARRAY
            The input at which the regular linear regression model should be evaluated.
            Please NOTE that the input array must be always an array of shape (n, spatial_dimension) where n is the amount of input points.

        Returns
        -------
        ARRAY
            The values of the regular linear regression model at the given input.
        """
        # Typing
        if not isinstance(input, ARRAY):
            raise TypeError(f"The input must be of type {ARRAY}.")
        if not input.dtype == FLOAT_DTYPE:
            raise TypeError(
                f"The data type of the input must be a floating point data type. ({FLOAT_DTYPE})"
            )
        if not isinstance(matrix, bool):
            raise TypeError("The matrix flag must be of type bool.")
        # Integrity
        if not len(input.shape) == 1:
            raise ValueError("The input array must be one dimensional.")
        # Compute the evaluation matrix
        E = np.array([phi(input) for phi in self._model]).T
        if matrix:
            return E
        else:
            if len(self._coeffs.shape) == 1:
                return E.dot(self._coeffs)
            elif len(self._coeffs.shape) == 2:
                return (E @ self._coeffs.T).T
            else:
                raise ValueError("The coefficients must be either one or two dimensional.")

    def _generate_multi_index_set(self) -> ARRAY:
        """Method for generating the multi index set.

        Returns
        -------
        ARRAY
            The multi index set of the regular linear regression model.

        Notes
        -----
        - NOTE This method can be overridden by the concrete child class to work with a different type of multi index set.
        """
        return generate_multi_index_set(
            self._params_dimension, [itr_p.shape[0] for itr_p in self._params]
        )

    def _compute_regression_matrix(self) -> ARRAY:
        """Method for computing the regression matrix of the regular linear regression model.

        Returns
        -------
        ARRAY
            The regression matrix of the laplace transformation of the regular linear regression model
            on the predefined grid points computed via the standard quadrature rule in scipy from -gp_max to gp_max,
            where gp_max is the maximum absolute value of the grid points.

        Notes
        -----
        - NOTE The regression matrix is independent of the values of the coefficients of the regular linear regression model.
        - TODO Detailed Balance?, gp_max? , runtime? ..."""
        # Integrity
        if self._grid_points is None:
            raise ValueError(
                "The grid points must be set before computing the regression matrix."
            )
        # Compute the regression matrix
        reg_mat = np.zeros(
            (self._grid_points.shape[0], self._degree), dtype=FLOAT_DTYPE
        )
        gp_max = np.abs(self._grid_points).max()
        for i, phi in enumerate(self._model):
            reg_mat[:, i] = np.array(
                [
                    quad(
                        lambda omega: np.exp(-t * omega) * phi(omega),
                        -gp_max,
                        gp_max,
                    )[0]
                    for t in self._grid_points
                ],
                dtype=FLOAT_DTYPE,
            )
        return reg_mat

    def _model_function(self, param: List[ARRAY], input: ARRAY) -> ARRAY:
        """Method for computing one model function for a given set of parameters.

        Parameters
        ----------
        param: List[ARRAY]
            The parameters of the model function.
        input: Array
            The input of the model function.

        Returns
        -------
        callable
            The model function of the regular linear regression model.

        Notes
        -----
        - NOTE This method must be overridden by the concrete child class to implement the actual model function.
        """
        raise NotImplementedError(
            "The model method must be implemented by the concrete child class."
        )


if __name__ == "__main__":
    pass
