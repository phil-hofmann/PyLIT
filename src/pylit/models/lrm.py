import warnings
import numpy as np

import numpy as np
from typing import List
from pylit.settings import FLOAT_DTYPE, INT_DTYPE
from pylit.utils import generate_multi_index_set, to_string


class LinearRegressionModel:
    """
    Notes:
    - NOTE This class is a parent class which should not be instantiated.
    - NOTE Please inherit from this class and override the following methods:
        - kernel
        - ltransform
    """

    def __init__(
        self,
        name: str,
        tau: np.ndarray,
        params: List[np.ndarray],
    ):
        """Initializes the linear regression model.

        Args:
            name : str
                The name of the linear regression model.
            tau : np.ndarray
                The nodes of the linear regression model.
            params : List[np.ndarray]
                The parameters of the linear regression model.
        """
        super().__init__()
        self.tau = tau
        self.params = params
        self._pdim = len(params)
        self._multi_index_set = self._generate_multi_index_set()
        self._degree = self._multi_index_set.shape[0]
        self._coeffs = np.zeros(self._degree, dtype=FLOAT_DTYPE)

    @property
    def tau(self) -> np.ndarray:
        """Nodes."""
        return self._tau

    @tau.setter
    def tau(self, tau: np.ndarray) -> None:
        """Nodes setter.

        Args:
            tau : np.ndarray
                The nodes of the linear regression model.

        Raises:
            ValueError
                If the nodes are None.
            ValueError
                If the nodes are not one-dimensional.
        """

        # Warning
        if hasattr(self, "_reg_mat") and self._reg_mat is not None:
            warnings.warn("WARNING: You must re-compute the regression matrix.")

        # Typing
        if tau is None:
            raise ValueError("The nodes cannot be None.")

        # Type Conversion
        tau = np.asarray(tau).astype(FLOAT_DTYPE)

        # Integrity
        if not tau.ndim == 1:
            raise ValueError("The spatial dimension of the nodes must be one.")

        # Assign
        self._tau = tau

    @property
    def params(self) -> List[np.ndarray[FLOAT_DTYPE]]:
        """Parameters."""
        return self._params

    @params.setter
    def params(self, params: List[np.ndarray[FLOAT_DTYPE]]) -> None:
        """Parameters setter.

        Args:
            params : List[np.ndarray]
                The parameters of the linear regression model.

        Raises:
            TypeError
                If the parameters are not a list.
            ValueError
                If the amount of parameters is less than one.
            ValueError
                If any parameter array is not one-dimensional.
            ValueError
                If any parameter has less than one configuration.
        """

        # Warning
        if hasattr(self, "_coeffs") and np.linalg.norm(self.coeffs) > 0:
            warnings.warn(
                "WARNING: You must map the old coefficients to the new coefficients."
            )
        if hasattr(self, "_reg_mat") and self._reg_mat is not None:
            warnings.warn("WARNING: You must recompute the regression matrix.")

        # Typing
        if params is None:
            raise TypeError("The parameters cannot be None.")
        if not isinstance(params, List):
            raise TypeError("The parameters must be a list.")

        # Type Conversion
        params = [np.asarray(param).astype(FLOAT_DTYPE) for param in params]

        # Integrity
        if len(params) == 0:
            raise ValueError("The amount of parameters must be at least one.")
        if any(len(itr_p.shape) != 1 for itr_p in params):
            raise ValueError("Every parameter array must be one dimensional.")
        if any(itr_p.shape[0] == 0 for itr_p in params):
            raise ValueError(
                "There need to be at least one configuration for every parameter."
            )

        # Assign
        self._params = params

    @property
    def pdim(self) -> INT_DTYPE:
        """Dimension of the parameter space."""
        return self._pdim

    @pdim.setter
    def pdim(self, pdim: INT_DTYPE) -> None:
        raise PermissionError(
            "The dimension of the model cannot be changed directly. It is determined by the parameters."
        )

    @property
    def multi_index_set(self) -> np.ndarray[INT_DTYPE]:
        """Multi-Index Set."""
        return self._multi_index_set

    @multi_index_set.setter
    def multi_index_set(self, multi_index_set: np.ndarray[INT_DTYPE]) -> None:
        raise PermissionError(
            "The multi index set of the model cannot be changed directly. It is determined by the _generate_multi_index_set method."
        )

    @property
    def degree(self) -> INT_DTYPE:
        """Degree."""
        return self._degree

    @degree.setter
    def degree(self, degree: INT_DTYPE) -> None:
        """Degree setter."""
        raise PermissionError(
            "The degree of the model cannot be changed. It is determined by the parameters."
        )

    @property
    def coeffs(self) -> np.ndarray:
        """Coefficients."""
        return self._coeffs

    @coeffs.setter
    def coeffs(self, coeffs: np.ndarray[FLOAT_DTYPE]) -> None:
        """Coefficients setter."""

        # Type Conversion
        coeffs = np.asarray(coeffs).astype(FLOAT_DTYPE)

        # Integrity
        if not len(coeffs) == self._degree:
            raise ValueError(
                "The amount of coefficients must equal the degree of the model."
            )

        self._coeffs = coeffs

    @property
    def regression_matrix(self) -> np.ndarray:
        """Regression matrix."""
        return self._reg_mat

    @regression_matrix.setter
    def regression_matrix(self, reg_mat: np.ndarray) -> None:
        """Setter for the regression matrix.

        Args:
            reg_mat : np.ndarray
                The regression matrix.
        """

        # Warning
        if hasattr(self, "_reg_mat") and self._reg_mat is not None:
            warnings.warn(
                "WARNING: The regression matrix of the linear regression model has been set manually."
            )

        # Type Conversion
        reg_mat = np.asarray(reg_mat).astype(FLOAT_DTYPE)

        # Integrity
        if not reg_mat.ndim == 2:
            raise ValueError("The regression matrix must be a two dimensional array.")
        if not reg_mat.shape[0] == self._tau.shape[0]:
            raise ValueError(
                "The amount of rows of the regression matrix must equal the amount of nodes."
            )
        if not reg_mat.shape[1] == self._degree:
            raise ValueError(
                "The amount of columns of the regression matrix must equal the degree of the model."
            )

        # Assign
        self._reg_mat = reg_mat

    def __str__(self) -> str:
        """Class string method."""
        return to_string(self)

    def kernel(
        self,
        omega: np.ndarray,
        param: List[np.ndarray],
    ) -> np.ndarray[FLOAT_DTYPE]:
        """Method for computing one kernel function for a given set of parameters.

        Args:
            omega: np.ndarray
                The input of the model function.
            param: List[np.ndarray]
                The parameters of the model function.

        Returns:
            np.ndarray
                The value of the model function at the given input.

        Notes:
            - NOTE This method must be overridden by the concrete child class to implement the actual model function.
        """

        raise NotImplementedError(
            "The kernel method must be implemented by the concrete child class."
        )

    def ltransform(
        self,
        tau: np.ndarray,
        param: List[np.ndarray],
    ) -> np.ndarray:
        """Method for computing the Laplace transformation of the model function.

        Args:
            tau: np.ndarray
                The input of the Laplace transformation.
            param: List[np.ndarray]
                The parameters of the Laplace transformation.

        Notes:
            - NOTE This method must be overridden by the concrete child class to implement the actual model function.
        """

        raise NotImplementedError(
            "The ltransform method must be implemented by the concrete child class."
        )

    def forward(self) -> np.ndarray:
        """Method for evaluating the linear regression model at the nodes.

        Returns:
            np.ndarray
                The values of the linear regression model at the nodes.
        """

        return self.regression_matrix.dot(self._coeffs)

    def __call__(self, omega: np.ndarray, matrix: bool = False) -> np.ndarray:
        """Method for evaluating the linear regression model at the given input.

        Args:
            omega: np.ndarray
                The input at which the linear regression model should be evaluated.

        Returns:
            np.ndarray
                The values of the linear regression model at the given omegas.
        """
        # Type Conversion
        omega = np.asarray(omega).astype(FLOAT_DTYPE)

        # Integrity
        if not omega.ndim == 1:
            raise ValueError(f"The omega array must be one dimensional.")

        # Compute the evaluation matrix
        E = np.array(
            [
                self.kernel(
                    omega,
                    np.array(
                        [self._params[j][mi[j]] for j in range(self._pdim)],
                        dtype=FLOAT_DTYPE,
                    ),
                )
                for mi in self._multi_index_set
            ]
        ).T

        if matrix:
            return E
        return E.dot(self._coeffs)

    def _generate_multi_index_set(self) -> np.ndarray:
        """Method for generating the multi index set.

        Returns:
            np.ndarray
                The multi index set of the linear regression model.
        """

        return generate_multi_index_set(
            self._pdim, [itr_p.shape[0] for itr_p in self._params]
        )

    def compute_regression_matrix(self) -> np.ndarray:
        """Computation of the regression matrix.

        Returns:
            np.ndarray
                The regression matrix of the laplace transformation of the linear regression model
                on the predefined grid points computed via the standard quadrature rule in scipy from -gp_max to gp_max,
                where gp_max is the maximum absolute value of the grid points.

        Notes:
            - NOTE The regression matrix is independent of the values of the coefficients of the linear regression model.
        """

        tau, params = self._tau, self._params
        reg_mat = np.zeros((tau.shape[0], self._degree), dtype=FLOAT_DTYPE)

        for i in range(self._degree):
            mi = self._multi_index_set[i]
            param = [params[j][mi[j]] for j in range(self._pdim)]
            row = self.ltransform(tau, param)
            reg_mat[:, i] = row
        self._reg_mat = reg_mat
