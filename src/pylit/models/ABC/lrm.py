import abc
from typing import List
from pylit.global_settings import ARRAY, INT_DTYPE

class LinearRegressionModelABC(abc.ABC):

    """Abstract Base Class for the linear regression model.
       
    Notes
    -----
    The linear regression model is a linear model, which is used to approximate a function by a linear combination of model functions.
    Through the regression matrix, the coefficients of the linear model can be computed by solving a linear system of equations."""
    
    _grid_points: ARRAY = None
    _eval_mat: ARRAY = None
    _reg_mat: ARRAY = None

    @property
    @abc.abstractmethod
    def coeffs(self) -> ARRAY:

        """Abstract container which stores the coefficients.

        Returns
        -------
        ARRAY
            The coefficients of the linear regression model.

        Notes
        -----
        This is a placeholder of the ABC, which is overwritten by the concrete implementation."""

        pass

    @property
    @abc.abstractmethod
    def spatial_dimension(self) -> INT_DTYPE:

        """Abstract container which stores the spatial dimension.

        Returns
        -------
        INT_DTYPE
            The spatial dimension of the linear regression model.

        Notes
        -----
        This is a placeholder of the ABC, which is overwritten by the concrete implementation."""

        pass

    @property
    @abc.abstractmethod
    def degree(self) -> INT_DTYPE:

        """Abstract container which stores the degree.

        Returns
        -------
        INT_DTYPE
            The degree of the linear regression model.

        Notes
        -----
        This is a placeholder of the ABC, which is overwritten by the concrete implementation."""

        pass

    @property
    @abc.abstractmethod
    def grid_points(self) -> ARRAY:

        """Abstract container which stores the grid points.

        Returns
        -------
        ARRAY
            The grid points of the linear regression model.

        Notes
        -----
        This is a placeholder of the ABC, which is overwritten by the concrete implementation."""

        pass

    @property
    @abc.abstractmethod
    def regression_matrix(self) -> ARRAY:

        """Abstract function which provides the regression matrix.

        Returns
        -------
        ARRAY
            The regression matrix of the linear regression model.

        Notes
        -----
        This is a placeholder of the ABC, which is overwritten by the concrete implementation."""
        
        pass
    
    @grid_points.setter
    @abc.abstractmethod
    def grid_points(self, grid_points: ARRAY) -> None:

        """Abstract setter method for the grid points of the linear model.

        Parameters
        ----------
        grid_points : ARRAY
            The grid points of the linear regression model.

        Returns
        -------
        None

        Notes
        -----
        This is a placeholder of the ABC, which is overwritten by the concrete implementation."""
        
        pass

    @coeffs.setter
    @abc.abstractmethod
    def coeffs(self, coeffs: ARRAY) -> None:

        """Abstract setter method for the coefficients of the linear model.

        Parameters
        ----------
        coeffs : ARRAY
            The coefficients of the linear model.

        Returns
        -------
        None

        Notes
        -----
        This is a placeholder of the ABC, which is overwritten by the concrete implementation."""
        
        pass
    
    @abc.abstractmethod
    def forward(self) -> ARRAY:

        """Abstract function which provides the forward computation of the linear regression model.

        Returns
        -------
        ARRAY
            The result of the forward computation of the linear regression model.

        Notes
        -----
        This is a placeholder of the ABC, which is overwritten by the concrete implementation."""
        
        pass

    @abc.abstractmethod
    def __call__(self, input: ARRAY) -> ARRAY:

        """Abstract function which provides the evaluation at the given input.

        This function is called, if an instance of the linear regression model is called e.g. : ``lrm(input)``

        Parameters
        ----------
        input : ARRAY
            The input at which the linear regression model should be evaluated.

        Returns
        -------
        ARRAY
            The values of the linear regression model at the given input.

        Notes
        -----
        This is a placeholder of the ABC, which is overwritten by the concrete implementation."""

        pass

if __name__ == '__main__':
    pass