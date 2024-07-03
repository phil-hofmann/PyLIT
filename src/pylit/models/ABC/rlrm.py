import abc
from typing import List
from pylit.global_settings import ARRAY, INT_DTYPE
from pylit.models.ABC.lrm import LinearRegressionModelABC

class RegularLinearRegressionModelABC(LinearRegressionModelABC):

    """Abstract Base Class for the regular linear regression model.
    
    Notes
    -----
    The regular linear regression model is a linear regression model with a multi index set.
    The multi index set indicates the regularity of the linear models parameter space."""

    @property
    @abc.abstractmethod
    def params(self) -> List[ARRAY]:

        """Abstract container which stores the parameters.

        Returns
        -------
        ARRAY
            The parameters of the regular linear regression model.

        Notes
        -----
        This is a placeholder of the ABC, which is overwritten by the concrete implementation."""

        pass
    
    @property
    @abc.abstractmethod
    def params_dimension(self) -> INT_DTYPE:
            
        """Abstract container which stores the dimension of the parameter space.

        Returns
        -------
        INT_DTYPE
            The amount of different parameters of the regular linear regression model.

        Notes
        -----
        This is a placeholder of the ABC, which is overwritten by the concrete implementation."""

        pass

    @property
    @abc.abstractmethod
    def multi_index_set(self) -> ARRAY:

        """Abstract container which stores the multi index.

        Returns
        -------
        ARRAY
            The multi index set of the regular linear regression model.

        Notes
        -----
        This is a placeholder of the ABC, which is overwritten by the concrete implementation."""

        pass

    @property
    @abc.abstractmethod
    def model(self)->List[callable]:
            
        """Abstract container which stores all the model functions.

        Returns
        -------
        List[callable]
            The model functions of the regular linear regression model.

        Notes
        -----
        This is a placeholder of the ABC, which is overwritten by the concrete implementation."""

        pass
    
    @params.setter
    @abc.abstractmethod
    def params(self, params: List[ARRAY]) -> None:

        """Abstract setter method for the parameters.

        Parameters
        ----------
        params : List[ARRAY]
            The parameters of the model.

        Returns
        -------
        None

        Notes
        -----
        This is a placeholder of the ABC, which is overwritten by the concrete implementation."""

        pass
    
    @abc.abstractmethod
    def _generate_multi_index_set(self)->ARRAY:
            
        """Abstract function which generates the multi index set.

        Returns
        -------
        ARRAY
            The multi index set of the regular linear regression model.

        Notes
        -----
        This is a placeholder of the ABC, which is overwritten by the concrete implementation."""

        pass
    
    @abc.abstractmethod
    def _model_function(self, param: List[ARRAY], input: ARRAY)->ARRAY:
            
        """Abstract function which provides one model function for the given parameters.

        Parameters
        ----------
        param : List[ARRAY]
            One parameter specification.

        input: ARRAY
            The input of the model function.

        Returns
        -------
        callable
            The model function with the given parameters.

        Notes
        -----
        This is a placeholder of the ABC, which is overwritten by the concrete implementation."""

        pass