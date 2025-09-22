import warnings
import numpy as np

import numpy as np
from typing import List
from pylit.settings import FLOAT_DTYPE, INT_DTYPE
from pylit.utils import generate_multi_index_set, to_string


class LinearRegressionModel:
    r"""
    Abstract base class for parametric linear regression models of the form

    .. math::
        f(\omega) = \sum_{\boldsymbol{\alpha} \in \mathcal{I}} c_\boldsymbol{\alpha} \, K(\omega; \boldsymbol{\theta}_\boldsymbol{\alpha}),

    where
        - :math:`\omega \in \mathbb{R}` is the input variable,
        - :math:`\mathcal{I}` is a multi–index set determined by the parameter configurations,
        - :math:`c_\boldsymbol{\alpha} \in \mathbb{R}` are the model coefficients,
        - :math:`K(\cdot; \boldsymbol{\theta}_\boldsymbol{\alpha})` is a kernel function depending on a parameter tuple
          :math:`\boldsymbol{\theta}_{\boldsymbol{\alpha}} = (\theta_{\alpha_1}, \ldots, \theta_{\alpha_p})`,
        - :math:`p = \mathrm{pdim}` is the number of parameter groups.

    Given
        - a finite set of nodes :math:`\tau = (\tau_1, \ldots, \tau_m) \in \mathbb{R}^m`,
        - discrete parameter sets :math:`\{ \theta_{j,k} \}_{k=1}^{n_j}` for each parameter group
          :math:`j = 1,\ldots,p`,

    the model constructs the Cartesian multi–index set

    .. math::
        \mathcal{I} = \{ (i_1, \ldots, i_p) \mid 1 \leq i_j \leq n_j \},

    with degree :math:`|\mathcal{I}| = \prod_{j=1}^p n_j`.

    The regression matrix :math:`R \in \mathbb{R}^{m \times |\mathcal{I}|}` is defined entrywise by

    .. math::
        R_{r,\boldsymbol{\alpha}} = \mathcal{L}\big[K(\cdot; \boldsymbol{\theta}_\boldsymbol{\alpha}) \big](\tau_r),

    where :math:`\mathcal{L}` denotes a Laplace-type transformation specified in
    :meth:`ltransform`.

    The forward model at the nodes is then

    .. math::
        f(\tau) = R \boldsymbol{c},

    and evaluation at arbitrary inputs :math:`\omega` uses the kernel representation.

    Notes:
        - This class is **abstract** and should not be instantiated directly.
        - Concrete subclasses must override:
            * :meth:`kernel` — defines :math:`K(\omega; \boldsymbol{\theta})`,
            * :meth:`ltransform` — defines :math:`\mathcal{L}[K](\tau; \boldsymbol{\theta})`.
        - Coefficients :math:`\boldsymbol{c}` are free variables, whereas :math:`R` depends only on the chosen parameters and nodes.
    """

    def __init__(
        self,
        name: str,
        tau: np.ndarray,
        params: List[np.ndarray],
    ):
        """
        Initializes the linear regression model.

        Args:
            name:
                The name of the linear regression model.
            tau:
                The discrete time axis of the linear regression model.
            params:
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
        return self._tau

    @tau.setter
    def tau(self, tau: np.ndarray) -> None:
        """Setter for the discrete time axis.

        Args:
            tau:
                The discrete time axis of the linear regression model.

        Raises:
            ValueError: If ``tau`` is None.
            ValueError: If ``tau`` is not one-dimensional.
        """

        # Warning
        if hasattr(self, "_reg_mat") and self._reg_mat is not None:
            warnings.warn("WARNING: You must re-compute the regression matrix.")

        # Typing
        if tau is None:
            raise ValueError("`tau` cannot be None.")

        # Type Conversion
        tau = np.asarray(tau).astype(FLOAT_DTYPE)

        # Integrity
        if not tau.ndim == 1:
            raise ValueError("The spatial dimension of `tau` must be one.")

        # Assign
        self._tau = tau

    @property
    def params(self) -> List[np.ndarray[FLOAT_DTYPE]]:
        return self._params

    @params.setter
    def params(self, params: List[np.ndarray[FLOAT_DTYPE]]) -> None:
        """Parameters setter.

        Args:
            params:
                The parameters of the linear regression model.

        Raises:
            TypeError: If the parameters are not a list.
            ValueError: If the amount of parameters is less than one.
            ValueError: If any parameter array is not one-dimensional.
            ValueError: If any parameter has less than one configuration.
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
        return self._pdim

    @pdim.setter
    def pdim(self, pdim: INT_DTYPE) -> None:
        raise PermissionError(
            "The dimension of the model cannot be changed directly. It is determined by the parameters."
        )

    @property
    def multi_index_set(self) -> np.ndarray[INT_DTYPE]:
        return self._multi_index_set

    @multi_index_set.setter
    def multi_index_set(self, multi_index_set: np.ndarray[INT_DTYPE]) -> None:
        raise PermissionError(
            "The multi index set of the model cannot be changed directly. It is determined by the _generate_multi_index_set method."
        )

    @property
    def degree(self) -> INT_DTYPE:
        return self._degree

    @degree.setter
    def degree(self, degree: INT_DTYPE) -> None:
        r"""Attempted manual setter for the model degree.

        The degree :math:`d = |\mathcal{I}|` is defined as the cardinality of the
        multi–index set :math:`\mathcal{I}` generated by the parameters. It is
        not a free variable: changing the parameters automatically determines the
        degree.

        Args:
            degree:
                Unused. Any attempt to assign to ``degree`` will raise an error.

        Raises:
            PermissionError:
                Always, since the degree is derived from the parameter
                configuration and cannot be set manually.
        """
        raise PermissionError(
            "The degree of the model cannot be changed. It is determined by the parameters."
        )

    @property
    def coeffs(self) -> np.ndarray:
        return self._coeffs

    @coeffs.setter
    def coeffs(self, coeffs: np.ndarray[FLOAT_DTYPE]) -> None:
        r"""Set the coefficient vector of the linear regression model.

        The coefficient vector :math:`\boldsymbol{c} \in \mathbb{R}^d` determines the linear
        combination of kernel functions in the model

        .. math::
            f(\omega) = \sum_{\boldsymbol{\alpha} \in \mathcal{I}} c_boldsymbol{\alpha} \, K(\omega; \boldsymbol{\theta}_boldsymbol{\alpha}),

        where :math:`d = |\mathcal{I}|` is the model degree (i.e., the size of the
        multi–index set).

        Args:
            coeffs:
                One-dimensional array of length equal to the model degree, containing
                the coefficients :math:`c_boldsymbol{\alpha}`.

        Raises:
            ValueError: If the length of ``coeffs`` does not equal the degree of the model.
        """

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
        return self._reg_mat

    @regression_matrix.setter
    def regression_matrix(self, reg_mat: np.ndarray) -> None:
        r"""Set the regression matrix manually.

        Args:
            reg_mat:
                Two-dimensional array of shape ``(m, d)``.
                Must have as many rows as nodes :math:`m = |\tau|` and as many
                columns as the model degree :math:`d`.

        Raises:
            ValueError: If ``reg_mat`` is not two-dimensional.
            ValueError: If the number of rows does not match the number of nodes.
            ValueError: If the number of columns does not match the model degree.

        Warns:
            UserWarning: If an existing regression matrix is being overwritten.
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
        return to_string(self)

    def kernel(
        self,
        omega: np.ndarray,
        param: List[np.ndarray],
    ) -> np.ndarray[FLOAT_DTYPE]:
        r"""Evaluate a single kernel function for a given set of parameters.

        The kernel function :math:`K(\omega; \boldsymbol{\theta})` defines the
        model contribution for a specific parameter tuple
        :math:`\boldsymbol{\theta} \in \Theta`. Concrete subclasses must implement
        this method.

        Args:
            omega:
                One-dimensional array representing the discrete
                frequency axis at which the kernel is evaluated.
            param:
                List of parameter arrays corresponding to
                the parameter tuple :math:`\boldsymbol{\theta}` for this kernel.

        Returns:
            Values of the kernel function :math:`K(\omega; \boldsymbol{\theta})`
            at the input frequencies.

        Notes:
            This method must be overridden in the concrete child class to define
            the actual kernel function.
        """

        raise NotImplementedError(
            "The kernel method must be implemented by the concrete child class."
        )

    def ltransform(
        self,
        tau: np.ndarray,
        param: List[np.ndarray],
    ) -> np.ndarray:
        r"""Evaluate the Laplace-transformed kernel function at the discrete time axis.

        For a given parameter tuple :math:`\boldsymbol{\theta}`, this method
        computes the Laplace transform of the kernel function:

        .. math::
            \mathcal{L}[K(\cdot; \boldsymbol{\theta})](\tau),

        where :math:`\tau` is the discrete time axis. Concrete subclasses must
        implement this method to define the Laplace-transformed model function.

        Args:
            tau:
                One-dimensional array representing the discrete
                time axis at which the Laplace transform is evaluated.
            param:
                List of parameter arrays corresponding to
                the parameter tuple :math:`\boldsymbol{\theta}`.

        Returns:
            Values of the Laplace-transformed kernel function at the discrete time axis.

        Notes:
            This method must be overridden in the concrete child class to define
            the Laplace-transformed model.
        """

        raise NotImplementedError(
            "The ltransform method must be implemented by the concrete child class."
        )

    def forward(self) -> np.ndarray:
        r"""Evaluation of the Laplace-transformed model at the discrete time axis.

        Computes the model values at the discrete time axis :math:`\tau` using the
        regression matrix :math:`R` and the coefficient vector :math:`\mathbf{c}`:

        .. math::
            R \mathbf{c},

        where
            * :math:`R` is the regression matrix of shape ``(m, d)``,
            * :math:`\mathbf{c}` is the coefficient vector of length ``d``,
            * :math:`m` is the number of nodes in the discrete time axis.

        Returns:
            np.ndarray: One-dimensional array of length ``m`` containing the model
            values at the discrete time axis :math:`\tau`.
        """

        return self.regression_matrix.dot(self._coeffs)

    def __call__(self, omega: np.ndarray, matrix: bool = False) -> np.ndarray:
        r"""Evaluate the linear regression model at given inputs.

        For input vector :math:`\omega = (\omega_1,\ldots,\omega_n) \in \mathbb{R}^n`,
        the method constructs the evaluation matrix

        .. math::
            E_{i,\boldsymbol{\alpha}} =
            K(\omega_i; \boldsymbol{\theta}_{\boldsymbol{\alpha}}),

        where :math:`\boldsymbol{\alpha} \in \mathcal{I}` indexes the multi–index set
        of parameter combinations. The model evaluation is then

        .. math::
            f(\omega) = E \boldsymbol{c},

        with coefficient vector :math:`\boldsymbol{c} \in \mathbb{R}^d`,
        where :math:`d = |\mathcal{I}|`.

        Args:
            omega:
                One-dimensional array of inputs :math:`\omega`
                at which the model should be evaluated.
            matrix:
                If ``True``, return the evaluation matrix
                :math:`E \in \mathbb{R}^{n \times d}` instead of
                the model output. Defaults to ``False``.

        Returns:
            np.ndarray:
                - If ``matrix=False``: one-dimensional array of length ``n`` containing :math:`f(\omega)`.
                - If ``matrix=True``: two-dimensional array of shape ``(n, d)`` containing the evaluation matrix :math:`E`.

        Raises:
            ValueError: If ``omega`` is not one-dimensional.
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
        return generate_multi_index_set(
            self._pdim, [itr_p.shape[0] for itr_p in self._params]
        )

    def compute_regression_matrix(self) -> np.ndarray:
        r"""Compute the regression matrix.

        The regression matrix :math:`R \in \mathbb{R}^{m \times d}` encodes the
        evaluation of the Laplace-transformed kernels at the discrete time axis
        :math:`\tau` with the given parameter grid :math:`\{ \boldsymbol{\theta}_{\alpha} \}`.

        Returns:
            np.ndarray: Two-dimensional array of shape ``(m, d)`` containing the
            regression matrix :math:`R`.

        Notes:
            The regression matrix depends only on ``tau`` and ``params``,
            but not on ``coeffs``.
        """

        tau, params = self._tau, self._params
        reg_mat = np.zeros((tau.shape[0], self._degree), dtype=FLOAT_DTYPE)

        for i in range(self._degree):
            mi = self._multi_index_set[i]
            param = [params[j][mi[j]] for j in range(self._pdim)]
            row = self.ltransform(tau, param)
            reg_mat[:, i] = row
        self._reg_mat = reg_mat
