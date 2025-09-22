import numpy as np
from pathlib import Path
from typing import Literal
from dataclasses import dataclass


@dataclass
class Configuration:
    """Represents the configuration of an optimization problem.

    This dataclass stores all paths, numerical parameters, and
    algorithmic options required to run the solver.

    Args:
        path_F:
            Input path of the Laplace transformed data. (required)
        path_D:
            Input path of the Default model. (required)
        path_S:
            Output path of the model. | None = None
        path_L_S:
            Output path of the laplace transformed model. | None = None
        path_prep:
            Output path of the Preparation dataclass. | None = None
        path_res:
            Output path of the Result dataclass. | None = None
        selection_name:
            Strategy for selecting the models parameters. Choices include
            ``simulated_annealing`` (default) and ``heuristic``.
        n:
            Number of support points in the frequency domain :math:`\omega`.
        window:
            The window size when searching for the optimal kernel widths.
            Only used in ``heuristic``.
        widths:
            The total number of kernel widths.
        non_negative:
            Enforces non-negativity for the Default model.
        detailed_balance:
            If ``True``, imposes the detailed balance condition.
        model_name:
            Name of the kernel model to use for optimization.
        method_name:
            Optimization method to be applied. Choices include regularization- and fit-based approaches.
        lambd:
            Regularization parameter. Can be a scalar, an array of values,
            or ``None``.
        optimizer_name:
            The numerical optimizer used to solve the problem.
        tol:
            Convergence tolerance for the optimizer.
        maxiter:
            Maximum number of iterations allowed for the optimization.
        adaptive:
            Whether to use the decorator function :func:`~pylit.optimizer.adaptive.adaptive`.
        adaptive_residuum_mode:
            If ``True``, the residuum_mode in :func:`~pylit.optimizer.adaptive.adaptive` is enabled.
        c0:
            Initial guess for the solution. If ``None``, a default (zeros) is used.
        svd:
            Whether to apply SVD-based dimensionality reduction before solving.
            Should be only applied to regularization methods ``l1_reg`` and ``l2_reg``.
        protocol:
            If ``True``, prints a protocol of the optimization run.
    """

    # Paths
    path_F: Path
    path_D: Path
    path_S: Path | None = None
    path_L_S: Path | None = None
    path_prep: Path | None = None
    path_res: Path | None = None

    # Parameter selection
    selection_name: Literal["simulated_annealing", "heuristic"] = "simulated_annealing"
    n: int = 100
    window: int = 5
    widths: int = 5

    # Model
    non_negative: bool = True
    detailed_balance: bool = True
    model_name: Literal[
        "Gauss",
        "Laplace",
        "Cauchy",
        "Uniform",
    ] = "Gauss"

    # Method
    method_name: Literal[
        "l1_reg",
        "l2_reg",
        "tv_reg",
        "var_reg",
        "l2_fit",
        "max_entropy_fit",
        "cdf_l2_fit",
    ] = "l1_reg"
    lambd: np.ndarray | float | None = None

    # Optimizer
    optimizer_name: Literal[
        "nnls",
        "nesterov",
        "adam",
    ] = "nesterov"
    tol: float = 10e-16
    maxiter: int = 1_000
    adaptive: bool = False
    adaptive_residuum_mode: bool = False
    c0: np.ndarray | None = None
    svd: bool = False
    protocol: bool = False

    def __post_init__(self):
        if self.model_name == "Uniform" and self.adaptive:
            raise ValueError(
                "Adaptive regularization is not supported for the Uniform model."
            )


@dataclass
class Preparation:
    r"""Represents the output of the preparation of the data.

    Args:
        tau:
            Discrete time axis :math:`\tau` on which the input data :math:`F(\tau)` is defined.
        F:
            Raw input data defined on :math:`\tau`.
        beta:
            Inverse temperature parameter :math:`\beta = 1/T`.
        max_F:
            Maximum value of :math:`F(\tau)`, used for normalization.
        scaled_F:
            Normalized version of :math:`F(\tau)` to improve numerical stability.
        omega:
            Discrete frequency axis :math:`\omega` on which the default model :math:`D(\omega)` is defined.
        D:
            Default model for the dynamic structure factor :math:`S(\omega)`.
        int_D:
            Integral of the default model :math:`D(\omega)` over omega, representing the zeroth moment (normalization).
        scaled_D:
            Normalized version of :math:`D(\omega)`, ensuring unit integral.
        exp_D:
            Expectation value (first moment) of the default model :math:`D(\omega)`.
        std_D:
            Standard deviation of the default model :math:`D(\omega)`.
        moments_D:
            The i-th entry corresponds to the moment :math:`\mu_i = \int d\omega D(\omega)`, with indices covering :math:`i=-1, 0, 1, ..., 10`.
        forward_D:
            Laplace transform of ``scaled_D`` onto :math:`\tau`, computed using the trapezoidal rule as

            .. math ::

                \text{forward_D}_i = \sum_j w_j \ \text{scaled_D}_j \ \exp(-\omega_j \tau_i),

            where :math:`w_j` are the trapezoidal integration weights corresponding
            to the frequency grid :math:`\omega`.
        eps_D:
            Pointwise error between F and forward_D.
        max_eps_D:
            Maximum absolute error between F and forward_D.
    """

    # τ, F, β, Scaled F
    tau: np.ndarray
    F: np.ndarray
    beta: float
    max_F: float
    scaled_F: np.ndarray

    # ω, D, μ(D), σ(D), µ_α(D), L(D)[τ], Error, Maximal Error, Scaled D
    omega: np.ndarray
    D: np.ndarray
    int_D: float
    scaled_D: np.ndarray
    exp_D: float
    std_D: float
    moments_D: np.ndarray
    forward_D: np.ndarray
    eps_D: np.ndarray
    max_eps_D: float


@dataclass
class Result:
    r"""Represents the output of the optimizer.

    Args:
        eps:
            The values of the objective functions evaluated at the solutions ``coefficients``.
        residuals: np.ndarray
            The residual norms from :eq:`(*) <lsq-problem>` evaluated at the solutions ``coefficients``.
        mu:
            Discrete support points of the models in the frequency domain :math:`\omega`.
        sigma:
            Kernel widths of the models.
        coefficients:
            The final iterates of the optimization, representing the solutions.
        S:
            Evaluated model in the frequency domain at :math:`\omega`:

            .. math::
                S_i = \sum_j \text{coefficients}_j K_j(\omega_i).

            *The scaling and detailed balance corrections are not included here,
            as they are applied automatically by*
            :func:`~pylit.core.decorators.linear_scaling_decorator` *and*
            :func:`~pylit.core.decorators.detailed_balance_decorator`.
        eps_S:
        exp_S: np.ndarray
            Expected value (first moment) of S, computed as

            .. math::
                \rho_i = \frac{\max(S_i, 0)}{\sum_j \max(S_j, 0)}, \quad
                \langle S \rangle = \sum_i \omega_i \rho_i
        std_S:
            Standard deviation of S, computed as

            .. math::
                \sigma_S = \sqrt{\sum_i \rho_i (\omega_i - \langle S \rangle)^2}
        moments_S: np.ndarray
            Higher-order moments of S for indices :math:`\alpha = -1, 0, 1, \dots, 10`,
            computed as

            .. math::
                \mu_\alpha = \sum_i \omega_i^\alpha S_i
        forward_S:
            Forward-transformed S onto the original :math:`\tau` grid,
            computed via a kernel or Laplace transform:

            .. math::
                \text{forward_S}_i = \sum_j \text{coefficients}_j \mathcal{L}(K)(\tau_i)

            *The scaling and detailed balance corrections are not included here,
            as they are applied automatically by*
            :func:`~pylit.core.decorators.linear_scaling_decorator` *and*
            :func:`~pylit.core.decorators.detailed_balance_decorator`.
        eps_S:
            Pointwise reconstruction error between the forward-transformed
            model and the observed data.
        max_eps_S:
            Maximum absolute reconstruction error in ``eps_S``.
    """

    # Optimization
    eps: np.ndarray
    residuals: np.ndarray

    # Model
    mu: np.ndarray
    sigma: np.ndarray
    coefficients: np.ndarray

    # S
    S: np.ndarray
    exp_S: np.ndarray
    std_S: np.ndarray
    moments_S: np.ndarray

    # L(S)[τ]
    forward_S: np.ndarray
    eps_S: np.ndarray
    max_eps_S: np.ndarray


@dataclass
class Method:
    """Represents a method for solving an optimization problem.

    Args:
        name:
            Name of the optimization method.
        f:
            The objective function to minimize.
        grad_f:
            Function computing the gradient of ``f``.
        solution:
            Routine that computes the solution directly,
            without the non-negative constraint, if available.
        lr:
            Learning rate function that returns a step size,
            depending on the regression matrix `R`.
    """

    name: str
    f: callable
    grad_f: callable
    solution: callable
    lr: callable


@dataclass
class Solution:
    """Represents a solution to an optimization problem.

    Args:
        x:
            The final iterate of the optimization, representing the solution vector.
        eps:
            The value of the objective function (depending on the chosen method) evaluated at the solution ``x``.
        residuum:
            The residual norm from :eq:`(*) <lsq-problem>` evaluated at the solution ``x``.
    """

    x: np.ndarray
    eps: float
    residuum: float
