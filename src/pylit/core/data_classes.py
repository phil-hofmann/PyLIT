import numpy as np
from pathlib import Path
from typing import Literal
from dataclasses import dataclass, field


@dataclass
class Configuration:
    """Represents the configuration of the optimization problem.

    Attributes:
        path_F: Path (required)
        path_D: Path (required)
        path_S: Path | None = None
        path_L_S: Path | None = None
        path_prep: Path | None = None
        path_res: Path | None = None

        non_negative: bool = True
        detailed_balance: bool = True
        tau_scaling: float | None = 1.0
        model_name: Literal[
            "Gauss",
            "Laplace",
            "Cauchy",
            "Uniform",
        ] = "Gauss"
        n: int = 100
        y_tol: float = 0.01
        window: int = 5
        widths: int = 100
        fat_tol: float = 0.01

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

        optimizer_name: Literal[
            "nnls",
            "nesterov",
            "adam",
        ] = "nesterov"
        tol: float = 10e-16
        maxiter: int = 1000
        adaptive: bool = True
        adaptive_residuum_mode: bool = False
        c0: np.ndarray | None = None
        svd: bool = False
        protocol: bool = False
    """

    # Paths
    path_F: Path
    path_D: Path
    path_S: Path | None = None
    path_L_S: Path | None = None
    path_prep: Path | None = None
    path_res: Path | None = None

    # Model
    non_negative: bool = True
    detailed_balance: bool = True
    tau_scaling: float | None = 1.0
    model_name: Literal[
        "Gauss",
        "Laplace",
        "Cauchy",
        "Uniform",
    ] = "Gauss"
    n: int = 100
    y_tol: float = 0.01
    window: int = 5
    widths: int = 100
    fat_tol: float = 0.01

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
    maxiter: int = 1000
    adaptive: bool = True
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
    """Represents the output of the preparation of the data.

    Attributes:
        tau: np.ndarray
        F: np.ndarray
        beta: float
        max_F: float
        scaled_F: np.ndarray

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
    """Represents the output of the optimizer.

    Attributes:
        eps: np.ndarray
        residuals: np.ndarray

        mu: np.ndarray
        sigma: np.ndarray
        coefficients: np.ndarray

        S: np.ndarray
        exp_S: np.ndarray
        std_S: np.ndarray
        moments_S: np.ndarray

        forward_S: np.ndarray
        eps_S: np.ndarray
        max_eps_S: np.ndarray
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

    Attributes:
        name: str
        f: callable
        grad_f: callable
        solution: callable
        lr: callable
    """

    name: str
    f: callable
    grad_f: callable
    solution: callable
    lr: callable


@dataclass
class Solution:
    """Represents a solution to an optimization problem.

    Attributes:
        x: np.ndarray
        eps: float
        residuum: float
    """

    x: np.ndarray
    eps: float
    residuum: float
