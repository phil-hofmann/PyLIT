from dataclasses import dataclass, field
from pylit.global_settings import ARRAY, FLOAT_DTYPE, INT_DTYPE
from pylit.backend.core.utils import empty_array


@dataclass
class Configuration:
    name: str = ""
    scaleMaxF: bool = True
    PosS: bool = True
    ExtS: bool = True
    trapzS: bool = True
    noiseActive: bool = False
    noiseName: str = ""
    noiseParams: dict = field(default_factory=dict)
    noiseConvActive: bool = False
    noiseConvName: str = ""
    noiseConvParams: dict = field(default_factory=dict)
    methodName: str = ""
    methodParams: dict = field(default_factory=dict)
    svd: bool = False
    optimName: str = ""
    optimParams: dict = field(default_factory=dict)
    x0Reset: bool = False
    adaptiveActive: bool = False
    adaptiveResiduumMode: bool = False
    scalingName: str = ""
    modelName: str = ""
    modelParams: dict = field(default_factory=dict)
    plot_coeffs: bool = True
    plot_model: bool = True
    plot_forward_model: bool = True
    plot_error_model: bool = True
    plot_error_forward_model: bool = True


@dataclass
class Preparation:
    omega: ARRAY = field(default_factory=empty_array)
    modifiedOmega: ARRAY = field(default_factory=empty_array)
    modifiedOmegaMin: FLOAT_DTYPE = 0.0
    modifiedOmegaMax: FLOAT_DTYPE = 0.0
    F: ARRAY = field(default_factory=empty_array)
    modifiedF: ARRAY = field(default_factory=empty_array)
    tau: ARRAY = field(default_factory=empty_array)
    tauMin: FLOAT_DTYPE = 0.0
    tauMax: FLOAT_DTYPE = 0.0
    S: ARRAY = field(default_factory=empty_array)
    modifiedS: ARRAY = field(default_factory=empty_array)
    expS: FLOAT_DTYPE = 0.0
    stdS: FLOAT_DTYPE = 0.0
    forwardModifiedS: ARRAY = field(default_factory=empty_array)
    forwardModifiedSAbsError: ARRAY = field(default_factory=empty_array)
    forwardModifiedSMaxError: FLOAT_DTYPE = 0.0


@dataclass
class Output:
    valsF: ARRAY = field(default_factory=empty_array)
    valsS: ARRAY = field(default_factory=empty_array)
    coefficients: ARRAY = field(default_factory=empty_array)
    eps: ARRAY = field(default_factory=empty_array)
    residuals: ARRAY = field(default_factory=empty_array)
    integral: ARRAY = field(default_factory=empty_array)


@dataclass
class Method:
    """Represents a method for solving an optimization problem."""

    name: str
    f: callable
    grad_f: callable
    solution: callable
    lr: callable


@dataclass
class Solution:
    """Represents a solution to an optimization problem."""

    x: ARRAY
    eps: FLOAT_DTYPE
    residuum: FLOAT_DTYPE
