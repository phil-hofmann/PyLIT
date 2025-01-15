from dataclasses import dataclass, field
from pylit.global_settings import ARRAY, FLOAT_DTYPE, INT_DTYPE
from pylit.backend.core.utils import empty_array


@dataclass
class Configuration:
    name: str = ""
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
    modelName: str = ""
    modelParams: dict = field(default_factory=dict)


@dataclass
class Preparation:
    # τ, F:
    tau: ARRAY = field(default_factory=empty_array)
    tauMin: FLOAT_DTYPE = 0.0
    tauMax: FLOAT_DTYPE = 0.0
    F: ARRAY = field(default_factory=empty_array)
    noiseF: ARRAY = field(default_factory=empty_array)

    # ω, D:
    omega: ARRAY = field(default_factory=empty_array)
    omegaMin: FLOAT_DTYPE = 0.0
    omegaMax: FLOAT_DTYPE = 0.0
    D: ARRAY = field(default_factory=empty_array)
    expD: FLOAT_DTYPE = 0.0
    stdD: FLOAT_DTYPE = 0.0
    freqMomentsD: ARRAY = field(default_factory=empty_array)

    # L(D)[τ]:
    forwardD: ARRAY = field(default_factory=empty_array)
    forwardDAbsError: ARRAY = field(default_factory=empty_array)
    forwardDMaxError: FLOAT_DTYPE = 0.0


@dataclass
class Output:

    # Optimization:
    eps: ARRAY = field(default_factory=empty_array)
    residuals: ARRAY = field(default_factory=empty_array)

    # Model:
    timeScaling: bool = True
    normalization: bool = True
    coefficients: ARRAY = field(default_factory=empty_array)

    # S:
    S: ARRAY = field(default_factory=empty_array)
    expS: ARRAY = field(default_factory=empty_array)
    stdS: ARRAY = field(default_factory=empty_array)
    freqMomentsS: ARRAY = field(default_factory=empty_array)

    # L(S)[τ]:
    forwardS: ARRAY = field(default_factory=empty_array)
    forwardSAbsError: ARRAY = field(default_factory=empty_array)
    forwardSMaxError: ARRAY = field(default_factory=empty_array)


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
