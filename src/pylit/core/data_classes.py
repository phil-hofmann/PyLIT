from dataclasses import dataclass, field
from pylit.global_settings import ARRAY, TOL, MAX_ITER
from pylit.core.utils import empty_array


@dataclass
class Configuration:
    name: str = ""
    noiseActive: bool = False
    noiseName: str = ""
    noiseParams: dict = field(default_factory=dict)
    noiseConvActive: bool = False
    noiseConvName: str = ""
    noiseConvParams: dict = field(default_factory=dict)
    scaleMaxF: bool = True
    PosS: bool = True
    ExtS: bool = True
    trapzS: bool = True
    methodName: str = ""
    methodParams: dict = field(default_factory=dict)
    optimName: str = ""
    optimMaxIter: int = MAX_ITER
    optimTol: float = TOL
    scalingName: str = ""
    scalingParams: dict = field(default_factory=dict)
    modelName: str = ""
    modelParams: dict = field(default_factory=dict)


@dataclass
class Preparation:
    omega: ARRAY = field(default_factory=empty_array)
    modifiedOmega: ARRAY = field(default_factory=empty_array)
    modifiedOmegaMin: float = 0.0
    modifiedOmegaMax: float = 0.0
    F: ARRAY = field(default_factory=empty_array)
    modifiedF: ARRAY = field(default_factory=empty_array)
    tau: ARRAY = field(default_factory=empty_array)
    tauMin: float = 0.0
    tauMax: float = 0.0
    S: ARRAY = field(default_factory=empty_array)
    modifiedS: ARRAY = field(default_factory=empty_array)
    expS: float = 0.0
    stdS: float = 0.0


@dataclass
class Output:
    valsF: ARRAY = field(default_factory=empty_array)
    valsS: ARRAY = field(default_factory=empty_array)
    coefficients: ARRAY = field(default_factory=empty_array)
    integral: ARRAY = field(default_factory=empty_array)
