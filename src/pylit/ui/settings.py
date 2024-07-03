import math
import plotly
import matplotlib as mpl

from pathlib import Path
from pylit.global_settings import TOL, ARRAY, FLOAT_DTYPE, INT_DTYPE
from pylit.ui.options import Options, Option
from pylit.ui.param_map import ParamMap, Param

MODEL_NAMES = ["rpa", "static", "ideal"]
PATH_PROJECT = str(Path(__file__).parent.parent.parent.parent)
PATH_DATA = PATH_PROJECT + "/data"  # TODO os join
PATH_EXPERIMENTS = PATH_PROJECT + "/experiments"  # TODO os join
PATH_SETTINGS = PATH_PROJECT + "/app_settings.json"  # TODO os join
MPL_COLORS = mpl.colors
PLOTLY_COLORS = plotly.colors.qualitative.Plotly
IGNORE_FIRST_COLUMNS = 2
SCI_NUM_STEP = 1e-15
NUM_STEP = 1e-2

MODELS = Options([
    Option(ref="GaussRLRM", name="Linear Gaussian Regression"),
    Option(ref="DeltaDiracRLRM", name="Linear Delta Dirac Regression"),
    Option(ref="UniformRLRM", name="Linear Uniform Regression"),
])

OPTIMIZER = Options([
    Option(ref="nn_nesterov", name="Non Negative Nesterov"),
    Option(ref="nn_adam", name="Non Negative ADAM"),
    Option(ref="nn_bro", name="Non Negative Bro"),
    # Option(name="nn_trust", label="Non Negative Trust-Region"),
])

OPTIM_PARAM_MAP = ParamMap([
    Param(name="maxiter", label="Maximal Iterations", type=INT_DTYPE),
    Param(name="tol", label="Tolerance", default=TOL, type=FLOAT_DTYPE, min_value=1e-30, max_value=1e-1, step=1e-30),
])

SCALINGS = Options([
    Option(ref="linear", name="Linear"),
])

SCALINGS_PARAM_MAP = ParamMap([
    Param(name="right_end_point", ignore=True, label="Right End Point",
          default=1.0, type=FLOAT_DTYPE, min_value=0.0, max_value=5.0),
])

METHODS = Options([
    Option(ref="lsq_l1_reg", name="Lasso Regularization"),
    Option(ref="lsq_l2_reg", name="Ridge Regularization"),
    Option(ref="lsq_tv_reg", name="Total Variation Regularization"),
    Option(ref="lsq_var_reg", name="Variance Regularization"),
    Option(ref="lsq_l2_fit", name="L2 Fitness"),
    Option(ref="lsq_max_entropy_fit", name="Maximum Entropy Fitness"),
    Option(ref="lsq_cdf_l2_fit", name="CDF L2 Fitness"),
    Option(ref="lsq_cdf_l2_fit_autograd", name="CDF L2 Fitness autograd"),
])

METHODS_PARAM_MAP = ParamMap([
    Param(name="lambd", type=ARRAY, optional=True, optional_label="Variation of lambd"),
    Param(name="E", ignore=True, default=True),
    Param(name="S", ignore=True, default=True),
    Param(name="omegas", ignore=True, default=True),
])

NOISES_IID = Options([
    Option(ref="WhiteNoise", name="Gaussian"),
    Option(ref="UniformNoise", name="Uniform"),
    Option(ref="BernoulliNoise", name="Bernoulli"),
    Option(ref="PoissonNoise", name="Poisson"),
])

NOISES_IID_PARAM_MAP = ParamMap([
    Param(name="mean", type=FLOAT_DTYPE, default=0.0, min_value=-1.0, max_value=1.0, step=0.1),
    Param(name="std", type=FLOAT_DTYPE, default=0.01, min_value=0.0, max_value=1.0, step=0.001),
    Param(name="low", type=FLOAT_DTYPE, default=-0.1, min_value=-1.0, max_value=1.0, step=0.01),
    Param(name="high", type=FLOAT_DTYPE, default=0.1, min_value=-1.0, max_value=1.0, step=0.01),
    Param(name="prob", type=FLOAT_DTYPE, default=0.5, min_value=0.0, max_value=1.0, step=0.01),
    Param(name="lam", type=FLOAT_DTYPE, default=0.1, min_value=0.0, max_value=1.0, step=0.01),
])

NOISES_CONV = Options([
    Option(ref="BinomKernel", name="Binomial"),
    Option(ref="UniformKernel", name="Uniform"),
])

NOISES_CONV_PARAM_MAP = ParamMap([
    Param(name="prob", type=FLOAT_DTYPE, default=0.5, min_value=0.0, max_value=1.0, step=0.01),
    Param(name="window", type=INT_DTYPE, default=5, min_value=1, max_value=20, step=1),
])
