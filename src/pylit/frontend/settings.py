from pylit.global_settings import TOL, FLOAT_DTYPE, INT_DTYPE
from pylit.frontend.core import Options, Option, ParamMap, Param

MODELS = Options(
    [
        Option(ref="GaussRLRM", name="Linear Gaussian Regression"),
        Option(ref="DeltaDiracRLRM", name="Linear Delta Dirac Regression"),
        Option(ref="UniformRLRM", name="Linear Uniform Regression"),
    ]
)

OPTIMIZER = Options(
    [
        Option(ref="nn_nesterov", name="Non Negative Nesterov"),
        Option(ref="nn_adam", name="Non Negative ADAM"),
        Option(ref="nn_bro", name="Non Negative Bro"),
    ]
)

OPTIM_PARAM_MAP = ParamMap(
    [
        Param(
            name="maxiter", label="Maximal Iterations", my_type=INT_DTYPE, default=100
        ),
        Param(
            name="tol",
            label="Tolerance",
            default=TOL,
            my_type=FLOAT_DTYPE,
            step=1e-30,
        ),
        Param(name="S", ignore=True, default=True),
        Param(name="R", ignore=True),
        Param(name="F", ignore=True),
        Param(name="x0", ignore=True),
        Param(name="method", ignore=True),
    ]
)

SCALINGS = Options(
    [
        Option(ref="linear", name="Linear"),
    ]
)

SCALINGS_PARAM_MAP = ParamMap(
    [
        Param(
            name="right_end_point",
            ignore=True,
            label="Right End Point",
            default=1.0,
            my_type=FLOAT_DTYPE,
        ),
    ]
)

METHODS = Options(
    [
        Option(ref="lsq_l1_reg", name="Lasso Regularization"),
        Option(ref="lsq_l2_reg", name="Ridge Regularization"),
        Option(ref="lsq_tv_reg", name="Total Variation Regularization"),
        Option(ref="lsq_var_reg", name="Variance Regularization"),
        Option(ref="lsq_l2_fit", name="L2 Fitness"),
        Option(ref="lsq_max_entropy_fit", name="Maximum Entropy Fitness"),
        Option(ref="lsq_cdf_l2_fit", name="CDF L2 Fitness"),
        Option(ref="lsq_cdf_l2_fit_autograd", name="CDF L2 Fitness autograd"),
    ]
)

METHODS_PARAM_MAP = ParamMap(
    [
        Param(
            name="lambd",
            my_type=FLOAT_DTYPE,
            variation=True,
        ),
        Param(name="E", ignore=True),
        Param(name="S", ignore=True),
        Param(name="omegas", ignore=True),
    ]
)

NOISES_IID = Options(
    [
        Option(ref="WhiteNoise", name="Gaussian"),
        Option(ref="UniformNoise", name="Uniform"),
        Option(ref="BernoulliNoise", name="Bernoulli"),
        Option(ref="PoissonNoise", name="Poisson"),
    ]
)

NOISES_IID_PARAM_MAP = ParamMap(
    [
        Param(
            name="mean",
            my_type=FLOAT_DTYPE,
            default=0.0,
            step=0.1,
        ),
        Param(
            name="std",
            my_type=FLOAT_DTYPE,
            default=0.01,
            step=0.001,
        ),
        Param(
            name="low",
            my_type=FLOAT_DTYPE,
            default=-0.1,
            step=0.01,
        ),
        Param(
            name="high",
            my_type=FLOAT_DTYPE,
            default=0.1,
            step=0.01,
        ),
        Param(
            name="prob",
            my_type=FLOAT_DTYPE,
            default=0.5,
            step=0.01,
        ),
        Param(
            name="lam",
            my_type=FLOAT_DTYPE,
            default=0.1,
            step=0.01,
        ),
    ]
)

NOISES_CONV = Options(
    [
        Option(ref="BinomKernel", name="Binomial"),
        Option(ref="UniformKernel", name="Uniform"),
    ]
)

NOISES_CONV_PARAM_MAP = ParamMap(
    [
        Param(
            name="prob",
            my_type=FLOAT_DTYPE,
            default=0.5,
            step=0.01,
        ),
        Param(name="window", my_type=INT_DTYPE, default=5, step=1),
    ]
)
