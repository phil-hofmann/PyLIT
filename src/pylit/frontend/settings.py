import numpy as np
from pylit.global_settings import TOL, FLOAT_DTYPE, INT_DTYPE
from pylit.frontend.core import Options, Option, ParamMap, Param
from pylit.backend.core import Experiment
from pylit.global_settings import ARRAY

MODELS = Options(
    [
        Option(ref="GaussRLRM", name="Linear Gaussian Regression"),
        Option(ref="LaplaceRLRM", name="Linear Laplace Regression"),
        # Option(ref="LogisticRLRM", name="Linear Logistic Regression"), # TODO fix some issue...
        # Option(ref="CauchyRLRM", name="Linear Cauchy Regression"), # TODO fix some issue...
    ]
)


def MODEL_PARAM_MAP(exp: Experiment) -> ParamMap:
    return ParamMap(
        [
            Param(
                name="omegas",
                my_type=ARRAY,
                lower_value=np.round(
                    exp.prep.modifiedOmegaMin,
                    2,
                ),
                upper_value=np.round(exp.prep.modifiedOmegaMax, 2),
                num_value=int(len(exp.prep.modifiedOmega) / 20),
            ),
            Param(
                name="sigmas",
                my_type=ARRAY,
                lower_value=np.round(exp.prep.stdS, 2),
                upper_value=np.round(10 * exp.prep.stdS, 2),
                num_value=int(1 / exp.prep.stdS),
            ),
            Param(
                name="beta",
                default=1.0,
                my_type=FLOAT_DTYPE,
                ignore=True,
            ),
            Param(
                name="order",
                default="0,1",
                my_type=str,
                ignore=True,
            ),
        ]
    )


OPTIMIZER = Options(
    [
        Option(
            ref="nn_nesterov",
            name="Non Negative Nesterov",
        ),
        Option(
            ref="nn_adam",
            name="Non Negative ADAM",
        ),
        # Option(
        #     ref="nn_bro",
        #     name="Non Negative Bro",
        # ),
    ]
)

OPTIM_PARAM_MAP = ParamMap(
    [
        Param(
            name="maxiter",
            label="Maximal Iterations",
            my_type=INT_DTYPE,
            default=1000,
        ),
        Param(
            name="tol",
            label="Tolerance",
            default=TOL,
            my_type=FLOAT_DTYPE,
            step=1e-30,
        ),
        Param(
            name="S",
            ignore=True,
            my_type=bool,
            default=True,
        ),
        Param(
            name="R",
            ignore=True,
            my_type=bool,
            default=True,
        ),
        Param(
            name="F",
            ignore=True,
            my_type=bool,
            default=True,
        ),
        Param(
            name="x0",
            ignore=True,
            my_type=bool,
            default=True,
        ),
        Param(
            name="method",
            ignore=True,
            my_type=bool,
            default=True,
        ),
    ]
)

SCALINGS = Options(
    [
        Option(
            ref="linear",
            name="Linear",
        ),
    ]
)

METHODS = Options(
    [
        Option(
            ref="lsq_l1_reg",
            name="Lasso Regularization",
        ),
        Option(
            ref="lsq_l2_reg",
            name="Ridge Regularization",
        ),
        Option(
            ref="lsq_tv_reg",
            name="Total Variation Regularization",
        ),
        Option(
            ref="lsq_var_reg",
            name="Variance Regularization",
        ),
        Option(
            ref="lsq_l2_fit",
            name="L2 Fitness",
        ),
        Option(
            ref="lsq_max_entropy_fit",
            name="Maximum Entropy Fitness",
        ),
        Option(
            ref="lsq_cdf_l2_fit",
            name="CDF L2 Fitness",
        ),
    ]
)


def METHODS_PARAM_MAP(exp: Experiment) -> ParamMap:
    lambd_default = 1.0 if not exp.prepared else exp.prep.forwardModifiedSMaxError
    upper_value = 1.0 if not exp.prepared else exp.prep.forwardModifiedSMaxError
    return ParamMap(
        [
            Param(
                name="lambd",
                my_type=FLOAT_DTYPE,
                default=lambd_default,
                variation=True,
                lower_value=0.0,
                upper_value=upper_value,
                num_value=10,
            ),
            Param(
                name="E",
                ignore=True,
                # my_type=ARRAY,
                my_type=bool,
                default=True
            ),
            Param(
                name="S",
                ignore=True,
                # my_type=ARRAY,
                my_type=bool,
                default=True,
            ),
            Param(
                name="omegas",
                ignore=True,
                # my_type=ARRAY,
                my_type=bool,
                default=True,
            ),
        ]
    )


NOISES_IID = Options(
    [
        Option(
            ref="WhiteNoise",
            name="Gaussian",
        ),
        Option(
            ref="UniformNoise",
            name="Uniform",
        ),
        Option(
            ref="BernoulliNoise",
            name="Bernoulli",
        ),
        Option(
            ref="PoissonNoise",
            name="Poisson",
        ),
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
        Option(
            ref="BinomKernel",
            name="Binomial",
        ),
        Option(
            ref="UniformKernel",
            name="Uniform",
        ),
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
        Param(
            name="window",
            my_type=INT_DTYPE,
            default=5,
            step=1,
        ),
    ]
)
