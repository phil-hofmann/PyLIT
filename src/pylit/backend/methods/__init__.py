# Methods: Regularization
from .lsq_l1_reg import get as lsq_l1_reg
from .lsq_l2_reg import get as lsq_l2_reg
from .lsq_tv_reg import get as lsq_tv_reg
from .lsq_var_reg import get as lsq_var_reg

# Methods: Fitness
from .lsq_l2_fit import get as lsq_l2_fit
from .lsq_max_entropy_fit import get as lsq_max_entropy_fit
from .lsq_cdf_l2_fit import get as lsq_cdf_l2_fit
