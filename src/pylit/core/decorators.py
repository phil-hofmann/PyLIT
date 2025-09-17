from pylit.models.lrm import LinearRegressionModel
from pylit.core.linear_scaling import TauLinearScaling, OmegaLinearScaling
from pylit.core.detailed_balance import TauDetailedBalance, OmegaDetailedBalance


def linear_scaling_decorator(
    lrm: LinearRegressionModel,
) -> LinearRegressionModel:
    r"""Decorator for the Linear Regression Model.

    This decorator is used to scale the time axis :math:`\tau` to [0,1].

    Args:
        lrm :
            The  linear regression model.

    Returns:
        The scaled linear regression model.
    """

    # Apply the scaling decorators
    lrm.kernel = OmegaLinearScaling(lrm.tau)(lrm.kernel)
    lrm.ltransform = TauLinearScaling(lrm.tau)(lrm.ltransform)

    return lrm


def detailed_balance_decorator(
    lrm: LinearRegressionModel,
) -> LinearRegressionModel:
    r"""Decorator for the LinearRegressionModel.

    This decorator is used to apply the detailed balance.

    Args:
        lrm:
            The  linear regression model.

    Returns:
        The detailed balanced linear regression model.
    """

    # Apply the detailed balance decorators
    lrm.kernel = OmegaDetailedBalance(lrm.tau)(lrm.kernel)
    lrm.ltransform = TauDetailedBalance(lrm.tau)(lrm.ltransform)

    return lrm
