from pylit.models.lrm import LinearRegressionModel
from pylit.core.linear_scaling import ForwardLinearScaling, InverseLinearRescaling
from pylit.core.detailed_balance import ForwardDetailedBalance, InverseDetailedBalance


def linear_scaling_decorator(
    lrm: LinearRegressionModel,
    beta: float,
) -> LinearRegressionModel:
    """Decorator for the Linear Regression Model.

    This decorator is used to scale the Linear Regression Model with given right end point.

    Parameters:
        lrm : LinearRegressionModel
            The  linear regression model.
        beta : float
            The new right end point of the nodes tau.

    Returns:
        LinearRegressionModel:
            The scaled linear regression model.
    """

    # Apply the scaling decorators to the regression matrix
    lrm.kernel = InverseLinearRescaling(lrm.tau, beta)(lrm.kernel)

    # Apply the scaling decorators to the regression matrix
    lrm.ltransform = ForwardLinearScaling(lrm.tau, beta)(lrm.ltransform)

    return lrm


def detailed_balance_decorator(
    lrm: LinearRegressionModel,
    beta: float,
) -> LinearRegressionModel:
    """Decorator for the Linear Regression Model.

    This decorator is used to apply the detailed balance to the Linear Regression Model.

    Parameters:
        lrm : LinearRegressionModel
            The  linear regression model.
        beta : float
            The new parameter beta for the detailed balance.

    Returns:
        LinearRegressionModel:
            The detailed balanced linear regression model.
    """

    # Apply the scaling decorators to the regression matrix
    lrm.kernel = InverseDetailedBalance(lrm.tau)(lrm.kernel)

    # Apply the scaling decorators to the regression matrix
    lrm.ltransform = ForwardDetailedBalance(lrm.tau)(lrm.ltransform)

    return lrm
