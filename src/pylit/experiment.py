import inspect
import numpy as np
import pandas as pd

from pylit import (
    methods,
    models,
    optimizer,
)
from pylit.settings import (
    FLOAT_DTYPE,
    MOMENT_ORDERS,
)
from pylit.utils import (
    exp_std,
    complete_detailed_balance,
    moments,
    import_xY,
    save_to_json,
    y_tol_fit,
    fat_tol_fit,
)
from pylit.core.data_classes import Configuration, Preparation, Result
from pylit.core.decorators import linear_scaling_decorator, detailed_balance_decorator
from pylit.optimizer import adaptiveRF


def prepare(config: Configuration) -> Preparation:
    # τ, F, β, Scaled F
    tau, F = import_xY(config.path_F)
    beta = np.max(tau)
    max_F = np.max(F, axis=1)
    scaled_F = F / max_F if max_F != 0.0 else np.copy(F)

    # ω, D, Integral D, Scaled D, μ(D), σ(D), µ_α(D), L(D)[τ], Error, Maximal Error
    omega, D = import_xY(config.path_D)
    D = D[0]
    if config.non_negative:
        D = np.maximum(0.0, D)
    if config.detailed_balance:
        omega, D = complete_detailed_balance(omega, D, beta)
    int_D = np.trapezoid(D, omega)
    scaled_D = D / int_D if int_D != 0.0 else np.copy(D)
    mu_D, sigma_D = exp_std(omega, scaled_D)
    moments_D = moments(omega, scaled_D, MOMENT_ORDERS)  # NOTE REM
    forward_D = np.array(
        [np.trapezoid(scaled_D * np.exp(-omega * t), omega) for t in tau],
        dtype=FLOAT_DTYPE,
    )
    eps_D = np.abs(forward_D - scaled_F)
    max_eps_D = np.max(eps_D)

    prep = Preparation(
        tau=tau,
        F=F,
        beta=beta,
        max_F=max_F,
        scaled_F=scaled_F,
        omega=omega,
        D=D,
        int_D=int_D,
        scaled_D=scaled_D,
        exp_D=mu_D,
        std_D=sigma_D,
        moments_D=moments_D,
        forward_D=forward_D,
        eps_D=eps_D,
        max_eps_D=max_eps_D,
    )

    if config.path_prep is not None:
        save_to_json(prep, config.path_prep)

    return prep


def itransform(config: Configuration, prep: Preparation) -> Result:
    # Type Conversion
    config.lambd = (
        np.asarray(config.lambd).astype(FLOAT_DTYPE)
        if config.lambd is not None
        else np.array([0.0], dtype=FLOAT_DTYPE)
    )
    config.c0 = (
        np.asarray(config.c0).astype(FLOAT_DTYPE) if config.c0 is not None else None
    )

    # Integrity
    if config.lambd.ndim == 0:
        config.lambd = config.lambd.reshape(1)
    if config.c0 is not None and config.c0.ndim == 1:
        config.c0 = config.c0.reshape(1, -1)
    elif config.c0 is not None and config.c0.ndim != 2:
        raise ValueError("The initial guess must be a one- or two-dimensional array.")

    # Get Model, Method, and Optimizer
    ModelClass = getattr(models, config.model_name)
    model_params = inspect.signature(ModelClass.__init__).parameters.keys()
    if not "tau" in model_params:
        raise ValueError("The model does not have a 'tau' parameter.")
    if not "mu" in model_params:
        raise ValueError("The model does not have a 'mu' parameter.")

    method_func = getattr(methods, config.method_name)
    optimizer_func = getattr(optimizer, config.optimizer_name)

    omega_l, omega_r = 0.0, y_tol_fit(prep.scaled_D, prep.omega, config.y_tol)

    if config.detailed_balance:
        omega_l = 0.0

    mu_S = np.linspace(omega_l, omega_r, config.n, dtype=FLOAT_DTYPE)

    sigma_S = (
        fat_tol_fit(
            omega=prep.omega,
            D=prep.scaled_D,
            mu=mu_S,
            model_name=config.model_name,
            tol=config.fat_tol,
            widths=config.widths,
            window=config.window,
        )
        if "sigma" in model_params
        else None
    )

    mu_S *= prep.beta
    sigma_S = sigma_S * prep.beta if sigma_S is not None else None

    model = (
        ModelClass(tau=prep.tau, mu=mu_S, sigma=sigma_S)
        if sigma_S is not None
        else ModelClass(tau=prep.tau, mu=mu_S)
    )

    # Regression Matrix
    if config.tau_scaling is not None:
        model = linear_scaling_decorator(lrm=model, beta=config.tau_scaling)
    if config.detailed_balance is not None and config.tau_scaling is not None:
        model = detailed_balance_decorator(lrm=model, beta=config.tau_scaling)
    elif config.detailed_balance is not None:
        model = detailed_balance_decorator(lrm=model, beta=prep.beta)
    model.compute_regression_matrix()

    # Method Arguments
    args = {
        "omegas": prep.omega,
        "D": prep.scaled_D,
        "E": model(prep.omega, matrix=True),
    }
    args = {
        key: value
        for key, value in args.items()
        if key in inspect.signature(method_func).parameters
    }

    # Optimize
    R = model.regression_matrix
    m = R.shape[1]
    first_param_len = len(model.params[0])
    solutions = []

    for i, _lambd in enumerate(config.lambd):
        solution_row = []
        for j, _scaled_F in enumerate(prep.scaled_F):
            _max_F = prep.max_F[j]
            method = method_func(**args, lambd=_lambd)
            c0 = (
                config.c0[i] / _max_F
                if config.c0 is not None
                else np.zeros(m, dtype=FLOAT_DTYPE)
            )
            if config.adaptive:

                def optim_RFx0(_R, _F, _x0):
                    return optimizer_func(
                        R=_R,
                        F=_F,
                        x0=_x0,
                        method=method,
                        maxiter=config.maxiter,
                        tol=config.tol,
                        protocol=config.protocol,
                        svd=config.svd,
                    )

                # Optimize adaptively
                solution = adaptiveRF(
                    R=R,
                    F=_scaled_F,
                    x0=c0,
                    steps=first_param_len,
                    optim_RFx0=optim_RFx0,
                    residuum_mode=config.adaptive_residuum_mode,
                )

                solution_row.append(solution)
            else:
                # Optimize directly
                solution = optimizer_func(
                    R=R,
                    F=_scaled_F,
                    x0=c0,
                    method=method,
                    maxiter=config.maxiter,
                    tol=config.tol,
                    protocol=config.protocol,
                    svd=config.svd,
                )

                solution_row.append(solution)

            # Rescale coefficients by F_max
            solution_row[-1].x *= _max_F
        solutions.append(solution_row)

    # Map solutions
    eps = np.array([[item.eps for item in row] for row in solutions], dtype=FLOAT_DTYPE)
    residuals = np.array(
        [[item.residuum for item in row] for row in solutions], dtype=FLOAT_DTYPE
    )
    coefficients = np.array(
        [[item.x for item in row] for row in solutions], dtype=FLOAT_DTYPE
    )

    # Evaluate Model
    forward_S, S = [], []

    for row in coefficients:
        forward_S_row, S_row = [], []
        for coeffs in row:
            model.coeffs = coeffs
            forward_S_row.append(model.forward())
            S_row.append(model(prep.omega))
        forward_S.append(forward_S_row)
        S.append(S_row)

    exp_S, std_S = np.array(
        [
            [
                exp_std(
                    prep.omega,
                    _S,
                )
                for _S in row
            ]
            for row in S
        ],
        dtype=FLOAT_DTYPE,
    ).T

    moments_S = np.array(
        [
            [
                moments(
                    prep.omega,
                    _S,
                    MOMENT_ORDERS,
                )
                for _S in row
            ]
            for row in S
        ],
        dtype=FLOAT_DTYPE,
    )

    eps_S = np.array(
        [np.abs(row - prep.F) for row in forward_S],
        dtype=FLOAT_DTYPE,
    )

    max_eps_S = np.amax(eps_S, axis=1)

    # Store S, L(S) to CSV TODO!
    if config.path_S is not None:
        S_df = pd.DataFrame([prep.omega, *S]).T
        S_df.to_csv(config.path_S, index=False, header=False)

    if config.path_L_S is not None:
        L_S_df = pd.DataFrame([prep.tau, *forward_S]).T
        L_S_df.to_csv(config.path_L_S, index=False, header=False)

    res = Result(
        eps=eps,
        residuals=residuals,
        mu=mu_S,
        sigma=sigma_S,
        coefficients=coefficients,
        S=S,
        exp_S=exp_S,
        std_S=std_S,
        moments_S=moments_S,
        forward_S=forward_S,
        eps_S=eps_S,
        max_eps_S=max_eps_S,
    )

    if config.path_res is not None:
        save_to_json(res, config.path_res)

    return res
