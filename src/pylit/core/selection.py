import numpy as np
from pylit import models
from typing import Tuple
from scipy.optimize import nnls
from pylit.settings import FLOAT_DTYPE
from scipy.ndimage import gaussian_filter1d
from pylit.utils import find_zero, find_max_cutoff
from pylit.core.data_classes import Configuration, Preparation
from pylit.core.decorators import detailed_balance_decorator

# Simulated Annealing


def simulated_annealing(
    config: Configuration,
    prep: Preparation,
    has_sigma: bool,
    max_iter: int = 1000,
) -> np.ndarray:
    r"""
    Perform parameter selection using simulated annealing.

    This method optimizes kernel centers and widths by iteratively proposing
    new parameter values and accepting them based on a Metropolis–Hastings
    acceptance criterion.

    Args:
        config:
            Configuration object.
        prep:
            Preparation object.
        has_sigma:
            Whether the model uses kernel widths as parameters.
        max_iter:
            Maximum number of iterations performed during the optimization procedure.

    Returns:
        The optimized kernel centers and, if available, the optimized kernel widths.
    """

    # Guess mu and sigma
    mu = (
        _select_omega_by_distribution(
            prep.omega, prep.scaled_D, config.n, config.detailed_balance
        )
        if has_sigma
        else _select_omega_by_gradient(
            prep.omega, prep.scaled_D, config.n, config.detailed_balance
        )
    )
    interval_widths = np.abs(np.diff(mu))
    geometric_mean_of_max_min = np.sqrt(
        np.amax(interval_widths) * np.amin(interval_widths)
    )
    sigma = np.geomspace(
        2 * geometric_mean_of_max_min, geometric_mean_of_max_min / 10.0, config.widths
    )

    # ````
    # Build model
    ModelClass = getattr(models, config.model_name)
    model = ModelClass(prep.tau, mu, sigma) if has_sigma else ModelClass(prep.tau, mu)
    if config.detailed_balance:
        model = detailed_balance_decorator(lrm=model)

    # Forward pass
    E = model(prep.omega, matrix=True)
    D = prep.scaled_D
    c, _ = nnls(E, D)
    proposal_D = E @ c
    # ````

    # Annealing states
    SA_beta = 1.5  # Metropolis Hastings Beta
    step_size = 0.1  # Gaussian noise scale
    eps = np.sum((D - proposal_D) ** 2)

    best_mu, best_sigma, best_eps = (
        np.copy(mu),
        np.copy(sigma) if has_sigma else None,
        eps,
    )

    for _ in range(max_iter):
        # Propose parameters
        proposal_mu = np.sort(mu * np.exp(step_size * np.random.normal(size=mu.shape)))
        proposal_sigma = (
            sigma * np.exp(step_size * np.random.normal(size=sigma.shape))
            if has_sigma
            else None
        )

        # ````
        # Build model
        model = (
            ModelClass(prep.tau, proposal_mu, proposal_sigma)
            if has_sigma
            else ModelClass(prep.tau, proposal_mu)
        )
        if config.detailed_balance:
            model = detailed_balance_decorator(lrm=model)

        # Forward pass
        E = model(prep.omega, matrix=True)
        c, _ = nnls(E, D)
        proposal_D = E @ c
        # ````

        proposal_eps = np.sum((D - proposal_D) ** 2)

        # Metropolis Hastings (MH)
        delta_eps = proposal_eps - eps
        if delta_eps >= 0 and np.random.rand() >= np.exp(-SA_beta * delta_eps):
            continue  # reject proposal

        # Accept proposal
        mu, sigma, eps = proposal_mu, proposal_sigma, proposal_eps

        # --- Update best solution ---
        if proposal_eps <= config.tol:
            best_mu, best_sigma = np.copy(mu), np.copy(sigma)
            break

        if proposal_eps <= best_eps:
            best_mu, best_sigma, best_eps = np.copy(mu), np.copy(sigma), proposal_eps
            SA_beta *= 0.9
            step_size *= 0.9

    return np.abs(best_mu), best_sigma


def _select_omega_by_distribution(
    omega: np.ndarray,
    D_omega: np.ndarray,
    n: int,
    detailed_balance: bool,
    std: float = 5.0,
    cutoff: float = 1e-8,
):
    r"""
    Select distribution of kernel centers based on where
    :math:`D(\omega)` has the most mass.

    Args:
        omega:
            1D array of input values.
        D_omega:
            1D array of function values corresponding to w.
        n:
            Number of points in the new redistributed array.
        std:
            Standard deviation for Gaussian smoothing to suppress spikes and
            Dirac-like features.
        cutoff:
            Minimum threshold below which values of :math:`D(\omega)` are ignored.

    Returns:
        New omega values redistributed based on the CDF."""

    # Cutoff omega and D
    left = (
        find_zero(omega)
        if detailed_balance
        else len(omega) - 1 - find_max_cutoff(np.flip(D_omega))
    )
    right = find_max_cutoff(D_omega, cutoff)
    omega = omega[left:right]
    D_omega = D_omega[left:right]

    # Severely limit Dirac deltas
    D_omega = gaussian_filter1d(np.abs(D_omega), sigma=std)
    scaled_D_omega = D_omega / np.sum(D_omega)

    # Build the cumulative distribution function
    cdf = np.cumsum(scaled_D_omega)

    # Sample uniformly along the CDF to get new omega values
    uniform_samples = np.linspace(0.0, 1.0, n)
    new_omega = np.interp(uniform_samples, cdf, omega)

    return new_omega


def _select_omega_by_gradient(
    omega: np.ndarray,
    D_omega: np.ndarray,
    n: int,
    detailed_balance: bool,
    std: float = 5,
    cutoff: float = 1e-12,
):
    r"""
    Select distribution of kernel centers based on where the gradient
    of :math:`D(\omega)` has the most mass.

    Args:
        omega:
            1D array of input values.
        D_omega:
            1D array of function values corresponding to w.
        n:
            Number of points in the new redistributed array.
        std:
            Standard deviation for Gaussian smoothing to suppress spikes and
            Dirac-like features.
        cutoff:
            Minimum threshold below which values of :math:`D(\omega)` are ignored.

    Returns:
        np.ndarray: New omega values redistributed based on gradient magnitude."""

    # Cutoff omega and D
    left = (
        find_zero(omega)
        if detailed_balance
        else len(omega) - 1 - find_max_cutoff(np.flip(D_omega))
    )
    right = find_max_cutoff(D_omega, cutoff)
    omega = omega[left:right]
    D_omega = D_omega[left:right]

    # Severely limit Dirac deltas
    D_omega = gaussian_filter1d(np.abs(D_omega), sigma=std)

    # Compute the absolute values of the gradient
    grad_D_omega = np.abs(np.gradient(D_omega, omega))
    scaled_grad_D_omega = grad_D_omega / np.sum(grad_D_omega)

    # Build the cumulative distribution function
    cdf = np.cumsum(scaled_grad_D_omega)

    # Sample uniformly along the CDF to get new omega values
    uniform_samples = np.linspace(0.0, 1.0, n)
    new_omega = np.interp(uniform_samples, cdf, omega)

    return new_omega


# Heuristic


def heuristic(
    config: Configuration, prep: Preparation, has_sigma: bool
) -> Tuple[np.ndarray, np.ndarray]:
    r"""
    Perform parameter selection using a heuristic approach.

    Args:
        config:
            Configuration object.
        prep:
            Preparation object.
        has_sigma:
            Whether the model uses kernel widths as parameters.

    Returns:
        The heuristic kernel centers and, if applicable, the kernel widths."""

    mu_S = np.linspace(
        0.0,
        _y_tol_fit(prep.scaled_D, prep.omega),
        config.n,
        dtype=FLOAT_DTYPE,
    )

    sigma_S = (
        _fat_tol_fit(
            omega=prep.omega,
            D=prep.scaled_D,
            mu=mu_S,
            model_name=config.model_name,
            widths=config.widths,
            window=config.window,
        )
        if has_sigma
        else None
    )

    return mu_S, sigma_S


def _y_tol_fit(
    D: np.ndarray,
    omega: np.ndarray,
    y_tol: float = 0.01,
) -> Tuple[float, float]:
    r"""
    Computes the threshold :math:`\omega` at a certain tolerance level.

    Args:
        D:
            The target values.
        omega:
            The support points.
        y_tol:
            Tolerance parameter used to determine an effective frequency threshold.

    Returns:
        The threshold value :math:`\omega`.
    """

    I = np.trapezoid(D, omega)

    for i in range(len(omega)):
        T_I = np.trapezoid(D[:i], omega[:i])
        if T_I / I >= np.abs(1.0 - y_tol):
            return omega[i]
    return None


def _width_start_values(
    omega: np.ndarray,
    D: np.ndarray,
) -> Tuple[float, float]:
    r"""
    Initialise start values for the kernel widths.

    Args:
        omega: np.ndarray
            The support points.
        D: np.ndarray
            The target values.

    Returns:
        The lower and upper bounds of the kernel widths.
    """
    e = np.trapezoid(omega * D, omega)
    var = np.trapezoid((omega - e) ** 2 * D, omega)
    sigma = np.sqrt(var)
    return 0.0001 * sigma, sigma


def _fat_tol_fit(
    omega: np.ndarray,
    D: np.ndarray,
    mu: np.ndarray,
    model_name: str,
    widths: int,
    window: int,
    fat_tol: float = 0.01,
) -> np.ndarray:
    r"""
    Find the best kernel widths to fit the data.

    Args:
        omega:
            The support points.
        D:
            The target values.
        mu:
            The support points for the model.
        widths:
            The total number of kernel widths.
        window:
            The window size when searching for the optimal kernel widths.
        fat_tol:
            Tolerance used when selecting optimal kernel widths for the model.

    Returns:
        The best kernel widths under the given specifications.
    """

    n_mu, idx_peak = len(mu), np.argmax(D)
    peak_val = D[idx_peak]
    sigma_lower, sigma_upper = _width_start_values(omega, D)
    sigma = np.logspace(np.log10(sigma_lower), np.log10(sigma_upper), widths)[::-1]
    ModelClass = getattr(models, model_name)
    model = ModelClass(np.array([], dtype=FLOAT_DTYPE), mu, sigma)
    E = np.array(
        [
            model.kernel(omega, param=[model.params[0][mi[0]], model.params[1][mi[1]]])
            for mi in model.multi_index_set
        ],
        dtype=FLOAT_DTYPE,
    ).T
    best_sigma = None
    for i in range(len(sigma) - 2):
        _sigma = sigma[i : i + window]
        l, r = i * n_mu, (i + window) * n_mu
        _E = E[:, l:r]
        _c, _ = nnls(_E, D)
        _D = _E @ _c

        if _D[idx_peak] == 0:
            continue
        _peak_val = _D[idx_peak]
        _eps = np.max(np.abs(_peak_val - peak_val)) / peak_val
        if _eps <= np.abs(1.0 - fat_tol):
            best_sigma = np.copy(_sigma)
            break

    if best_sigma is None:
        raise ValueError("No sigmas fulfill the tolerance.")
    return best_sigma
