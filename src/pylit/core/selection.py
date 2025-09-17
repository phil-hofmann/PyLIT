import numpy as np
from pylit import models
from typing import Tuple
from scipy.optimize import nnls
from pylit.settings import FLOAT_DTYPE
from scipy.ndimage import gaussian_filter1d
from pylit.utils import find_zero, find_max_cutoff

# Initial guess

# TODO c0...

# Simulated Annealing

def simulatedAnnealingAlg(
    omega: np.ndarray,
    tau: np.ndarray,
    D: np.ndarray,
    mu: np.ndarray,
    model_name: str,
    tol: float,
    widths: int,
    window: int,
    detailed_balance: bool,
    beta: float
) -> np.ndarray:
    """Finds the best kernel widths, in order to fit the data.
    
    Args:
        omega: np.ndarray
            The support points.
        D: np.ndarray
            The target values.
        mu: np.ndarray
            The support points for the model.
        tol: float
            The tolerance level.
        widths: int
            The total number of kernel widths.
        window: int
            The window size when searching for the best kernel widths.
    
    Returns:
        np.ndarray
            The best kernel widths under the given specifications."""

    # 0) declare auxillary variables
    SA_beta = 1.5 # Metropolis Hastings Beta
    step_size=0.1 # standard deviation of Gaussian noise is 0.1 of the current value.
    max_SA_iterations = widths # repurpose widths parameter for max_SA_iterations
    print('model name', model_name)
    if (model_name != 'Uniform'):
        interval_widths = np.abs(np.diff(mu))
        max_interval_width = np.amax(interval_widths)
        min_interval_width = np.amin(interval_widths)
        geometric_mean_of_max_min = np.sqrt(max_interval_width * min_interval_width)
        sigma = np.geomspace(2*geometric_mean_of_max_min, geometric_mean_of_max_min/10., window)

        # 2/n, 1/(10n), window
    else:
        sigma=None

    # Instantiate model
    ModelClass = getattr(models, model_name)
    if (model_name != 'Uniform'):
        model = ModelClass(tau, mu, sigma)
    else:
        model = ModelClass(tau, mu)

    model = detailed_balance_decorator(lrm=model, beta=beta)
    model.compute_regression_matrix()
    R = model.regression_matrix
    E = model(omega, matrix=True)
    c, _ = nnls(E, D)
    proposal_D = E @ c

    # 4) instantiate SA diagnostics and SA result
    sigma_list = [sigma];
    svdvals_list = [ np.linalg.svdvals(R) ];
    D_list = [ proposal_D ]

    # NOTE: this seems wrong.
    eps_list = [ np.mean(np.linalg.norm(D - proposal_D)**2) ]
    
    best_mu = np.copy(mu)
    if (model_name != 'Uniform'):
        best_sigma = np.copy(sigma)
    else:
        best_sigma = None
    best_c0 = np.copy(c)
    best_E = np.copy(E)
    best_index = 0
    for i in range(max_SA_iterations):
        # Propose new sigma values by adding Gaussian noise in log space (to stay positive)
        proposal_mu = mu*np.exp( step_size*np.random.normal(size=mu.shape) )
        proposal_mu = np.sort(proposal_mu) # sort just in case!
        if (model_name != 'Uniform'):
            proposal_sigma = sigma*np.exp( step_size*np.random.normal(size=sigma.shape) )
            #scaling = eps_list[-1]/eps_list[0]
            #proposal_sigma = sigma*np.exp( scaling*np.random.normal(size=sigma.shape) )
        else:
            proposal_sigma = None

        # evaluate proposal_sigma
        if (model_name != 'Uniform'):
            model = ModelClass(tau, proposal_mu, proposal_sigma)
        else:
            model = ModelClass(tau, proposal_mu)
        model = detailed_balance_decorator(lrm=model, beta=beta)
        E = model(omega, matrix=True)
        c, _ = nnls(E, D) # invert evaluation to find the coefficients
        proposal_D = E @ c # infer what solution was reach by nnls

        # NOTE: this seems wrong.
        proposal_eps = np.mean(np.linalg.norm(D - proposal_D)**2) # assess chi_2 of appropriate interval

        # Metropolis Hastings (MH) update
        delta_eps = proposal_eps - eps_list[-1]
        if delta_eps < 0 or np.random.rand() < np.exp(-SA_beta * delta_eps):

            # update mu and
            mu = proposal_mu
            sigma = proposal_sigma

            # collect new sigmas and associated chi-sq
            eps_list.append( proposal_eps )
            sigma_list.append( proposal_sigma )

            # collect new singular values
            model.compute_regression_matrix()
            R = model.regression_matrix
            svdvals_list.append( np.linalg.svdvals(R) )
            # check is this error is below threshold
            if proposal_eps <= tol:
                best_index += 1
                best_c0 = np.copy(c)
                best_E = np.copy(E)
                best_mu = np.copy(proposal_mu)
                best_sigma = np.copy(proposal_sigma)
                print('MH iteration:' ,i, 'proposal_eps: ', proposal_eps, ' updated best_sigma: ', best_sigma)
                break
            # check if this error the global best and cool down the algorithm
            elif( np.all( proposal_eps <= eps_list) ):
                SA_beta *= 0.9
                step_size *= 0.9
                best_index += 1
                best_c0 = np.copy(c)
                best_E = np.copy(E)
                best_mu = np.copy(proposal_mu)
                best_sigma = np.copy(proposal_sigma)
                print('MH iteration:' ,i, 'proposal_eps: ', proposal_eps, ' updated best_sigma: ', best_sigma)

    #if best_sigma is None:
    #    raise ValueError("Keine Sigmas erfüllen die Toleranzanforderung.")
    return np.abs(best_mu), best_sigma, best_c0

def _select_omega_by_distribution(omega:np.ndarray, D_omega:np.ndarray, n:int, detailed_balance:bool, std:float=5.0, cutoff:float=1e-8):
    r"""Select distribution of kernel centers based on where
    :math:`D(\omega)` has the most mass.
    
    Args:
        omega:
            1D array of input values.
        D_omega: 
            1D array of function values corresponding to w.
        n: 
            Number of points in the new redistributed array.
    
    Returns:
        np.ndarray: 
            New omega values redistributed based on the CDF."""

    # Cutoff omega and D
    left = find_zero(omega) if detailed_balance else len(omega)-1-find_max_cutoff(np.flip(D_omega))
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
        omega:np.ndarray, D_omega:np.ndarray, n:int, detailed_balance: bool, std=5, cutoff=1e-12,
    ):
    r"""Select distribution of kernel centers based on where the gradient
    of :math:`D(\omega)` has the most mass.
    
    Args:
        omega: 
            1D array of input values.
        D_omega: 
            1D array of function values corresponding to w.
        n: 
            Number of points in the new redistributed array.
    
    Returns:
        np.ndarray: New omega values redistributed based on gradient magnitude."""

    # Cutoff omega and D
    left = find_zero(omega) if detailed_balance else len(omega)-1-find_max_cutoff(np.flip(D_omega))
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

def heuristic():
    pass

def _y_tol_fit(
    D: np.ndarray, omega: np.ndarray, tol: float = 0.95
) -> Tuple[float, float]:
    """
    Computes the threshold ω at a certain tolerance level.

    Args:
        D: np.ndarray
            The target values.
        omega: np.ndarray
            The support points.
        tol: float
            The tolerance level. (0 < tol <= 1)

    Returns:
        omega: float
            The threshold value ω.
    """

    I = np.trapezoid(D, omega)

    for i in range(len(omega)):
        T_I = np.trapezoid(D[:i], omega[:i])
        if T_I / I >= np.abs(1.0 - tol):
            return omega[i]
    return None


def _width_start_values(
    omega: np.ndarray,
    D: np.ndarray,
) -> Tuple[float, float]:
    """
    Initialise start values for the kernel widths.

    Args:
        omega: np.ndarray
            The support points.
        D: np.ndarray
            The target values.

    Returns:
        Tuple[float, float]
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
    tol: float,
    widths: int,
    window: int,
) -> np.ndarray:
    """
    Finds the best kernel widths, in order to fit the data.

    Args:
        omega: np.ndarray
            The support points.
        D: np.ndarray
            The target values.
        mu: np.ndarray
            The support points for the model.
        tol: float
            The tolerance level.
        widths: int
            The total number of kernel widths.
        window: int
            The window size when searching for the optimal kernel widths.

    Returns:
        np.ndarray
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
        if _eps <= np.abs(1.0 - tol):
            best_sigma = np.copy(_sigma)
            break

    if best_sigma is None:
        raise ValueError("No sigmas fulfill the tolerance.")
    return best_sigma
