import os
import json
import inspect
import numpy as np

from pylit import models
from pathlib import Path
from typing import Tuple, List
from scipy.optimize import nnls
from pylit.settings import FLOAT_DTYPE


def to_string(obj):
    attributes = vars(obj)
    mystr = obj.__class__.__name__ + " object:\n"

    for attr, value in attributes.items():
        if isinstance(value, np.ndarray) or isinstance(value, list):
            if isinstance(value[0], np.ndarray) or isinstance(value[0], list):
                value = (
                    "[["
                    + str(value[0][0])
                    + ", ... ,"
                    + str(value[0][-1])
                    + "], ... ,"
                    + "["
                    + str(value[-1][0])
                    + ", ... ,"
                    + str(value[-1][-1])
                    + "]]"
                )
            else:
                value = "[" + str(value[0]) + ", ... ," + str(value[-1]) + "]"
        elif isinstance(value, float):
            value = "{:.2e}".format(value)
        elif isinstance(value, dict):
            keys = list(value.keys())
            values = list(value.values())
            value = "{" + f"{keys[0]}: {values[0]}, ... ,{keys[-1]}: {values[-1]}" + "}"
        mystr += f"{attr}: {value}\n"

    return mystr


def import_xY(path: Path) -> Tuple[np.ndarray, np.ndarray]:

    if not os.path.isfile(path):
        raise ValueError(f"File: {path} does not exist.")

    # Fetch x, Y from the data file
    data = np.genfromtxt(path, delimiter=",")

    # Data Checks
    if data is None:
        raise ValueError("Data file is empty.")
    if len(data.shape) != 2:
        raise ValueError("Data file must be a 2D array.")
    if data.shape[1] < 2:
        raise ValueError("Data file must have at least two columns.")

    # Extract x, Y
    x, Y = data[:, 0], data[:, 1:]

    # Sort x, Y in ascending order of x
    idx = np.argsort(x)
    x, Y = x[idx], Y[idx, :]

    return x, Y.T


class NumpyArrayEncoder(json.JSONEncoder):
    """Custom JSON encoder for numpy arrays."""

    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)


def save_to_json(obj, filename):
    """
    Serializes an object to JSON and saves it to a file.
    """
    with open(filename, "w") as file:
        json.dump(obj.__dict__, file, cls=NumpyArrayEncoder, indent=4)


def _numpy_array_decoder(obj):
    """Custom decoder function that converts lists to numpy arrays if they contain only numeric types."""
    for key, value in obj.items():
        if isinstance(value, list):  # Check if the value is a list
            try:
                # Attempt to convert list to a numpy array
                obj[key] = np.array(value)
            except ValueError:
                # If conversion fails, keep the original list
                pass
    return obj


def load_from_json(obj, filename):
    """Loads an object from a JSON file, filtering out any keys not present in the __init__ method."""
    with open(filename, "r") as file:
        obj_dict = json.load(file, object_hook=_numpy_array_decoder)

    # Get the parameter names of the __init__ method
    init_params = inspect.signature(obj.__init__).parameters

    # Filter out keys not present in the __init__ method
    filtered_dict = {k: v for k, v in obj_dict.items() if k in init_params}

    return obj(**filtered_dict)


def exp_std(omega: np.ndarray, rho: np.ndarray) -> Tuple[float, float]:
    """Calculate the corrected sample variance for the input data array.

    Args:
        omega: np.ndarray
            The input data array of omega values.

        rho: np.ndarray
            The (probably unscaled and negative) density function of the input data array.

    Returns:
        Tuple[float, float]:
            Returns the expected value (mu) and the standard deviation (sigma)."""

    # Type Conversion
    omega = np.asarray(omega).astype(FLOAT_DTYPE)
    rho = np.asarray(rho).astype(FLOAT_DTYPE)

    # Integrity
    if omega.size != rho.size:
        raise ValueError(
            "The input data array and the density function must have the same size."
        )

    # Calculate the expected value and the standard deviation
    rho = np.amax(
        [rho, np.zeros_like(rho)], axis=0
    )  # Ensure that the density function is non-negative
    rho = rho / np.sum(rho)  # Normalize the density function
    mu = np.sum(omega * rho)  # Expected value
    sigma = np.sqrt(np.sum((omega - mu) ** 2 * rho))  # Standard deviation

    return mu, sigma


def complete_detailed_balance(
    omega: np.ndarray, D: np.ndarray, beta: float
) -> Tuple[np.ndarray, np.ndarray]:
    """Complete the detailed balance of the input data omega and D.

    Args:
        omega : np.ndarray
            The input array of (non-negative) omega values.
        D : np.ndarray
            The input array of D(omega) values.
        beta : float
            The beta parameter for the detailed balance.

    Returns:
        Tuple[np.ndarray, np.ndarray
            The completed omega and D values in terms of detailed balance.
    """

    # Type Conversion
    omega = np.asarray(omega).astype(FLOAT_DTYPE)
    D = np.asarray(D).astype(FLOAT_DTYPE)
    beta = float(beta)

    # Integrity
    if np.any(omega < 0.0):
        raise ValueError(
            "The omega values must be non-negative when the detailed balance is applied."
        )

    # Sort omega ascending
    idx = np.argsort(omega)
    omega, D = omega[idx], D[idx]

    # S(-ω) = exp(-βω) S(ω) for ω in (0, +∞)
    omega_neg = -omega[::-1]
    D_neg = np.exp(beta * omega_neg) * D[::-1]

    # Concatenate
    omega = np.concatenate((omega_neg, omega))
    D = np.concatenate((D_neg, D))

    return omega, D


def moments(omega: np.ndarray, rho: np.ndarray, alphas: np.ndarray) -> np.ndarray:
    # Type Conversion
    omega = np.asarray(omega).astype(FLOAT_DTYPE)
    rho = np.asarray(rho).astype(FLOAT_DTYPE)
    alphas = np.asarray(alphas).astype(FLOAT_DTYPE)

    # Integrity
    if omega.size != rho.size:
        raise ValueError(
            "The input data array and the density function must have the same size."
        )

    # Compute the moments
    return np.array([np.sum(omega**alpha * rho) for alpha in alphas], dtype=FLOAT_DTYPE)


def generate_multi_index_set(dimension: int, degrees: List[int]) -> np.ndarray:
    """Compute the multi indices with respect to the dimension and the degree.

    Args:
        dimension : int
            The dimension of the multi indices.
        degree : List[int]
            The degree of the multi indices.

    Returns:
        np.ndarray
            The multi indices.
    """

    # Compute multi-indices
    meshgrid_arrays = np.meshgrid(
        *([np.arange(degrees[idx_d]) for idx_d in range(dimension)])
    )

    # Reshape multi-indices and return the correct multi-index set
    return np.array([meshgrid_array.ravel() for meshgrid_array in meshgrid_arrays]).T


def diff_interval(tau1: FLOAT_DTYPE, tau0: FLOAT_DTYPE) -> Tuple[callable, callable]:
    def psy(tau: FLOAT_DTYPE) -> FLOAT_DTYPE:
        return (tau - tau0) / (tau1 - tau0)

    def psy_inv(tau: FLOAT_DTYPE) -> FLOAT_DTYPE:
        return tau0 + tau * (tau1 - tau0)

    return psy, psy_inv


def y_tol_fit(
    D: np.ndarray, omega: np.ndarray, exp_D: float, tol: float = 0.9
) -> Tuple[float, float]:
    """
    Computes the threshold ω at a certain tolerance level.

    Args:
        D: np.ndarray
            The target values.
        omega: np.ndarray
            The support points.
        exp_D: float
            The expected value of D.
        tol: float
            The tolerance level. (0 < tol <= 1)

    Returns:
        omega: float
            The threshold value ω.
    """
    # Integrity
    if tol <= 0.0 or tol > 1.0:
        raise ValueError("The tolerance level must be within the interval (0, 1].")

    idx_exp = np.argmin(np.abs(omega - exp_D))
    idx_l, idx_r = idx_exp, idx_exp
    I = np.trapezoid(D, omega)
    if np.isclose(I, 0.0):
        raise ValueError("The integral of D(ω) is zero.")

    for i in range(len(omega)):
        I_lr = np.trapezoid(D[idx_l : idx_r + 1], omega[idx_l : idx_r + 1])
        ratio = I_lr / I
        if ratio >= tol:
            return omega[idx_l], omega[idx_r]
        I_r = (
            np.trapezoid(D[idx_l : idx_r + 2], omega[idx_l : idx_r + 2])
            if idx_r + 2 < len(omega)
            else 0.0
        )
        I_l = (
            np.trapezoid(D[idx_l - 1 : idx_r + 1], omega[idx_l - 1 : idx_r + 1])
            if idx_l > 0
            else 0.0
        )
        if I_r >= I_l:
            idx_r += 1
        else:
            idx_l -= 1


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
    var = np.trapezoid((D - e) ** 2 * omega, omega)
    sigma = np.sqrt(var)
    return 0.0005 * sigma, 0.09 * sigma  # TODO Unclear why these values are chosen


def fat_tol_fit(
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
            The window size when searching for the best kernel widths.

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
        if _eps <= tol:
            best_sigma = np.copy(_sigma)
            break

    if best_sigma is None:
        raise ValueError("Keine Sigmas erfüllen die Toleranzanforderung.")
    return best_sigma

if __name__ == "__main__":
    print(
        fat_tol_fit(
            np.array([1, 2, 3, 4, 5], dtype=FLOAT_DTYPE),
            np.array([1, 2, 3, 4, 5], dtype=FLOAT_DTYPE),
            np.array([1, 2, 3, 4, 5], dtype=FLOAT_DTYPE),
            "Gauss",
            0.9,
            5,
            2,
        )
    )