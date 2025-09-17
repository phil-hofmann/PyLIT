import os
import json
import inspect
import numpy as np

from pathlib import Path
from typing import Tuple, List
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


def find_zero(array: np.ndarray) -> int:
    """Finds the index where the sign flips from negative to positive.
    
    Args:
        array:
            One-dimensional array of values.
    Returns:
        Index `i+1` such that arr[i] < 0 and arr[i+1] >= 0. Returns None if no flip is found."""
    for i in range(len(array) - 1):
        if array[i] < 0 and array[i + 1] >= 0:
            return i+1
    return None


def find_max_cutoff(array: np.ndarray, cutoff: float):
    """
    Find the first index after the global maximum where the array falls below a cutoff.

    Args:
        array:
            One-dimensional array of values.
        cutoff:
            Threshold value.

    Returns:
        Index of the first element smaller than the ``cutoff`` when scanning
        forward from the global maximum. If no such element is found,
        returns None.
    """
    i_start = np.argmax(array)
    for i in range(i_start, len(array)):
        if (array[i] < cutoff):
            return i
    return None
