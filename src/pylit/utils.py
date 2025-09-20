import os
import json
import inspect
import numpy as np

from pathlib import Path
from typing import Tuple, List
from pylit.settings import FLOAT_DTYPE


def to_string(obj):
    """
    Create a human-readable string representation of an object and its attributes.

    Handles special formatting for arrays, lists, floats, and dictionaries.

    Args:
        obj:
            Any object with attributes accessible via ``vars(obj)``.

    Returns:
        str:
            A string summarizing the object's attributes.
    """

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
    """
    Load and preprocess data from a CSV file into x and Y arrays.

    The first column is treated as the x-values, while the remaining
    columns are treated as Y-values. The data is sorted by x.

    Args:
        path:
            Path to the CSV file.

    Returns:
        Tuple[np.ndarray, np.ndarray]:
            - x (1D array): Sorted x-values.
            - Y (2D array): Corresponding Y-values (transposed for row-wise access).

    Raises:
        ValueError: If the file does not exist, is empty, or has invalid format.
    """

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
    """Custom JSON encoder that converts NumPy arrays to lists."""

    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)


def save_to_json(obj, filename: Path):
    """
    Serialize an object's attributes to JSON and save to file.

    NumPy arrays are converted to lists for compatibility.

    Args:
        obj:
            Object with a ``__dict__`` attribute.
        filename:
            Path to save the JSON file.
    """

    with open(filename, "w") as file:
        json.dump(obj.__dict__, file, cls=NumpyArrayEncoder, indent=4)


def _numpy_array_decoder(obj) -> dict:
    """
    Decode JSON data by converting lists to numpy arrays when possible.

    Args:
        obj:
            Dictionary parsed from JSON.

    Returns:
        Dictionary with lists converted to numpy arrays when appropriate.s.
    """
    for key, value in obj.items():
        if isinstance(value, list):  # Check if the value is a list
            try:
                # Attempt to convert list to a numpy array
                obj[key] = np.array(value)
            except ValueError:
                # If conversion fails, keep the original list
                pass
    return obj


def load_from_json(obj, filename: Path):
    """
    Load and reconstruct an object from JSON file.

    Filters out any keys not present in the ``__init__`` method
    of the target class.

    Args:
        obj:
            Class type to instantiate.
        filename:
            Path to the JSON file.

    Returns:
        Instance of ``obj`` constructed with filtered attributes.
    """

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
        omega:
            The input data array of omega values.
        rho:
            Density function values, must match the size of ``omega``.

    Returns:
        Tuple[float, float]:
            - mu: Expected value.
            - sigma: Standard deviation."""

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
    r"""Extend the default model :math:`D(\omega)` to satisfy the detailed balance.

    Args:
        omega:
            Array of non-negative discrete frequency axis.
        D:
            The default model :math:`D(\omega)` evaluated at ``omega``.
        beta:
            Inverse temperature parameter :math:`\beta`

    Returns:
        Tuple[np.ndarray, np.ndarray]:
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
    """
    Compute statistical moments of a distribution.

    Args:
        omega:
            Array of discrete frequency axis.
        rho:
            Density function values, must match the size of ``omega``.
        alphas:
            Powers for which to compute the moments.

    Returns:
        Array of computed moments.
    """

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
    """Generate a set of multi-indices.

    Args:
        dimension:
            Length of multi-indices. (In practice, only ``dimension=1`` is used.)
        degrees:
            Maximum degree in each dimension.


    Returns:
        Array of multi-indices, shape (n_combinations, dimension).
    """

    # Compute multi-indices
    meshgrid_arrays = np.meshgrid(
        *([np.arange(degrees[idx_d]) for idx_d in range(dimension)])
    )

    # Reshape multi-indices and return the correct multi-index set
    return np.array([meshgrid_array.ravel() for meshgrid_array in meshgrid_arrays]).T


def diff_interval(tau1: FLOAT_DTYPE, tau0: FLOAT_DTYPE) -> Tuple[callable, callable]:
    """Construct diffeomorphic transformations between two intervals.

    Args:
        tau1:
            Upper bound of the target interval.
        tau0:
            Lower bound of the target interval.

    Returns:
        Tuple[callable, callable]:
            - Forward mapping diffeomorphism ``psy(tau)``.
            - Inverse mapping diffeomorphism ``psy_inv(tau)``.
    """

    def psy(tau: FLOAT_DTYPE) -> FLOAT_DTYPE:
        return (tau - tau0) / (tau1 - tau0)

    def psy_inv(tau: FLOAT_DTYPE) -> FLOAT_DTYPE:
        return tau0 + tau * (tau1 - tau0)

    return psy, psy_inv


def find_zero(array: np.ndarray) -> int:
    """Finds the index where the sign changes from negative to positive.

    Args:
        array:
            One-dimensional array of values.
    Returns:
        Index `i+1` such that arr[i] < 0 and arr[i+1] >= 0.
        Returns None if no sign change is found.
    """
    for i in range(len(array) - 1):
        if array[i] < 0 and array[i + 1] >= 0:
            return i + 1
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
        Index of the first element smaller than the ``cutoff`` after the (global) maximum.
        Returns None if no such element is found.
    """
    i_start = np.argmax(array)
    for i in range(i_start, len(array)):
        if array[i] < cutoff:
            return i
    return None
