import json
import numpy as np
from typing import Tuple
from pylit.global_settings import ARRAY, FLOAT_DTYPE


def save_to_json(obj, filename):
    """
    Serializes an object to JSON and saves it to a file.
    """
    with open(filename, "w") as file:
        json.dump(obj.__dict__, file, indent=4)


def load_from_json(obj, filename):
    """
    Loads an object from a JSON file.
    """
    with open(filename, "r") as file:
        obj_dict = json.load(file)
    return obj(**obj_dict)


def empty_array():
    return np.array([])


def exp_std(x: ARRAY, rho: ARRAY) -> Tuple[FLOAT_DTYPE, FLOAT_DTYPE]:
    """Calculate the corrected sample variance for the input data array.

    Parameters:
    -----------
        x: ARRAY
            The input data array.

        rho: ARRAY
            The (probably unscaled and negative) density function of the input data array.

    Returns:
    --------
        Tuple[FLOAT_DTYPE, FLOAT_DTYPE]:
            Returns the expected value and the standard deviation."""

    rho = rho - np.min(rho) * any(rho < 0.0)  # Shift the density function
    rho = rho / np.sum(rho)  # Normalize the density function
    mean = np.sum(x * rho)  # Expected value
    std = np.sqrt(np.sum((x - mean) ** 2 * rho))  # Standard deviation

    return mean, std


def extend_on_negative_axis(nodes: ARRAY) -> ARRAY:
    """Extend the input array to the negative axis.

    Args:
    -----
    nodes : ARRAY
        The array containing the nodes which are greater or equal zero.

    Raises:
    -------
    ValueError:
        If the nodes are not non-negative.

    Returns
    -------
    ARRAY:
        The nodes extended symmetrical to the negative axis.
    """

    # Type check
    if not isinstance(nodes, ARRAY):
        raise TypeError("The nodes must be an array.")

    # Type Conversion
    nodes = nodes.astype(FLOAT_DTYPE)

    # Integrity
    if np.any(nodes < 0.0):
        raise ValueError("The nodes must be non-negative.")

    return np.concatenate((-np.flip(nodes), nodes))


def extend_S(S_pos: ARRAY, omega_pos: ARRAY, beta: FLOAT_DTYPE) -> ARRAY:
    """Extend the S(omega) function to the negative axis.

    With given values of S(omega) having omega >= 0
    this function returns S(omega) with the left side, thus where
    omega < 0.

    Args:
    -----
    S_pos : ARRAY
        The values of S(omega) with omega >= 0.
    omega_pos : ARRAY
        The non negative omega points.
    beta : FLOAT_DTYPE
        The decay factor.

    Returns:
    --------
    ARRAY:
        S(omega) values.
    """

    S_neg = np.flip(S_pos)
    S_neg = S_pos[:] * np.exp(-beta * omega_pos[:])
    S_ext = np.concatenate((np.flip(S_neg), S_pos))

    return S_ext
