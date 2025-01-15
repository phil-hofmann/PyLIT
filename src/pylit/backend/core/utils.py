import json
import inspect
import numpy as np
from typing import Tuple
from pylit.global_settings import ARRAY, FLOAT_DTYPE


class NumpyArrayEncoder(json.JSONEncoder):
    """Custom JSON encoder for numpy arrays."""

    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()  # Convert ndarray to list
        # Let the base class default method raise the TypeError
        return json.JSONEncoder.default(self, obj)


def save_to_json(obj, filename):
    """
    Serializes an object to JSON and saves it to a file.
    """
    with open(filename, "w") as file:
        json.dump(obj.__dict__, file, cls=NumpyArrayEncoder, indent=4)


def numpy_array_decoder(obj):
    """
    Custom decoder function that converts lists to numpy arrays
    if they contain only numeric types.
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


def load_from_json(obj, filename):
    """
    Loads an object from a JSON file, filtering out any keys not present in the __init__ method.
    """
    with open(filename, "r") as file:
        obj_dict = json.load(file, object_hook=numpy_array_decoder)

    # Get the parameter names of the __init__ method
    init_params = inspect.signature(obj.__init__).parameters

    # Filter out keys not present in the __init__ method
    filtered_dict = {k: v for k, v in obj_dict.items() if k in init_params}

    return obj(**filtered_dict)


def empty_array():
    return np.array([])


def exp_std(x: ARRAY, rho: ARRAY) -> Tuple[FLOAT_DTYPE, FLOAT_DTYPE]:
    """Calculate the corrected sample variance for the input data array.

    Args:
        x: ARRAY
            The input data array.

        rho: ARRAY
            The (probably unscaled and negative) density function of the input data array.

    Returns:
        Tuple[FLOAT_DTYPE, FLOAT_DTYPE]:
            Returns the expected value and the standard deviation."""

    # Type Conversion
    x = np.asarray(x).astype(FLOAT_DTYPE)
    rho = np.asarray(rho).astype(FLOAT_DTYPE)

    # Integrity
    if x.size != rho.size:
        raise ValueError(
            "The input data array and the density function must have the same size."
        )

    # Calculate the expected value and the standard deviation
    rho = np.amax([rho, np.zeros_like(rho)], axis=0) # Ensure that the density function is non-negative
    rho = rho / np.sum(rho)  # Normalize the density function
    mean = np.sum(x * rho)  # Expected value
    std = np.sqrt(np.sum((x - mean) ** 2 * rho))  # Standard deviation

    return mean, std


def moments(
    x: ARRAY, rho: ARRAY, alphas: ARRAY
) -> ARRAY:
    # Type Conversion
    x = np.asarray(x).astype(FLOAT_DTYPE)
    rho = np.asarray(rho).astype(FLOAT_DTYPE)
    alphas = np.asarray(alphas).astype(FLOAT_DTYPE)

    # Integrity
    if x.size != rho.size:
        raise ValueError(
            "The input data array and the density function must have the same size."
        )

    # Calculate the moments
    return np.array([np.sum(x**alpha * rho) for alpha in alphas], dtype=FLOAT_DTYPE) # Moments


def complete_detailed_balance(
    omegas: ARRAY, S: ARRAY, beta: FLOAT_DTYPE
) -> Tuple[ARRAY, ARRAY]:
    """Extend the input array to the negative axis.

    Args:
    -----
    omegas : ARRAY
        The input array of omegas in ascending order.
    S : ARRAY
        The input array of S(omega) values.
    beta : FLOAT_DTYPE
        The decay factor.

    Raises:
    -------
    ValueError:
        If the omegas are not in ascending order.

    Returns
    -------
    Tuple[ARRAY, ARRAY]:
        The completed omega and S values.
    """

    # Type check
    if not isinstance(omegas, ARRAY):
        raise TypeError("The input array of omegas must be an array.")

    # Type Conversion
    omegas = omegas.astype(FLOAT_DTYPE)

    # Integrity
    if not np.all(omegas[:-1] <= omegas[1:]):
        raise ValueError("The input array of omegas must be in ascending order.")

    # S(ω) = exp(βω) S(-ω) for ω in (-∞, +∞)
    idx_pos = np.where(omegas >= 0)
    idx_neg = np.where(omegas < 0)

    omegas_pos, S_pos = omegas[idx_pos], S[idx_pos]
    omegas_neg, S_neg = omegas[idx_neg], S[idx_neg]

    # Complete ω > 0
    omegas_neg_to_pos = np.abs(omegas_neg)
    S_neg_to_pos = S_neg * np.exp(beta * omegas_neg_to_pos)

    omegas_pos_complete_unordered = np.concatenate((omegas_neg_to_pos, omegas_pos))
    S_pos_complete_unordered = np.concatenate((S_neg_to_pos, S_pos))

    idx_pos_complete = np.argsort(omegas_pos_complete_unordered)
    omegas_pos_complete = omegas_pos_complete_unordered[idx_pos_complete]
    S_pos_complete = S_pos_complete_unordered[idx_pos_complete]

    # Complete ω < 0
    omegas_pos_to_neg = -omegas_pos
    S_pos_to_neg = S_pos * np.exp(beta * omegas_pos_to_neg)

    omegas_neg_complete_unordered = np.concatenate((omegas_pos_to_neg, omegas_neg))
    S_neg_complete_unordered = np.concatenate((S_pos_to_neg, S_neg))

    idx_neg_complete = np.argsort(omegas_neg_complete_unordered)
    omegas_neg_complete = omegas_neg_complete_unordered[idx_neg_complete]
    S_neg_complete = S_neg_complete_unordered[idx_neg_complete]

    # Glue the two parts together
    omegas_complete = np.concatenate((omegas_neg_complete, omegas_pos_complete))
    S_complete = np.concatenate((S_neg_complete, S_pos_complete))

    return omegas_complete, S_complete
