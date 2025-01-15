import numpy as np
from numba import njit

# from IPython.display import display, Math
from typing import Tuple, List

# from pylit.backend.models.ABC import LinearRegressionModelABC
from pylit.global_settings import ARRAY, FLOAT_DTYPE, INT_DTYPE


def extendS(
    S_pos: ARRAY,
    omega_pos: ARRAY,
    beta: FLOAT_DTYPE,
    S_only=False,
) -> Tuple[ARRAY, ARRAY]:
    """Extend the S(omega) function to the negative axis.

    With given values of S(omega) having omega >= 0
    this function returns S(omega) with the left side, thus where
    omega < 0.

    Parameters
    ----------
    S_pos : ARRAY
        The values of S(omega) with omega >= 0.
    omega_pos : ARRAY
        The non negative omega points.
    beta : float
        The decay factor.

    Returns
    -------
    omega : ARRAY
    S : ARRAY
        S(omega) values.

    Notes:
    ------
    1. Having as an Input a model (from abstract) and returning in future a decorator would be nice.
    """

    S_neg = np.flip(S_pos)
    S_neg = S_pos[:] * np.exp(-beta * omega_pos[:])
    S_ext = np.concatenate((np.flip(S_neg), S_pos))

    omega_ext = extend_on_negative_axis(omega_pos)

    return omega_ext, S_ext


def generate_multi_index_set(dimension: INT_DTYPE, degrees: List[INT_DTYPE]) -> ARRAY:
    """Compute the multi indices with respect to the dimension and the degree.

    Parameters
    ----------
    dimension : INT_DTYPE
        The dimension of the multi indices.
    degree : INT_DTYPE
        The degree of the multi indices.

    Returns
    -------
    ARRAY
        The multi indices."""

    # Compute multi-indices
    meshgrid_arrays = np.meshgrid(
        *([np.arange(degrees[idx_d]) for idx_d in range(dimension)])
    )

    # Reshape multi-indices and return the correct multi-index set
    return np.array([meshgrid_array.ravel() for meshgrid_array in meshgrid_arrays]).T


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


def svd(M: ARRAY) -> Tuple[ARRAY, ARRAY, ARRAY]:
    U, s, Vt = np.linalg.svd(M)
    V = Vt.T

    M, N = U.shape[0], V.shape[0]
    S = np.zeros((M, N), dtype=FLOAT_DTYPE)
    np.fill_diagonal(S, s[: min(M, N)])

    return U, S, V


def diff_interval(tau1: FLOAT_DTYPE, tau0: FLOAT_DTYPE) -> Tuple[callable, callable]:
    def psy(tau: FLOAT_DTYPE) -> FLOAT_DTYPE:
        return (tau - tau0) / (tau1 - tau0)

    def psy_inv(tau: FLOAT_DTYPE) -> FLOAT_DTYPE:
        return tau0 + tau * (tau1 - tau0)

    return psy, psy_inv


# TODO deprecated
# def lrm_analysis(model: LinearRegressionModelABC, invertible: bool = False):
#     # Regression Matrix
#     R = model.regression_matrix

#     # Condition number
#     display(Math("\|R\|_1 = " + "{:.2e}".format(np.linalg.norm(R, ord=1))))

#     if invertible:
#         R_inv = np.linalg.inv(R)
#         display(Math("\|R^{-1}\|_1 = " + "{:.2e}".format(np.linalg.norm(R_inv, ord=1))))

#     else:
#         R_plus = np.linalg.pinv(R)
#         display(Math("\|R^+\|_1 = " + "{:.2e}".format(np.linalg.norm(R_plus, ord=1))))

#     # SVD Precision
#     U, S, V = svd(R)
#     display(
#         Math(
#             "\|R - U \cdot S \cdot V^\\top\|_1 = "
#             + "{:.2e}".format(np.linalg.norm(R - U @ S @ V.T, ord=1))
#         )
#     )


def local_maxima(nodes: ARRAY, func_values: ARRAY) -> ARRAY:
    """Find the indices where the function represented by func_values has local maxima.

    Parameters:
    -----------
        func_values: ARRAY
            ARRAY representing the values of the function.

    Returns:
    --------
        ARRAY:
            Returns ARRAY of tuples containing (node, function_value) at local maxima.

    Notes:
    ------
    - TODO make nodes optional"""

    # Calculate the first differences (backward differences) of the function values
    diff_values = np.diff(func_values)

    # Find indices where the differences change sign from positive to negative
    sign_changes = np.where(diff_values[:-1] * diff_values[1:] < 0)[0] + 1

    # Filter out the local maxima (values greater than both neighbors)
    local_maxima_indices = sign_changes[
        np.where(func_values[sign_changes] > func_values[sign_changes - 1])[0]
    ]

    # Create an array of tuples containing nodes and corresponding function values
    maxima_entries = np.array(
        [(nodes[idx], func_values[idx]) for idx in local_maxima_indices]
    )

    return maxima_entries


def str_to_int_array(string: str) -> ARRAY:
    """Convert a string of integers separated by commas to an array.

    Parameters:
    -----------
        string: str
            The string of integers separated by commas.

    Returns:
    --------
        ARRAY:
            The array containing the integers."""

    return np.array([INT_DTYPE(x) for x in string.split(",")], dtype=INT_DTYPE)


def int_array_to_str(int_array):
    """
    Convert a list of integers to a string representation.

    Parameters:
        int_array (list): List of integers.

    Returns:
        str: String representation of the list of integers.
    """
    return ",".join(str(x) for x in int_array)


def add_white_noise(data: ARRAY, percent: FLOAT_DTYPE = 5.0) -> ARRAY:
    """Add white noise to the input data array.

    Parameters:
    -----------
        data: ARRAY
            The input data array.

        percent: FLOAT_DTYPE
            The percentage of the maximum absolute value of the input data array. (Default: 5.0)

    Returns:
    --------
        ARRAY:
            Returns the noisy data array."""

    return data + np.random.normal(0, (percent / 100) * np.max(np.abs(data)), len(data))


def global_maximum(nodes, func_values) -> ARRAY:
    """Find the global maximum value, corresponding node, and corresponding function value.

    Parameters:
    -----------
        nodes: ARRAY
            Array representing the nodes.

        func_values: ARRAY
            Array representing the values of the function.

    Returns:
    --------
        ARRAY:
            Returns the node and the function value at the global maximum.

    Notes:
    ------
    - TODO make nodes optional"""

    local_maxima_entries = local_maxima(nodes, func_values)

    if len(local_maxima_entries) == 0:
        raise ValueError("No local maxima found.")

    global_max_entry = max(local_maxima_entries, key=lambda x: x[1])

    return global_max_entry


def print_str_dict(d, indent=0):
    output = ""
    for key, value in d.items():
        if isinstance(value, dict):
            output += "\t" * indent + str(key) + ":\n"
            output += print_str_dict(value, indent + 4)
        else:
            if isinstance(value, np.ndarray):
                value = str(value[0]) + " ... " + str(value[-1])
            output += "\t" * indent + str(key) + ": " + str(value) + "\n"
    return output


def print_dict(d, indent=0):
    for key, value in d.items():
        if isinstance(value, dict):
            print("\t" * indent + str(key) + ":")
            print_dict(value, indent + 4)
        else:
            if isinstance(value, np.ndarray):
                value = str(value[0]) + " ... " + str(value[-1])
            print("\t" * indent + str(key) + ": " + str(value))


def print_list_of_dicts(list_of_dicts, indent=0):
    # NOTE: Deprecated !?!
    for item in list_of_dicts:
        for key, value in item.items():
            if isinstance(value, dict):
                print("\t" * indent + str(key) + ":")
                print_dict(value, indent + 4)
            else:
                print("\t" * indent + str(key) + ": " + str(value))
        print()


def print_list_of_objects(list_of_objs):
    for obj in list_of_objs:
        print(to_string(obj))


def convert_np_arrays_to_lists(data):
    """
    Recursively converts NumPy arrays to nested lists in a dictionary.

    Parameters:
    - data (dict): Input dictionary possibly containing NumPy arrays.

    Returns:
    - dict: Dictionary with NumPy arrays converted to nested lists.
    """
    if isinstance(data, np.ndarray):
        # If data is a NumPy array, convert it to a list
        return data.tolist()
    elif isinstance(data, dict):
        # If data is a dictionary, recursively process its values
        return {key: convert_np_arrays_to_lists(value) for key, value in data.items()}
    elif isinstance(data, (list, tuple)):
        # If data is a list or tuple, recursively process its elements
        return [convert_np_arrays_to_lists(item) for item in data]
    else:
        # For other types (e.g., scalar values), return as is
        return data


@njit
def jit_sub_mat_by_index_set(mat: ARRAY, I: ARRAY) -> ARRAY:
    n = len(I)
    sub = np.zeros((n, n), dtype=mat.dtype)

    if len(mat.shape) == 2:
        for i_, i in enumerate(I):
            for j_, j in enumerate(I):
                sub[i_][j_] = mat[i][j]

    else:
        raise NotImplementedError("Unsupported number of dimensions for mat.")

    return sub


@njit
def jit_sub_vec_by_index_set(vec: ARRAY, I: ARRAY) -> ARRAY:
    n = len(I)
    sub = np.zeros(n, dtype=vec.dtype)

    if len(vec.shape) == 1:
        for i_, i in enumerate(I):
            sub[i_] = vec[i]

    else:
        raise NotImplementedError("Unsupported number of dimensions for vec.")

    return sub


@njit
def argmax(array):
    max_value = array[0]
    max_index = 0
    for i in range(1, len(array)):
        if array[i] > max_value:
            max_value = array[i]
            max_index = i
    return max_index


@njit
def trapz_mat(x: ARRAY):
    # Calculate differences in x
    dx = x[1:] - x[:-1]

    # Create a matrix to hold the values for the trapezoidal rule
    trapz_mat = np.zeros((len(x) - 1, len(x)))

    # Fill the matrix with the values for the trapezoidal rule
    np.fill_diagonal(trapz_mat[:, :-1], dx)
    np.fill_diagonal(trapz_mat[:, 1:], dx)

    return trapz_mat


@njit
def svd_optim(R: ARRAY, F: ARRAY, x0: ARRAY):
    U, S_diag, VT = np.linalg.svd(R)
    S = np.zeros_like(R)
    np.fill_diagonal(S, S_diag)
    V, UT = VT.T, U.T
    F_prime = np.copy(UT @ F)
    x0_prime = VT @ x0
    return S, F_prime, x0_prime, V
