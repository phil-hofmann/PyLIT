import pylit 
import numpy as np
import numba as nb
from pylit.optimize import Solution

import scipy as sp

from pylit.global_settings import FLOAT_DTYPE, INT_DTYPE, ARRAY, TOL

def nn_trust(
    R: ARRAY,
    F: ARRAY,
    x0: ARRAY,
    method,
    maxiter=None,
    tol=None,
    protocol=False,
) -> Solution:
    """Solves the optimization problem using the trust region non-negative method."""

    # Integrity
    if not R.shape[0] == len(F):
        raise ValueError(
            "The number of rows of R and the length of F must be equal"
        )

    # Type Conversion
    R = R.astype(FLOAT_DTYPE)
    F = F.astype(FLOAT_DTYPE)
    x0 = x0.astype(FLOAT_DTYPE)

    # Prepare
    n, m = R.shape
    maxiter = 2 * n if maxiter is None else INT_DTYPE(maxiter)
    tol = 10 * max(m, n) * TOL if tol is None else FLOAT_DTYPE(tol)

    # Subroutine
    x = _nn_trust(
        R,
        F,
        method.f,
        method.grad_f,
        x0,
        maxiter,
        tol,
        method.pr,
        protocol,
    )
    fx = method.f(x, R, F)

    return Solution(x, fx, FLOAT_DTYPE(0.5 * np.sum((R @ x - F) ** 2)))

# @nb.njit # NOTE: Numba does not support scipy.optimize
def _nn_trust(
    R, F, f, grad_f, x0, maxiter, tol, pr, protocol
) -> ARRAY:
    
    # Initialize variables:
    x = np.copy(x0)

    # Subroutine implementation:

    cons_ = { # Non negative constraint
        "type": "ineq",
        "fun": pr if pr is not None else lambda x: x,
    } 
    f_ = lambda x: f(x, R, F)
    grad_f_ = lambda x: grad_f(x, R, F)

    x = sp.optimize.minimize(
        f_,
        x,
        jac=grad_f_,
        constraints=cons_,
        method="trust-constr",
        tol=tol,
        options={"disp": protocol, "maxiter": maxiter},
    ).x

    return x

# # NOTE: Deprecated:
# def trust_nn_lsq_l1_reg(
#     R: ARRAY,
#     F: ARRAY,
#     E: ARRAY = None,
#     lambd: FLOAT_DTYPE = 1.0,
#     c0: ARRAY = None,
#     svd=True,
#     maxiter: INT_DTYPE = 2000,
#     tol=TOL,
# ) -> ARRAY:
#     """Solves the optimization problem using SVD and Lp Regularization.

#     Parameters
#     ----------
#     R : matrix
#         Regression matrix.
#     F : array
#         Target values.
#     E : array
#         Evaluation matrix.
#     lambd : float
#         The scaling parameter.
#     c0 : array
#         Initial guess.
#     svd : bool
#         If True, the SVD decomposition is used. (TODO)
#     maxiter : int
#         Maximum number of iterations.
#     tol : float
#         Tolerance.

#     Raises
#     ------
#     ValueError
#         If the number of rows of R and the length of F are not equal.
#         If the scaling parameter is negative.
#         If the length of the initial guess is not equal to the number of columns of R.

#     Returns
#     -------
#     array
#         Solution.

#     Note:
#     -----
#     - NOTE DEPRECATED
#     - NOTE AUTOGRAD NOT ACTIVATED ANYMORE
#     - TODO Cite Otsuki
#     - TODO Add svd=False mode"""

#     # Integrity
#     M, N = R.shape

#     if not M == len(F):
#         raise ValueError(
#             "The number of rows of R and the length of F must be equal"
#         )

#     if not E is None:
#         # if not E.shape[0] == len(F):
#         #    raise ValueError('The number of rows of E and the length of F must be equal')

#         if not E.shape[1] == N:
#             raise ValueError(
#                 "The number of columns of E and the number of columns of R must be equal"
#             )

#     if not lambd >= 0.0:
#         raise ValueError("The sclaing parameter must be non-negative.")

#     if c0 is not None:
#         if not len(c0) == N:
#             raise ValueError(
#                 "The length of the initial guess must be equal to the number of columns of R."
#             )

#     # Type Conversion
#     R = R.astype(FLOAT_DTYPE)
#     F = F.astype(FLOAT_DTYPE)
#     E = (
#         E.astype(FLOAT_DTYPE)
#         if E is not None
#         else np.eye(N, dtype=FLOAT_DTYPE)
#     )
#     lambd = FLOAT_DTYPE(lambd)
#     c0 = (
#         np.zeros(N, dtype=FLOAT_DTYPE)
#         if c0 is None
#         else c0.astype(FLOAT_DTYPE)
#     )

#     # Single Value Decomposition:
#     # ---------------------------
#     # U, V  ... orthonormal matrices
#     # S     ... diagonal matrix with singular values
#     # R = U @ S @ Vt
#     # Ut @ R @ c - Ut @ F = S @ Vt @ c - Ut @ F

#     U, S, V = pylit.utils.svd(R)
#     # print(f"||R - U @ S @ Vt|| = {np.linalg.norm(R - U @ S @ V.T)}")

#     Ut, Vt = U.T, V.T
#     M, N = U.shape[0], V.shape[0]

#     # Redefine the problem as:
#     # -----------------------
#     # c_prime := Vt @ c
#     # F_prime := Ut @ F
#     # min 0.5*||S @ c_prime - F_prime||_2^2 + lambd*||c_prime||_p

#     F_prime = Ut @ F
#     c0_prime = Vt @ c0

#     print(
#         f"||S @ c0_prime - F_prime|| = {np.linalg.norm(S @ c0_prime - F_prime)}"
#     )

#     def loss(c_prime):
#         return 0.5 * np.sum((S @ c_prime - F_prime) ** 2) + (lambd) * np.sum(
#             np.abs(c_prime)
#         )

#     jac = autograd.jacobian(loss)  # TODO write explicitly
#     hess = autograd.hessian(loss)  # TODO write explicitly
#     cons = {
#         "type": "ineq",
#         "fun": lambda c_prime: E @ V @ c_prime,
#     }  # NOTE Evaluation of model should be positive

#     # print(f"||grad(c0_prime)|| = {np.linalg.norm(jac(c0_prime))}")

#     c_prime = spopt.minimize(
#         loss,
#         c0_prime,
#         jac=jac,
#         constraints=cons,
#         method="trust-constr",
#         tol=tol,
#         options={"disp": True, "maxiter": maxiter},
#     ).x

#     # print(f"||grad(c_prime)|| = {np.linalg.norm(jac(c_prime))}")

#     print(
#         f"||S @ c_prime - F_prime|| = {np.linalg.norm(S @ c_prime - F_prime)}"
#     )

#     c = V @ c_prime

#     return c

if __name__ == "__main__":
    pass