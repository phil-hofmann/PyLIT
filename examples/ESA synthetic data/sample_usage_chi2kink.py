import sys

import numpy as np

pi = np.pi
from pathlib import Path
from pylit import prepare, itransform
from pylit.core.data_classes import Configuration

# chi2 kink algorithm
from scipy.optimize import curve_fit


def main():
    if len(sys.argv) != 7:
        print("Usage: python sample_usage_chi2kink.py <kernel_name> <method_name>")
        print("                                       <path_F> <path_D>")
        print("                                       <estimate_file> <posterior_file>")
        sys.exit(1)

    # -------------------------------------------------- #
    # 0) Handle function arguments and declare variables #
    # -------------------------------------------------- #
    kernel_name = sys.argv[1]  # selects kernels, must be either 'Gauss' or 'Uniform'
    method_name = sys.argv[
        2
    ]  # selects regularizer, must be 'l2_fit', 'max_entropy_fit', 'cdf_l2_fit'
    path_F = sys.argv[3]  # file containing F(tau)
    path_D = sys.argv[4]  # file containing S(w) Bayesian prior
    estimate_file = sys.argv[5]  # file to save DSF estimate
    SSE_file = sys.argv[6]  # file to save sum of squared errors (SSE)
    # as a function of regularization weight

    # Declare a grid of regularization weights to compute the solution.
    Nlambda = 61
    lambda_min = 1e-14
    lambda_max = 1e3
    lambdas = 10 ** np.linspace(np.log10(lambda_min), np.log10(lambda_max), Nlambda)
    SSE_lambda = np.zeros((Nlambda,))  # SSE at each lambda value

    # number of chi2kink values used to compute final estimate
    NC2K = 10
    chi2kink_vals = np.linspace(
        2.0, 2.5, NC2K
    )  # values recommended by Kaufman and Held
    print(f"lambda_min {lambda_min:.3e} lambda_max {lambda_max:.3e} Nlambda {Nlambda}")
    print(f"chi2kink_vals", chi2kink_vals)

    # --------------------------------------- #
    # 1) Run PyLIT to solve for DSF from ITCF #
    # --------------------------------------- #
    # Declare configuration
    config = Configuration(
        path_F=Path(path_F),
        path_D=Path(path_D),
        # path_S=Path(estimate_file), # store results
        n=50,  # number of sites to place gaussian kernels to use to fit the signal
        model_name=kernel_name,  # kernel choice
        method_name=method_name,  # regularization choice
        lambd=lambdas,
        optimizer_name="nesterov",
        tol=1e-12,  # algorithm runs until tolerance on the OLS criteria
        adaptive=False,
        maxiter=15_000,
    )
    # conduct the inversion
    prep = prepare(config)
    res = itransform(config, prep)
    # NOTE:
    # solution S(w) for various lambda stored as:
    # x_{lambda}(w) = res.S[j][0], where j = 0, .., Nlambda
    # The Laplace transformed solutions for various lambdas are stored as:
    # L{x_{lambda}(w)}(tau) = res.forward_S[j][0], for j = 0, ..., Nlambda

    # ------------------------------- #
    # 2) Conduct chi^2-kink algorithm #
    # ------------------------------- #
    # compute SSE at each regularization weight value
    for i in range(Nlambda):
        SSE_lambda[i] = np.linalg.norm(res.forward_S[i][0] - prep.F[0]) ** 2

    # fit logistic growth curve to the chi^2(\lambda).
    params, error = fit_logistic(np.log10(lambdas), np.log10(SSE_lambda))
    [_, _, c, d] = params

    # estimate best lambda values according to chi^2-kink
    best_lambda_vals = 10 ** (c - chi2kink_vals / d)
    print("best regularization weights", best_lambda_vals)

    # --------------------------------------------#
    # 3) Compute mean and error of best solutions #
    # --------------------------------------------#
    # Compute solution at the Best regularization weights
    # Declare configuration
    config = Configuration(
        path_F=Path(path_F),
        path_D=Path(path_D),
        # path_S=Path(estimate_file), # store results
        n=50,  # number of sites to place gaussian kernels to use to fit the signal
        model_name=kernel_name,  # kernel choice
        method_name=method_name,  # regularization choice
        lambd=best_lambda_vals,
        optimizer_name="nesterov",
        tol=1e-12,  # algorithm runs until tolerance on the OLS criteria
        adaptive=False,
        maxiter=15_000,
    )
    # conduct the inversion
    prep = prepare(config)
    res = itransform(config, prep)

    # Compute the mean and variance of the best solutions and SSE
    Nomegas = len(res.S[0][0]) // 2  # number of grid points are in positive w domain
    x_arr = np.zeros((len(res.S[0][0]), NC2K))
    SSE = 0
    for j in range(NC2K):
        x_arr[:, j] = res.S[j][0]
        SSE += np.linalg.norm(res.forward_S[j][0] - prep.F[0]) ** 2
    SSE /= NC2K
    x_avg = np.average(x_arr, axis=1)
    x_err = np.std(x_arr, axis=1, ddof=1)  # compute the sample standard deviation

    # -------------------- #
    # 4) Store the Results #
    # -------------------- #
    print("SSE at best lambda", SSE)
    # Remove the omega<0 domain and save results
    omegas = prep.omega[Nomegas:]
    Skw_estimate = x_avg[Nomegas:]
    Skw_err = x_err[Nomegas:]
    np.savetxt(
        estimate_file,
        np.column_stack((omegas, Skw_estimate, Skw_err)),
        header="omega(Hartree) S(1/Hartree) dS(1/Hartree)",
        delimiter=",",
    )
    np.savetxt(
        SSE_file,
        np.column_stack((lambdas, SSE_lambda, np.zeros_like(SSE_lambda))),
        header="lambda, SSE(lambda), dSSE",
        delimiter=",",
    )


# Define the logistic function
def logistic_function(alpha, a, b, c, d):
    return a + b / (1 + np.exp(-d * (alpha - c)))


# Fit the data using curve_fit
def fit_logistic(alpha, y):
    initial_guess = [1.0, 1.0, 0.0, 1.0]  # Initial guess for (a, b, c, d)
    params, pcov = curve_fit(
        logistic_function, alpha, y, p0=initial_guess, maxfev=50000
    )
    return params, np.sqrt(np.diag(pcov))


if __name__ == "__main__":
    main()
