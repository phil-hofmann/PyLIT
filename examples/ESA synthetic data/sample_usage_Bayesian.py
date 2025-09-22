import sys

import numpy as np

pi = np.pi
from pathlib import Path
from pylit import prepare, itransform
from pylit.core.data_classes import Configuration


def main():
    if len(sys.argv) != 7:
        print("Usage: python script.py <kernel_name> <method_name> <path_F> <path_D> ")
        print("                        <estimate_file> <posterior_file>")
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
    posterior_file = sys.argv[6]  # file to save posterior distribution

    # Declare a grid of regularization weights to compute the solution.
    Nlambda = 61
    lambda_min = 1e-9
    lambda_max = 1e2
    lambdas = 10 ** np.linspace(np.log10(lambda_min), np.log10(lambda_max), Nlambda)
    print(f"lambda_min {lambda_min:.3e} lambda_max {lambda_max:.3e} Nlambda {Nlambda}")

    # ------------------------------------ #
    # 1) Run PyLIT to solve for DSF from ITCF #
    # ------------------------------------ #
    # configure PyLIT settings
    config = Configuration(
        path_F=Path(path_F),
        path_D=Path(path_D),
        # path_S=Path(estimate_file), # store results if desired
        n=50,  # number of library functions (e.g. # of gaussian kernels)
        model_name=kernel_name,
        method_name=method_name,
        lambd=lambdas,  # lambda values to try
        optimizer_name="nesterov",
        tol=1e-12,  # algorithm runs until tolerance on the OLS criteria
        adaptive=False,
        maxiter=15_000,  # max number of nesterov alg. iterations
    )

    # conduct the inversion at all regularization weights
    prep = prepare(config)
    res = itransform(config, prep)
    # NOTE:
    # solution S(w) for various lambda stored as:
    # x_{lambda}(w) = res.S[j][0], where j = 0, .., Nlambda
    # The Laplace transformed solutions for various lambdas are stored as:
    # L{x_{lambda}(w)}(tau) = res.forward_S[j][0], for j = 0, ..., Nlambda

    # ------------------------------------------- #
    # 2) Conduct Gull's Bayesian Weighting scheme #
    # ------------------------------------------- #
    # Load in data
    F_data = np.loadtxt(path_F, delimiter=",")  # [tau, F(tau)-full, F(tau)-JK0,...]
    taus_invHartree = F_data[:, 0]
    F = F_data[:, 1]

    D_data = np.loadtxt(path_D, delimiter=",")  # [tau, F(tau)-full, F(tau)-JK0,...]
    omegas_Hartree = D_data[:, 0]
    Nomegas = len(omegas_Hartree)
    dw_Hartree = omegas_Hartree[1] - omegas_Hartree[0]
    omegas_negpos = np.concatenate([-np.flip(omegas_Hartree), omegas_Hartree])
    D_pos = D_data[:, 1]
    D_negpos = fill_in_negative_omega_with_detailed_balance(D_pos, omegas_Hartree)

    # Instantiate discrete double sided Laplace Kernel
    EE, TT = np.meshgrid(omegas_negpos, taus_invHartree)
    DoubleSidedLaplaceKernel = TT * EE
    for j in range(len(taus_invHartree)):
        for k in range(len(omegas_negpos)):
            DoubleSidedLaplaceKernel[j, k] = np.exp(-DoubleSidedLaplaceKernel[j, k])

    # Compute the Bayesian posterior (weighting function)
    objfxn = np.zeros((Nlambda,))
    SSE_lambda = np.zeros((Nlambda,))
    for j in range(Nlambda):  # compute posterior on alpha grid
        # consider solution at a given lambda
        x = res.S[j][0]

        # compute the Least Squares cost function
        L = 0.5 * np.linalg.norm(res.forward_S[j][0] - prep.F[0]) ** 2
        SSE_lambda[j] = 2 * L

        # compute the regularization value
        if method_name == "l2_fit":
            regularization_term = lambdas[j] * np.linalg.norm(x - D_negpos) ** 2
        elif method_name == "max_entropy_fit":
            regularization_term = lambdas[j] * np.nansum(
                D_negpos - x + x * np.log(x / D_negpos)
            )
        elif method_name == "cdf_l2_fit":
            regularization_term = (
                lambdas[j] * np.linalg.norm(np.cumsum(x) - np.cumsum(D_negpos)) ** 2
            )

        # Compute the Gull Term
        B = np.sqrt(x) * np.eye(len(x))
        LAMBDA = DoubleSidedLaplaceKernel @ B
        LAMBDA = (dw_Hartree**2) * LAMBDA.T @ LAMBDA
        LAMBDA_eigvals, _ = np.linalg.eigh(LAMBDA)
        if np.any((lambdas[j] + LAMBDA_eigvals) < 0):
            Gull_term = -np.inf
        elif np.any(LAMBDA_eigvals) == 0.0:
            Gull_term = -np.inf
        else:
            Gull_term = 0.5 * np.sum(np.log(lambdas[j] / (lambdas[j] + LAMBDA_eigvals)))

        PalphaHm = 1  # Laplace's Rule;
        # PalphaHm= 1/lambdas[j] # Jeffery's Rule;

        objfxn[j] = -L - regularization_term + Gull_term
        print(
            "alpha:",
            lambdas[j],
            "Fidelity",
            L,
            "regularization",
            regularization_term,
            "Gull term",
            Gull_term,
        )

    # compute the weighted average according to Bayesian Posterior
    max_index = np.argmax(objfxn)
    # print('weight argument', objfxn[:] - objfxn[max_index])
    P_tot = 0.0
    x_avg = np.zeros_like(res.S[j][0])
    xsq_avg = np.zeros_like(res.S[j][0])
    acceptance = np.zeros((Nlambda,))
    for j in range(Nlambda):
        # accept or reject alpha based on weighting function
        weight = np.exp(objfxn[j] - objfxn[max_index])
        if weight > 0.1:
            x_avg += (res.S[j][0]) * weight
            xsq_avg += (res.S[j][0] * res.S[j][0]) * weight
            P_tot += weight
            acceptance[j] = 1
    x_avg /= P_tot
    xsq_avg /= P_tot

    # -------------------- #
    # 3) Store the Results #
    # -------------------- #
    # compute the value of the final estimate
    xtmp = np.reshape(x_avg, (-1, 1))
    SSE = (
        np.linalg.norm(
            dw_Hartree * (DoubleSidedLaplaceKernel @ xtmp).flatten() - prep.F[0]
        )
        ** 2
    )

    # Remove the omega<0 domain from x_avg
    Skw_estimate = x_avg[Nomegas:]
    Skw_err = np.sqrt(xsq_avg - x_avg * x_avg)[Nomegas:]
    Prob_estimate = np.exp(objfxn - np.amax(objfxn))
    Peak_estimate = omegas_Hartree[np.argmax(x_avg[Nomegas:])]

    print("acceptance arr", acceptance)
    print("SSE across lambda", SSE_lambda)
    print("MEM peak estimate: %.3e" % (Peak_estimate))
    np.savetxt(
        estimate_file,
        np.column_stack((omegas_Hartree, Skw_estimate, Skw_err)),
        header="omega(Hartree) S(1/Hartree) dS(1/Hartree)",
        delimiter=",",
    )
    np.savetxt(
        posterior_file,
        np.column_stack((lambdas, Prob_estimate, acceptance)),
        header="alpha, P(alpha), avg_acceptance",
        delimiter=",",
    )


def fill_in_negative_omega_with_detailed_balance(Skw_pos_omega, pos_omega_Hartree):
    beta_invHartree = pos_omega_Hartree[-1]
    print("beta_invHartree shape", np.shape(beta_invHartree))
    Skw_neg_omega = np.exp(-beta_invHartree * pos_omega_Hartree) * np.flip(
        Skw_pos_omega
    )
    return np.concatenate([Skw_neg_omega, Skw_pos_omega])


if __name__ == "__main__":
    main()
