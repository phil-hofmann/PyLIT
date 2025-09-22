if __name__ == "__main__":
    import numpy as np
    from pathlib import Path
    from pylit import prepare, itransform
    from pylit.core.data_classes import Configuration

    path_F = Path(__file__).parent / "F10.csv"
    path_D = Path(__file__).parent / "S10.csv"
    path_S = Path(__file__).parent / "S_out.csv"
    path_L_S = Path(__file__).parent / "L_S_out.csv"
    path_prep = Path(__file__).parent / "prep.json"
    path_res = Path(__file__).parent / "res.json"
    lambdas = np.array([10e-6], dtype=np.float64)

    config = Configuration(
        path_F=path_F,
        path_D=path_D,
        path_S=path_S,
        path_L_S=path_L_S,
        path_prep=path_prep,
        path_res=path_res,
        ### Parameter selection default values:
        selection_name="simulated_annealing",
        n=100,
        # window=5,
        widths=5,
        ### Model default values:
        non_negative=True,
        detailed_balance=True,
        model_name="Gauss",
        ### Method:
        method_name="l1_reg",
        lambd=lambdas,
        ### Optimizer:
        optimizer_name="nesterov",  # default value
        tol=10e-10,
        adaptive=False,
        adaptive_residuum_mode=False,  # default value
        maxiter=1_000,  # default value
        c0=None,  # default value
        svd=False,  # default value
        protocol=True,
    )

    # Shortcut:
    config = Configuration(
        path_F=path_F,
        path_D=path_D,
        path_S=path_S,
        path_L_S=path_L_S,
        path_prep=path_prep,
        path_res=path_res,
        ### Method:
        method_name="l2_fit",
        lambd=lambdas,
        ### Optimizer:
        tol=10e-10,
        adaptive=False,
        protocol=True,
    )

    print(f"✅ Configuration instantiated.")
    prep = prepare(config)
    print(f"✅ Prepared data.")
    res = itransform(config, prep)
    print(f"✅ Finished inverse transform.")
    print(f"beta: {prep.beta}")
    print(f"mu: {res.mu}, sigma: {res.sigma}")

    import matplotlib.pyplot as plt

    n = len(res.S)
    for i, s in enumerate(res.S):
        plt.plot(prep.omega, s[0])
    plt.plot(prep.omega, prep.scaled_D * prep.max_F, color="black", linestyle="--")
    plt.show()

    for i, forward_s in enumerate(res.forward_S):
        plt.plot(prep.tau, forward_s[0])
    plt.plot(prep.tau, prep.forward_D * prep.max_F, color="green")
    plt.plot(prep.tau, prep.F[0], color="black", linestyle="--")
    plt.show()

    for i, eps_s in enumerate(res.eps_S):
        plt.plot(prep.tau, eps_s[0])
    plt.plot(prep.tau, prep.eps_D[0], color="black", linestyle="--")
    plt.show()
