if __name__ == "__main__":
    import numpy as np
    from pathlib import Path
    from pylit import prepare, itransform
    from pylit.core.data_classes import Configuration

    path_F = Path("/Users/philhofmann/Documents/pylit-workspace/data/raw_F/F10.csv")
    path_D = Path("/Users/philhofmann/Documents/pylit-workspace/data/raw_S/S10.csv")

    path_S = Path("/Users/philhofmann/Downloads/my_S.csv")
    path_L_S = Path("/Users/philhofmann/Downloads/my_L_S.csv")

    lambdas = np.array([10e-4], dtype=np.float64)

    config = Configuration(
        path_F=path_F,
        path_D=path_D,
        path_S=path_S,
        path_L_S=path_L_S,
        adaptive=False,
        optimizer_name="adam",
        method_name="l2_fit",
        lambd=lambdas,
        maxiter=1_000,
        detailed_balance=True,
        model_name="Gauss",
        protocol=True,
        tol=10e-10,
    )
    prep = prepare(config)
    res = itransform(config, prep)

    # print(f"beta: {prep.beta}")
    # print(f"mu: {res.mu}, sigma: {res.sigma}")

    import matplotlib as mpl
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

    # print("Shape S=", np.array(res.S).shape)
