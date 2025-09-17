# Welcome to Pylit 🚀!

**✨Python Laplace Inverse Transformation✨**

This software provides a Python implementation of the inverse Laplace Transformation.

## 📜 License

The project is licensed under the [MIT License](LICENSE.txt).

## 💬 Citations

- Inverse Laplace transform in quantum many-body theory. [Alexander Benedix Robles](a.benedix-robles@hzdr.de), [Phil-Alexander Hofmann](mailto:philhofmann@outlook.com), [Tobias Dornheim](t.dornheim@hzdr.de), [Michael Hecht](m.hecht@hzdr.de)

## 👥 Team and Support

- [Phil-Alexander Hofmann](https://github.com/philippocalippo/)
- [Alexander Benedix Robles](https://github.com/alexanderbenedix/)
- [Thomas Chuna](https://github.com/chunatho)
- [Dr. Tobias Dornheim](https://www.casus.science/de-de/team-members/dr-tobias-dornheim/) - [CASUS](https://www.casus.science/) ([HZDR](https://www.hzdr.de/)),
- [Prof. Dr. Michael Hecht](https://www.casus.science/de-de/team-members/michael-hecht/) - [CASUS](https://www.casus.science/) ([HZDR](https://www.hzdr.de/)),


We would like to acknowledge the support and resources provided by the [Center for Advanced Systems Understanding](https://www.casus.science/) ([Helmholtz-Zentrum Dreden-Rossendorf](https://www.hzdr.de/)) where the development of this project took place.


## ⚙️ Example Usage

```python
if __name__ == "__main__":
    import numpy as np
    from pathlib import Path
    from pylit import prepare, itransform
    from pylit.core.data_classes import Configuration

    path_F = Path("F.csv")
    path_D = Path("D.csv")

    config = Configuration(
        path_F=path_F,
        path_D=path_D,
        adaptive=False,
        optimizer_name="nesterov",
        method_name="l1_reg",
        lambd=np.array([10e-8], dtype=np.float64),
        maxiter=1_000,
        detailed_balance=True,
        model_name="Uniform",
    )
    prep = prepare(config)
    res = itransform(config, prep)

    print(f"beta: {prep.beta}")
    print(f"mu: {res.mu}, sigma: {res.sigma}")

    import matplotlib.pyplot as plt

    plt.plot(prep.omega, res.S[0][0], color="red")
    plt.plot(prep.omega, prep.scaled_D * prep.max_F, color="black", linestyle="--")
    plt.plot(res.mu, np.zeros_like(res.mu), "o", color="blue")
    plt.show()

    plt.plot(prep.tau, res.forward_S[0][0], color="red")
    plt.plot(prep.tau, prep.forward_D * prep.max_F, color="green")
    plt.plot(prep.tau, prep.F[0], color="black", linestyle="--")
    plt.show()

    plt.plot(prep.tau, res.eps_S[0][0], color="red")
    plt.plot(prep.tau, prep.eps_D[0], color="black", linestyle="--")
    plt.show()
```

Happy experimenting! 🎉

## 🏗️ DIY

**!WIP!**

How to implement [Methods](#), [Models](#) and [Optimizers](#) by yourself.
