# Welcome to Pylit 🚀!

**✨Python Laplace Inverse Transformation✨**

This software provides a Python implementation of the inverse Laplace Transformation.

### 📜 License

The project is licensed under the [MIT License](LICENSE.txt).

### 💬 Citations

- Inverse Laplace transform in quantum many-body theory. [Alexander Benedix Robles](a.benedix-robles@hzdr.de), [Phil-Alexander Hofmann](mailto:philhofmann@outlook.com), [Tobias Dornheim](t.dornheim@hzdr.de), [Michael Hecht](m.hecht@hzdr.de)

### 👥 Team and Support

- [Phil-Alexander Hofmann](https://github.com/philippocalippo/)
- [Alexander Benedix Robles](https://github.com/alexanderbenedix/)

### 🙏 Acknowledgments

We would like to acknowledge:

- [Dr. Tobias Dornheim](https://www.casus.science/de-de/team-members/dr-tobias-dornheim/) - [CASUS](https://www.casus.science/) ([HZDR](https://www.hzdr.de/)),
- [Prof. Dr. Michael Hecht](https://www.casus.science/de-de/team-members/michael-hecht/) - [CASUS](https://www.casus.science/) ([HZDR](https://www.hzdr.de/)),

and the support and resources provided by the [Center for Advanced Systems Understanding](https://www.casus.science/) ([Helmholtz-Zentrum Dreden-Rossendorf](https://www.hzdr.de/)) where the development of this project took place.

### 📝 Remarks

This project originated from a prototype developed by Alexander Benedix Robles. His initial work laid the foundation for the current implementation.

## 💻 Installation

### Including in Your Project (Poetry)

To include this package in your project using **Poetry**, follow these steps:

1. **(Optional) Create and activate a virtual environment with Python 3.12 or newer**

```bash
python3.12 -m venv .venv
source .venv/bin/activate  # On Linux/macOS
.venv\Scripts\activate     # On Windows
```

2. **Ensure that pip is installed (it should be by default)**

```bash
python -m ensurepip --upgrade
```

3. **Ensure that poetry is installed**

```bash
pip show poetry || pip install poetry
```

4. **If you don’t already have a pyproject.toml file, initialize one**

```bash
poetry init
```

5. **Use Poetry to install `pylit` directly from the GitHub repository**

```bash
poetry add git+https://github.com/phil-hofmann/pylit.git
```

**To install from a specific branch, use**

```bash
poetry add git+https://github.com/phil-hofmann/pylit.git@branch_name
```

6. **Enter the Poetry shell for working in your project**

```bash
poetry shell
```

7. **Exit the Poetry shell when you're done**

```bash
exit
```

8. **Deactivate the virtual environment**

```bash
deactivate
```

### Including in Your Project (Anaconda)

To include this package in your project using **Anaconda**, follow these steps:

1. **Create a Conda environment with Python 3.12 or newer (if not already set up)**

```bash
conda create --name venv python>=3.12
```

2. **Activate the environment**

```bash
conda activate venv
```

3. **Ensure that pip is installed (it should be by default)**

```bash
conda install pip && pip --version
```

4. **Use pip to install pylit directly from the GitHub repository**

```bash
pip install git+https://github.com/phil-hofmann/pylit.git
```

To install from a specific branch, use

```bash
pip install git+https://github.com/phil-hofmann/pylit.git@branch_name
```

5. **When you're done working, deactivate the environment**

```bash
conda deactivate
```

### Setting Up the Repository on Your Local Machine

1. **Clone the repository**

```bash
git clone https://github.com/phil-hofmann/pylit.git
cd pylit
```

2. **Create a virtual environment**

```bash
python3 -m venv .venv
```

3. **Activate the virtual environment**

```bash
source .venv/bin/activate
```

4. \*\*Install pip (if not already installed)

```bash
python3 -m ensurepip --upgrade
```

5. **Install Poetry**

```bash
pip install poetry
```

6. **Install dependencies using Poetry**

```bash
poetry install
```

7. **Deactivate the virtual environment**

```bash
deactivate
```

### ⚙️ Usage: Example

```python
if __name__ == "__main__":
    import pylit
    import numpy as np
    from pathlib import Path

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

### 🏗️ DIY

How to implement [Methods](#), [Models](#) and [Optimizers](#) by yourself.
