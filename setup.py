from setuptools import setup, find_packages

if __name__ == "__main__":
    setup(
        name="pylit",
        author=["Phil-Alexander Hofmann", "Alexander Benedix Robles"],
        author_email=["philhofmann@outlook.com", "alexbenedix@gmail.com"],
        description="A package for the double sided inverse Laplace transformation.",
        python_requires=">=3.8",
        version="0.1",
        packages=find_packages(where="src"),
        package_dir={"": "src"},
        install_requires=[
            "numpy",
            "matplotlib",
            "scipy",
            "sphinx",
            "ipykernel",
            "sympy",
            "nbsphinx",
            "sphinx-book-theme",
            "numba",
            "streamlit",
            "plotly",
        ],
    )
