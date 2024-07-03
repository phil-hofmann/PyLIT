from setuptools import setup, find_packages

if __name__ == "__main__":
    setup(
        name="pylit",
        author=["Alexander Benedix Robles", "Phil Hofmann"],
        author_email=["alexbenedix@gmail.com", "philhofmann@outlook.com"],
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
            "autograd",
            "ipykernel",
            "sympy",
            "nbsphinx",
            "sphinx-book-theme",
            "numba",
            "python-docx",
            "pypandoc",
            "pdflatex",
            "streamlit",
            "plotly",
            "streamlit_option_menu",
        ],
    )
