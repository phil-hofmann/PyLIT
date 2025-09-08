import os
import sys

sys.path.insert(0, os.path.abspath(".."))
# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = "PyLIT"
copyright = "2025, Phil-Alexander Hofmann, Alexander Benedix Robles, Thomas Chuna"
author = "Phil-Alexander Hofmann, Alexander Benedix Robles, Thomas Chuna"
release = "0.1"

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.doctest",
    "sphinx.ext.napoleon",
    "sphinx_autodoc_typehints",
    "sphinx.ext.mathjax",
]

templates_path = ["_templates"]
exclude_patterns = []


# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = "furo"
html_static_path = ["_static"]
html_logo = "logo.png"
# This appears as the main title
html_title = " "

# This appears as a subheading under the logo
html_theme_options = {
    "sidebar_hide_name": False,  # show project name/subtitle
    "announcement": "v.0.2",  # optional announcement banner
}

# For Furo specifically, you can also set:
html_theme_options.update({
    "navigation_with_keys": True,  # optional
})