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
copyright = "2025, Phil-Alexander Hofmann, Alexander Benedix Robles, Thomas Chuna, Tobias Dornheim, Michael Hecht"
author = "Phil-Alexander Hofmann, Alexander Benedix Robles, Thomas Chuna, Tobias Dornheim, Michael Hecht"
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

html_theme = "pydata_sphinx_theme"
html_static_path = ["_static"]
html_logo = "logo.png"
html_title = ""
html_theme_options = {
    "logo": {
        "text": "",
    },
    "navbar_end": ["navbar-icon-links"],
    "secondary_sidebar_items": [],
    "navbar_align": "content",
    "show_prev_next": False,
    "announcement": "v0.2",
}
html_theme_options.update(
    {
        "icon_links": [
            {
                "name": "GitHub",
                "url": "https://github.com/phil-hofmann/pylit",
                "icon": "fa-brands fa-github",
            }
        ],
        "use_edit_page_button": False,
        "navigation_depth": 4,
    }
)
html_sidebars = {
    "installation": [],
    "get-started": [],
    "modules": [],
}
