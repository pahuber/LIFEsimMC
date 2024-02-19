"""Sphinx configuration."""
project = "LIFEsim 2"
author = "Philipp A. Huber"
copyright = "2024, Philipp A. Huber"
extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx_click",
    "myst_parser",
]
autodoc_typehints = "description"
html_theme = "furo"
