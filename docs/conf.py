# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Path setup --------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.

import os
import sys
import urllib

sys.path.insert(0, os.path.abspath('../'))
# sys.path.insert(0, os.path.abspath('../'))

# -- Project information -----------------------------------------------------

project = 'LIFEsimMC'
author = 'Philipp A. Huber'
copyright = f'2024, Philipp A. Huber'
# html_theme_options = {
#     "logo_light": "_static/phringe2.png",
#     "logo_dark": "_static/phringe.png"
# }
html_theme = "furo"
master_doc = 'index'
html_static_path = ['_static']
html_css_files = ["style.css"]
pygments_style = "monokai"
pygments_dark_style = "monokai"
# html_logo = "_static/lifesimmc_logo.png"
html_title = "LIFEsimMC Docs"
html_theme_options = {
    "logo_only": True,  # Show only the logo, not the project name
    "sidebar_hide_name": True,
    "light_logo": "lifesimmc_logo_light.png",
    "dark_logo": "lifesimmc_logo.png",
}
html_context = {
    "display_github": False,  # Example of context variable
    "title": ""
}

# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.

extensions = ['sphinx_copybutton',
              'sphinx.ext.autodoc',
              'sphinx.ext.napoleon',
              'nbsphinx', ]
# 'recommonmark']

# -- Options for HTML output -------------------------------------------------


import os
import sys

sys.path.insert(0, os.path.abspath('../lifesimmc'))


def setup(app):
    urls = ["https://raw.githubusercontent.com/pahuber/PHRINGE/refs/heads/main/docs/source/observation.rst",
            "https://raw.githubusercontent.com/pahuber/PHRINGE/refs/heads/main/docs/source/instrument.rst",
            "https://raw.githubusercontent.com/pahuber/PHRINGE/refs/heads/main/docs/source/scene.rst",
            "https://raw.githubusercontent.com/pahuber/PHRINGE/refs/heads/main/docs/source/configuration.rst",
            "https://raw.githubusercontent.com/pahuber/PHRINGE/refs/heads/main/docs/source/all_sources/star.rst",
            "https://raw.githubusercontent.com/pahuber/PHRINGE/refs/heads/main/docs/source/all_sources/planet.rst",
            "https://raw.githubusercontent.com/pahuber/PHRINGE/refs/heads/main/docs/source/all_sources/local_zodi.rst",
            "https://raw.githubusercontent.com/pahuber/PHRINGE/refs/heads/main/docs/source/all_sources/exozodi.rst",
            ]

    for url in urls:
        # Extract the filename from the URL
        filename = os.path.basename(url)

        if 'star' in filename or 'planet' in filename or 'local_zodi' in filename or 'exozodi' in filename:
            local_path = os.path.join(os.path.dirname(__file__), "source/external/all_sources", filename)
        else:
            local_path = os.path.join(os.path.dirname(__file__), "source/external", filename)

        os.makedirs(os.path.dirname(local_path), exist_ok=True)
        urllib.request.urlretrieve(url, local_path)
