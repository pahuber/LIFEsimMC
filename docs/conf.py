# Configuration file for the Sphinx documentation builder.

import os
import sys
import urllib.request

# -- Path setup --------------------------------------------------------------

sys.path.insert(0, os.path.abspath(".."))
sys.path.insert(0, os.path.abspath("../lifesimmc"))
os.environ["PYTHONPATH"] = os.path.abspath("..")

# -- Project information -----------------------------------------------------

project = "LIFEsimMC"
author = "Philipp A. Huber"
copyright = "2024, Philipp A. Huber"

# -- General configuration ---------------------------------------------------

extensions = [
    "sphinx_copybutton",
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "nbsphinx",
    "recommonmark",
    "sphinx_contributors",
]

master_doc = "index"

# -- Notebook execution ------------------------------------------------------

nbsphinx_execute = "never"
nb_execution_mode = "off"

# -- HTML output -------------------------------------------------------------

html_theme = "shibuya"
html_title = "LIFEsimMC Docs"
html_static_path = ["_static"]

html_theme_options = {
    "light_logo": "_static/lifesimmc_dark.png",
    "dark_logo": "_static/lifesimmc_light.png",
    "github_url": "https://github.com/pahuber/LIFEsimMC",
    "globaltoc_expand_depth": 1,
    "accent_color": "blue",
}

html_context = {
    "source_type": "github",
    "source_user": "pahuber",
    "source_repo": "LIFEsimMC",
    "source_version": "main",
    "source_docs_path": "/docs/",
}

html_sidebars = {
    "**": [
        "sidebars/localtoc.html",
        "sidebars/repo-stats.html",
    ]
}

html_css_files = [
    "custom.css",
]

pygments_style = "friendly"
pygments_dark_style = "monokai"

exclude_patterns = [
    "_build",
    "Thumbs.db",
    ".DS_Store",
    "**/.ipynb_checkpoints",
]

# -- nbsphinx configuration --------------------------------------------------

nbsphinx_codecell_lexer = "python3"

# -- Matplotlib configuration ------------------------------------------------

try:
    import matplotlib as mpl

    mpl.rcParams["text.usetex"] = False
except ImportError:
    pass


# -- External documentation files --------------------------------------------

def setup(app):
    urls = [
        "https://raw.githubusercontent.com/pahuber/PHRINGE/refs/heads/main/docs/source/observation.rst",
        "https://raw.githubusercontent.com/pahuber/PHRINGE/refs/heads/main/docs/source/instrument.rst",
        "https://raw.githubusercontent.com/pahuber/PHRINGE/refs/heads/main/docs/source/scene.rst",
        "https://raw.githubusercontent.com/pahuber/PHRINGE/refs/heads/main/docs/source/configuration.rst",
        "https://raw.githubusercontent.com/pahuber/PHRINGE/refs/heads/main/docs/source/all_sources/star.rst",
        "https://raw.githubusercontent.com/pahuber/PHRINGE/refs/heads/main/docs/source/all_sources/planet.rst",
        "https://raw.githubusercontent.com/pahuber/PHRINGE/refs/heads/main/docs/source/all_sources/local_zodi.rst",
        "https://raw.githubusercontent.com/pahuber/PHRINGE/refs/heads/main/docs/source/all_sources/exozodi.rst",
    ]

    all_sources_files = {
        "star.rst",
        "planet.rst",
        "local_zodi.rst",
        "exozodi.rst",
    }

    for url in urls:
        filename = os.path.basename(url)

        if filename in all_sources_files:
            local_dir = os.path.join(os.path.dirname(__file__), "source", "external", "all_sources")
        else:
            local_dir = os.path.join(os.path.dirname(__file__), "source", "external")

        os.makedirs(local_dir, exist_ok=True)
        urllib.request.urlretrieve(url, os.path.join(local_dir, filename))
