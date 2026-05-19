.. _installation:

Installation
============

`LIFEsimMC` requires Python **>=3.10** to run.

.. _pip_install:

Installation From PyPI (Recommended)
------------------------------------

To install `LIFEsimMC` from PyPI, run the following command in your terminal:

.. code-block:: console

    pip install lifesimmc

Note that this might take a while, as the installation of the dependency `PyTorch` can take some time.

Installation From GitHub
------------------------
To install `LIFEsimMC` from GitHub, run the following command in your terminal:

.. code-block:: console

    pip install git+https://github.com/pahuber/LIFEsimMC


Alternatively, the repository can be cloned and installed from GitHub using:

.. code-block:: console

    git clone https://github.com/pahuber/LIFEsimMC.git
    cd LIFEsimMC
    pip install .

Test Installation
-----------------

You can check the installation by running the following code in a Python environment:

.. code-block:: python

    import lifesimmc
    print(lifesimmc.__version__)
