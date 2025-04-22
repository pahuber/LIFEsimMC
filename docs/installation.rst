.. _installation:

Installation
============

Prerequisites
-------------
* | **Python Installation:**
  | `LIFEsimMC` requires Python **3.10** or **3.11** to run. If you do not have Python installed, you can download it `here <https://www.python.org/downloads/>`_.
* | **Virtual Environment:**
  | We recommend installing `LIFEsimMC` in a virtual environment to avoid conflicts with other Python packages. For instructions on how to create and activate a virtual environment, see the `virtualenv user guide <https://virtualenv.pypa.io/en/latest/user_guide.html>`_.

.. _pip_install:

Installation From PyPI (Recommended)
------------------------------------

To install `LIFEsimMC` from PyPI, run the following command in your terminal:

.. code-block:: console

    pip install lifesimmc

Note that this might take a while, as the installation of the dependency ``PyTorch`` can take some time. You can check
the installation by running the following code in a Python environment:

.. code-block:: python

    import lifesimmc
    print(lifesimmc.__version__)


Installation From GitHub
------------------------
To install `LIFEsimMC` from GitHub, run the following command in your terminal:

.. code-block:: console

    pip install git+https://github.com/pahuber/LIFEsimMC


Alternatively, the repository can be cloned from GitHub using:

.. code-block:: console

    git clone https://github.com/pahuber/LIFEsimMC.git

After navigating to the cloned repository, the package can be installed using:

.. code-block:: console

    pip install .