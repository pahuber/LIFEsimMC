.. _seo:

Single-Epoch Observations
=========================

Overview
--------

The **single-epoch observation preset** provides a straightforward way to simulate a measurement using the **current
reference architecture of LIFE**. This includes

* the **generation of synthetic data** including **instrumental noise**,
* data **post-processing** (data whitening), and
* several tools for **signal extraction**.

It requires the **specification of the astrophysical scene** (e.g. properties of the observed target) and returns the **estimated planet SED** with corresponding uncertainties/covariances.

.. hint::
    A **public web interface** is available for ``LIFEsimMC``, providing access to the **GUI for the single-epoch observation** preset.
    See below for more information.

    .. raw:: html

       <p>
         <a class="button" href="https://huggingface.co/spaces/pahuber/LIFEsimMC-Test" target="_blank">
           Go to Public Web Interface (Beta)
         </a>
       </p>



Implementation & Assumptions
----------------------------

As the **reference design of LIFE is evolving** and instrument specifications and assumptions will change in the future,
the implementation of the single-epoch observation preset will also change. To account for this, the **preset
is versioned** and all versions are available `here <https://github.com/pahuber/LIFEsimMC/tree/main/lifesimmc/presets/single_epoch_observation/versions>`_.
The API **documentation of the current version including a list of all configurable parameters** can be found in the
:doc:`SingleEpochObservation <../source/presets/seo>` class documentation.

The current single-epoch observation preset includes the **following assumptions**:

* **Instrument Architecture:**
  Emma-X design with a 1:6 baseline ratio. Numerical values of the instrument parameters can be found `here <https://github.com/pahuber/LIFEsimMC/blob/main/lifesimmc/presets/single_epoch_observation/versions/single_epoch_observation_v1.py#L129>`_.
* **Noise Sources:**
  The noise consists of uncorrelated photon noise form astrophysical sources  (star, local zodi exozodi, planet) and correlated instrumental noise from instrumental perturbations
  (no, optimistic, or pessimistic levels as defined in `Huber et al. 2025 <https://doi.org/10.3847/1538-3881/adfb6b>`_).
* **Post-Processing:**
  Data whitening is applied to the synthetic data to reduce the impact of correlated instrumental noise (see `Huber et al. 2025 <https://doi.org/10.3847/1538-3881/adfb6b>`_).
* **Signal Extraction:**
  The planetary SED (, i.e., spectral energy distribution) is estimated using a numerical maximum likelihood estimator (see `Huber et al. 2025 <https://doi.org/10.3847/1538-3881/adfb6b>`_).
  It currently assumes the true planet position and SED as the initial values for the optimization.

Running a Single-Epoch Observation
----------------------------------

The single-epoch observation preset requires the **specification of the astrophysical scene**, including information on the
target star, exozodiacal dust, and the observed planet. Additionally, the **integration time** for the observation must be specified.

.. note::
    For a **documentation of all configurable parameters and default values**, check out the :doc:`SingleEpochObservation <../source/presets/seo>` class documentation.

The easiest way to run a single-epoch observation simulation with ``LIFEsimMC`` is through the **GUI**.
This can be done through the `public web interface <https://huggingface.co/spaces/pahuber/LIFEsimMC-Test>`_ or by running it **locally**.
However, it can also be run in a regular Python script through the **Python API**.

Graphical User Interface (GUI)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
.. _fig-gui:

.. figure:: ../_static/gui.png
   :alt: Single-Epoch Observation GUI
   :width: 100%
   :align: center

   Screenshot of the GUI of the single-epoch observation simulator.

To run the GUI (see screenshot below) locally (after `installing LIFEsimMC <../installation.rst>`_), open a console and run the following command:

.. code-block:: console

    lifesimmc-gui

The single-epoch observation simulator will then be hosted locally on your machine and you can just click on `http://127.0.0.1:7861 <http://127.0.0.1:7861>`_
to open it. Alternatively, you can open a browser and manually navigate to `http://127.0.0.1:7861 <http://127.0.0.1:7861>`_. A screenshot of the GUI
is shown in the figure above.

After specification of the astrophysical scene and the integration time (and optionally other parameters), the full simulation
can directly be run by clicking the "Run Simulation" button. The estimated planet SED and several other outputs are then displayed in the GUI.
All outputs can be downloaded as ``numpy`` arrays from within the GUI.


Python API
~~~~~~~~~~

All results offered by the GUI can also be accessed through the Python API. The following notebook illustrates how to
setup the astrophysical scene, run the single-epoch observation and plot all results.

.. include:: seo_api.ipynb
   :parser: myst_nb.docutils_


