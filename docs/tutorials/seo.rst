.. _seo:

Single-Epoch Observations
=========================

.. raw:: html

   <p>
     <a class="button" href="https://huggingface.co/spaces/pahuber/LIFEsimMC-Test" target="_blank">
       Go to Public Web Interface (Beta)
     </a>
   </p>

Graphical User Interface (GUI)
------------------------------

The easiest way to run a single-epoch observation simulation with ``LIFEsimMC`` is through the GUI (see screenshot below).
After specification of the astrophysical scene and the integration time (and optionally other parameters), the full simulation
can directly be run by clicking the "Run Simulation" button. The estimated planet SED and several other outputs are then displayed in the GUI.
All outputs can be downloaded as ``numpy`` arrays from within the GUI.

.. _fig-gui:

.. figure:: ../_static/gui.png
   :alt: Single-Epoch Observation GUI
   :width: 100%
   :align: center

   Screenshot of the GUI of the single-epoch observation simulator.

After `installing LIFEsimMC <../installation.rst>`_, you can start the GUI by running the following command in your terminal:

.. code-block:: console

    lifesimmc-gui

or, alternatively:

.. code-block:: console

    python -m lifesimmc.gui

For a documentation of all configurable parameters, check out the :doc:`SingleEpochObservation <../source/presets/seo>` class documentation.


Python API
----------

All results offered by the GUI can also be accessed through the Python API. The following notebook illustrates how to
setup the astrophysical scene, run the single-epoch observation and plot all results.

.. include:: seo_api.ipynb
   :parser: myst_nb.docutils_
