.. _config_file:

Config File Tutorial
====================



General Structure
------------------
A config file mus contain a dictionary named ``config`` with the keys ``observation``, ``instrument``,
and ``scene``, which are dictionaries themselves. E.g.:

.. code-block:: python

    config = {
        'observation': {...},
        'instrument': {...},
        'scene': {...}
    }

The key-value pairs of these sub-dictionaries are mapped to an ``Observation``, ``Instrument``, and ``Scene`` object, respectively,
upon initializing the ``Configuration`` object as described above.

Units
-----

Quantities that require units can be specified in three different ways:

1. As a string with an `AstroPy`-parseable unit suffix (e.g. ``'1.0 cm'``)
2. As an `AstroPy` quantity (e.g. ``1.0 * u.cm``)
3. As a float in SI units (e.g. ``0.01``; not recommended)

Example Config File
-------------------

The following show an example config file. For a documentation and meaning of all key words please refer to the
:doc:`Observation documentation <../source/external/observation>`, :doc:`Instrument documentation <../source/external/instrument>`,
:doc:`Scene documentation <../source/external/scene>`, :doc:`Star documentation <../source/external/all_sources/star>`,
:doc:`Planet documentation <../source/external/all_sources/planet>`, :doc:`ExoZodi documentation <../source/external/all_sources/exozodi>`,
and :doc:`LocalZodi documentation <../source/external/all_sources/local_zodi>`.


.. literalinclude::  ../_static/config.py
   :language: python


Usage in LIFEsimMC
------------------
A config file is used to setup the simulation and must be passed to the ``SetupModule`` via a ``Configuration`` object:

.. code-block:: python

    module = SetupModule(n_config_out='conf', configuration=Configuration(path=Path(r"path/to/config.py")))

Alternatively, if the ``config`` dictionary is not located in a separate file, but in the same file as the ``SetupModule``,
it can be passed directly to the ``Configuration`` object:

.. code-block:: python

    module = SetupModule(n_config_out='conf', configuration=Configuration(config_dict=config))

This creates a ``ConfigResource`` (see :doc:`ConfigResource documentation <../source/core/resources/config_resource>`)
named ``conf`` here that is added to the pipeline and can be accessed from all successive modules.

