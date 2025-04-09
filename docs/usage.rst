.. _usage:

Usage
=====

Creating a LIFEsimMC Pipeline
-----------------------------

`LIFEsimMC` features a `pipeline` architecture. Upon creating a ``Pipeline`` object, `modules` can be added to the pipeline
and executed in sequence. Information transfer between modules is handled via `resources`, which function as input and/or
output to the modules. The following code snippet gives a quick overview of the main workflow of `LIFEsimMC`.

.. code-block:: python

    # Create the pipeline
    pipeline = Pipeline()

    # Create and add a module with an output resource named 'ex_a' and another argument that is not a resource
    module = ExampleAModule(n_example_a_resource_out='ex_a', example_argument=42)
    pipeline.add_module(module)

    # Create and add a second module with an input ('ex_a') and output resource named 'ex_b'
    module = ExampleBModule(n_example_a_resource_in='ex_a', n_example_b_resource_out='ex_b')
    pipeline.add_module(module)

    # Run the pipeline
    pipeline.run()

    # Get the output resource named 'ex_b' of the second module
    ex_b = pipeline.get_resource('ex_b')


Specifying User Input
---------------------

`LIFEsimMC` requires the specification of an `observation`, the `instrument` parameters and the astrophysical `scene`.
This is done via the ``SetupModule``. There are two ways to set up the simulation:

Option 1: Using a Config File
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The simplest way to use `LIFEsimMC` is by using a config file (see :doc:`Config File Tutorial <tutorials/config_file>`) and a ``Configuration`` object (see :doc:`Configuration documentation <external/configuration>`):

.. code-block:: python

    module = SetupModule(n_config_out='config', configuration=Configuration(path=Path('path/to/config.py')))
    pipeline.add_module(module)

This module also produces an output ``ConfigResource`` named ``config``, which contains the ``Observation``, ``Instrument`` and ``Scene`` objects.


Option 2: Manually Creating Objects
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Alternatively, one can manually create the ``Observation``, ``Instrument`` and ``Scene`` objects (see :doc:`Observation documentation <external/observation>`, :doc:`Instrument documentation <external/instrument>` or :doc:`Scene documentation <external/scene>`):

.. code-block:: python

    obs = Observation(...)
    inst = Instrument(...)
    scene = Scene(...)

    module = SetupModule(n_config_out='config', observation=obs, instrument=inst, scene=scene))
    pipeline.add_module(module)

This may be required for more advanced use cases, e.g. when looping through different instrument configurations.

.. note::
    It is recommended to run `LIFEsimMC` on a GPU, as the simulation gets computationally expensive quickly and may take a substantial amount of time on CPUs.
