.. _usage:

Usage
=====

Creating a LIFEsimMC Pipeline
-----------------------------

`LIFEsimMC` features a `pipeline` architecture. Upon creating a ``Pipeline`` object, ``Module`` objects can be added to the pipeline
and executed in sequence. Information transfer between modules is handled via ``Resource`` objects, which function as input and/or
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

`LIFEsimMC` requires the specification of an ``Observation``, an ``Instrument`` and a ``Scene`` object
(see :doc:`Observation documentation <source/external/observation>`, :doc:`Instrument documentation <source/external/instrument>` or :doc:`Scene documentation <source/external/scene>`).
This is done via the ``SetupModule``. There are two ways to set up the simulation:

Option 1: Creating Objects Manually (Recommended)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The recommended way is to manually create the ``Observation``, ``Instrument`` and ``Scene`` objects.
After their creation, they can directly be given as input to the ``SetupModule``:

.. code-block:: python

    obs = Observation(...)
    inst = Instrument(...)
    scene = Scene(...)

    module = SetupModule(n_setup_out='setup', observation=obs, instrument=inst, scene=scene)
    pipeline.add_module(module)

Predefined instruments and observations are available for use with e.g. ``inst = LIFEReferenceDesign()`` or
``obs = LIFEObservation()`` (see :doc:`Tutorials <tutorials>`).

Option 2: Using a Config File
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Alternatively, `LIFEsimMC` can also be used with config files (see :doc:`Config File Tutorial <tutorials/config_file>`) and a ``Configuration`` object (see :doc:`Configuration documentation <external/configuration>`):

.. code-block:: python

    module = SetupModule(n_config_out='config', configuration=Configuration(path=Path('path/to/config.py')))
    pipeline.add_module(module)

This way all available simulation parameters can be configured in a single file, which requires a solid understanding of their interplay and is thus
only recommended for advanced users.

.. note::
    It is recommended to run `LIFEsimMC` on a GPU, as the simulation gets computationally expensive quickly and may take a substantial amount of time on CPUs.
