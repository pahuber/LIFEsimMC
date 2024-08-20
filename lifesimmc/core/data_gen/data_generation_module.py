from phringe.api import PHRINGE

from lifesimmc.core.base_module import BaseModule
from lifesimmc.core.context import Context


class DataGenerationModule(BaseModule):
    """Class representation of the data generation module."""

    def __init__(self, name: str, gpu: int, write_to_fits: bool = True, create_copy: bool = True):
        """Constructor method."""
        self.name = name
        self.gpu = gpu
        self.write_to_fits = write_to_fits
        self.create_copy = create_copy

    def apply(self, context) -> Context:
        """Use PHRINGE to generate synthetic data.

        :param context: The context object of the pipeline
        :return: The (updated) context object
        """
        print('Generating synthetic data...')
        phringe = PHRINGE()

        phringe.run(
            config_file_path=context.config_file_path,
            simulation=context.simulation,
            observation_mode=context.observation_mode,
            instrument=context.instrument,
            scene=context.scene,
            gpu=self.gpu,
            write_fits=self.write_to_fits,
            create_copy=self.create_copy
        )

        context.data = phringe.get_data()
        context._phringe = phringe

        print('Done')
        return context
