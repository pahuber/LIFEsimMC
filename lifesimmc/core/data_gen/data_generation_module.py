from pathlib import Path

from phringe.phringe import PHRINGE

from lifesimmc.core.base_module import BaseModule
from lifesimmc.core.context import Context


class DataGenerationModule(BaseModule):
    """Class representation of the data generation module."""

    def __init__(self, gpus: tuple[int], write_to_fits: bool = True, create_copy: bool = True,
                 output_path: Path = Path(".")):
        """Constructor method."""
        self.gpus = gpus
        self.write_to_fits = write_to_fits
        self.create_copy = create_copy
        self.output_path = output_path

    def apply(self, context) -> Context:
        """Use PHRINGE to generate synthetic data.

        :param context: The context object of the pipeline
        :return: The (updated) context object
        """
        phringe = PHRINGE()

        phringe.run(
            config_file_path=context.config_file_path,
            exoplanetary_system_file_path=context.exoplanetary_system_file_path,
            settings=context.settings,
            observatory=context.observatory,
            observation=context.observation,
            scene=context.scene,
            spectrum_files=context.spectrum_files,
            gpus=self.gpus,
            output_dir=self.output_path,
            write_fits=self.write_to_fits,
            create_copy=self.create_copy
        )

        context.data = phringe.get_data()

        return context
