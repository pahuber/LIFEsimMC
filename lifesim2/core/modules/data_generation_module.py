from pathlib import Path

from sygn.core.entities.observation import Observation
from sygn.core.entities.observatory.observatory import Observatory
from sygn.core.entities.scene import Scene
from sygn.core.entities.settings import Settings
from sygn.core.processing.data_generator import DataGenerator
from sygn.io.fits_writer import FITSWriter
from sygn.io.yaml_reader import YAMLReader

from lifesim2.core.base_module import BaseModule
from lifesim2.core.context import Context


class DataGenerationModule(BaseModule):
    """Class representation of the data generation module."""

    def __init__(
            self,
            config_file_path: Path,
            exoplanetary_system_file_path: Path,
            spectrum_file_path: Path = None,
            write_to_fits: bool = True,
            output_path: Path = Path("."),
    ):
        """Constructor method."""
        self.config_file_path = config_file_path
        self.exoplanetary_system_file_path = exoplanetary_system_file_path
        self.spectrum_file_path = spectrum_file_path
        self.write_to_fits = write_to_fits
        self.output_path = output_path

    def apply(self, context) -> Context:
        """Apply the module.

        :param context: The context object of the pipeline
        :return: The (updated) context object
        """

        config_dict = YAMLReader().read(self.config_file_path)
        system_dict = YAMLReader().read(self.exoplanetary_system_file_path)
        # TODO: add planet spectrum
        # planet_spectrum = TXTReader().read(self.spectrum_file_path) if self.spectrum_file_path else None

        context.settings = Settings(**config_dict["settings"])
        context.observation = Observation(**config_dict["observation"])
        context.observatory = Observatory(**config_dict["observatory"])
        context.scene = Scene(**system_dict)

        context.settings.prepare(context.observation, context.observatory)
        context.observation.prepare()
        context.observatory.prepare(
            context.settings, context.observation, context.scene
        )
        context.scene.prepare(context.settings, context.observatory, None)

        data_generator = DataGenerator(
            settings=context.settings,
            observation=context.observation,
            observatory=context.observatory,
            scene=context.scene,
        )
        context.data = data_generator.run()

        if self.write_to_fits:
            fits_writer = FITSWriter().write(context.data, self.output_path)

        return context
