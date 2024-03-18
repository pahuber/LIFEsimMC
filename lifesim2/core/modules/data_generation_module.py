from pathlib import Path

from phringe.phringe import PHRINGE

from lifesim2.core.base_module import BaseModule
from lifesim2.core.context import Context


class DataGenerationModule(BaseModule):
    """Class representation of the data generation module."""

    def __init__(self, write_to_fits: bool = True, create_copy: bool = True, output_path: Path = Path(".")):
        """Constructor method."""
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
            settings=context.settings,
            observatory=context.observatory,
            observation=context.observation,
            scene=context.scene,
            spectrum_files=context.spectrum_files,
            output_dir=self.output_path,
            write_fits=self.write_to_fits,
            create_copy=self.create_copy
        )

        context.data = phringe.get_data()

        return context
