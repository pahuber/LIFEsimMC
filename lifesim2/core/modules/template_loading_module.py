from pathlib import Path

from lifesim2.core.base_module import BaseModule
from lifesim2.core.context import Context
from lifesim2.io.fits_reader import FITSReader
from lifesim2.util.helpers import Template


class TemplateLoadingModule(BaseModule):
    """Class representation of the template loading module."""

    def __init__(self, template_directory: Path):
        """Constructor method."""
        self.template_directory = template_directory

    def apply(self, context) -> Context:
        """Apply the module.

        :param context: The context object of the pipelines
        :return: The (updated) context object
        """
        fits_reader = FITSReader()
        templates = []
        for file in self.template_directory.glob('*.fits'):
            template_information = file.stem.split('_')
            index_x = template_information[1]
            index_y = template_information[2]

            data = fits_reader.read(file)
            template = Template(data=data, x=int(index_x), y=int(index_y))
            templates.append(template)

        context.templates = templates
        return context
