from pathlib import Path

import torch
from phringe.phringe import PHRINGE
from tqdm.contrib.itertools import product

from lifesim2.core.base_module import BaseModule
from lifesim2.core.context import Context
from lifesim2.util.helpers import Template


class TemplateGenerationModule(BaseModule):
    """Class representation of the template generation module."""

    def __init__(self, gpus: tuple[int], write_to_fits: bool = True, create_copy: bool = True,
                 output_path: Path = Path(".")):
        """Constructor method."""
        self.gpus = gpus
        self.write_to_fits = write_to_fits
        self.create_copy = create_copy
        self.output_path = output_path

    def apply(self, context) -> Context:
        """Apply the module.

        :param context: The context object of the pipelines
        :return: The (updated) context object
        """
        settings_template = context.settings.copy()
        scene_template = context.scene.copy()

        # Turn of the planet orbital motion and only use the initial position of the planets. This matters, because the
        # sky coordinates for the planets are calculated based on their distance from the star and may vary for
        # different times of the observation, if the planet has moved a lot (to rule out undersampling issues when the
        # planet would get very close to the star).
        settings_template.has_planet_orbital_motion = False

        # Turn off noise sources so the scene.get_all_sources() only returns the planets in the data generator module
        # and the intensity response is ideal
        settings_template.has_stellar_leakage = False
        settings_template.has_local_zodi_leakage = False
        settings_template.has_exozodi_leakage = False
        settings_template.has_amplitude_perturbations = False
        settings_template.has_phase_perturbations = False
        settings_template.has_polarization_perturbations = False

        # For each planet in the scene generate the template data individually, i.e. with only one planet per data set,
        # since multiple planets per template don't make a lot of sense for template matching
        all_planets = scene_template.planets
        templates = []

        # Swipe the planet position through every point in the grid and generate the data for each position
        for index_planet, index_x, index_y in product(
                range(len(all_planets)),
                range(context.settings.grid_size),
                range(context.settings.grid_size)
        ):
            # Set the planet position to the current position in the grid
            scene_template.planets = [all_planets[index_planet]]
            scene_template.planets[0].grid_position = (index_x, index_y)

            # Generate the data
            # TODO: add FITS writing
            phringe = PHRINGE()
            phringe.run(
                settings=settings_template,
                observatory=context.observatory,
                observation=context.observation,
                scene=scene_template,
                spectrum_files=context.spectrum_files,
                gpus=self.gpus,
                output_dir=self.output_path,
                write_fits=self.write_to_fits,
                create_copy=self.create_copy
            )
            data = phringe.get_data()

            # Normalize the data to unit RMS
            data = torch.einsum('ijk, ij->ijk', data, 1 / torch.sqrt(torch.mean(data ** 2, axis=2)))

            template = Template(x=index_x, y=index_y, data=data)
            templates.append(template)

        context.templates = templates
        return context
