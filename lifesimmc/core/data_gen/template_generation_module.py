from datetime import datetime
from pathlib import Path

from phringe.phringe_ui import PHRINGE
from tqdm.contrib.itertools import product

from lifesimmc.core.base_module import BaseModule
from lifesimmc.core.context import Context
from lifesimmc.util.helpers import Template


class TemplateGenerationModule(BaseModule):
    """Class representation of the template generation module."""

    def __init__(
            self,
            gpu: int,
            write_to_fits: bool = True,
            create_copy: bool = True
    ):
        """Constructor method."""
        self.gpu = gpu
        self.write_to_fits = write_to_fits
        self.create_copy = create_copy

    def apply(self, context) -> Context:
        """Apply the module.

        :param context: The context object of the pipelines
        :return: The (updated) context object
        """
        # Generate the output directory if FITS files should be written
        if self.write_to_fits:
            template_dir = Path(datetime.now().strftime("%Y%m%d_%H%M%S.%f"))
            template_dir.mkdir(parents=True, exist_ok=True)

        simulation_template = context.simulation.copy()
        scene_template = context.scene.copy()

        # Only make templates for single planet systems
        if len(scene_template.planets) > 1:
            raise ValueError("Templates can only be created for single planet systems.")

        # Turn of the planet orbital motion and only use the initial position of the planets. This matters, because the
        # sky coordinates for the planets are calculated based on their distance from the star and may vary for
        # different times of the observation, if the planet has moved a lot (to rule out undersampling issues when the
        # planet would get very close to the star).
        simulation_template.has_planet_orbital_motion = False

        # Turn off noise sources so the scene.get_all_sources() only returns the planets in the data generator module
        # and the intensity response is ideal
        simulation_template.has_planet_signal = True
        simulation_template.has_stellar_leakage = False
        simulation_template.has_local_zodi_leakage = False
        simulation_template.has_exozodi_leakage = False
        simulation_template.has_amplitude_perturbations = False
        simulation_template.has_phase_perturbations = False
        simulation_template.has_polarization_perturbations = False

        templates = []

        # Swipe the planet position through every point in the grid and generate the data for each position
        print('Generating templates...')
        for index_x, index_y in product(range(context.simulation.grid_size), range(context.simulation.grid_size)):
            # Set the planet position to the current position in the grid
            scene_template.planets[0].grid_position = (index_x, index_y)

            # Generate the data
            phringe = PHRINGE()
            phringe.run(
                config_file_path=context.config_file_path,
                simulation=context.simulation,
                observation_mode=context.observation_mode,
                instrument=context.instrument,
                scene=context.scene,
                gpu=self.gpu,
                fits_suffix=f'_{index_x}_{index_y}',
                write_fits=self.write_to_fits,
                create_copy=False,
                create_directory=False,
                normalize=True
            )
            data = phringe.get_data()

            template = Template(x=index_x, y=index_y, data=data)
            templates.append(template)

        context.templates = templates
        print('Done')
        return context
