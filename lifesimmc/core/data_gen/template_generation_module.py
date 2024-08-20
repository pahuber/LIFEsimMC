from datetime import datetime
from pathlib import Path

from tqdm.contrib.itertools import product

from lifesimmc.core.base_module import BaseModule
from lifesimmc.util.helpers import Template


class TemplateGenerationModule(BaseModule):
    """Class representation of the template generation module."""

    def __init__(
            self,
            name: str,
            config_module: str,
            planet_name: str,
            write_to_fits: bool = True,
            create_copy: bool = True
    ):
        """Constructor method."""
        super().__init__(name)
        self.name = name
        self.config_module = config_module
        self.planet_name = planet_name
        self.write_to_fits = write_to_fits
        self.create_copy = create_copy
        self.templates = None

    def apply(self):
        """Generate templates for the planet with name planet_name at each point in the grid.
        """
        config_module = self.get_module_from_name(self.config_module)

        # Generate the output directory if FITS files should be written
        if self.write_to_fits:
            template_dir = Path(datetime.now().strftime("%Y%m%d_%H%M%S.%f"))
            template_dir.mkdir(parents=True, exist_ok=True)

        # simulation_template = config_module.simulation.copy()
        # scene_template = config_module.scene.copy()
        #
        # # Remove all planets frm the scene except the one with the name planet_name
        # scene_template.planets = [planet for planet in scene_template.planets if planet.name == self.planet_name]
        #
        # # Turn off the planet orbital motion and only use the initial position of the planets. This matters, because the
        # # sky coordinates for the planets are calculated based on their distance from the star and may vary for
        # # different times of the observation, if the planet has moved a lot (to rule out undersampling issues when the
        # # planet would get very close to the star).
        # simulation_template.has_planet_orbital_motion = False
        #
        # # Turn off noise sources so the scene.get_all_sources() only returns the planets in the data generator module
        # # and the intensity response is ideal
        # simulation_template.has_planet_signal = True
        # simulation_template.has_stellar_leakage = False
        # simulation_template.has_local_zodi_leakage = False
        # simulation_template.has_exozodi_leakage = False
        # simulation_template.has_amplitude_perturbations = False
        # simulation_template.has_phase_perturbations = False
        # simulation_template.has_polarization_perturbations = False

        templates = []

        # Swipe the planet position through every point in the grid and generate the data for each position
        print('Generating templates...')
        device = config_module.phringe._director._device
        time = config_module.phringe.get_time_steps().to(device)
        wavelength = config_module.phringe.get_wavelength_bin_centers().to(device)
        flux = config_module.phringe.get_spectral_flux_density('Earth').to(device)
        coord = [source for source in config_module.phringe._director._sources if source.name == 'Earth'][
            0].sky_coordinates

        for index_x, index_y in product(
                range(config_module.simulation.grid_size),
                range(config_module.simulation.grid_size)
        ):
            # Set the planet position to the current position in the grid
            # scene_template.planets[0].grid_position = (index_x, index_y)
            #

            # Get the current planet position in radians
            posx = coord[0, index_x, index_y].to(device)
            posy = coord[1, index_x, index_y].to(device)

            # Generate the data
            data = config_module.phringe.get_template(time, wavelength, posx, posy, flux)

            template = Template(x=index_x, y=index_y, data=data, planet_name=self.planet_name)
            templates.append(template)

        self.templates = templates

        print('Done')
