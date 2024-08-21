from datetime import datetime
from pathlib import Path

from tqdm.contrib.itertools import product

from lifesimmc.core.modules.base_module import BaseModule
from lifesimmc.core.resources.base_resource import BaseResource
from lifesimmc.core.resources.template_resource import TemplateResource
from lifesimmc.util.helpers import Template


class TemplateGenerationModule(BaseModule):
    """Class representation of the template generation module."""

    def __init__(
            self,
            r_config_in: str,
            r_template_out: str,
            write_to_fits: bool = True,
            create_copy: bool = True
    ):
        """Constructor method."""
        self.config_in = r_config_in
        self.template_out = r_template_out
        self.write_to_fits = write_to_fits
        self.create_copy = create_copy
        self.template_out = TemplateResource(r_template_out)

    def apply(self, resources: list[BaseResource]) -> TemplateResource:
        """Generate templates for a planet at each point in the grid.
        """
        config = self.get_resource_from_name(self.config_in)

        # Generate the output directory if FITS files should be written
        if self.write_to_fits:
            template_dir = Path(datetime.now().strftime("%Y%m%d_%H%M%S.%f"))
            template_dir.mkdir(parents=True, exist_ok=True)

        templates = []

        # Swipe the planet position through every point in the grid and generate the data for each position
        print('Generating templates...')

        device = config.phringe._director._device
        time = config.phringe.get_time_steps().to(device)
        wavelength = config.phringe.get_wavelength_bin_centers().to(device)
        flux = config.phringe.get_spectral_flux_density('Earth').to(device)
        coord = [source for source in config.phringe._director._sources if source.name == 'Earth'][
            0].sky_coordinates

        for index_x, index_y in product(
                range(config.simulation.grid_size),
                range(config.simulation.grid_size)
        ):
            # Set the planet position to the current position in the grid
            # scene_template.planets[0].grid_position = (index_x, index_y)
            #

            # Get the current planet position in radians
            posx = coord[0, index_x, index_y].to(device)
            posy = coord[1, index_x, index_y].to(device)

            # Generate the data
            data = config.phringe.get_template(time, wavelength, posx, posy, flux)

            template = Template(x=index_x, y=index_y, data=data)
            templates.append(template)

        self.template_out.templates = templates

        print('Done')
        return self.template_out
