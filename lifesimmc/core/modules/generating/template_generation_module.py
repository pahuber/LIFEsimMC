from datetime import datetime
from pathlib import Path

import numpy as np
import torch
from tqdm.contrib.itertools import product

from lifesimmc.core.modules.base_module import BaseModule
from lifesimmc.core.resources.base_resource import BaseResource
from lifesimmc.core.resources.template_resource import TemplateResource, TemplateResourceCollection


class TemplateGenerationModule(BaseModule):
    """Class representation of the template generation module to generate templates of planets with unit flux.

    :param n_config_in: The name of the input configuration resource
    :param n_template_out: The name of the output template resource collection
    :param fov: The field of view for which to generate the templates in radians
    :param write_to_fits: Whether the generated templates should be written to FITS files
    :param create_copy: Whether a copy of the input configuration should be created
    """

    def __init__(
            self,
            n_config_in: str,
            n_template_out: str,
            fov: float = None,
            write_to_fits: bool = True,
            create_copy: bool = True
    ):
        """Constructor method.

        :param n_config_in: The name of the input configuration resource
        :param n_template_out: The name of the output template resource collection
        :param fov: The field of view for which to generate the templates in radians
        :param write_to_fits: Whether the generated templates should be written to FITS files
        :param create_copy: Whether a copy of the input configuration should be created
        """
        self.n_config_in = n_config_in
        self.n_template_out = n_template_out
        self.fov = fov
        self.write_to_fits = write_to_fits
        self.create_copy = create_copy

    def apply(self, resources: list[BaseResource]) -> TemplateResourceCollection:
        """Generate templates for a planet at each point in the grid.

        :param resources: The resources to apply the module to
        :return: A list of template resources
        """
        r_config_in = self.get_resource_from_name(self.n_config_in)

        # Generate the output directory if FITS files should be written
        if self.write_to_fits:
            template_dir = Path(datetime.now().strftime("%Y%m%d_%H%M%S.%f"))
            template_dir.mkdir(parents=True, exist_ok=True)

        rc_template_out = TemplateResourceCollection(self.n_template_out)

        # Swipe the planet position through every point in the grid and generate the data for each position
        print('Generating templates...')

        device = r_config_in.phringe._director._device
        time = r_config_in.phringe.get_time_steps(as_numpy=False).to(device)
        wavelength = r_config_in.phringe.get_wavelength_bin_centers(as_numpy=False).to(device)
        wavelength_bin_width = r_config_in.phringe.get_wavelength_bin_widths(as_numpy=False).to(device)
        flux = torch.ones(wavelength.shape, device=device)

        hfov_max = r_config_in.phringe.get_field_of_view(as_numpy=False)[
                       0] / 2 if self.fov is None else self.fov / 2
        coord = np.linspace(-hfov_max, hfov_max, r_config_in.simulation.grid_size)

        for (ix, x), (iy, y) in product(enumerate(coord), enumerate(coord), total=len(coord) ** 2):
            # Set the planet position to the current position in the grid

            posx = torch.tensor(x, device=device)
            posy = torch.tensor(y, device=device)

            # Generate the data
            data = r_config_in.phringe.get_template_torch(time, wavelength, wavelength_bin_width, posx, posy, flux)

            r_template_out = TemplateResource(
                name='',
                x_coord=x,
                y_coord=y,
                x_index=ix,
                y_index=iy,
            )
            r_template_out.set_data(data)
            rc_template_out.collection.append(r_template_out)

        print('Done')
        return rc_template_out
