import torch
from phringe.util.grid import get_meshgrid

from lifesimmc.core.modules.base_module import BaseModule
from lifesimmc.core.resources.base_resource import BaseResource
from lifesimmc.core.resources.template_resource import TemplateResource


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
            fov: float,
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

    def apply(self, resources: list[BaseResource]) -> TemplateResource:
        """Generate templates for a planet at each point in the grid.

        :param resources: The resources to apply the module to
        :return: A list of template resources
        """
        print('Generating templates...')

        r_config_in = self.get_resource_from_name(self.n_config_in)
        time = r_config_in.phringe.get_time_steps()
        wavelength = r_config_in.phringe.get_wavelength_bin_centers()

        ir = r_config_in.phringe.get_instrument_response_theoretical(
            time, wavelength, self.fov, r_config_in.phringe.get_nulling_baseline()
        )

        template_counts = (
                ir
                * r_config_in.observation.detector_integration_time
                * r_config_in.phringe.get_wavelength_bin_widths()[None, :, None, None, None]
        )
        template_diff_counts = torch.zeros(
            (len(r_config_in.instrument.differential_outputs),) + template_counts.shape[1:],
            dtype=torch.float32,
            device=self.device
        )

        for i, diff_output in enumerate(r_config_in.instrument.differential_outputs):
            template_diff_counts[i] = template_counts[diff_output[0]] - template_counts[diff_output[1]]

        r_template_out = TemplateResource(
            name=self.n_template_out,
            grid_coordinates=get_meshgrid(self.fov, self.grid_size, self.device),
        )
        r_template_out.set_data(template_diff_counts)

        print('Done')
        return r_template_out
