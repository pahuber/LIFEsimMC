from itertools import product

import torch
from torch import Tensor

from lifesimmc.core.modules.base_module import BaseModule
from lifesimmc.core.resources.base_resource import BaseResource
from lifesimmc.core.resources.config_resource import ConfigResource
from lifesimmc.core.resources.image_resource import ImageResource
from lifesimmc.core.resources.spectrum_resource import SpectrumResource
from lifesimmc.util.grid import get_indices_of_maximum_of_2d_array


class AnalyticalMLEModule(BaseModule):
    def __init__(self, r_config_in: str, r_data_in: str, r_template_in: str, r_image_out: str, r_spectrum_out: str):
        self.config_in = r_config_in
        self.data_in = r_data_in
        self.template_in = r_template_in
        self.image_out = ImageResource(r_image_out)
        self.spectrum_out = SpectrumResource(r_spectrum_out)

    def _calculate_maximum_likelihood(self, data: Tensor, templates: list, config: ConfigResource) -> tuple:
        """Calculate the maximum likelihood estimate for the flux in units of photons at the position of the maximum of
        the cost function.

        :param context: The context object of the pipeline
        :return: The cost function and the optimum flux
        """
        data = data.to(config.phringe._director._device)

        cost_function = torch.zeros(
            (
                len(config.instrument.differential_outputs),
                config.simulation.grid_size,
                config.simulation.grid_size,
                len(config.instrument.wavelength_bin_centers)
            ),
            device=config.phringe._director._device
        )
        optimum_flux = torch.zeros(cost_function.shape, device=config.phringe._director._device)

        for index_x, index_y in product(range(config.simulation.grid_size), range(config.simulation.grid_size)):

            template = \
                [template for template in templates if template.x == index_x and template.y == index_y][
                    0]

            template_data = torch.einsum('ijk, ij->ijk', template.data,
                                         1 / torch.sqrt(torch.mean(template.data ** 2, axis=2)))

            matrix_c = self._get_matrix_c(data, template_data).to(config.phringe._director._device)
            matrix_b = self._get_matrix_b(data, template_data).to(config.phringe._director._device)

            for index_output in range(len(matrix_b)):
                matrix_b = torch.nan_to_num(matrix_b, 1)  # really 1?
                optimum_flux[index_output, index_x, index_y] = torch.diag(
                    torch.linalg.inv(matrix_b[index_output]) * matrix_c[index_output])

                # Set positivity constraint
                optimum_flux[index_output, index_x, index_y] = torch.where(
                    optimum_flux[index_output, index_x, index_y] >= 0,
                    optimum_flux[index_output, index_x, index_y],
                    0
                )

                # Calculate the cost function according to equation B.8
                cost_function[index_output, index_x, index_y] = (optimum_flux[index_output, index_x, index_y] *
                                                                 matrix_c[index_output])

        # Sum cost function over all wavelengths
        cost_function = torch.sum(torch.nan_to_num(cost_function, 0), axis=3)
        # cost_function[torch.isnan(cost_function)] = 0
        return cost_function, optimum_flux

    def _get_matrix_b(self, data: Tensor, template_data: Tensor) -> Tensor:
        """Calculate the matrix B according to equation B.3.

        :param data: The data
        :param template_data: The template data
        :return: The matrix B
        """
        data_variance = torch.var(data, axis=2)
        matrix_b_elements = torch.sum(template_data ** 2, axis=2) / data_variance
        matrix_b = torch.zeros(
            matrix_b_elements.shape[0],
            matrix_b_elements.shape[1],
            matrix_b_elements.shape[1],
            dtype=torch.float32
        )
        for index_output in range(len(matrix_b_elements)):
            matrix_b[index_output] = torch.diag(matrix_b_elements[index_output])

        return matrix_b

    def _get_matrix_c(self, data: Tensor, template_data: Tensor) -> Tensor:
        """Calculate the matrix C according to equation B.2.

        :param signal: The signal
        :param template_signal: The template signal
        :return: The matrix C
        """
        data_variance = torch.var(data, axis=2)
        return torch.sum(data * template_data, axis=2) / data_variance

    def _get_optimum_flux_at_cost_function_maximum(self, cost_functions, optimum_fluxes, config) -> Tensor:
        """Calculate the optimum flux at the position of the maximum of the cost function.

        :param cost_functions: The cost functions
        :param optimum_fluxes: The optimum fluxes
        :param context: The context object of the pipeline
        :return: The optimum flux at the position of the maximum of the cost function
        """
        optimum_flux_at_maximum = torch.zeros(
            (
                len(config.instrument.differential_outputs),
                len(config.instrument.wavelength_bin_centers)
            ),
            dtype=torch.float32
        )

        for index_output in range(len(optimum_flux_at_maximum)):
            index_x, index_y = get_indices_of_maximum_of_2d_array(cost_functions[index_output])
            optimum_flux_at_maximum[index_output] = optimum_fluxes[index_output, index_x, index_y]

        return optimum_flux_at_maximum

    def apply(self, resources: list[BaseResource]) -> SpectrumResource:

        print('Estimating fluxes using analytical maximum likelihood...')

        config = self.get_resource_from_name(self.config_in)
        data = self.get_resource_from_name(self.data_in).get_data()
        templates = self.get_resource_from_name(self.template_in).get_templates()

        cost_functions, optimum_fluxes = self._calculate_maximum_likelihood(data, templates, config)

        # Get the optimum flux at the position of the maximum of the cost function
        optimum_flux_at_maximum = self._get_optimum_flux_at_cost_function_maximum(
            cost_functions,
            optimum_fluxes,
            config
        )

        self.image_out.image = cost_functions
        self.spectrum_out.spectral_flux_density = optimum_flux_at_maximum

        # plt.imshow(self.image_out.image[0].cpu().numpy(), cmap='magma')
        # plt.colorbar()
        # plt.show()
        #
        # plt.plot(self.spectrum_out.spectral_flux_density[0])
        # plt.show()

        print('Done')
        return self.spectrum_out, self.image_out
