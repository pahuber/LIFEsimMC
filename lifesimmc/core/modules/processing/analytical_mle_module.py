from itertools import product

import torch
from torch import Tensor

from lifesimmc.core.modules.base_module import BaseModule
from lifesimmc.core.resources.base_resource import BaseResource
from lifesimmc.core.resources.config_resource import ConfigResource
from lifesimmc.core.resources.coordinate_resource import CoordinateResource
from lifesimmc.core.resources.flux_resource import FluxResource, FluxResourceCollection
from lifesimmc.core.resources.image_resource import ImageResource
from lifesimmc.util.grid import get_indices_of_maximum_of_2d_array


class AnalyticalMLEModule(BaseModule):
    def \
            __init__(
            self,
            n_config_in: str,
            n_data_in: str,
            n_template_in: str,
            n_image_out: str,
            n_flux_out: str,
            n_coordinate_out: str,
            use_true_position: bool = False
    ):
        self.n_config_in = n_config_in
        self.n_data_in = n_data_in
        self.n_template_in = n_template_in
        self.n_image_out = n_image_out
        self.n_flux_out = n_flux_out
        self.n_coordinate_out = n_coordinate_out
        self.use_true_position = use_true_position

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
                [template for template in templates if template.x_index == index_x and template.y_index == index_y][
                    0]
            template_data = template.get_data().to(config.phringe._director._device)[:, :, :, 0, 0]
            # template_data = torch.einsum('ijk, ij->ijk', template.data,
            #                              1 / torch.sqrt(torch.mean(template.data ** 2, axis=2)))

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

    def _get_optimum_flux_at_cost_function_maximum(self, cost_functions, optimum_fluxes, config, templates) -> Tensor:
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

        coordinates = []

        for index_output in range(len(optimum_flux_at_maximum)):
            if self.use_true_position:
                sky_brightness_distribution = config.phringe._director._planets[
                    0].sky_brightness_distribution  # TODO: Handel multiple planets
                # Get indices of only pixel that is not zero

                # if orbital motion is modeled, just use the initial position
                if sky_brightness_distribution.ndim == 4:
                    sky_brightness_distribution = sky_brightness_distribution[0]

                index_x, index_y = torch.nonzero(sky_brightness_distribution[0], as_tuple=True)
                # index_x, index_y = index_x[0].item(), index_y[0].item()
                x_coord = config.phringe._director._planets[0].sky_coordinates[
                    0, index_x[0].item(), index_y[0].item()].cpu().numpy()
                y_coord = config.phringe._director._planets[0].sky_coordinates[
                    1, index_x[0].item(), index_y[0].item()].cpu().numpy()
                # print(index_x, index_y)
                # plt.imshow(sky_brightness_distribution[0].cpu().numpy())
                # plt.colorbar()
                # plt.show()
                # sky_brightness_distribution[0, index_x, index_y] *= 10000
                # plt.imshow(sky_brightness_distribution[0].cpu().numpy())
                # plt.colorbar()
                # plt.show()

            else:
                index_x, index_y = get_indices_of_maximum_of_2d_array(cost_functions[index_output])
                template = \
                    [template for template in templates if template.x_index == index_x and template.y_index == index_y][
                        0]
                x_coord, y_coord = template.x_coord, template.y_coord

            optimum_flux_at_maximum[index_output] = optimum_fluxes[index_output, index_x, index_y]
            coordinates.append((x_coord, y_coord))

        return optimum_flux_at_maximum, coordinates

    def apply(self, resources: list[BaseResource]) -> tuple[
        FluxResourceCollection,
        ImageResource,
        CoordinateResource
    ]:
        """Perform analytical MLE on a grid of templates to crate a cost function map/image. For each grid point
        estimate the flux and return the flux of the grid point with the maximum of the cost function.

        :param resources: The resources to apply the module to
        :return: The resource
        """
        print('Performing analytical MLE...')

        r_config_in = self.get_resource_from_name(self.n_config_in)
        data_in = self.get_resource_from_name(self.n_data_in).get_data()
        templates_in = self.get_resource_from_name(self.n_template_in).collection

        cost_functions, optimum_fluxes = self._calculate_maximum_likelihood(data_in, templates_in, r_config_in)

        # Get the optimum flux at the position of the maximum of the cost function or at the true planet position
        optimum_flux_at_maximum, coordinates = self._get_optimum_flux_at_cost_function_maximum(
            cost_functions,
            optimum_fluxes,
            r_config_in,
            templates_in
        )

        r_image_out = ImageResource(self.n_image_out)
        r_image_out.image = cost_functions

        rc_flux_out = FluxResourceCollection(
            self.n_flux_out,
            'Collection of SpectrumResources, one for each differential output'
        )

        for index_output in range(len(optimum_flux_at_maximum)):
            flux = FluxResource(
                '',
                optimum_flux_at_maximum[index_output],
                r_config_in.phringe.get_wavelength_bin_centers(as_numpy=False),
                r_config_in.phringe.get_wavelength_bin_widths(as_numpy=False)
            )
            rc_flux_out.collection.append(flux)

        # TODO: Output coordinates for each differential output
        r_coordinates_out = CoordinateResource(self.n_coordinate_out, x=coordinates[0][0], y=coordinates[0][1])

        # plt.imshow(self.image_out.image[0].cpu().numpy(), cmap='magma')
        # plt.colorbar()
        # plt.show()
        #
        # plt.plot(optimum_flux_at_maximum[0])
        # plt.show()

        print('Done')
        return rc_flux_out, r_image_out, r_coordinates_out
