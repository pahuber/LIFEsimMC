from itertools import product
from typing import Tuple

import scipy
import torch
from matplotlib import pyplot as plt
from torch import Tensor, tensor

from lifesimmc.core.base_module import BaseModule
from lifesimmc.util.grid import get_indices_of_maximum_of_2d_array
from lifesimmc.util.helpers import Extraction


class MLMExtractionModule(BaseModule):
    """Class representation of the MLM extraction module."""

    def _calculate_maximum_likelihood(self, data, context) -> Tuple:
        """Calculate the maximum likelihood estimate for the flux in units of photons at the position of the maximum of
        the cost function.

        :param context: The context object of the pipeline
        :return: The cost function and the optimum flux
        """

        cost_function = torch.zeros(
            (
                context.observatory.beam_combination_scheme.number_of_differential_outputs,
                context.settings.grid_size,
                context.settings.grid_size,
                len(context.observatory.wavelength_bin_centers)
            )
        )
        optimum_flux = torch.zeros(cost_function.shape)

        for index_x, index_y in product(range(context.settings.grid_size), range(context.settings.grid_size)):

            template = [template for template in context.templates if template.x == index_x and template.y == index_y][
                0]

            template_data = torch.einsum('ijk, ij->ijk', template.data,
                                         1 / torch.sqrt(torch.mean(template.data ** 2, axis=2)))

            matrix_c = self._get_matrix_c(data, template_data)
            matrix_b = self._get_matrix_b(data, template_data)

            for index_output in range(len(matrix_b)):
                matrix_b = torch.nan_to_num(matrix_b, 1)
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
        cost_function[torch.isnan(cost_function)] = 0
        return cost_function, optimum_flux

    def _get_fluxes_uncertainties(self, cost_functions, cost_functions_white, optimum_fluxes_white,
                                  context) -> Tensor:
        """Return the uncertainties on the extracted fluxes by calculating the standard deviation of the extracted
        fluxes at positions around the center at a radius of the maximum cost function.

        :param cost_functions: The cost functions
        :param cost_functions_white: The whitened cost functions
        :param optimum_fluxes_white: The whitened optimum fluxes
        :param context: The context object of the pipeline
        :return: The uncertainties on the extracted fluxes
        """
        for index_output in range(len(cost_functions_white)):
            # Get extracted flux at positions around center at radius of maximum cost function
            height, width = cost_functions_white[index_output, :, :].shape
            index_max_x, index_max_y = get_indices_of_maximum_of_2d_array(cost_functions[index_output])

            # Create mask for circle around maximum of cost function
            center = (width // 2, height // 2)
            radius = torch.sqrt((index_max_x - width // 2) ** 2 + (index_max_y - width // 2) ** 2)

            width_linspace = torch.linspace(torch.asarray(0), torch.asarray(width - 1), width)
            height_linspace = torch.linspace(torch.asarray(0), torch.asarray(height - 1), height)
            y, x = torch.meshgrid(height_linspace, width_linspace)
            # y, x = torch.ogrid[:height, :width]
            mask = ((x - center[0]) ** 2 + (y - center[1]) ** 2 <= radius ** 2 + 0.5) & \
                   ((x - center[0]) ** 2 + (y - center[1]) ** 2 >= (radius - 1) ** 2 + 0.5)

            # remove planet position from mask
            # plt.imshow(mask)
            # plt.show()

            #################
            # get array of cost function values for each pixel in the mask for each output index
            cost_function_values_in_mask = torch.zeros(
                (context.observatory.beam_combination_scheme.number_of_differential_outputs, mask.sum()),
                dtype=torch.float32
            )
            for index_output in range(context.observatory.beam_combination_scheme.number_of_differential_outputs):
                cost_function_values_in_mask[index_output] = cost_functions_white[index_output, :, :][mask]

            a = 0

            ################

            masked_fluxes_white_flattened = torch.einsum(
                'ijk, ij -> ijk',
                optimum_fluxes_white[index_output, :, :],
                mask
            ).reshape(context.settings.grid_size ** 2, -1)

            uncertainties = torch.zeros(masked_fluxes_white_flattened.shape[1], dtype=torch.float32)

            for index in range(masked_fluxes_white_flattened.shape[1]):
                # Remove zero values from array, since they are not part of the mask and would change the standard
                # deviation
                non_zero_values = [el for el in masked_fluxes_white_flattened[:, index] if el > 0]
                uncertainties[index] = torch.std(torch.asarray(non_zero_values))
        return 0, cost_function_values_in_mask

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

    def _get_optimum_flux_at_cost_function_maximum(self, cost_functions, optimum_fluxes, context) -> Tensor:
        """Calculate the optimum flux at the position of the maximum of the cost function.

        :param cost_functions: The cost functions
        :param optimum_fluxes: The optimum fluxes
        :param context: The context object of the pipeline
        :return: The optimum flux at the position of the maximum of the cost function
        """
        optimum_flux_at_maximum = torch.zeros(
            (
                context.observatory.beam_combination_scheme.number_of_differential_outputs,
                len(context.observatory.wavelength_bin_centers)
            ),
            dtype=torch.float32
        )

        for index_output in range(len(optimum_flux_at_maximum)):
            index_x, index_y = get_indices_of_maximum_of_2d_array(cost_functions[index_output])
            optimum_flux_at_maximum[index_output] = optimum_fluxes[index_output, index_x, index_y]

        return optimum_flux_at_maximum

    def _get_whitened_data(self, optimum_fluxes, cost_functions, context) -> Tensor:
        """Return the whitened signal, i.e. the original signal with the most likely planet signal substracted.

        :param optimum_fluxes: The optimum fluxes
        :param cost_functions: The cost functions
        :param context: The context object of the pipeline
        :return: The whitened signal
        """
        for index_output in range(context.observatory.beam_combination_scheme.number_of_differential_outputs):
            index_x, index_y = get_indices_of_maximum_of_2d_array(cost_functions[index_output])
            signal_white = context.data.detach().clone()
            template = [template for template in context.templates if template.x == index_x and template.y == index_y][
                0]
            signal_white -= torch.einsum(
                'ij, ijk->ijk',
                optimum_fluxes[:, index_x, index_y],
                template.data
            )
        return signal_white

    def apply(self, context):
        """Apply the module.

        :param context: The context object of the pipelines
        :return: The (updated) context object
        """
        cost_functions, optimum_fluxes = self._calculate_maximum_likelihood(context.data, context)
        optimum_flux_at_maximum = self._get_optimum_flux_at_cost_function_maximum(
            cost_functions,
            optimum_fluxes,
            context
        )

        # Get the whitened data and the uncertainties on the extracted flux
        data_white = self._get_whitened_data(optimum_fluxes, cost_functions, context)
        cost_functions_white, optimum_fluxes_white = self._calculate_maximum_likelihood(data_white, context)

        # TODO: what about data whitening with poly subtraction?
        optimum_fluxes_uncertainties, cost_function_values_in_mask = self._get_fluxes_uncertainties(
            cost_functions,
            cost_functions,  # cost_functions_white,
            optimum_fluxes_white,
            context
        )

        # plt.hist(cost_function_values_in_mask[0])
        # plt.show()

        ##########
        mu_y_hat = torch.max(cost_functions)
        mu_x_hat = torch.mean(cost_function_values_in_mask)
        n = len(cost_function_values_in_mask[0])
        sigma_x_hat = torch.sqrt(torch.sum((cost_function_values_in_mask - mu_x_hat) ** 2) / (n - 1))
        t = (mu_y_hat - mu_x_hat) / (sigma_x_hat * torch.sqrt(tensor(1 + 1 / n)))

        # calc fpf from t using t-statistics
        fpf = scipy.stats.t.sf(t, n - 1)

        # convert fpf to sigma values
        sigma_fpf = scipy.stats.norm.ppf(1 - fpf)

        # print(mu_y_hat)
        # print(cost_function_values_in_mask)
        # print(mu_x_hat)
        # print(n)
        # print(sigma_x_hat)
        # print(t)
        # print(fpf)
        # print(sigma_fpf)
        ####

        context.extractions.append(
            Extraction(
                optimum_flux_at_maximum,
                optimum_fluxes_uncertainties,
                context.observatory.wavelength_bin_centers,
                cost_functions
            )
        )

        return context
