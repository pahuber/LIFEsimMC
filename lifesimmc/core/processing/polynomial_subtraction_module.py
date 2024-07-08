from itertools import product

import numpy as np
import torch
from matplotlib import pyplot as plt
from tqdm import tqdm

from lifesimmc.core.base_module import BaseModule
from lifesimmc.core.context import Context


class PolynomialSubtractionModule(BaseModule):
    """Class representation of the polynomial subtraction module."""

    def __init__(self):
        """Constructor method."""
        pass

    def apply(self, context) -> Context:
        """Apply the module.

        :param context: The context object of the pipeline
        :return: The (updated) context object
        """
        # Normalize data to unit RMS
        # context.data = torch.einsum('ijk, ij->ijk', context.data, 1 / torch.sqrt(torch.mean(context.data ** 2, axis=2)))

        # Convert to numpy array
        context.data = context.data.numpy()

        # Plot data before fit subtraction
        # plt.imshow(context.data[0])
        # plt.title('Data Before Fit Subtraction')
        # plt.colorbar()
        # plt.savefig('Data_Before_Fit_Subtraction.png', dpi=300)
        # plt.show()

        # Plot template before fit subtraction
        # template = \
        #     [template for template in context.templates if template.x == 8 and template.y == 4][0]
        # plt.imshow(template.data[0, :, 0:100])
        # plt.title('Template Before Fit Subtraction')
        # plt.colorbar()
        # plt.savefig('Data_Before_Fit_Subtraction.png', dpi=300)
        # plt.show()

        # Subtract polynomial fit from data and templates
        print('Subtracting polynomial fits...')
        for index_time in tqdm(range(len(context.data[0][0]))):
            for index_output in range(len(context.data)):
                data_spectral_column = np.nan_to_num(context.data[index_output][:, index_time])

                # Calculate polynomial fit
                coefficients = np.polyfit(
                    range(len(data_spectral_column)),
                    data_spectral_column,
                    3
                )
                fitted_function = np.poly1d(coefficients)

                # Subtract polynomial fit from data
                context.data[index_output][:, index_time] -= fitted_function(range(len(data_spectral_column)))

                # Loop through grid of templates
                for index_x, index_y in product(range(context.settings.grid_size), range(context.settings.grid_size)):
                    # Get template corresponding to grid position
                    template = \
                        [template for template in context.templates if template.x == index_x and template.y == index_y][
                            0]

                    # Get index of template in context.templates list
                    template_index = context.templates.index(template)

                    # Plot template before fit subtraction
                    if index_x == 8 and index_y == 4 and index_time == 0:
                        plt.plot(context.templates[template_index].data[index_output][:, index_time],
                                 context.observatory.wavelength_bin_centers, label='Template Pre-Subtraction')
                        plt.ylabel('Wavelength ($\mu$m)')
                        plt.xlabel('Photon Counts')
                        plt.title('Template Before Subtraction')
                        # plt.legend()
                        plt.show()

                    # Subtract polynomial fit from template
                    context.templates[template_index].data[index_output][:,
                    index_time] -= fitted_function(range(len(data_spectral_column)))

                    # Plot data and polynomial fit
                    if index_x == 8 and index_y == 4 and index_time == 0:
                        # Plot data and polynomial fit
                        plt.plot(data_spectral_column, context.observatory.wavelength_bin_centers, label='Data',
                                 color='#37B3FF', linewidth=2)
                        plt.plot(fitted_function(range(len(data_spectral_column))),
                                 context.observatory.wavelength_bin_centers, label='Fit', color='#FFC000', linewidth=2)
                        plt.ylabel('Wavelength ($\mu$m)')
                        plt.xlabel('Photon Counts')
                        # plt.title('Full Data and Polynomial Fit')
                        plt.legend()
                        plt.axis('off')
                        plt.savefig('Full_Data_Polynomial_Fit.svg', dpi=300, transparent=True)
                        plt.show()

                        # Plot template after fit subtraction
                        plt.plot(context.templates[template_index].data[index_output][:, index_time],
                                 context.observatory.wavelength_bin_centers, label='Template Post-Subtraction')
                        plt.ylabel('Wavelength ($\mu$m)')
                        plt.xlabel('Photon Counts')
                        plt.title('Template After Subtraction')
                        # plt.legend()
                        plt.show()

                        # Plot template data after fit subtraction
                        # plt.imshow(template.data[0, :, 0:100])
                        # plt.title('Template After Fit Subtraction')
                        # plt.colorbar()
                        # plt.savefig('Data_After_Fit_Subtraction.png', dpi=300)
                        # plt.show()

        context.data = torch.tensor(context.data)

        # Plot full data after fit
        # data = context.data
        # plt.imshow(context.data[0])
        # plt.title('Data After Fit Subtraction')
        # plt.colorbar()
        # plt.savefig('Data_After_Fit_Subtraction.png', dpi=300)
        # plt.show()

        # Plot full template after fit
        # template = \
        #     [template for template in context.templates if template.x == 8 and template.y == 4][0]
        # plt.imshow(template.data[0, :, 0:100])
        # plt.title('Template After Fit Subtraction')
        # plt.colorbar()
        # plt.savefig('Data_After_Fit_Subtraction.png', dpi=300)
        # plt.show()

        print('Done')
        return context
