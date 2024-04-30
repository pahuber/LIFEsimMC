from itertools import product

import numpy as np
import torch
from tqdm import tqdm

from lifesim2.core.base_module import BaseModule
from lifesim2.core.context import Context


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
        # context.data = torch.einsum('ijk, ij->ijk', context.data, 1 / torch.sqrt(torch.mean(context.data ** 2, axis=2)))
        context.data = context.data.numpy()
        # plt.imshow(context.data[0])
        # plt.title('Data Before Fit Subtraction')
        # plt.colorbar()
        # plt.savefig('Data_Before_Fit_Subtraction.png', dpi=300)
        # plt.show()
        #
        # template = \
        #     [template for template in context.templates if template.x == 8 and template.y == 4][0]
        # plt.imshow(template.data[0])
        # plt.title('Template Before Fit Subtraction')
        # plt.colorbar()
        # plt.savefig('Data_Before_Fit_Subtraction.png', dpi=300)
        # plt.show()

        for index_time in tqdm(range(len(context.data[0][0]))):
            for index_output in range(len(context.data)):
                data_spectral_column = np.nan_to_num(context.data[index_output][:, index_time])

                coefficients = np.polyfit(
                    range(len(data_spectral_column)),
                    data_spectral_column,
                    3
                )
                fitted_function = np.poly1d(coefficients)

                context.data[index_output][:, index_time] -= fitted_function(range(len(data_spectral_column)))

                for index_x, index_y in product(range(context.settings.grid_size), range(context.settings.grid_size)):
                    # Get template corresponding to grid position
                    template = \
                        [template for template in context.templates if template.x == index_x and template.y == index_y][
                            0]

                    # Get index of template in context.templates list
                    template_index = context.templates.index(template)

                    context.templates[template_index].data[index_output][:, index_time] -= fitted_function(
                        range(len(data_spectral_column)))

        context.data = torch.tensor(context.data)
        # data = context.data
        # plt.imshow(context.data[0])
        # plt.title('Data After Fit Subtraction')
        # plt.colorbar()
        # plt.savefig('Data_After_Fit_Subtraction.png', dpi=300)
        # plt.show()
        #
        # template = \
        #     [template for template in context.templates if template.x == 8 and template.y == 4][0]
        # plt.imshow(template.data[0])
        # plt.title('Template After Fit Subtraction')
        # plt.colorbar()
        # plt.savefig('Data_After_Fit_Subtraction.png', dpi=300)
        # plt.show()

        return context

        ##########
