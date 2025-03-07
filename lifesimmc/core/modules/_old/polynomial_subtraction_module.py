from typing import Union

import numpy as np
import torch
from tqdm import tqdm

from lifesimmc.core.modules.base_module import BaseModule
from lifesimmc.core.resources.base_resource import BaseResource
from lifesimmc.core.resources.data_resource import DataResource
from lifesimmc.core.resources.template_resource import TemplateResourceCollection, TemplateResource


class PolynomialSubtractionModule(BaseModule):
    """Class representation of the polynomial subtraction module."""

    def __init__(
            self,
            n_cov_in: str,
            n_config_in: str,
            n_data_in: str = None,
            n_template_in: str = None,
            n_data_out: str = None,
            n_template_out: str = None,
            overwrite: bool = False
    ):
        """Constructor method.

        :param n_cov_in: The name of the covariance matrix resource
        :param n_data_in: The name of the data resource
        :param n_template_in: The name of the template resource
        :param n_config_in: The name of the configuration resource
        :param n_data_out: The name of the output data resource
        :param n_template_out: The name of the output template resource
        :param overwrite: Whether to overwrite/delete the existing resources
        """
        self.n_cov_in = n_cov_in
        self.n_data_in = n_data_in
        self.n_template_in = n_template_in
        self.n_config_in = n_config_in
        self.n_data_out = n_data_out
        self.n_template_out = n_template_out
        self.overwrite = overwrite

    def apply(self, resources: list[BaseResource]) -> Union[None, BaseResource, tuple]:
        """Apply the module.

        :param context: The context object of the pipeline
        :return: The (updated) context object
        """
        print('Subtracting polynomial fits...')

        r_config_in = self.get_resource_from_name(self.n_config_in)
        data_in = self.get_resource_from_name(self.n_data_in).get_data() if self.n_data_in is not None else None
        templates_in = self.get_resource_from_name(
            self.n_template_in).collection if self.n_template_in is not None else None
        r_data_out = DataResource(self.n_data_out) if self.n_data_out is not None else None
        rc_template_out = TemplateResourceCollection(
            self.n_template_out,
            'Collection of TemplateResources, one for each point in the grid'
        ) if self.n_template_out is not None else None

        # Plot data before fit subtraction
        # plt.imshow(context.data[0])
        # plt.title('Data Before Fit Subtraction')
        # plt.colorbar()
        # plt.savefig('Data_Before_Fit_Subtraction.png', dpi=300)
        # plt.show()

        # # Plot template before fit subtraction
        # template = \
        #     [template for template in context.templates if template.x == 8 and template.y == 4][0]
        # plt.imshow(template.data[0, :, 0:100])
        # plt.title('Template Before Fit Subtraction')
        # plt.colorbar()
        # plt.savefig('Data_Before_Fit_Subtraction.png', dpi=300)
        # plt.show()

        # # Subtract polynomial fit from data and templates
        # print('Subtracting polynomial fits...')
        # context.templates_subtracted = copy.deepcopy(context.templates)
        #
        # # #############
        # plt.figure()
        # ax = plt.gca()
        # im = ax.imshow(context.data[0], cmap='Greys')
        # plt.ylabel('Spectral Channel', size=8)
        # plt.xlabel('Time (d)', size=8)
        # divider = make_axes_locatable(ax)
        # cax = divider.append_axes("right", size="5%", pad=0.05)
        # cb = plt.colorbar(im, cax=cax)
        # cb.set_label(label='Normalized Counts', size=8)
        # cb.ax.tick_params(labelsize=6)
        # plt.savefig('data_before.svg', dpi=500, bbox_inches='tight')
        # plt.show()
        # #############

        polynomial_fits = np.zeros(data_in.shape)

        for index_time in tqdm(range(len(data_in[0][0]))):
            for index_output in range(len(data_in)):
                data_spectral_column = np.nan_to_num(data_in[index_output][:, index_time])

                # Calculate polynomial fit
                coefficients = np.polyfit(
                    range(len(data_spectral_column)),
                    data_spectral_column,
                    3
                )
                fitted_function = np.poly1d(coefficients)

                # Subtract polynomial fit from data
                data_in[index_output][:, index_time] -= fitted_function(range(len(data_spectral_column)))
                polynomial_fits[index_output][:, index_time] = fitted_function(range(len(data_spectral_column)))

        r_data_out.set_data(data_in)

        for it in tqdm(range(len(templates_in))):
            template = templates_in[it]
            template_data = template.get_data().cpu().numpy()
            template_data -= polynomial_fits[:, :, :, None, None]

            rc_template_out.collection.append(
                TemplateResource(
                    name='',
                    x_coord=template.x_coord,
                    y_coord=template.y_coord,
                    x_index=template.x_index,
                    y_index=template.y_index,
                    _data=torch.tensor(template_data, device=r_config_in.phringe._director._device)
                )
            )

            # if self.overwrite:
            #     templates_in[it] = None

        print('Done')
        if self.n_data_out is not None and self.n_template_out is not None:
            return r_data_out, rc_template_out
        elif self.n_data_out is not None:
            return r_data_out
        elif self.n_template_out is not None:
            return rc_template_out
        else:
            return None

            # # Loop through grid of templates
            # for index_x, index_y in product(range(r_config_in.simulation.grid_size),
            #                                 range(r_config_in.simulation.grid_size)):
            #     # Get template corresponding to grid position
            #     template = \
            #         [template for template in context.templates if template.x == index_x and template.y == index_y][
            #             0]
            #
            #     # Get index of template in context.templates list
            #     template_index = context.templates.index(template)
            #
            #     # Plot template before fit subtraction
            #     # if index_x == 8 and index_y == 4 and index_time == 0:
            #     #     plt.plot(context.templates[template_index].data[index_output][:, index_time],
            #     #              context.observatory.wavelength_bin_centers, label='Template Pre-Subtraction')
            #     #     plt.ylabel('Wavelength ($\mu$m)')
            #     #     plt.xlabel('Photon Counts')
            #     #     plt.title('Template Before Subtraction')
            #     #     # plt.legend()
            #     #     plt.show()
            #
            #     # Subtract polynomial fit from template
            #     context.templates_subtracted[template_index].data[index_output][:,
            #     index_time] -= fitted_function(range(len(data_spectral_column)))
            # TODO: do we really want zero here?

            # # Plot data and polynomial fit
            # if index_x == 8 and index_y == 4 and index_time == 0:
            #     # Plot data and polynomial fit
            #     plt.plot(data_spectral_column, context.instrument.wavelength_bin_centers, label='Data',
            #              color='#37B3FF', linewidth=2)
            #     plt.plot(fitted_function(range(len(data_spectral_column))),
            #              context.instrument.wavelength_bin_centers, label='Fit', color='#FFC000', linewidth=2)
            #     plt.ylabel('Wavelength ($\mu$m)')
            #     plt.xlabel('Photon Counts')
            #     # plt.title('Full Data and Polynomial Fit')
            #     plt.legend()
            #     plt.axis('off')
            #     plt.savefig('Full_Data_Polynomial_Fit.svg', dpi=300, transparent=True)
            #     plt.show()
            #
            #     # Plot template after fit subtraction
            # plt.plot(context.templates[template_index].data[index_output][:, index_time],
            #          context.observatory.wavelength_bin_centers, label='Template Post-Subtraction')
            # plt.ylabel('Wavelength ($\mu$m)')
            # plt.xlabel('Photon Counts')
            # plt.title('Template After Subtraction')
            # # plt.legend()
            # plt.show()

            # Plot template data after fit subtraction
            # plt.imshow(template.data[0, :, 0:100])
            # plt.title('Template After Fit Subtraction')
            # plt.colorbar()
            # # plt.savefig('Data_After_Fit_Subtraction.png', dpi=300)
            # plt.show()
        #
        # context.data = torch.tensor(context.data)
        # context.polyfits = torch.tensor(polynomial_fits)

        # template = \
        #     [template for template in context.templates if template.x == 8 and template.y == 4][0]
        # plt.imshow(template.data[0, :, 0:100])
        # plt.title('Template After Fit Subtraction')
        # plt.colorbar()
        # # plt.savefig('Data_Before_Fit_Subtraction.png', dpi=300)
        # plt.show()

        # ###########
        #
        # plt.figure()
        # ax = plt.gca()
        # im = ax.imshow(context.data[0], cmap='Greys')
        # context.templates_subtracted = copy.deepcopy(context.templates)
        # plt.ylabel('Spectral Channel', size=8)
        # plt.xlabel('Time (d)', size=8)
        # divider = make_axes_locatable(ax)
        # cax = divider.append_axes("right", size="5%", pad=0.05)
        # cb = plt.colorbar(im, cax=cax)
        # cb.set_label(label='Counts', size=8)
        # cb.ax.tick_params(labelsize=6)
        # plt.savefig('data_after.svg', dpi=500, bbox_inches='tight')
        # plt.show()
        # ############

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

        # print('Done')
        # return context
