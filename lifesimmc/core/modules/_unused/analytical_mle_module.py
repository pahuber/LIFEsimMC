import torch
from matplotlib import pyplot as plt

from lifesimmc.core.modules.base_module import BaseModule
from lifesimmc.core.resources.base_resource import BaseResource
from lifesimmc.core.resources.coordinate_resource import CoordinateResource
from lifesimmc.core.resources.flux_resource import FluxResourceCollection
from lifesimmc.core.resources.image_resource import ImageResource


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
        data = self.get_resource_from_name(self.n_data_in).get_data()
        r_templates_in = self.get_resource_from_name(self.n_template_in)
        template_data = r_templates_in.get_data()

        # Normalize template data with their variance along axis 2
        # template_data = template_data / torch.var(template_data, axis=2, keepdim=True) ** 0.5

        # Calculate matrix C according to equation B.2
        data_variance = torch.var(data, axis=2)
        sum = torch.sum(torch.einsum('ijk, ijklm->ijklm', data, template_data), axis=2)
        vector_c = torch.einsum('ijkl, ij->ijkl', sum, 1 / data_variance)

        # Calculate matrix B according to equation B.3
        sum = torch.sum(template_data ** 2, axis=2)
        vector_b = torch.nan_to_num(torch.einsum('ijkl, ij->ijkl', sum, 1 / data_variance), 1)

        b_shape = vector_b.shape
        matrix_b = torch.zeros(
            b_shape[0],
            b_shape[1],
            b_shape[1],
            b_shape[2],
            b_shape[3],
            device=self.device,
            dtype=torch.float32
        )
        idx = torch.arange(b_shape[1], device=self.device)
        matrix_b[0, idx, idx, :, :] = vector_b[0]  # TODO: fix for multiple outputs

        # Calculate the optimum flux according to equation B.6 and set positivity constraint
        b_perm = matrix_b.permute(0, 3, 4, 1, 2)
        b_inv = torch.linalg.inv(b_perm).permute(0, 3, 4, 1, 2)

        optimum_flux = torch.einsum('ijklm, iklm->ijlm', b_inv, vector_c)

        # Calculate the cost function according to equation B.8
        optimum_flux = torch.where(optimum_flux >= 0, optimum_flux, 0)
        cost_function = optimum_flux * vector_c
        cost_function = torch.sum(torch.nan_to_num(cost_function, 0), axis=1)

        plt.imshow(cost_function[0].cpu().numpy(), cmap='magma')
        plt.colorbar()
        plt.show()

        # Get indices of the maximum of the cost function
        flat = cost_function.view(cost_function.shape[0], -1)
        max_indices = flat.argmax(dim=1)
        max_rows = max_indices // cost_function.shape[2]  # row index
        max_cols = max_indices % cost_function.shape[2]  # column index
        max_coords = torch.stack((max_rows, max_cols), dim=1)
        print(r_templates_in.grid_coordinates[0][max_coords[0][0]][max_coords[0][1]])
        print(r_templates_in.grid_coordinates[1][max_coords[0][0]][max_coords[0][1]])

        # Get the optimum flux at the position of the maximum of the cost function
        rows, cols = max_coords[:, 0], max_coords[:, 1]
        batch_indices = torch.arange(optimum_flux.shape[0])
        optimum_flux_at_maximum = optimum_flux[batch_indices, :, rows, cols]

        wavelengths = r_config_in.phringe.get_wavelength_bin_centers().cpu().numpy()
        spec = r_config_in.phringe.get_source_spectrum('Earth')
        plt.plot(wavelengths, spec.cpu().numpy())
        plt.plot(wavelengths, optimum_flux_at_maximum[0].cpu().numpy())
        plt.show()

        plt.plot(wavelengths, spec.cpu().numpy() / optimum_flux_at_maximum[0].cpu().numpy())
        plt.plot(wavelengths, spec.cpu().numpy())
        plt.show()

        #
        # r_image_out = ImageResource(self.n_image_out)
        # r_image_out.image = cost_functions
        #
        # rc_flux_out = FluxResourceCollection(
        #     self.n_flux_out,
        #     'Collection of SpectrumResources, one for each differential output'
        # )
        #
        # for index_output in range(len(optimum_flux_at_maximum)):
        #     flux = FluxResource(
        #         '',
        #         optimum_flux_at_maximum[index_output],
        #         r_config_in.phringe.get_wavelength_bin_centers(as_numpy=False),
        #         r_config_in.phringe.get_wavelength_bin_widths(as_numpy=False)
        #     )
        #     rc_flux_out.collection.append(flux)
        #
        # # TODO: Output coordinates for each differential output
        # r_coordinates_out = CoordinateResource(self.n_coordinate_out, x=coordinates[0][0], y=coordinates[0][1])

        # plt.imshow(self.image_out.image[0].cpu().numpy(), cmap='magma')
        # plt.colorbar()
        # plt.show()
        #
        # plt.plot(optimum_flux_at_maximum[0])
        # plt.show()

        print('Done')
        # return rc_flux_out, r_image_out, r_coordinates_out
