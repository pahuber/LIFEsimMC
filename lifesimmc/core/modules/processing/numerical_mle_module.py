from typing import Union

import numpy as np
import torch
from lmfit import minimize, Parameters
from matplotlib import pyplot as plt

from lifesimmc.core.modules.base_module import BaseModule
from lifesimmc.core.resources.base_resource import BaseResource
from lifesimmc.core.resources.coordinate_resource import CoordinateResource
from lifesimmc.core.resources.flux_resource import FluxResource, FluxResourceCollection


class MLFluxPosEstimationModule(BaseModule):
    def __init__(
            self,
            n_config_in: str,
            n_data_in: str,
            n_flux_out: str,
            n_cov_in: Union[str, None],
            n_template_in: str = None,
            n_coordinate_out: str = None,
    ):
        self.n_config_in = n_config_in
        self.n_data_in = n_data_in
        self.n_template_in = n_template_in
        self.n_cov_in = n_cov_in
        self.n_flux_out = n_flux_out
        self.n_coordinate_out = n_coordinate_out

    def _get_analytical_initial_guess(self, data, template_data, grid_coordinates):

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

        # Get the optimum flux at the position of the maximum of the cost function
        rows, cols = max_coords[:, 0], max_coords[:, 1]
        batch_indices = torch.arange(optimum_flux.shape[0])
        optimum_flux_at_maximum = optimum_flux[batch_indices, :, rows, cols].cpu().numpy()

        plt.plot(optimum_flux_at_maximum[0])
        plt.show()

        return optimum_flux_at_maximum, grid_coordinates[0][max_coords[0][0]][max_coords[0][1]].cpu().numpy(), \
            grid_coordinates[1][max_coords[0][0]][max_coords[0][1]].cpu().numpy()

    def apply(self, resources: list[BaseResource]) -> Union[FluxResourceCollection, CoordinateResource]:
        print('Performing numerical MLE...')

        r_config_in = self.get_resource_from_name(self.n_config_in)
        r_templates_in = self.get_resource_from_name(self.n_template_in)
        r_cov_in = self.get_resource_from_name(self.n_cov_in) if self.n_cov_in is not None else None
        rc_flux_out = FluxResourceCollection(
            self.n_flux_out,
            'Collection of SpectrumResources, one for each differential output'
        )
        r_coordinate_out = CoordinateResource(self.n_coordinate_out)

        times = r_config_in.phringe.get_time_steps().cpu().numpy()
        wavelengths = r_config_in.phringe.get_wavelength_bin_centers().cpu().numpy()
        wavelength_bin_widths = r_config_in.phringe.get_wavelength_bin_widths().cpu().numpy()
        i_cov_sqrt = r_cov_in.i_cov_sqrt.cpu().numpy()
        data_in = self.get_resource_from_name(self.n_data_in).get_data()
        template_data = r_templates_in.get_data()
        grid_coordinates = r_templates_in.grid_coordinates

        # Set up parameters and initial conditions
        flux_init, posx_init, posy_init = self._get_analytical_initial_guess(data_in, template_data, grid_coordinates)
        data_in = data_in.cpu().numpy()
        hfov_max = r_config_in.phringe.get_field_of_view()[-1].cpu().numpy() / 2  # TODO: /14 Check this

        params = Parameters()

        params.add('pos_x', value=posx_init, min=-hfov_max, max=hfov_max)
        params.add('pos_y', value=posy_init, min=-hfov_max, max=hfov_max)

        # Perform MLE for each differential output
        for i in range(len(r_config_in.instrument.differential_outputs)):
            i_cov_sqrt_i = i_cov_sqrt[i]

            for j in range(len(flux_init[i])):
                params.add(f'flux_{j}', value=flux_init[i, j])  # , min=0, max=1e7)

            def residual_data(params, target):
                posx = params['pos_x'].value
                posy = params['pos_y'].value
                flux = np.array([params[f'flux_{z}'].value for z in range(len(flux_init[i]))])
                model = r_config_in.phringe._get_template_diff_counts(
                    times,
                    wavelengths,
                    wavelength_bin_widths,
                    flux,
                    posx,
                    posy
                )[i]
                model = i_cov_sqrt_i @ model
                return model - target

            out = minimize(residual_data, params, args=(data_in[i],), method='leastsq')
            cov_out = out.covar

            fluxes = np.array([out.params[f'flux_{k}'].value for k in range(len(flux_init[i]))])
            posx = out.params['pos_x'].value
            posy = out.params['pos_y'].value

            if r_cov_in is None:
                print("Covariance matrix could not be estimated. Try different method.")
                rc_flux_out.collection.append(
                    FluxResource(
                        name='',
                        spectral_irradiance=torch.tensor(fluxes),
                        wavelength_bin_centers=torch.tensor(wavelengths),
                        wavelength_bin_widths=torch.tensor(wavelength_bin_widths),
                    )
                )
                break

            stds = np.sqrt(np.diag(cov_out))
            ############
            # cov_out = cov_out[:10, :10]
            # plt.imshow(cov_out)
            # plt.colorbar()
            # plt.show()
            # mean = np.zeros(cov_out.shape[0])
            # data_fake = np.random.multivariate_normal(mean, cov_out, 1000)
            # df = pd.DataFrame(data_fake)
            # sns.pairplot(df)
            # plt.show()
            # std_dev = np.sqrt(np.diag(cov_out))
            #
            # # Step 2: Create the correlation matrix by dividing each element by the product of corresponding standard deviations
            # correlation_matrix = cov_out / np.outer(std_dev, std_dev)
            #
            # # Step 3: Fill diagonal with 1s since correlation of a variable with itself is 1
            # np.fill_diagonal(correlation_matrix, 1)
            #
            # plt.imshow(correlation_matrix)
            # plt.colorbar()
            # plt.show()

            ############

            posx_err = stds[0]
            posy_err = stds[1]
            flux_err = stds[2:]

            rc_flux_out.collection.append(
                FluxResource(
                    name='',
                    spectral_irradiance=torch.tensor(fluxes),
                    wavelength_bin_centers=torch.tensor(wavelengths),
                    wavelength_bin_widths=torch.tensor(wavelength_bin_widths),
                    err_low=torch.tensor(flux_err),
                    err_high=torch.tensor(flux_err),
                    cov=cov_out
                )
            )

            # update this for all outputs
            r_coordinate_out.x = posx
            r_coordinate_out.y = posy
            r_coordinate_out.x_err_low = posx_err
            r_coordinate_out.x_err_high = posx_err
            r_coordinate_out.y_err_low = posy_err
            r_coordinate_out.y_err_high = posy_err

        print('Done')
        return rc_flux_out, r_coordinate_out
