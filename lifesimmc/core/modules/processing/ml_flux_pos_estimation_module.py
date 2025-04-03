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
        data_variance = torch.var(data, axis=0)
        sum = torch.sum(torch.einsum('ij, ijkl->ijkl', data, template_data), axis=0)
        vector_c = torch.einsum('ijk, i->ijk', sum, 1 / data_variance)

        # Calculate matrix B according to equation B.3
        sum = torch.sum(template_data ** 2, axis=0)
        vector_b = torch.nan_to_num(torch.einsum('ijk, i->ijk', sum, 1 / data_variance), 1)

        # Create diagonal matrix B
        b_shape = vector_b.shape
        eye = torch.eye(b_shape[0], device=self.device).unsqueeze(-1).unsqueeze(-1)
        x_diag = vector_b.unsqueeze(1)
        matrix_b = eye * x_diag

        # Calculate the optimum flux according to equation B.6 and set positivity constraint
        b_perm = matrix_b.permute(2, 3, 0, 1)
        b_inv = torch.linalg.inv(b_perm).permute(2, 3, 0, 1)

        optimum_flux = torch.einsum('ijkl, jkl->ikl', b_inv, vector_c)

        # Calculate the cost function according to equation B.8
        optimum_flux = torch.where(optimum_flux >= 0, optimum_flux, 0)
        cost_function = optimum_flux * vector_c
        cost_function = torch.sum(torch.nan_to_num(cost_function, 0), axis=0)

        plt.imshow(cost_function.cpu().numpy(), cmap='magma')
        plt.colorbar()
        plt.show()

        # Get the optimum flux at the position of the maximum of the cost function
        flat_idx = cost_function.argmax()  # scalar
        row = flat_idx // cost_function.shape[1]
        col = flat_idx % cost_function.shape[1]
        optimum_flux_at_maximum = optimum_flux[:, row.item(), col.item()].cpu().numpy()  # TODO: fix this scaling

        # plt.plot(optimum_flux_at_maximum)
        # plt.show()

        # Get the coordinates of the maximum
        x_coord = grid_coordinates[0][row.item(), col.item()].cpu().numpy()
        y_coord = grid_coordinates[1][row.item(), col.item()].cpu().numpy()

        return optimum_flux_at_maximum, x_coord, y_coord

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
        # i_cov_sqrt_block = torch.block_diag(*i_cov_sqrt).cpu().numpy()
        data_in = self.get_resource_from_name(self.n_data_in).get_data()
        template_data = r_templates_in.get_data()
        grid_coordinates = r_templates_in.grid_coordinates

        # Flatten data along differential outputs and times axes
        data_in = data_in.permute(0, 2, 1)
        data_in = data_in.reshape((-1,) + data_in.shape[2:])
        template_data = template_data.permute(0, 2, 1, 3, 4)
        template_data = template_data.reshape((-1,) + template_data.shape[2:])

        # Set up parameters and initial conditions
        flux_init, posx_init, posy_init = self._get_analytical_initial_guess(data_in, template_data, grid_coordinates)
        data_in = data_in.cpu().numpy()
        hfov_max = r_config_in.phringe.get_field_of_view()[-1].cpu().numpy() / 2  # TODO: /14 Check this

        params = Parameters()

        for j in range(len(flux_init)):
            params.add(f'flux_{j}', value=flux_init[j])
        params.add('pos_x', value=posx_init, min=-hfov_max, max=hfov_max)
        params.add('pos_y', value=posy_init, min=-hfov_max, max=hfov_max)

        # Perform MLE
        def residual_data(params, target):
            posx = params['pos_x'].value
            posy = params['pos_y'].value
            flux = np.array([params[f'flux_{z}'].value for z in range(len(flux_init))])
            model = r_config_in.phringe._get_template_diff_counts(
                times,
                wavelengths,
                wavelength_bin_widths,
                flux,
                posx,
                posy
            )
            for i in range(model.shape[0]):
                model[i] = i_cov_sqrt[i] @ model[i]
            model = np.transpose(model, (0, 2, 1))
            model = model.reshape(data_in.shape)
            return model - target

        out = minimize(residual_data, params, args=(data_in,), method='leastsq')
        cov_out = out.covar

        fluxes = np.array([out.params[f'flux_{k}'].value for k in range(len(flux_init))])
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

        stds = np.sqrt(np.diag(cov_out))
        flux_err = stds[0:-2]
        posx_err = stds[-2]
        posy_err = stds[-1]

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
