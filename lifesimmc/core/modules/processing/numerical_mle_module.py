from typing import Union

import numpy as np
import torch
from lmfit import minimize, Parameters

from lifesimmc.core.modules.base_module import BaseModule
from lifesimmc.core.resources.base_resource import BaseResource
from lifesimmc.core.resources.spectrum_resource import SpectrumResource, SpectrumResourceCollection


class NumericalMLEModule(BaseModule):
    def __init__(
            self,
            n_config_in: str,
            n_data_in: str,
            n_coordinate_in: str,
            n_spectrum_out: str,
            n_cov_in: Union[str, None] = None):
        self.n_config_in = n_config_in
        self.n_data_in = n_data_in
        self.n_coordinate_in = n_coordinate_in
        self.n_cov_in = n_cov_in
        self.n_spectrum_out = n_spectrum_out

    def apply(self, resources: list[BaseResource]) -> SpectrumResourceCollection:
        print('Performing numerical MLE...')

        r_config_in = self.get_resource_from_name(self.n_config_in)
        data_in = self.get_resource_from_name(self.n_data_in).get_data().cpu().numpy()
        r_cov_in = self.get_resource_from_name(self.n_cov_in) if self.n_cov_in is not None else None
        rc_spectrum_out = SpectrumResourceCollection(self.n_spectrum_out)

        if r_cov_in is not None:
            i_cov_sqrt = r_cov_in.i_cov_sqrt.cpu().numpy()
        else:
            diag_matrix = np.eye(data_in.shape[1])
            i_cov_sqrt = np.tile(diag_matrix, (data_in.shape[0], 1, 1))

        times = r_config_in.phringe.get_time_steps(as_numpy=True)
        wavelengths = r_config_in.phringe.get_wavelength_bin_centers(as_numpy=True)
        wavelength_bin_widths = r_config_in.phringe.get_wavelength_bin_widths(as_numpy=True)

        r_coordinates_in = self.get_resource_from_name(self.n_coordinate_in)

        params = Parameters()
        posx = np.array(r_coordinates_in.x)
        posy = np.array(r_coordinates_in.y)

        # flux_init = config.phringe.get_spectral_flux_density('Earth', as_numpy=True)
        flux_init = np.ones(wavelengths.shape) * 1e5
        hfov_max = np.sqrt(2) * r_config_in.phringe.get_field_of_view(as_numpy=True)[-1] / 2 / 14  # TODO: Check this
        # print(hfov_max)

        for i in range(len(flux_init)):
            # params.add(f'flux_{i}', value=0.9 * flux_init[i], min=0, max=1e7)
            params.add(f'flux_{i}', value=5e5, min=0, max=1e7)
        params.add('pos_x', value=posx, min=-hfov_max, max=hfov_max)
        params.add('pos_y', value=posy, min=-hfov_max, max=hfov_max)

        for i in range(len(r_config_in.instrument.differential_outputs)):
            def residual_data(params, target):
                posx = np.array(params['pos_x'].value)
                posy = np.array(params['pos_y'].value)
                flux = np.array(
                    [params[f'flux_{z}'].value for z in range(len(flux_init))]
                )
                model = (i_cov_sqrt_i @
                         r_config_in.phringe.get_template_numpy(times, wavelengths, wavelength_bin_widths, posx, posy,
                                                                flux)[
                         self.i, :, :, 0, 0])
                return model - target

            i_cov_sqrt_i = i_cov_sqrt[i]
            self.i = i
            out = minimize(residual_data, params, args=(data_in[i],), method='leastsq')
            r_cov_in = out.covar

            fluxes = np.array([out.params[f'flux_{i}'].value for i in range(len(flux_init))])
            posx = out.params['pos_x'].value
            posy = out.params['pos_y'].value

            if r_cov_in is None:
                print("Covariance matrix could not be estimated. Try different method.")
                rc_spectrum_out.collection.append(
                    SpectrumResource(
                        name='',
                        spectral_irradiance=torch.tensor(fluxes),
                        wavelength_bin_centers=torch.tensor(wavelengths),
                        wavelength_bin_widths=torch.tensor(wavelength_bin_widths),
                    )
                )
                break

            stds = np.sqrt(np.diag(r_cov_in))

            flux_err = np.array([stds[i] for i in range(len(flux_init))])
            posx_err = stds[-2]
            posy_err = stds[-1]

            rc_spectrum_out.collection.append(
                SpectrumResource(
                    name='',
                    spectral_irradiance=torch.tensor(fluxes),
                    wavelength_bin_centers=torch.tensor(wavelengths),
                    wavelength_bin_widths=torch.tensor(wavelength_bin_widths),
                    err_low=torch.tensor(flux_err),
                    err_high=torch.tensor(flux_err)
                )
            )

        print('Done')
        return rc_spectrum_out
