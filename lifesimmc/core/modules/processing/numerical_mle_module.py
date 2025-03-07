from typing import Union

import numpy as np
import torch
from lmfit import minimize, Parameters

from lifesimmc.core.modules.base_module import BaseModule
from lifesimmc.core.resources.base_resource import BaseResource
from lifesimmc.core.resources.coordinate_resource import CoordinateResource
from lifesimmc.core.resources.flux_resource import FluxResource, FluxResourceCollection


class NumericalMLEModule(BaseModule):
    def __init__(
            self,
            n_config_in: str,
            n_data_in: str,
            n_flux_out: str,
            n_coordinate_in: str = None,
            n_coordinate_out: str = None,
            n_flux_in: str = None,
            n_cov_in: Union[str, None] = None):
        self.n_config_in = n_config_in
        self.n_data_in = n_data_in
        self.n_flux_in = n_flux_in
        self.n_coordinate_in = n_coordinate_in
        self.n_cov_in = n_cov_in
        self.n_flux_out = n_flux_out
        self.n_coordinate_out = n_coordinate_out

    def apply(self, resources: list[BaseResource]) -> Union[FluxResourceCollection, CoordinateResource]:
        print('Performing numerical MLE...')

        r_config_in = self.get_resource_from_name(self.n_config_in)
        data_in = self.get_resource_from_name(self.n_data_in).get_data().cpu().numpy()
        rc_flux_in = self.get_resource_from_name(self.n_flux_in) if self.n_flux_in is not None else None
        r_cov_in = self.get_resource_from_name(self.n_cov_in) if self.n_cov_in is not None else None
        r_coordinates_in = self.get_resource_from_name(
            self.n_coordinate_in) if self.n_coordinate_in is not None else None
        rc_flux_out = FluxResourceCollection(
            self.n_flux_out,
            'Collection of SpectrumResources, one for each differential output'
        )
        r_coordinate_out = CoordinateResource(self.n_coordinate_out)

        times = r_config_in.phringe.get_time_steps(as_numpy=True)
        wavelengths = r_config_in.phringe.get_wavelength_bin_centers(as_numpy=True)
        wavelength_bin_widths = r_config_in.phringe.get_wavelength_bin_widths(as_numpy=True)

        # Handle case where no covariance matrix is given, i.e. no whitening is performed
        if r_cov_in is not None:
            i_cov_sqrt = r_cov_in.i_cov_sqrt.cpu().numpy()
        else:
            diag_matrix = np.eye(data_in.shape[1])
            i_cov_sqrt = np.tile(diag_matrix, (data_in.shape[0], 1, 1))

        # flux_init = config.phringe.get_spectral_flux_density('Earth', as_numpy=True)
        # flux_init = np.ones(wavelengths.shape) * 1e5
        if rc_flux_in is not None:
            flux_init = rc_flux_in.collection[
                0].spectral_irradiance.cpu().numpy().tolist()  # TODO: specify which spectrum to use
        else:
            flux_init = np.ones(wavelengths.shape)

        hfov_max = np.sqrt(2) * r_config_in.phringe.get_field_of_view(as_numpy=True)[-1] / 2  # TODO: /14 Check this
        # print(hfov_max)

        # Set up parameters and initial conditions
        params = Parameters()
        posx = np.array(r_coordinates_in.x) if r_coordinates_in is not None else np.array(hfov_max / 2)
        posy = np.array(r_coordinates_in.y) if r_coordinates_in is not None else np.array(hfov_max / 2)

        for i in range(len(flux_init)):
            # params.add(f'flux_{i}', value=0.9 * flux_init[i], min=0, max=1e7)
            params.add(f'flux_{i}', value=flux_init[i])  # , min=0, max=1e7)
        params.add('pos_x', value=posx, min=-hfov_max, max=hfov_max)
        params.add('pos_y', value=posy, min=-hfov_max, max=hfov_max)

        # Perform MLE for each differential output
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
                # return (model[:, model.shape[1] - target.shape[1]:] - target)
                return model - target

            i_cov_sqrt_i = i_cov_sqrt[i]
            self.i = i
            # var_i = np.var(data_in[i])
            out = minimize(residual_data, params, args=(data_in[i],), method='leastsq')
            cov_out = out.covar

            fluxes = np.array([out.params[f'flux_{i}'].value for i in range(len(flux_init))])
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

            flux_err = np.array([stds[i] for i in range(len(flux_init))])
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
