from typing import Union

import numpy as np
import torch
from lmfit import minimize, Parameters
from matplotlib import pyplot as plt

from lifesimmc.core.modules.base_module import BaseModule
from lifesimmc.core.resources.base_resource import BaseResource
from lifesimmc.core.resources.spectrum_resource import SpectrumResource
from lifesimmc.util.helpers import Spectrum


class NumericalMLEModule(BaseModule):
    def __init__(self, r_config_in: str, r_data_in: str, r_spectrum_out: str, r_cov_in: Union[str, None] = None):
        self.config_in = r_config_in
        self.data_in = r_data_in
        self.cov_in = r_cov_in if r_cov_in is not None else None
        self.spectrum_out = SpectrumResource(r_spectrum_out)

    def apply(self, resources: list[BaseResource]) -> SpectrumResource:
        print('Performing numerical MLE...')

        config = self.get_resource_from_name(self.config_in)
        data = self.get_resource_from_name(self.data_in).get_data().cpu().numpy()
        cov = self.get_resource_from_name(self.cov_in) if self.cov_in is not None else None
        if cov is not None:
            icov2 = cov.icov2
        else:
            icov2 = torch.diag(torch.ones(data.shape[1], device=config.phringe._director._device)).unsqueeze(0).repeat(
                data.shape[0], 1, 1)

        times = config.phringe.get_time_steps(as_numpy=False)
        wavelengths = config.phringe.get_wavelength_bin_centers(as_numpy=False)
        wavelength_bin_widths = config.phringe.get_wavelength_bin_widths(as_numpy=False)

        params = Parameters()
        posx = -3.45e-7
        posy = 3.45e-7

        flux_init = config.phringe.get_spectral_flux_density('Earth', as_numpy=False)
        hfov_max = np.sqrt(2) * config.phringe.get_field_of_view(as_numpy=False)[-1] / 2

        for i in range(len(flux_init)):
            params.add(f'flux_{i}', value=1 * flux_init[i].cpu().numpy(), min=0, max=1e7)
            # params.add(f'flux_{i}', value=5e5, min=0, max=1e9)
        params.add('pos_x', value=posx, min=-hfov_max, max=hfov_max)
        params.add('pos_y', value=posy, min=-hfov_max, max=hfov_max)

        icov2 = icov2.cpu().numpy()

        for i in range(len(config.instrument.differential_outputs)):
            def residual_data(params, target):
                posx = params['pos_x'].value.to(config.phringe._director._device)
                posy = params['pos_y'].value.to(config.phringe._director._device)
                flux = torch.tensor(
                    [params[f'flux_{z}'].value for z in range(len(flux_init))],
                    device=config.phringe._director._device
                )
                model = (icov2i @
                         config.phringe.get_template_torch(times, wavelengths, wavelength_bin_widths, posx, posy, flux)[
                         self.i, :, :, 0, 0].cpu().numpy())
                return model - target

            icov2i = icov2[i]
            self.i = i
            out = minimize(residual_data, params, args=(data[i],), method='leastsq')
            cov = out.covar
            # print(out.params)
            # plt.plot(out.residual)
            # plt.show()
            # print(out.params)
            # print(out.uvars)
            # print(out.errorbars)

            fluxes = [out.params[f'flux_{z}'].value for z in range(len(flux_init))]
            plt.plot(range(len(fluxes[:-1])), fluxes[:-1], color='black',
                     label='Fit')
            plt.show()

            if cov is None:
                print("Covariance matrix could not be estimated. Try different method.")
                break

            stds = np.sqrt(np.diag(cov))

            fluxes = [out.params[f'flux_{i}'].value for i in range(len(flux_init))]
            posx = out.params['pos_x'].value
            posy = out.params['pos_y'].value

            flux_err = [stds[i] for i in range(len(flux_init))]
            posx_err = stds[-2]
            posy_err = stds[-1]

            self.spectrum_out.spectra.append(
                Spectrum(fluxes, flux_err, flux_err, config.phringe.get_wavelength_bin_centers(as_numpy=False),
                         config.phringe.get_wavelength_bin_widths(as_numpy=False)))

            # best_fit = get_model(xdata, out.params['pos_x'].value, out.params['pos_y'].value,
            #                      *[out.params[f'flux_{i}'].value for i in range(len(flux_real))])
            # Plot best flux and snr in a 2x1 grid
            # flux_init = flux_init.cpu().numpy()
            # plt.subplot(2, 1, 1)
            # plt.plot(flux_init[:-1], linestyle='dashed', label='True', color='black')
            # plt.errorbar(range(len(fluxes[:-1])), fluxes[:-1], yerr=flux_err[:-1], fmt='o', color='black',
            #              label='Fit')
            # plt.plot(range(len(fluxes[:-1])), fluxes[:-1], fmt='o', color='black',
            #          label='Fit')
            # plt.fill_between(
            #     range(len(flux_init[:-1])),
            #     np.array(flux_init[:-1]) - np.array(flux_err[:-1]),
            #     np.array(flux_init[:-1]) + np.array(flux_err[:-1]),
            #     color="k", alpha=0.2, label='1-$\sigma$'
            # )
            # plt.legend()
            # plt.ylim(0, 1e6)
            #
            # # snr
            # bins = config.phringe._director._wavelength_bin_widths.cpu().numpy()[:-1]
            # snr = fluxes / np.array(flux_err)
            # snr_total = np.round(np.sqrt(np.sum(snr ** 2)), 1)
            #
            # plt.subplot(2, 1, 2)
            # plt.step(bins, snr[:-1], where='mid')
            # plt.title(f'SNR: {snr_total}')
            #
            # plt.tight_layout()
            # plt.show()

        print('Done')
        return self.spectrum_out
