import sys

import emcee
import numpy as np
import torch
from matplotlib import pyplot as plt

from lifesimmc.core.resources.base_resource import BaseResource
from lifesimmc.core.resources.coordinate_resource import CoordinateResource
from lifesimmc.core.resources.spectrum_resource import SpectrumResource, SpectrumResourceCollection

sys.path.append("/home/huberph/lifesimmc")
from lifesimmc.core.modules.base_module import BaseModule


class MCMCModule(BaseModule):
    """Module to estimate the flux using a parametric maximum likelihood estimation"""

    def __init__(
            self,
            n_config_in: str,
            n_data_in: str,
            n_coordinate_in: str,
            n_spectrum_in: str,
            n_coordinate_out: str,
            n_spectrum_out: str,
            walkers_multiplier: int = 4,
            steps: int = 100,
            n_cov_in: str = None):
        """Constructor method."""
        self.n_config_in = n_config_in
        self.n_data_in = n_data_in
        self.n_coordinate_in = n_coordinate_in
        self.n_spectrum_in = n_spectrum_in
        self.n_coordinate_out = n_coordinate_out
        self.n_spectrum_out = n_spectrum_out
        self.n_walkers_multiplier = walkers_multiplier
        self.n_steps = steps
        self.n_cov_in = n_cov_in

    def get_scaled_blackbody_spectrum(self, wavelengths, temperature: float, radius: float) -> np.ndarray:
        """Get a scaled blackbody spectrum.

        :param temperature: The temperature of the blackbody
        :param radius: The radius of the blackbody
        :return: The scaled blackbody spectrum
        """
        solid_angle = np.pi * (radius / self.context.scene.star.distance) ** 2
        from scipy.constants import c, h, k
        return 2 * h * c ** 2 / wavelengths ** 5 / (
                np.exp((h * c / (
                        k * wavelengths * temperature))) - 1) / c * wavelengths / h * solid_angle

    def apply(self, resources: list[BaseResource]) -> tuple:
        """Apply module
        """
        print('Estimating flux using MCMC...')

        r_config_in = self.get_resource_from_name(self.n_config_in)
        data_in = self.get_resource_from_name(self.n_data_in).get_data().cpu().numpy().astype(np.float64)
        r_cov_in = self.get_resource_from_name(self.n_cov_in) if self.n_cov_in is not None else None
        rc_spectrum_in = self.get_resource_from_name(self.n_spectrum_in)
        rc_spectrum_out = SpectrumResourceCollection(
            self.n_spectrum_out,
            'Collection of SpectrumResources, one for each differential output'
        )
        r_coordinates_out = CoordinateResource(self.n_coordinate_out)

        if r_cov_in is not None:
            i_cov_sqrt = r_cov_in.i_cov_sqrt
        else:
            i_cov_sqrt = torch.diag(
                torch.ones(data_in.shape[1], device=r_config_in.phringe._director._device)).unsqueeze(0).repeat(
                data_in.shape[0], 1, 1)

        def get_model(time, x_pos, y_pos, *flux):
            """Return the data model for all wavelengths."""

            # dit = self.dit
            # wl_bin = self.wl_bin
            # # eta_t = self.eta_t
            # model = np.zeros((len(self.wavelengths), len(time)))
            flux = torch.tensor(flux).to(r_config_in.phringe._director._device)
            x_pos = torch.tensor(x_pos).to(r_config_in.phringe._director._device)
            y_pos = torch.tensor(y_pos).to(r_config_in.phringe._director._device)
            model = icov2i @ \
                    r_config_in.phringe.get_template_torch(time, self.wavelengths, self.wavelength_bin_widths, x_pos,
                                                           y_pos,
                                                           flux).cpu().numpy()[
                    self.i, :, :, 0, 0]
            # for i, wavelength in enumerate(self.wavelengths):
            #     self.wavelength = wavelength
            #     model[i] = get_model_single(time, flux[i], x_pos, y_pos) * wl_bin[i] * dit
            # if context.icov2 is not None:
            #     return context.icov2 @ model
            return model

        def lnlike(theta, x, y, yerr):
            posx, posy, *flux = theta
            model = get_model(x, posx, posy, *flux)
            # inv_sigma2 = 1.0 / (yerr ** 2 + model ** 2 * np.exp(2 * lnf))
            # return -0.5 * (np.sum((y - model) ** 2 * inv_sigma2 - np.log(inv_sigma2)))

            return -0.5 * np.sum(np.einsum('ij, i -> ij', (y - model) ** 2, 1 / yerr ** 2))

            # return -0.5 * np.sum(((y - model) ** 2 / yerr ** 2))

        def lnprior(theta):
            posx, posy, *flux = theta
            hfov = np.max(self.fovs) / 2
            # TODO: add smoothness prior
            if 0 <= np.asarray(flux)[:-1].all() < np.inf and -hfov < posx < hfov and -hfov < posy < hfov:
                # if shape is similar to blackbody
                # if (context.bb_min < flux).all() and (flux < context.bb_max).all():
                return 0.0
                # # calc derivative of flux and of a blacbody spectrum
                # diff_flux = np.diff(flux)
                # diff_bb = np.diff(self.get_scaled_blackbody_spectrum(self.wavelengths, 300, 7e6))
                # # check that sign of derivative is the same
                # if np.sign(diff_flux).all() == np.sign(diff_bb).all():
                #     return 0.0
            return -np.inf

        def lnprob(theta, x, y, yerr):
            lp = lnprior(theta)
            if not np.isfinite(lp):
                return -np.inf
            return lp + lnlike(theta, x, y, yerr)

        # Definitions
        #########################################################################
        # Working

        time = r_config_in.phringe.get_time_steps(as_numpy=False)
        self.wavelengths = r_config_in.phringe.get_wavelength_bin_centers(as_numpy=False)
        self.wavelength_bin_widths = r_config_in.phringe.get_wavelength_bin_widths(as_numpy=False)
        self.fovs = r_config_in.phringe.get_field_of_view(as_numpy=True)
        i_cov_sqrt = i_cov_sqrt.cpu().numpy()

        # Define MCMC
        flux_init = rc_spectrum_in.collection[
            0].spectral_irradiance.cpu().numpy().tolist()  # TODO: specify which spectrum to use

        r_coordinates_in = self.get_resource_from_name(self.n_coordinate_in)

        initial_guess = [r_coordinates_in.x, r_coordinates_in.y]
        initial_guess.extend(flux_init)
        ndim = len(initial_guess)
        nwalkers = self.n_walkers_multiplier * ndim
        nsteps = self.n_steps

        spectra_flux_densities = []
        err_lows = []
        err_highs = []

        for p in range(len(r_config_in.instrument.differential_outputs)):
            self.i = p
            icov2i = i_cov_sqrt[p]
            data_in = data_in[p]
            # Define data uncertainty
            # yerr = np.ones(len(data)) * 1.5 * np.max(abs(get_model(time, *initial_guess)))
            yerr = np.ones(len(data_in)) * 0.1 * np.max(abs(data_in))
            # yerr = 0.8 * (data - get_model(time, *initial_guess))
            # print(yerr.shape)
            # yerr = 0.1 * data

            # Initialize walkers
            pos = np.zeros((nwalkers, ndim))
            pos_posx = initial_guess[0] + 1e-10 * np.random.randn(nwalkers)
            pos_posy = initial_guess[1] + 1e-10 * np.random.randn(nwalkers)
            # pos_a = initial_guess[2] + 1e-4 * np.random.randn(nwalkers)
            pos[:, 0] = pos_posx
            pos[:, 1] = pos_posy
            # pos[:, 2] = pos_a
            for i in range(2, len(initial_guess)):
                pos_flux = initial_guess[i] + 1e3 * np.random.randn(nwalkers)
                pos[:, i] = pos_flux
            # pos = [np.array(initial_guess) + 1e-8 * np.random.randn(ndim) for i in range(nwalkers)]

            # Plot initial guess and data in two imshow subplots that are on top of each other
            # plt.subplot(2, 1, 1)
            # plt.imshow(get_model(time, *initial_guess), label='Initial guess')
            # plt.colorbar()
            # plt.title('Initial guess')
            # plt.subplot(2, 1, 2)
            # plt.imshow(data_in, label='Data')
            # plt.colorbar()
            # plt.title('Data')
            # plt.savefig('init.pdf')
            # plt.show()
            # plt.close()

            # Run MCMC
            sampler = emcee.EnsembleSampler(nwalkers, ndim, lnprob, args=(time, data_in, yerr))
            sampler.run_mcmc(pos, nsteps, progress=True)

            # Working
            ########################################################################################
            # Analysis

            # Acceptance rate
            acceptance_rate = np.mean(sampler.acceptance_fraction)
            print(f"Acceptance rate: {acceptance_rate}")

            # Plot chains
            samples = sampler.get_chain()
            fig, axes = plt.subplots(ndim, figsize=(10, 7), sharex=True)
            labels = ["flux", "pos x", "pos y"]
            for i in range(ndim):
                ax = axes[i]
                ax.plot(samples[:, :, i], "k", alpha=0.3)
                ax.set_xlim(0, len(samples))
                # ax.set_ylabel(labels[i])
                # ax.yaxis.set_label_coords(-0.1, 0.5)
            axes[-1].set_xlabel("step number")
            # plt.show()
            # plt.close()

            # Flatten samples
            # discard = int(nsteps * 0.2)
            # thin = discard // 10
            tau = emcee.autocorr.integrated_time(sampler.get_chain(), tol=0)
            discard = int(3 * max(tau))
            thin = int(max(tau) / 2)

            flat_samples = sampler.get_chain(discard=discard, thin=thin, flat=True)
            # flat_samples = sampler.get_chain(flat=True)

            # Corner plot
            # fig = corner(flat_samples)
            # plt.show()

            # Get percentiles and data best values
            best = []
            err_low = []
            err_high = []
            best1 = []
            err_low1 = []
            err_high1 = []
            for i in range(ndim):
                # mcmc = np.percentile(flat_samples[:, i], [16, 50, 84])
                # q = np.diff(mcmc)
                # print(f'label: {mcmc[1]} +{q[1]} -{q[0]}')
                quantiles = np.percentile(flat_samples[:, i], [16, 50, 84])
                median = quantiles[1]
                err_l1 = quantiles[0] - median
                err_h1 = quantiles[2] - median

                # best1.append(median)
                # err_low1.append(err_l1)
                # err_high1.append(err_h1)
                best.append(median)
                err_low.append(err_l1)
                err_high.append(err_h1)

                # best.append(mcmc[1])
                # err_low.append(q[0])
                # err_high.append(q[1])

            # best = np.abs(best)
            err_low = np.abs(err_low)
            err_high = np.abs(err_high)
            best_flux = best[3:]
            err_low_flux = err_low[3:]
            err_high_flux = err_high[3:]
            best_pos = best[:2]
            err_low_pos = err_low[:2]
            err_high_pos = err_high[:2]

            rc_spectrum_out.collection.append(
                SpectrumResource(
                    name='',
                    spectral_irradiance=torch.tensor(best_flux),
                    err_low=torch.tensor(err_low_flux),
                    err_high=torch.tensor(err_high_flux),
                    wavelength_bin_centers=torch.tensor(self.wavelengths),
                    wavelength_bin_widths=r_config_in.phringe._director._wavelength_bin_widths
                )
            )

            r_coordinates_out.x = best_pos[0]
            r_coordinates_out.y = best_pos[1]
            r_coordinates_out.err_low_x = err_low_pos[0]
            r_coordinates_out.err_high_x = err_high_pos[0]
            r_coordinates_out.err_low_y = err_low_pos[1]
            r_coordinates_out.err_high_y = err_high_pos[1]

            # TODO: automate this for all extractions
            # for q, extraction in enumerate(context.extractions):
            #     context.extractions[i] = Extraction(
            #         flux=best_flux,
            #         flux_err_low=err_low_flux,
            #         flux_err_high=err_high_flux,
            #         wavelengths=extraction.wavelengths,
            #         cost_function=extraction.cost_function
            #     )
            #
            # calculate 1 sigma uncertainties of flux
            # err_low_flux = np.abs(np.array(err_low_flux) - np.array(best_flux))
            # err_high_flux = np.abs(np.array(err_high_flux) - np.array(best_flux))
            # print(err_low_flux)
            # print(err_high_flux)

            # # Plot data and fit
            # plt.subplot(3, 1, 1)
            # plt.imshow(get_model(time, *best), label='Fit', cmap='Greys')
            # plt.colorbar()
            # plt.title('Fit')
            # plt.subplot(3, 1, 2)
            # plt.imshow(data_in, label='Data')
            # plt.colorbar()
            # plt.title('Data')
            # plt.subplot(3, 1, 3)
            # plt.imshow(data_in - get_model(time, *best), label='Diff')
            # plt.colorbar()
            # plt.title('Diff')
            # plt.savefig('data_fit.pdf')
            # plt.show()
            # plt.savefig('data_fit.svg')
            # plt.close()

            # Plot data and true spectrum and error bars
            # wl = config.phringe.get_wavelength_bin_centers().cpu().numpy()
            # wl = wl[:-1]
            # flux_init = flux_init[:-1]
            # yerr = np.stack([err_low_flux, err_high_flux])
            # plt.errorbar(wl, best_flux, yerr=yerr, fmt=".k", capsize=0)
            # plt.scatter(wl, best_flux, label='Data', color='black')
            # plt.plot(wl, flux_init, label='True', color='black', linestyle='dashed')
            # # plt.fill_between(range(len(best[:-1])), np.array(best[:-1]) - np.array(err_low[:-1]),
            # #                  np.array(best[:-1]) + np.array(err_high[:-1]),
            # #                  color="k", alpha=0.2)
            # plt.fill_between(wl, np.array(flux_init) - np.array(err_low_flux),
            #                  np.array(flux_init) + np.array(err_high_flux),
            #                  color="k", alpha=0.2, label='1-$\sigma$')
            # plt.title('True vs. Estimated Spectrum')
            # plt.xlabel('Wavelength ($\mu$m)')
            # plt.ylabel('Flux Density (ph s$^{-1}$ m$^{-2}$ $\mu$m$^{-1}$)')
            # plt.ticklabel_format(axis='y', style='sci', scilimits=(0, 0))
            # plt.legend()
            # plt.ylim(0, 1e6)
            # plt.tight_layout()
            # plt.savefig('spectrum.svg')
            # plt.show()
            # plt.close()

            # Plot SNR
            # bins = r_config_in.phringe._director._wavelength_bin_widths.cpu().numpy()[:-1]
            # err_mean_flux = np.mean([err_low_flux, err_high_flux], axis=0)
            # snr = best_flux / err_mean_flux
            # snr2 = np.sqrt(np.asarray(best_flux) ** 2 / err_mean_flux)
            # snr3 = best_flux / np.sqrt(err_mean_flux)
            #
            # snr_total = np.sqrt(np.sum(snr ** 2))
            # snr_total = np.sum(best_flux) / np.sqrt(np.sum(err_mean_flux ** 2))
            # snr_total2 = np.sqrt(np.sum(snr2 ** 2))
            # snr_total3 = np.sqrt(np.sum(snr3 ** 2))
            #
            # plt.step(bins, snr, label='SNR')
            # # plt.plot(snr2, label='SNR2')
            # plt.title(f'SNR {snr_total}')
            # plt.xlabel('Wavelength ($\mu$m)')
            # plt.ylabel('SNR')
            # # plt.legend()
            # plt.savefig('snr.pdf')
            # plt.show()
            # plt.close()

            # inds = np.random.randint(len(flat_samples), size=100)
            # for ind in inds:
            #     sample = flat_samples[ind]
            #     plt.plot(get_model(time, *sample), "gray", alpha=0.1)
            # # plt.errorbar(time, y, yerr=yerr, fmt=".k", capsize=0)
            # plt.plot(data, 'orange', label='Data')
            # plt.plot(get_model(time, *best), "k", label="Fit")
            # plt.legend(fontsize=14)
            # # plt.xlim(0, 10)
            # plt.xlabel("x")
            # plt.ylabel("y")
            # plt.show()

        # return context

        print('Done')
        return rc_spectrum_out, r_coordinates_out
