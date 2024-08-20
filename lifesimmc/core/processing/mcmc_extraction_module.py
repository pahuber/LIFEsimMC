import sys

import emcee
import numpy as np
import sympy as sp
from matplotlib import pyplot as plt
from phringe.api import PHRINGE

from lifesimmc.util.helpers import Extraction

sys.path.append("/home/huberph/lifesimmc")
from lifesimmc.core.base_module import BaseModule


class MCMCExtractionModule(BaseModule):
    """Module to estimate the flux using a parametric maximum likelihood estimation"""

    def __init__(self, gpus: tuple[int]):
        """Constructor method."""
        self.fov = None
        self.wavelength = None
        self.context = None
        self.nulling_baseline = None
        self.diff_ir = None
        self._intensity_response = None
        self.gpus = gpus

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

    def get_analytical_expr_counts(self):
        """Return data (counts) calculated using a parametric model."""

        t, flux, pos_x, pos_y, lam, nb, br, eta, ap, mod = sp.symbols('t flux pos_x pos_y lam nb br eta ap mod')
        # Rotation matrix
        argument = 2 * sp.pi / mod * t
        c = sp.cos(argument)
        s = sp.sin(argument)
        rotation_matrix = np.stack([
            c, -s,
            s, c
        ]).reshape(2, 2)

        # X static
        baseline_ratio = br
        x_static = nb / 2 * np.asarray(
            [[baseline_ratio, baseline_ratio, -baseline_ratio, -baseline_ratio], [1, -1, -1, 1]])

        # X rotating --> collector positions
        array_pos = rotation_matrix @ x_static

        # Complex amplitudes
        complex_amplitudes = sp.zeros(array_pos.shape[1], dtype=complex)
        for i in range(array_pos.shape[1]):
            complex_amplitudes[i] = ap / 2 * sp.sqrt(
                eta) * sp.exp(
                1j * 2 * sp.pi * (
                        array_pos[0, i] * pos_x + array_pos[1, i] * pos_y) / lam)

        # Get CATM X
        catm = 1 / 2 * np.asarray([[0, 0, sp.sqrt(2), sp.sqrt(2)],
                                   [sp.sqrt(2), sp.sqrt(2), 0, 0],
                                   [1, -1, -sp.exp(1j * sp.pi / 2),
                                    sp.exp(1j * sp.pi / 2)],
                                   [1, -1, sp.exp(1j * sp.pi / 2),
                                    -sp.exp(1j * sp.pi / 2)]])

        # Intensity response
        ir = sp.zeros(catm.shape[1])
        for j in range(catm.shape[1]):
            for k in range(catm.shape[0]):
                ir[j] += catm[j, k] * complex_amplitudes[k]
            ir[j] = sp.Abs(ir[j]) ** 2
        indices = self.context.instrument.beam_combiner.get_differential_output_pairs()
        diff_ir = ir[indices[0][0]] - ir[indices[0][1]]

        return flux * diff_ir

    def get_analytical_expr_ir(self):
        """Return the intensity response calculated using a parametric model."""

        t, flux, pos_x, pos_y, lam, nb, br, eta, ap, mod = sp.symbols('t flux pos_x pos_y lam nb br eta ap mod')
        # Rotation matrix
        argument = 2 * sp.pi / mod * t
        c = sp.cos(argument)
        s = sp.sin(argument)
        rotation_matrix = np.stack([
            c, -s,
            s, c
        ]).reshape(2, 2)

        # X static
        baseline_ratio = br
        x_static = nb / 2 * np.asarray(
            [[baseline_ratio, baseline_ratio, -baseline_ratio, -baseline_ratio], [1, -1, -1, 1]])

        # X rotating --> collector positions
        array_pos = rotation_matrix @ x_static

        # Complex amplitudes
        complex_amplitudes = sp.zeros(array_pos.shape[1], dtype=complex)
        for i in range(array_pos.shape[1]):
            complex_amplitudes[i] = ap / 2 * sp.sqrt(
                eta) * sp.exp(
                1j * 2 * sp.pi * (
                        array_pos[0, i] * pos_x + array_pos[1, i] * pos_y) / lam)

        # Get CATM X
        catm = 1 / 2 * np.asarray([[0, 0, sp.sqrt(2), sp.sqrt(2)],
                                   [sp.sqrt(2), sp.sqrt(2), 0, 0],
                                   [1, -1, -sp.exp(1j * sp.pi / 2),
                                    sp.exp(1j * sp.pi / 2)],
                                   [1, -1, sp.exp(1j * sp.pi / 2),
                                    -sp.exp(1j * sp.pi / 2)]])

        # Intensity response
        ir = sp.zeros(catm.shape[1])
        for j in range(catm.shape[1]):
            for k in range(catm.shape[0]):
                ir[j] += catm[j, k] * complex_amplitudes[k]
            ir[j] = sp.Abs(ir[j]) ** 2

        return ir

    def apply(self, context):
        """Apply module"""

        def get_model_single(time, flux, x_pos, y_pos):
            """Return the data model for a single wavelength."""

            from numpy import exp, pi, sin, cos, sqrt
            t = time
            flux = flux
            pos_x = x_pos
            pos_y = y_pos
            ap = self.context.instrument.aperture_diameter.numpy()
            eta = self.context.instrument.unperturbed_instrument_throughput.numpy()
            lam = self.wavelength
            nb = self.nulling_baseline
            br = self.context.observation_mode.baseline_ratio
            mod = self.context.observation_mode.modulation_period
            I = 1j

            expr = flux * (-abs(-0.25 * ap * sqrt(eta) * exp(0.5 * I * pi) * exp(2.0 * I * pi * (
                    pos_x * (-br * nb * cos(2 * pi * t / mod) / 2 - nb * sin(2 * pi * t / mod) / 2) + pos_y * (
                    -br * nb * sin(2 * pi * t / mod) / 2 + nb * cos(
                2 * pi * t / mod) / 2)) / lam) + 0.25 * ap * sqrt(eta) * exp(0.5 * I * pi) * exp(
                2.0 * I * pi * (
                        pos_x * (-br * nb * cos(2 * pi * t / mod) / 2 + nb * sin(2 * pi * t / mod) / 2) + pos_y * (
                        -br * nb * sin(2 * pi * t / mod) / 2 - nb * cos(
                    2 * pi * t / mod) / 2)) / lam) + 0.25 * ap * sqrt(eta) * exp(2.0 * I * pi * (
                    pos_x * (br * nb * cos(2 * pi * t / mod) / 2 - nb * sin(2 * pi * t / mod) / 2) + pos_y * (
                    br * nb * sin(2 * pi * t / mod) / 2 + nb * cos(
                2 * pi * t / mod) / 2)) / lam) - 0.25 * ap * sqrt(eta) * exp(2.0 * I * pi * (
                    pos_x * (br * nb * cos(2 * pi * t / mod) / 2 + nb * sin(2 * pi * t / mod) / 2) + pos_y * (
                    br * nb * sin(2 * pi * t / mod) / 2 - nb * cos(2 * pi * t / mod) / 2)) / lam)) ** 2 + abs(
                0.25 * ap * sqrt(eta) * exp(0.5 * I * pi) * exp(2.0 * I * pi * (
                        pos_x * (-br * nb * cos(2 * pi * t / mod) / 2 - nb * sin(2 * pi * t / mod) / 2) + pos_y * (
                        -br * nb * sin(2 * pi * t / mod) / 2 + nb * cos(
                    2 * pi * t / mod) / 2)) / lam) - 0.25 * ap * sqrt(eta) * exp(0.5 * I * pi) * exp(
                    2.0 * I * pi * (pos_x * (
                            -br * nb * cos(2 * pi * t / mod) / 2 + nb * sin(2 * pi * t / mod) / 2) + pos_y * (
                                            -br * nb * sin(2 * pi * t / mod) / 2 - nb * cos(
                                        2 * pi * t / mod) / 2)) / lam) + 0.25 * ap * sqrt(eta) * exp(
                    2.0 * I * pi * (pos_x * (
                            br * nb * cos(2 * pi * t / mod) / 2 - nb * sin(2 * pi * t / mod) / 2) + pos_y * (
                                            br * nb * sin(2 * pi * t / mod) / 2 + nb * cos(
                                        2 * pi * t / mod) / 2)) / lam) - 0.25 * ap * sqrt(eta) * exp(
                    2.0 * I * pi * (pos_x * (
                            br * nb * cos(2 * pi * t / mod) / 2 + nb * sin(2 * pi * t / mod) / 2) + pos_y * (
                                            br * nb * sin(2 * pi * t / mod) / 2 - nb * cos(
                                        2 * pi * t / mod) / 2)) / lam)) ** 2)
            return expr

        def get_model(time, x_pos, y_pos, *flux):
            """Return the data model for all wavelengths."""

            dit = self.dit
            wl_bin = self.wl_bin
            # eta_t = self.eta_t
            model = np.zeros((len(self.wavelengths), len(time)))
            for i, wavelength in enumerate(self.wavelengths):
                self.wavelength = wavelength
                model[i] = get_model_single(time, flux[i], x_pos, y_pos) * wl_bin[i] * dit
            if context.icov2 is not None:
                return context.icov2 @ model
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

        # Generate data
        phringe = PHRINGE()

        phringe.run(
            config_file_path=context.config_file_path,
            exoplanetary_system_file_path=context.exoplanetary_system_file_path,
            settings=context.simulation,
            observatory=context.instrument,
            observation=context.observation_mode,
            scene=context.scene,
            spectrum_files=context.spectrum_files,
            gpus=self.gpus,
            output_dir='.',
            write_fits=False,
            create_copy=False,
            detailed=True
        )

        # Extract data from PHRINGE
        self._intensity_response = phringe.get_intensity_response()
        # self.diff_ir = (self._intensity_response['Earth'][:, 2, :, :, :] - self._intensity_response['Earth'][:, 3, :, :,
        #                                                                    :]).numpy().astype(np.float64)
        self.nulling_baseline = phringe._director.nulling_baseline  # phringe.get_nulling_baseline()
        self.context = context
        # wavelength_index = 10
        self.wavelengths = phringe.get_wavelength_bin_centers().numpy()
        # self.wavelength = phringe.get_wavelength_bin_centers().numpy()[wavelength_index]
        # self.fov = phringe.get_field_of_view()[wavelength_index] * 10
        self.fovs = phringe.get_field_of_view().numpy() * 10
        self.dit = phringe._director._detector_integration_time
        self.wl_bin = phringe._director._instrument_wavelength_bin_widths.cpu().numpy()
        self.eta_t = phringe._director._unperturbed_instrument_throughput.cpu().numpy()
        self.polyfits = context.polyfits.numpy() if context.polyfits is not None else None

        # # Get analytical expression
        # expr = self.get_analytical_expr()
        # print(expr)
        # expr = self.get_analytical_expr_ir()
        # print(expr)

        # Initial guess
        flux_real = (np.float64(phringe._director._planets[0].spectral_flux_density.cpu().numpy())).tolist()
        # flux_init = context.bb_best  # np.ones(len(self.wavelengths)) * 500000
        flux_init = np.asarray(flux_real)
        xdata = phringe.get_time_steps().numpy()
        # ydata = context.data[0, wavelength_index].numpy().astype(np.float64)
        ydata = context.data[0].numpy().astype(np.float64)

        # Define MCMC
        initial_guess = [3.4e-7, 3.4e-7]
        initial_guess.extend(flux_init)
        ndim = len(initial_guess)
        nwalkers = 10 * ndim
        nsteps = 400

        # Define data uncertainty
        # yerr = np.ones(len(ydata)) * 1.5 * np.max(abs(get_model(xdata, *initial_guess)))
        yerr = np.ones(len(ydata)) * 0.1 * np.max(abs(ydata))
        # yerr = 0.8 * (ydata - get_model(xdata, *initial_guess))
        # print(yerr.shape)
        # yerr = 0.001 * ydata

        # Initialize walkers
        pos = np.zeros((nwalkers, ndim))
        pos_posx = initial_guess[0] + 1e-8 * np.random.randn(nwalkers)
        pos_posy = initial_guess[1] + 1e-8 * np.random.randn(nwalkers)
        # pos_a = initial_guess[2] + 1e-4 * np.random.randn(nwalkers)
        pos[:, 0] = pos_posx
        pos[:, 1] = pos_posy
        # pos[:, 2] = pos_a
        for i in range(2, len(initial_guess)):
            pos_flux = initial_guess[i] + 1e4 * np.random.randn(nwalkers)
            pos[:, i] = pos_flux
        # pos = [np.array(initial_guess) + 1e-8 * np.random.randn(ndim) for i in range(nwalkers)]

        # Plot initial guess and data in two imshow subplots that are on top of each other
        plt.subplot(2, 1, 1)
        plt.imshow(get_model(xdata, *initial_guess), label='Initial guess')
        plt.colorbar()
        plt.title('Initial guess')
        plt.subplot(2, 1, 2)
        plt.imshow(ydata, label='Data')
        plt.colorbar()
        plt.title('Data')
        plt.savefig('init.pdf')
        plt.show()
        plt.close()

        # Run MCMC
        sampler = emcee.EnsembleSampler(nwalkers, ndim, lnprob, args=(xdata, ydata, yerr))
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
        plt.show()
        plt.close()

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

        best = np.abs(best)
        err_low = np.abs(err_low)
        err_high = np.abs(err_high)
        best_flux = best[3:]
        err_low_flux = err_low[3:]
        err_high_flux = err_high[3:]
        best_pos = best[:2]
        err_low_pos = err_low[:2]
        err_high_pos = err_high[:2]

        # TODO: automate this for all extractions
        for i, extraction in enumerate(context.extractions):
            context.extractions[i] = Extraction(
                flux=best_flux,
                flux_err_low=err_low_flux,
                flux_err_high=err_high_flux,
                wavelengths=extraction.wavelengths,
                cost_function=extraction.cost_function
            )

        # calculate 1 sigma uncertainties of flux
        # err_low_flux = np.abs(np.array(err_low_flux) - np.array(best_flux))
        # err_high_flux = np.abs(np.array(err_high_flux) - np.array(best_flux))
        # print(err_low_flux)
        # print(err_high_flux)

        # # Plot data and fit
        plt.subplot(3, 1, 1)
        plt.imshow(get_model(xdata, *best), label='Fit', cmap='Greys')
        plt.colorbar()
        plt.title('Fit')
        plt.subplot(3, 1, 2)
        plt.imshow(ydata, label='Data')
        plt.colorbar()
        plt.title('Data')
        plt.subplot(3, 1, 3)
        plt.imshow(ydata - get_model(xdata, *best), label='Diff')
        plt.colorbar()
        plt.title('Diff')
        plt.savefig('data_fit.pdf')
        plt.show()
        plt.savefig('data_fit.svg')
        plt.close()

        # Plot data and true spectrum and error bars
        yerr = np.stack([err_low_flux[:-1], err_high_flux[:-1]])
        plt.errorbar(range(len(best_flux[:-1])), best_flux[:-1], yerr=yerr, fmt=".k", capsize=0)
        plt.scatter(range(len(best_flux[:-1])), best_flux[:-1], label='Data', color='black')
        plt.plot(flux_real[:-1], label='True', color='black', linestyle='dashed')
        # plt.fill_between(range(len(best[:-1])), np.array(best[:-1]) - np.array(err_low[:-1]),
        #                  np.array(best[:-1]) + np.array(err_high[:-1]),
        #                  color="k", alpha=0.2)
        plt.fill_between(range(len(flux_real[:-1])), np.array(flux_real[:-1]) - np.array(err_low_flux),
                         np.array(flux_real[:-1]) + np.array(err_high_flux),
                         color="k", alpha=0.2, label='1-$\sigma$')
        plt.title('True vs. Estimated Spectrum')
        plt.xlabel('Wavelength ($\mu$m)')
        plt.ylabel('Flux Density (ph s$^{-1}$ m$^{-2}$ $\mu$m$^{-1}$)')
        plt.ticklabel_format(axis='y', style='sci', scilimits=(0, 0))
        plt.legend()
        plt.ylim(0, 1e6)
        plt.tight_layout()
        plt.savefig('spectrum.svg')
        plt.show()
        plt.close()

        # Plot SNR
        bins = phringe._director._instrument_wavelength_bin_widths.cpu().numpy()[:-1]
        err_mean_flux = np.mean([err_low_flux, err_high_flux], axis=0)
        snr = best_flux / err_mean_flux
        snr2 = np.sqrt(np.asarray(best_flux) ** 2 / err_mean_flux)
        snr3 = best_flux / np.sqrt(err_mean_flux)

        snr_total = np.sqrt(np.sum(snr ** 2))
        snr_total = np.sum(best_flux) / np.sqrt(np.sum(err_mean_flux ** 2))
        snr_total2 = np.sqrt(np.sum(snr2 ** 2))
        snr_total3 = np.sqrt(np.sum(snr3 ** 2))

        plt.step(bins, snr, label='SNR')
        # plt.plot(snr2, label='SNR2')
        plt.title(f'SNR {snr_total}')
        plt.xlabel('Wavelength ($\mu$m)')
        plt.ylabel('SNR')
        # plt.legend()
        plt.savefig('snr.pdf')
        plt.show()
        plt.close()

        # inds = np.random.randint(len(flat_samples), size=100)
        # for ind in inds:
        #     sample = flat_samples[ind]
        #     plt.plot(get_model(xdata, *sample), "gray", alpha=0.1)
        # # plt.errorbar(xdata, y, yerr=yerr, fmt=".k", capsize=0)
        # plt.plot(ydata, 'orange', label='Data')
        # plt.plot(get_model(xdata, *best), "k", label="Fit")
        # plt.legend(fontsize=14)
        # # plt.xlim(0, 10)
        # plt.xlabel("x")
        # plt.ylabel("y")
        # plt.show()

        return context
