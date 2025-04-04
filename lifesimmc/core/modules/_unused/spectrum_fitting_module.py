import numpy as np
from matplotlib import pyplot as plt
from scipy.optimize import curve_fit

from lifesimmc.core.modules.base_module import BaseModule
from lifesimmc.core.context import Context


class SpectrumFittingModule(BaseModule):
    """Class representation of the spectrum fitting module.
    """

    def __init__(self):
        """Constructor method.
        """
        pass

    def apply(self, context: Context) -> Context:
        """Fit a blackbody spectrum to the data.

        :param context: The context object of the pipeline
        :return: The (updated) context object
        """

        def get_scaled_blackbody_spectrum(wavelengths, temperature: float, radius: float) -> np.ndarray:
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

        self.context = context

        for extraction in context.extractions:
            print((extraction.flux_err_low + extraction.flux_err_high) / 2)

            popt, pcov = curve_fit(
                get_scaled_blackbody_spectrum,
                extraction.wavelengths.numpy()[:-1],
                extraction.flux,
                p0=[300.0, 7.0e6],
                sigma=(extraction.flux_err_low + extraction.flux_err_high) / 2
            )

            # print(popt, pcov)
            temperature = popt[0]
            radius = popt[1]

            print(f'Temperature: {temperature} +/- {np.sqrt(pcov[0, 0])} K')
            print(f'Radius: {radius / 1e3} +/- {np.sqrt(pcov[1, 1]) / 1e3} km')

            plt.scatter(extraction.wavelengths[:-1], extraction.flux, label='Data')

            plt.errorbar(extraction.wavelengths[:-1], extraction.flux,
                         yerr=(extraction.flux_err_low + extraction.flux_err_high) / 2, fmt=".k", capsize=0)

            plt.plot(extraction.wavelengths[:-1],
                     get_scaled_blackbody_spectrum(extraction.wavelengths[:-1], temperature, radius),
                     label='Fit')
            plt.legend()
            plt.show()

        return context
