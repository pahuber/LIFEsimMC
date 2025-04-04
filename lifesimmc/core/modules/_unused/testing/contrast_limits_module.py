import numpy as np
import torch
from phringe.api import PHRINGE
from scipy.optimize import fsolve
from tqdm import tqdm

from lifesimmc.core.modules.base_module import BaseModule
from lifesimmc.core.resources.base_resource import BaseResource
from lifesimmc.core.resources.contrast_limits_resource import ContrastLimitsResource


def _get_xtx(
        radius_0: float,
        radius: float,
        phringe: PHRINGE,
        pos_x: np.ndarray,
        pos_y: np.ndarray,
        flux_planet: float,
        i_cov_sqrt: torch.Tensor,
        time_steps: torch.Tensor,
        wavelength_bin_centers: torch.Tensor,
        wavelength_bin_widths: torch.Tensor
):
    model = phringe.get_template_numpy(
        time_steps,
        wavelength_bin_centers,
        wavelength_bin_widths,
        pos_x,
        pos_y,
        flux_planet / radius_0 ** 2 * radius ** 2
    )
    x = (i_cov_sqrt @ model[0, :, :, 0, 0]).flatten()
    return x @ x


class ContrastLimitsModule(BaseModule):
    """Class representation of the contrast limits module.

    """

    def __init__(
            self,
            n_config_in: str,
            n_test_in: str,
            n_contrast_limits_out: str,
            n_cov_in: str = None,
            reps: int = 10,
            pixels: int = 10,
            fov_max: float = 400.
    ):
        """Constructor method.

        :param n_config_in: Name of the input configuration resource.
        :param n_test_in: Name of the input test resource.
        :param n_contrast_limits_out: Name of the output contrast limits resource.
        :param n_cov_in: Name of the input covariance resource.
        :param reps: Number of repetitions to create average
        :param pixels: Number of pixels.
        :param fov_max: Field of view maximum.
        """
        self.n_config_in = n_config_in
        self.n_test_in = n_test_in
        self.n_contrast_limits_out = n_contrast_limits_out
        self.n_cov_in = n_cov_in
        self.reps = reps
        self.pixels = pixels
        self.fov_max = fov_max / 206264806.7

    def apply(self, resources: list[BaseResource]) -> ContrastLimitsResource:

        print("Calculating contrast limits...")

        r_config_in = self.get_resource_from_name(self.n_config_in)
        r_test_in = self.get_resource_from_name(self.n_test_in).collection[0]
        r_cov_in = self.get_resource_from_name(self.n_cov_in) if self.n_cov_in is not None else None

        positions = np.linspace(-self.fov_max / 2, self.fov_max / 2, self.pixels)
        planet_radius_0 = r_config_in.phringe._director._planets[0].radius
        phringe = r_config_in.phringe
        flux_planet = phringe._director._planets[0].spectral_flux_density.cpu().numpy()
        flux_star = phringe._director._star.spectral_flux_density.cpu().numpy()
        time_steps = phringe.get_time_steps(as_numpy=True)
        wavelength_bin_centers = phringe.get_wavelength_bin_centers(as_numpy=True)
        wavelength_bin_widths = phringe.get_wavelength_bin_widths(as_numpy=True)
        mag_map = np.zeros((self.reps, len(wavelength_bin_centers), self.pixels, self.pixels))
        radius_map = np.zeros((self.reps, self.pixels, self.pixels))

        def _func(radius):
            return _get_xtx(
                planet_radius_0,
                radius,
                phringe,
                np.array([pos_x]),
                np.array([pos_y]),
                flux_planet,
                r_cov_in.i_cov_sqrt,
                time_steps,
                wavelength_bin_centers,
                wavelength_bin_widths
            )

        for rep in tqdm(range(self.reps)):
            for ix, iy in zip(positions, positions):
                ix = int(ix)
                iy = int(iy)
                pos_x = positions[ix]
                pos_y = positions[iy]

                radius_limit = fsolve(lambda x: _func(x) - r_test_in.xsi, x0=np.array(planet_radius_0))
                flux_planet = flux_planet / planet_radius_0 ** 2 * radius_limit ** 2
                mag_diff = -2.5 * np.log10(flux_planet / flux_star)

                radius_map[rep, ix, iy] = radius_limit
                mag_map[rep, :, ix, iy] = mag_diff

        # Calculate average
        avg_mag_map = np.mean(mag_map, axis=0)
        avg_radius_map = np.mean(radius_map, axis=0)

        r_contrast_limits_out = ContrastLimitsResource(
            name=self.n_contrast_limits_out,
            magnitude_map=torch.tensor(avg_mag_map),
            radius_map=torch.tensor(avg_radius_map)
        )

        return r_contrast_limits_out
