from dataclasses import dataclass

from torch import Tensor

from lifesimmc.core.resources.base_resource import BaseResource


@dataclass
class FluxResource(BaseResource):
    """Class representation of a flux resource. If multiple planets are present, each list element corresponds to
    a different planet. The order of the elements in the list corresponds to the order of the planets in the
    configuration file.

    :param spectral_irradiance: Spectral irradiance.
    :param err_low: Lower bound of the error.
    :param err_high: Upper bound of the error.
    :param wavelength_bin_centers: Wavelength bin centers.
    :param wavelength_bin_widths: Wavelength bin widths.
    """
    spectral_irradiance: list[Tensor]
    wavelength_bin_centers: Tensor
    wavelength_bin_widths: Tensor
    planet_name: list[str] = None
    err_low: list[Tensor] = None
    err_high: list[Tensor] = None
    covariance: list[Tensor] = None
