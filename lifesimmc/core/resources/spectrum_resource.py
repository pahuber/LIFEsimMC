from dataclasses import dataclass, field

from torch import Tensor

from lifesimmc.core.resources.base_resource import BaseResource, BaseResourceCollection


@dataclass
class SpectrumResource(BaseResource):
    """Class representation of a spectrum resource.

    :param spectral_irradiance: Spectral irradiance of the spectrum.
    :param err_low: Lower bound of the error.
    :param err_high: Upper bound of the error.
    :param wavelength_bin_centers: Wavelength bin centers.
    :param wavelength_bin_widths: Wavelength bin widths.
    """
    spectral_irradiance: Tensor
    wavelength_bin_centers: Tensor
    wavelength_bin_widths: Tensor
    err_low: Tensor = None
    err_high: Tensor = None
    planet_name: str = None


@dataclass
class SpectrumResourceCollection(BaseResourceCollection):
    """Class representation of a collection of spectrum resources.

    :param collection: The collection of spectrum resources.
    """
    collection: list[SpectrumResource] = field(default_factory=list)
