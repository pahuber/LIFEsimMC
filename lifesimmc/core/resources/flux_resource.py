from dataclasses import dataclass, field

from torch import Tensor

from lifesimmc.core.resources.base_resource import BaseResource, BaseResourceCollection


@dataclass
class FluxResource(BaseResource):
    """Class representation of a flux resource.

    :param spectral_irradiance: Spectral irradiance.
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
class FluxResourceCollection(BaseResourceCollection):
    """Class representation of a collection of flux resources.

    :param collection: The collection of flux resources.
    """
    collection: list[FluxResource] = field(default_factory=list)
