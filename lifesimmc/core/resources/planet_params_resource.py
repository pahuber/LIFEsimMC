from dataclasses import dataclass, field

from torch import Tensor

from lifesimmc.core.resources.base_resource import BaseResource


@dataclass
class PlanetParams:
    """Class representation of a planet parameter.

    :param name: Name of the planet.
    :param sed_wavelength_bin_centers: Wavelength bin centers of the SED.
    :param sed_wavelength_bin_widths: Wavelength bin widths of the SED.
    :param pos_x: X position of the planet.
    :param pos_y: Y position of the planet.
    :param pos_x_err_low: Lower bound of the error in the X position.
    :param pos_x_err_high: Upper bound of the error in the X position.
    :param pos_y_err_low: Lower bound of the error in the Y position.
    :param pos_y_err_high: Upper bound of the error in the Y position.
    :param sed: Spectral energy distribution of the planet.
    :param sed_err_low: Lower bound of the error in the SED.
    :param sed_err_high: Upper bound of the error in the SED.
    :param covariance: Covariance of the parameters.
    """

    name: str
    sed_wavelength_bin_centers: Tensor
    sed_wavelength_bin_widths: Tensor
    pos_x: float = None
    pos_y: float = None
    pos_x_err_low: float = None
    pos_x_err_high: float = None
    pos_y_err_low: float = None
    pos_y_err_high: float = None
    sed: Tensor = None
    sed_err_low: Tensor = None
    sed_err_high: Tensor = None
    covariance: Tensor = None


@dataclass
class PlanetParamsResource(BaseResource):
    """Class representation of a planet parameter resource.
    """
    params: list[PlanetParams] = field(default_factory=list)
