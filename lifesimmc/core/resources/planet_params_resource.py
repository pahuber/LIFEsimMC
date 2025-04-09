from dataclasses import dataclass, field

from torch import Tensor

from lifesimmc.core.resources.base_resource import BaseResource


@dataclass
class PlanetParams:
    """Class representation of a planet parameter.

    Parameters
    ----------

    name : str
        Name of the planet.
    sed_wavelength_bin_centers : Tensor
        Wavelength bin centers of the SED.
    sed_wavelength_bin_widths : Tensor
        Wavelength bin widths of the SED.
    pos_x : float
        X position of the planet.
    pos_y : float
        Y position of the planet.
    pos_x_err_low : float
        Lower bound of the error in the X position.
    pos_x_err_high : float
        Upper bound of the error in the X position.
    pos_y_err_low : float
        Lower bound of the error in the Y position.
    pos_y_err_high : float
        Upper bound of the error in the Y position.
    sed : Tensor
        Spectral energy distribution of the planet.
    sed_err_low : Tensor
        Lower bound of the error in the SED.
    sed_err_high : Tensor
        Upper bound of the error in the SED.
    covariance : Tensor
        Covariance of the parameters.
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

    Parameters
    ----------

    params : list[PlanetParams]
        List of planet parameters.
    """
    params: list[PlanetParams] = field(default_factory=list)
