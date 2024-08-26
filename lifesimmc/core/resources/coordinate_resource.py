from dataclasses import dataclass

from lifesimmc.core.resources.base_resource import BaseResource


@dataclass
class CoordinateResource(BaseResource):
    """Class representation of a coordinate resource.

    :param x: The x coordinate
    :param y: The y coordinate
    :param x_err_low: The low x error
    :param x_err_high: The high x error
    :param y_err_low: The low y error
    :param y_err_high: The high y error
    """
    x: float = None
    y: float = None
    x_err_low: float = None
    x_err_high: float = None
    y_err_low: float = None
    y_err_high: float = None
