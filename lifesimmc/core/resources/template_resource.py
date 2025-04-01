from copy import deepcopy
from dataclasses import dataclass

from torch import Tensor

from lifesimmc.core.resources.base_resource import BaseResource


@dataclass
class TemplateResource(BaseResource):
    """Class representation of a template resource.

    :param x_coord: The x-coordinate of the template.
    :param y_coord: The y-coordinate of the template.
    :param x_index: The x-index of the template.
    :param y_index: The y-index of the template.
    :param data: The data of the template.
    """
    # x_coord: float = None
    # y_coord: float = None
    # x_index: int = None
    # y_index: int = None
    _data: Tensor = None
    grid_coordinates: Tensor = None

    def get_data(self):
        return deepcopy(self._data)

    def set_data(self, data: Tensor):
        self._data = data

# @dataclass
# class TemplateResourceCollection(BaseResourceCollection):
#     """Class representation of a collection of template resources.
#
#     :param collection: The collection of template resources.
#     """
#     collection: list[TemplateResource] = field(default_factory=list)
