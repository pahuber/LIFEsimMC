from copy import deepcopy
from dataclasses import dataclass

from torch import Tensor

from lifesimmc.core.resources.base_resource import BaseResource


@dataclass
class TemplateResource(BaseResource):
    """Class representation of a template resource.

    Parameters
    ----------
    name : str
        The name of the resource.
    _data : Tensor
        The data of the resource.
    """
    _data: Tensor = None
    grid_coordinates: tuple[Tensor, Tensor] = None

    def get_data(self):
        return deepcopy(self._data)

    def set_data(self, data: Tensor):
        self._data = data
