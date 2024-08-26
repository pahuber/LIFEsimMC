from copy import deepcopy
from dataclasses import dataclass, field

from torch import Tensor

from lifesimmc.core.resources.base_resource import BaseResource


@dataclass
class DataResource(BaseResource):
    """Class representation of the data resource.
    :param _data: The data to be stored.
    """
    _data: Tensor = None

    def get_data(self):
        return deepcopy(self._data)

    def set_data(self, data: Tensor):
        self._data = data
