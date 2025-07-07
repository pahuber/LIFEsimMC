from copy import copy
from dataclasses import dataclass

from torch import Tensor

from lifesimmc.core.resources.base_resource import BaseResource


@dataclass
class DataResource(BaseResource):
    """Class representation of the data resource.

    Parameters
    ----------
    _data : Tensor
        The data to be stored.
    """
    _data: Tensor = None

    def get_data(self) -> Tensor:
        """Get the data stored in the resource.

        Returns
        -------
        Tensor
            A deep copy of the data stored in the resource.
        """
        return copy(self._data)

    def set_data(self, data: Tensor):
        """Set the data in the resource.

        Parameters
        ----------
        data : Tensor
            The data to be stored in the resource.
        """
        self._data = data
