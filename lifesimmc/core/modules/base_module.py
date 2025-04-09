from abc import ABC, abstractmethod
from typing import Union

from lifesimmc.core.resources.base_resource import BaseResource


class BaseModule(ABC):
    """Class representation of the base module.

    Parameters
    ----------
    seed : int or None
        The random seed.
    gpu_index : int or None
        The index of the GPU to use.
    grid_size : int or None
        The size of the grid.
    time_step_size : float or None
        The size of the time step.
    device : str or None
        The device to use.
    """

    def __init__(self):
        """Constructor method."""
        self.seed = None
        self.gpu_index = None
        self.grid_size = None
        self.time_step_size = None
        self.device = None
        self.resources = None

    def get_resource_from_name(self, name: str):
        """Get the resource from the name.

        :param name: The name of the resource
        :return: The resource
        """
        resource = self.resources.get(name)
        return resource

    @abstractmethod
    def apply(self, resources: list[BaseResource]) -> Union[None, BaseResource, tuple]:
        """Apply the module.

        :param resources: The resources to apply the module to
        :return: The resource
        """
        pass
