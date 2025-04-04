from abc import ABC, abstractmethod
from typing import Union

from lifesimmc.core.resources.base_resource import BaseResource


class BaseModule(ABC):
    """Class representation of the base module."""

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
