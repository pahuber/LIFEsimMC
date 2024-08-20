from abc import ABC, abstractmethod
from typing import Union

from lifesimmc.core.resources.base_resource import BaseResource


class BaseModule(ABC):
    """Class representation of the base module."""

    def __init__(self):
        """Constructor method."""
        self.gpu = None
        # self.preceding_modules = None

    def get_resource_from_name(self, name: str):
        """Get the resource from the name.

        :param name: The name of the resource
        :return: The resource
        """
        resource = [resource for resource in self.resources if resource.name == name][0]
        return resource

    @abstractmethod
    def apply(self, resources: list[BaseResource]) -> Union[None, BaseResource]:
        """Apply the module.
        """
        pass
