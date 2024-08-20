from abc import ABC, abstractmethod


class BaseModule(ABC):
    """Class representation of the base module."""

    def __init__(self, name: str):
        """Constructor method."""
        self.name = name
        self.gpu = None
        self.preceding_modules = None

    def get_module_from_name(self, name: str):
        """Get the module from the name.

        :param name: The name of the module
        :return: The module
        """
        module = [module for module in self.preceding_modules if module.name == name][0]
        return module

    @abstractmethod
    def apply(self):
        """Apply the module.
        """
        pass
