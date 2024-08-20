from abc import ABC, abstractmethod

from lifesimmc.core.context import Context


class BaseModule(ABC):
    """Class representation of the base module."""

    def __init__(self, name: str):
        """Constructor method."""
        self.name = name

    @abstractmethod
    def apply(self, context: Context) -> Context:
        """Apply the module.

        :param context: The context object of the pipelines
        :return: The (updated) context object
        """
        pass
