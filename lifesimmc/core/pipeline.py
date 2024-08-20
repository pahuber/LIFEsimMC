from copy import copy

from lifesimmc.core.base_module import BaseModule


class Pipeline:
    """Class representation of the pipeline."""

    def __init__(self, gpu: int = None):
        """Constructor method."""
        self.gpu = gpu
        self._modules = []

    def add_module(self, module: BaseModule):
        """Add a module to the pipeline.

        :param module: The module to add
        """
        module.preceding_modules = copy(self._modules)
        module.gpu = self.gpu
        self._modules.append(module)

    def run(self):
        """Run the pipeline with all the modules that have been added. Remove the modules after running."""
        for module in self._modules:
            module.apply()
        self._modules = []
