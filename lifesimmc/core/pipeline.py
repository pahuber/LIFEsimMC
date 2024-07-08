from lifesimmc.core.base_module import BaseModule
from lifesimmc.core.context import Context


class Pipeline:
    """Class representation of the pipeline."""

    def __init__(self):
        """Constructor method."""
        self._modules = []
        self._context = Context()

    def add_module(self, module: BaseModule):
        """Add a module to the pipeline.

        :param module: The module to add
        """
        self._modules.append(module)

    def run(self):
        """Run the pipeline with all the modules that have been added. Remove the modules after running."""
        for module in self._modules:
            self._context = module.apply(context=self._context)
        self._modules = []
