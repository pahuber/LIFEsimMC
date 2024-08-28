from typing import Union

from lifesimmc.core.modules.base_module import BaseModule
from lifesimmc.core.resources.base_resource import BaseResource, BaseResourceCollection


class Pipeline:
    """Class representation of the pipeline."""

    def __init__(self, gpu: int = None):
        """Constructor method."""
        self.gpu = gpu
        self._modules = []
        self._resources = []

    def add_module(self, module: BaseModule):
        """Add a module to the pipeline.

        :param module: The module to add
        """
        module.resources = self._resources
        module.gpu = self.gpu
        self._modules.append(module)

    def get_resource(self, name: str) -> Union[BaseResource, BaseResourceCollection, None]:
        """Get a resource by name.

        :param name: The name of the resource
        :return: The resource
        """
        for resource in self._resources:
            if resource.name == name:
                return resource
        print(f"Resource {name} not found.")
        return None

    def run(self):
        """Run the pipeline with all the modules that have been added. Remove the modules after running."""
        for module in self._modules:
            resource = module.apply(self._resources)
            if resource is not None:
                if isinstance(resource, tuple):
                    for res in resource:
                        self._resources.append(res)
                else:
                    self._resources.append(resource)
        self._modules = []
