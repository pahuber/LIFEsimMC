from typing import Union

import torch
from phringe.main import PHRINGE

from lifesimmc.core.modules.base_module import BaseModule
from lifesimmc.core.resources.base_resource import BaseResource


class Pipeline:
    """Class representation of the pipeline."""

    def __init__(
            self,
            seed: int = None,
            gpu_index: int = None,
            grid_size: int = 40,
            time_step_size: float = None,
            device: torch.device = None
    ):
        """Constructor method."""
        self.seed = seed
        self.gpu_index = gpu_index
        self.grid_size = grid_size
        self.time_step_size = time_step_size
        self.device = PHRINGE()._get_device(self.gpu_index)
        self._modules = []
        self._resources = []

    def add_module(self, module: BaseModule):
        """Add a module to the pipeline.

        :param module: The module to add
        """
        module.resources = self._resources
        module.seed = self.seed
        module.gpu_index = self.gpu_index
        module.grid_size = self.grid_size
        module.time_step_size = self.time_step_size
        module.device = self.device
        self._modules.append(module)

    def get_resource(self, name: str) -> Union[BaseResource, None]:
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
