from typing import Union

import torch
from phringe.util.device import get_device

from lifesimmc.core.modules.base_module import BaseModule
from lifesimmc.core.resources.base_resource import BaseResource


class Pipeline:
    """Class representation of the pipeline.

    Parameters
    ----------
    seed : int
        The seed for the random number generator.
    gpu_index : int
        The index of the GPU to use.
    grid_size : int
        The size of the grid.
    time_step_size : float
        The size of the time step.
    device : torch.device
        The device to use for the simulation.
    _modules : list
        The list of modules in the pipeline.
    _resources : dict
        The dictionary of resources in the pipeline.
    """

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
        self.device = get_device(self.gpu_index) if device is None else device
        self._modules = []
        self._resources = {}

    def add_module(self, module: BaseModule):
        """Add a module to the pipeline.

        Parameters
        ----------
        module : BaseModule
            The module to add to the pipeline.
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

        Parameters
        ----------
        name : str
            The name of the resource to get.

        Returns
        -------
        BaseResource or None
            The resource if found, otherwise None.
        """
        if name in self._resources:
            return self._resources[name]
        else:
            print(f"Resource {name} not found.")
            return None

    def run(self):
        """Run the pipeline with all the modules that have been added. Remove the modules after running."""
        for module in self._modules:
            resource = module.apply(self._resources)
            if resource is not None:
                if isinstance(resource, tuple):
                    for res in resource:
                        self._resources[res.name] = res
                else:
                    self._resources[resource.name] = resource
        self._modules = []
