from typing import Union, Callable

from lifesimmc.core.modules.base_module import BaseModule


def get_transformations_from_resource_name(module: BaseModule, resource_name: Union[str, tuple[str]]) -> list[Callable]:
    if resource_name is None:
        transformations = []
    elif isinstance(resource_name, tuple):
        transformations = [module.get_resource_from_name(
            name).transformation for name in resource_name]
    else:
        transformations = [module.get_resource_from_name(
            resource_name).transformation]
    return transformations
