from dataclasses import dataclass
from typing import Callable

from torch import Tensor


@dataclass
class TransformationResource:
    """Class representation of the transformation resource.

    :param name: The name of the resource
    :param transformation: The transformation function
    """
    name: str
    transformation: Callable[[Tensor], Tensor]
