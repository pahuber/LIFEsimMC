from dataclasses import dataclass
from typing import Callable

from torch import Tensor


@dataclass
class TransformationResource:
    """Class representation of the transformation resource.

    Parameters
    ----------
    name : str
        The name of the transformation.
    transformation : Callable[[Tensor], Tensor]
        The transformation function that takes a Tensor as input and returns a Tensor.
    """
    name: str
    transformation: Callable[[Tensor], Tensor]
