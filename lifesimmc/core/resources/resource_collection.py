from dataclasses import dataclass, field
from typing import TypeVar, Generic

from lifesimmc.core.resources.base_resource import BaseResource

T = TypeVar("T")


@dataclass
class ResourceCollection(Generic[T], BaseResource):
    """
    Represents a collection of resources.

    This class is designed to manage a collection of resources in the form of a
    list. It provides a structured way to store and manipulate a list of resources
    efficiently in various use cases.

    Attributes
    ----------
    collection : list
        A list that stores the collection of resources.
    """
    collection: list[T] = field(default_factory=list)
