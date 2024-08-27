from dataclasses import dataclass, field


@dataclass
class BaseResource:
    """Class representation of the base resource.

    :param name: The name of the resource
    """
    name: str


@dataclass
class BaseResourceCollection:
    """Class representation of the base resource collection.

    :param name: The name of the resource collection
    :param collection: The collection of resources
    """
    name: str
    description: str
    collection: list[BaseResource] = field(default_factory=list)
