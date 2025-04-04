from dataclasses import dataclass


@dataclass
class BaseResource:
    """Class representation of the base resource.

    :param name: The name of the resource
    """
    name: str
