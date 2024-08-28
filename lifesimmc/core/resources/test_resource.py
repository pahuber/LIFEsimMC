from dataclasses import dataclass, field

from lifesimmc.core.resources.base_resource import BaseResource, BaseResourceCollection


@dataclass
class TestResource(BaseResource):
    """Class representation of a test resource.
    """
    test_statistic: float = None
    xsi: float = None


@dataclass
class TestResourceCollection(BaseResourceCollection):
    """Class representation of a collection of test resources.
    """
    collection: list[TestResource] = field(default_factory=list)
