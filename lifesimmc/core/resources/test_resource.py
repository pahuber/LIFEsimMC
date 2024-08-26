from lifesimmc.core.resources.base_resource import BaseResource


class TestResource(BaseResource):
    """Class representation of a test resource.
    """
    test_statistic: float = None
    xsi: float = None
