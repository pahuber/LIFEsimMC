from lifesimmc.core.resources.base_resource import BaseResource


class TestResource(BaseResource):

    def __init__(self, name: str):
        super().__init__(name)
        self.test_statistic = None
        self.xsi = None
