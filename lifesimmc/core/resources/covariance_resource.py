from lifesimmc.core.resources.base_resource import BaseResource


class CovarianceResource(BaseResource):
    def __init__(self, name: str):
        super().__init__(name)
        self.cov = None
        self.icov2 = None
