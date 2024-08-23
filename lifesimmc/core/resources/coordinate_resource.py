from lifesimmc.core.resources.base_resource import BaseResource


class CoordinateResource(BaseResource):
    def __init__(self, name: str):
        super().__init__(name)
        self.coordinates = []
