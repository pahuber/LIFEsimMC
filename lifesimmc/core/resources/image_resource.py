from lifesimmc.core.resources.base_resource import BaseResource


class ImageResource(BaseResource):
    def __init__(self, name: str):
        super().__init__(name)
        self.image = None
