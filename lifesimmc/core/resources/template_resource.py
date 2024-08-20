from lifesimmc.core.resources.base_resource import BaseResource


class TemplateResource(BaseResource):
    def __init__(self, name: str):
        super().__init__(name)
        self.templates = []
