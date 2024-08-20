from copy import copy

from lifesimmc.core.resources.base_resource import BaseResource


class TemplateResource(BaseResource):
    def __init__(self, name: str):
        super().__init__(name)
        self._templates = []

    def get_templates(self):
        return copy(self._templates)

    def add_template(self, template):
        self._templates.append(template)
