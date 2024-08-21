from copy import deepcopy

from lifesimmc.core.resources.base_resource import BaseResource


class DataResource(BaseResource):
    def __init__(self, name: str):
        super().__init__(name)
        self._data = None

    def get_data(self):
        return deepcopy(self._data)

    def set_data(self, data):
        self._data = data
