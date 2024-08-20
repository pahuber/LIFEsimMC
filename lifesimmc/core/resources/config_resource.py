from lifesimmc.core.resources.base_resource import BaseResource


class ConfigResource(BaseResource):
    def __init__(self, name: str):
        super().__init__(name)
        self.phringe = None
        self.config_file_path = None
        self.simulation = None
        self.observation_mode = None
        self.instrument = None
        self.scene = None
