from lifesimmc.core.resources.base_resource import BaseResource


class SpectrumResource(BaseResource):
    def __init__(self, name: str):
        super().__init__(name)
        self.spectral_flux_density = None
