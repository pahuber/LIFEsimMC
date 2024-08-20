from lifesimmc.core.modules.base_module import BaseModule


class AnalyticalMLEModule(BaseModule):
    def __init__(self, name: str, data_in: str, template_in: str, cov_in: str, data_out: str):
        super().__init__(name)
        self.data_in = data_in
        self.template_in = template_in
        self.cov_in = cov_in
        self.data_out = data_out
        self.data = None

    def apply(self):
        pass
