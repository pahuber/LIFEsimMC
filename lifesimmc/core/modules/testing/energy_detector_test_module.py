from scipy.stats import ncx2

from lifesimmc.core.modules.base_module import BaseModule
from lifesimmc.core.resources.base_resource import BaseResource
from lifesimmc.core.resources.test_resource import TestResource


class EnergyDetectorTestModule(BaseModule):
    def __init__(self, data_in: str, test_out: str, pfa: float):
        self.data_in = data_in
        self.test_out = TestResource(test_out)
        self.pfa = pfa

    def apply(self, resources: list[BaseResource]) -> TestResource:
        print("Performing energy detector test...")

        data = self.get_resource_from_name(self.data_in).get_data()
        num_of_diff_outputs = len(data)

        self.test_out.test_statistic = []
        self.test_out.xsi = []

        for i in range(num_of_diff_outputs):
            dataf = data[i].flatten()
            ndim = dataf.numel()

            test = (dataf @ dataf) / ndim
            xsi = ncx2.ppf(1 - self.pfa, df=ndim, nc=0)

            self.test_out.test_statistic.append(test)
            self.test_out.xsi.append(xsi)

        print('Done')
        return self.test_out
