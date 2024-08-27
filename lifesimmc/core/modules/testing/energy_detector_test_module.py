from scipy.stats import ncx2

from lifesimmc.core.modules.base_module import BaseModule
from lifesimmc.core.resources.base_resource import BaseResource
from lifesimmc.core.resources.test_resource import TestResource, TestResourceCollection


class EnergyDetectorTestModule(BaseModule):
    """Class representation of an energy detector test module.

    :param n_data_in: Name of the input data resource.
    :param n_test_out: Name of the output test resource.
    :param pfa: Probability of false alarm.
    """

    def __init__(self, n_data_in: str, n_test_out: str, pfa: float):
        """Constructor method.

        :param n_data_in: Name of the input data resource.
        :param n_test_out: Name of the output test resource.
        :param pfa: Probability of false alarm.
        """
        self.n_data_in = n_data_in
        self.n_test_out = n_test_out
        self.pfa = pfa

    def apply(self, resources: list[BaseResource]) -> TestResourceCollection:
        """Apply the energy detector test.

        :param resources: List of resources.
        :return: Test resource collection.
        """
        print("Performing energy detector test...")

        data = self.get_resource_from_name(self.n_data_in).get_data()
        rc_test_out = TestResourceCollection(self.n_test_out)
        num_of_diff_outputs = len(data)

        for i in range(num_of_diff_outputs):
            dataf = data[i].flatten()
            ndim = dataf.numel()

            test = (dataf @ dataf) / ndim
            xsi = ncx2.ppf(1 - self.pfa, df=ndim, nc=0)

            r_test_out = TestResource(
                name='',
                test_statistic=test,
                xsi=xsi,
            )
            rc_test_out.collection.append(r_test_out)

        print('Done')
        return rc_test_out
