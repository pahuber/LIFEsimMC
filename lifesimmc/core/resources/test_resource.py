from dataclasses import dataclass

from lifesimmc.core.resources.base_resource import BaseResource


@dataclass
class TestResource(BaseResource):
    """Class representation of a test resource.

    Parameters
    ----------

    test_statistic_h1 : float
        The test statistic under the alternative hypothesis.
    test_statistic_h0 : float
        The test statistic under the null hypothesis.
    threshold_xsi : float
        The threshold for the test statistic.
    model_length_xtx : float
        The length of the model.
    dimensions : int
        The number of dimensions.
    detection_probability : float
        The probability of detection.
    probability_false_alarm : float
        The probability of false alarm.
    """
    test_statistic_h1: float = None
    test_statistic_h0: float = None
    threshold_xsi: float = None
    model_length_xtx: float = None
    dimensions: int = None
    detection_probability: float = None
    probability_false_alarm: float = None
    p_value: float = None
