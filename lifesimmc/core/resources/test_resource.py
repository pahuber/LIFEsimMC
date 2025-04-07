from dataclasses import dataclass

from lifesimmc.core.resources.base_resource import BaseResource


@dataclass
class TestResource(BaseResource):
    """Class representation of a test resource.
    """
    test_statistic_h1: float = None
    test_statistic_h0: float = None
    threshold_xsi: float = None
    model_length_xtx: float = None
    dimensions: int = None
    detection_probability: float = None
    probability_false_alarm: float = None
