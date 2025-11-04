from typing import Union

from astropy.units import Quantity
from phringe.core.observation import Observation
from phringe.util.baseline import OptimalNullingBaseline


class LIFEReferenceObservation(Observation):
    def __init__(
            self,
            total_integration_time: Union[str, float, Quantity],
            detector_integration_time: Union[str, float, Quantity],
            nulling_baseline: Union[str, float, Quantity, OptimalNullingBaseline],
    ):
        """Constructor method.

        Parameters
        ----------
        total_integration_time : str or float or Quantity
            The total integration time in units of time.
        detector_integration_time : str or float or Quantity
            The detector integration time in units of time.
        optimized_star_separation : str or float or Quantity
            The optimized star separation in units of angle or the string 'habitable-zone'.
        """
        super().__init__(
            total_integration_time=total_integration_time,
            detector_integration_time=detector_integration_time,
            modulation_period=total_integration_time,
            solar_ecliptic_latitude='0 deg',
            nulling_baseline=nulling_baseline
        )
