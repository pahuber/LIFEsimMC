from typing import Union

from astropy.units import Quantity
from phringe.core.entities.observation import Observation


class LIFEObservation(Observation):
    def __init__(
            self,
            total_integration_time: Union[str, float, Quantity],
            detector_integration_time: Union[str, float, Quantity],
            optimized_star_separation: Union[str, float, Quantity],
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
            optimized_differential_output=0,
            optimized_star_separation=optimized_star_separation,
            optimized_wavelength='15 um'
        )
