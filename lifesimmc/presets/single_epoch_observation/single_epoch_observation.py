from abc import abstractmethod
from typing import Union

import numpy as np
from astropy.units import Quantity

from lifesimmc.presets.base_preset import BasePreset


class SingleEpochObservation(BasePreset):
    """Abstract class representing a single epoch observation."""
    _IS_FACTORY = True
    _LATEST_VERSION = "1"

    @classmethod
    def _get_preset_mappings(cls) -> dict[str, type]:
        from lifesimmc.presets.single_epoch_observation.versions.single_epoch_observation_v1 import (
            SingleEpochObservationV1,
        )

        return {
            "1": SingleEpochObservationV1,
        }

    @abstractmethod
    def extract_sed(self, units: Union[str, Quantity] = 'ph/s/m3') -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Get the extracted SED and associated uncertainties of the observation.

        Parameters
        ----------
        Union[str, Quantity]
            The units to return the input SED in. Defaults to 'ph/s/m3'.

        Returns
        -------
        tuple[np.ndarray, np.ndarray, np.ndarray]
            A tuple containing the SED, standard deviations, and covariance matrix.
        """
        pass

    @abstractmethod
    def get_detection_significance(self) -> float:
        """Get the detection significance of the observation based on the Neyman-Pearson test.

        Returns
        -------
        float
            The detection significance of the observation in units of sigma.
        """
        pass

    @abstractmethod
    def get_input_sed(self, units: Union[str, Quantity] = 'ph/s/m3') -> np.ndarray:
        """Get the input SED of the observation.

        Parameters
        ----------
        Union[str, Quantity]
            The units to return the input SED in. Defaults to 'ph/s/m3'.

        Returns
        -------
        np.ndarray
            The input SED of the planet.
        """
        pass

    @abstractmethod
    def get_matched_filter(self) -> np.ndarray:
        """Get the matched filter map for the observation. This returns a map containing the matched filter values at
        each point in the FOV based on data and normalized templates.

        Returns
        np.ndarray
            The matched filter map.
        """
        pass

    @abstractmethod
    def get_wavelength_bin_centers(self) -> np.ndarray:
        """Get the wavelength bin centers of the observation.

        Returns
        -------
        np.ndarray
            The wavelength bin centers.
        """
        pass
