from abc import abstractmethod

import numpy as np

from lifesimmc.presets.base_preset import BasePreset


class SingleEpochObservation(BasePreset):
    _IS_FACTORY = True

    @classmethod
    def _get_preset_mappings(cls):
        from lifesimmc.presets.single_epoch_observation.versions.single_epoch_observation_v1 import (
            SingleEpochObservationV1,
        )

        return {
            "latest": SingleEpochObservationV1,
            "1": SingleEpochObservationV1,
        }

    @abstractmethod
    def extract_sed(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Get the extracted SED and associated uncertainties of the observation.

        Returns
        -------
        tuple[np.ndarray, np.ndarray, np.ndarray]
            A tuple containing the SED, standard deviations, and covariance matrix.
        """
        pass

    @abstractmethod
    def get_detection_significance(self) -> float:
        pass

    @abstractmethod
    def get_input_sed(self) -> np.ndarray:
        """Get the input SED of the observation.

        Returns
        -------
        np.ndarray
            The input SED of the planet.
        """
        pass

    @abstractmethod
    def get_matched_filter(self) -> np.ndarray:
        pass

    @abstractmethod
    def get_wavelength_bin_centers(self) -> np.ndarray:
        pass
