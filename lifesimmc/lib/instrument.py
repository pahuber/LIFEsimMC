from enum import Enum

import astropy.units as u
from phringe.core.instrument import Instrument
from phringe.core.perturbations.power_law_psd_perturbation import PowerLawPSDPerturbation
from phringe.lib.array_configuration import XArrayConfiguration
from phringe.lib.beam_combiner import DoubleBracewell


class InstrumentalNoise(Enum):
    """Enum class for instrumental noise types.

    Parameters
    ----------
    instrumental_noise : str
        The type of instrumental noise.
    """
    NONE = 0
    OPTIMISTIC = 1
    PESSIMISTIC = 2


class LIFEReferenceDesign(Instrument):
    def __init__(self, instrumental_noise: InstrumentalNoise = InstrumentalNoise.NONE):

        if instrumental_noise == InstrumentalNoise.OPTIMISTIC:
            amplitude_perturbation = PowerLawPSDPerturbation(coefficient=1, rms=0.1 * u.percent)
            phase_perturbation = PowerLawPSDPerturbation(coefficient=1, rms=1.5 * u.nm, chromatic=True)
            polarization_perturbation = PowerLawPSDPerturbation(coefficient=1, rms=0.001 * u.rad)

        elif instrumental_noise == InstrumentalNoise.PESSIMISTIC:
            amplitude_perturbation = PowerLawPSDPerturbation(coefficient=1, rms=1 * u.percent)
            phase_perturbation = PowerLawPSDPerturbation(coefficient=1, rms=15 * u.nm, chromatic=True)
            polarization_perturbation = PowerLawPSDPerturbation(coefficient=1, rms=0.01 * u.rad)

        elif instrumental_noise == InstrumentalNoise.NONE:
            amplitude_perturbation = None
            phase_perturbation = None
            polarization_perturbation = None

        super().__init__(
            array_configuration_matrix=XArrayConfiguration.acm,
            complex_amplitude_transfer_matrix=DoubleBracewell.catm,
            kernels=DoubleBracewell.kernels,
            aperture_diameter=3.5 * u.m,
            nulling_baseline_min=10 * u.m,
            nulling_baseline_max=60 * u.m,
            spectral_resolving_power=50,
            wavelength_min=4 * u.um,
            wavelength_max=18.5 * u.um,
            wavelength_bands_boundaries=[],
            throughput=0.12,
            quantum_efficiency=0.7,
            amplitude_perturbation=amplitude_perturbation,
            phase_perturbation=phase_perturbation,
            polarization_perturbation=polarization_perturbation,
        )
