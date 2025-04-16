from enum import Enum

import astropy.units as u
from phringe.core.entities.instrument import Instrument
from phringe.core.entities.perturbations.amplitude_perturbation import AmplitudePerturbation
from phringe.core.entities.perturbations.phase_perturbation import PhasePerturbation
from phringe.core.entities.perturbations.polarization_perturbation import PolarizationPerturbation
from phringe.lib.array_configuration import XArrayConfiguration
from phringe.lib.beam_combiner import DoubleBracewellBeamCombiner


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
        super().__init__(
            array_configuration_matrix=XArrayConfiguration.acm,
            complex_amplitude_transfer_matrix=DoubleBracewellBeamCombiner.catm,
            differential_outputs=DoubleBracewellBeamCombiner.diff_out,
            sep_at_max_mod_eff=DoubleBracewellBeamCombiner.sep_at_max_mod_eff,
            aperture_diameter=3.5 * u.m,
            baseline_maximum=600 * u.m,
            baseline_minimum=8 * u.m,
            spectral_resolving_power=50,
            wavelength_min=4 * u.um,
            wavelength_max=18.5 * u.um,
            throughput=0.12,
            quantum_efficiency=0.7,
        )

        if instrumental_noise == InstrumentalNoise.OPTIMISTIC:
            ampl_pert = AmplitudePerturbation(rms='0.1 %', color='pink')
            phase_pert = PhasePerturbation(rms='1.5 nm', color='pink')
            pol_pert = PolarizationPerturbation(rms='0.001 rad', color='pink')
            self.add_perturbation(ampl_pert)
            self.add_perturbation(phase_pert)
            self.add_perturbation(pol_pert)

        elif instrumental_noise == InstrumentalNoise.PESSIMISTIC:
            ampl_pert = AmplitudePerturbation(rms='1 %', color='pink')
            phase_pert = PhasePerturbation(rms='15 nm', color='pink')
            pol_pert = PolarizationPerturbation(rms='0.01 rad', color='pink')
            self.add_perturbation(ampl_pert)
            self.add_perturbation(phase_pert)
            self.add_perturbation(pol_pert)
