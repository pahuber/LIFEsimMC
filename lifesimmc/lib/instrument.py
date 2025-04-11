import astropy.units as u
from phringe.core.entities.instrument import Instrument
from phringe.lib.array_configuration import XArrayConfiguration
from phringe.lib.beam_combiner import DoubleBracewellBeamCombiner


class LIFEIdeal(Instrument):
    def __init__(self):
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


class LIFEPerturbedOptimistic(Instrument):
    def __init__(self):
        super().__init__(
            array_configuration_matrix=XArrayConfiguration.acm,
            complex_amplitude_transfer_matrix=DoubleBracewellBeamCombiner.catm,
            differential_outputs=DoubleBracewellBeamCombiner.diff_out,
            sep_at_max_mod_eff=DoubleBracewellBeamCombiner.sep_at_max_mod_eff,
            aperture_diameter=3 * u.m,
            baseline_maximum=600 * u.m,
            baseline_minimum=8 * u.m,
            spectral_resolving_power=50,
            wavelength_min=4 * u.um,
            wavelength_max=18.5 * u.um,
            throughput=0.12,
            quantum_efficiency=0.7,
            perturbations={
                'amplitude': {
                    'rms': '0.1 %',
                    'color': 'pink',
                },
                'phase': {
                    'rms': '1.5 nm',
                    'color': 'pink',
                },
                'polarization': {
                    'rms': '0.001 rad',
                    'color': 'pink',
                },
            }
        )


class LIFEPerturbedPessimistic(Instrument):
    def __init__(self):
        super().__init__(
            array_configuration_matrix=XArrayConfiguration.acm,
            complex_amplitude_transfer_matrix=DoubleBracewellBeamCombiner.catm,
            differential_outputs=DoubleBracewellBeamCombiner.diff_out,
            sep_at_max_mod_eff=DoubleBracewellBeamCombiner.sep_at_max_mod_eff,
            aperture_diameter=3 * u.m,
            baseline_maximum=600 * u.m,
            baseline_minimum=8 * u.m,
            spectral_resolving_power=50,
            wavelength_min=4 * u.um,
            wavelength_max=18.5 * u.um,
            throughput=0.12,
            quantum_efficiency=0.7,
            perturbations={
                'amplitude': {
                    'rms': '1 %',
                    'color': 'pink',
                },
                'phase': {
                    'rms': '15 nm',
                    'color': 'pink',
                },
                'polarization': {
                    'rms': '0.01 rad',
                    'color': 'pink',
                },
            }
        )
