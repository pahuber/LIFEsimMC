from phringe.lib.array_architecture import LIFEBaselineArchitecture

config = {
    'observation': {
        'solar_ecliptic_latitude': '0 deg',
        'total_integration_time': '10 d',
        'detector_integration_time': '0.1 d',
        'modulation_period': '10 d',
        'optimized_differential_output': 0,
        'optimized_star_separation': '0.1 arcsec',
        'optimized_wavelength': '10 um',
    },
    'instrument': {
        'array_configuration_matrix': LIFEBaselineArchitecture.acm,
        'complex_amplitude_transfer_matrix': LIFEBaselineArchitecture.catm,
        'differential_outputs': LIFEBaselineArchitecture.diff_out,
        'sep_at_max_mod_eff': LIFEBaselineArchitecture.sep_at_max_mod_eff,
        'aperture_diameter': '3.5 m',
        'baseline_maximum': '600 m',
        'baseline_minimum': '5 m',
        'spectral_resolving_power': 50,
        'wavelength_min': '4 um',
        'wavelength_max': '18.5 um',
        'throughput': 0.12,
        'quantum_efficiency': 0.7,
        'perturbations': {
            # 'amplitude': {
            #     'rms': '0.1 %',
            #     'color': 'pink',
            # },
            # 'phase': {
            #     'rms': '1.5 nm',
            #     'color': 'pink',
            # },
            # 'polarization': {
            #     'rms': '0.001 rad',
            #     'color': 'pink',
            # },
        }
    },
    'scene': {
        'star': {
            'name': 'Sun',
            'distance': '10 pc',
            'mass': '1 Msun',
            'radius': '1 Rsun',
            'temperature': '5700 K',
            'right_ascension': '10 hourangle',
            'declination': '45 deg',
        },
        'exozodi': {
            'level': 3
        },
        'local_zodi': {},
        'planets': [
            {
                'name': 'Earth',
                'has_orbital_motion': False,
                'mass': '1 Mearth',
                'radius': '1 Rearth',
                'temperature': '254 K',
                'semi_major_axis': '1 au',
                'eccentricity': '0',
                'inclination': '70 deg',
                'raan': '90 deg',
                'argument_of_periapsis': '0 deg',
                'true_anomaly': '45 deg',
                'input_spectrum': None,
                # 'host_star_distance': '10 pc',
                # 'host_star_mass': '1 Msun',
            },
            # {
            #     'name': 'Mars',
            #     'mass': '1 Mearth',
            #     'radius': '1 Rearth',
            #     'temperature': '288 K',
            #     'semi_major_axis': '1 au',
            #     'eccentricity': '0',
            #     'inclination': '0 deg',
            #     'raan': '0 deg',
            #     'argument_of_periapsis': '45 deg',
            #     'true_anomaly': '0 deg',
            #     'input_spectrum': None
            # },
        ],
    },
}
