from typing import Union

import numpy as np
import torch
from astropy import units as u
from astropy.units import Quantity
from phringe.core.instrument import Instrument
from phringe.core.observation import Observation
from phringe.core.perturbations.power_law_psd_perturbation import PowerLawPSDPerturbation
from phringe.core.scene import Scene
from phringe.lib.baseline import OptimalNullingBaseline
from phringe.lib.beam_combiner import DoubleBracewell

from lifesimmc.core.modules.generating.data_generation_module import DataGenerationModule
from lifesimmc.core.modules.generating.template_generation_module import TemplateGenerationModule
from lifesimmc.core.modules.loading.setup_module import SetupModule
from lifesimmc.core.modules.processing.matched_filter_module import MatchedFilterModule
from lifesimmc.core.modules.processing.ml_parameter_estimation_module import MLSEDEstimationModule
from lifesimmc.core.modules.processing.neyman_pearson_test_module import NeymanPearsonTestModule
from lifesimmc.core.modules.processing.zca_whitening_module import ZCAWhiteningModule
from lifesimmc.core.pipeline import Pipeline
from lifesimmc.lib.instrument import InstrumentalNoise
from lifesimmc.presets.single_epoch_observation.single_epoch_observation import SingleEpochObservation
from lifesimmc.util.library import XArrayConfiguration
from lifesimmc.util.spectrum import convert_spectral_units


class SingleEpochObservationV1(SingleEpochObservation):
    """Single-epoch observation preset (version 1).

    Parameters
    ----------
    scene : Scene
        The astrophysical scene containing the target system (e.g., star, planets,
        and background sources).

    total_integration_time : str or float or Quantity
        Total observation time.

    detector_integration_time : str or float or Quantity, optional
        Integration time per detector frame. If None, defaults to
        ``total_integration_time / 200``.

    modulation_period : str or float or Quantity, optional
        Period of the interferometric modulation (e.g., array rotation).
        If None, defaults to ``total_integration_time``.

    solar_ecliptic_latitude : str, optional
        Solar ecliptic latitude used to estimate local zodiacal light contribution.

    nulling_baseline : str or float or Quantity, optional
        Nulling baseline of the interferometer. Can be specified directly or
        derived from an optimal baseline prescription.

    aperture_diameter : Quantity, optional
        Diameter of each telescope aperture.

    nulling_baseline_min : Quantity, optional
        Minimum allowed nulling baseline.

    nulling_baseline_max : Quantity, optional
        Maximum allowed nulling baseline.

    spectral_resolving_power : int, optional
        Spectral resolving power of the instrument.

    wavelength_min : Quantity, optional
        Minimum wavelength of the observation band.

    wavelength_max : Quantity, optional
        Maximum wavelength of the observation band.

    throughput : float, optional
        Total optical throughput of the instrument.

    quantum_efficiency : float, optional
        Detector quantum efficiency.

    instrumental_noise : InstrumentalNoise, optional
        Level of instrumental perturbations (e.g., NONE, OPTIMISTIC, PESSIMISTIC).

    template_fov_rad : float, optional
        Field of view radius used for template generation.

    seed : int, optional
        Random seed for reproducibility.

    grid_size : int, optional
        Number of pixels per spatial dimension for image-based calculations.

    device : torch.device, optional
        Compute device used for the simulation (CPU or GPU).
    """

    def __init__(
            self,
            scene: Scene,
            total_integration_time: Union[str, float, Quantity],
            detector_integration_time: Union[str, float, Quantity] = None,
            modulation_period: Union[str, float, Quantity] = None,
            solar_ecliptic_latitude: str = '0 deg',
            nulling_baseline: Union[str, float, Quantity] = OptimalNullingBaseline(
                angular_star_separation='habitable-zone',
                wavelength='15 um',
                sep_at_max_mod_eff=DoubleBracewell.sep_at_max_mod_eff[0],
            ),
            aperture_diameter: float = 3.5 * u.m,
            nulling_baseline_min: float = 10 * u.m,
            nulling_baseline_max: float = 100 * u.m,
            spectral_resolving_power: int = 50,
            wavelength_min: float = 4 * u.um,
            wavelength_max: float = 18.5 * u.um,
            throughput: float = 0.15,
            quantum_efficiency: float = 0.6,
            instrumental_noise: InstrumentalNoise = InstrumentalNoise.NONE,
            template_fov_rad: float = 1e-6,
            seed: int = None,
            grid_size: int = 40,
            device: torch.device = torch.device('cpu'),
    ):
        """Initialize the single-epoch observation preset.

        Sets up the instrument, observation configuration, and internal
        state required to run the simulation and analysis pipeline.
        """
        super().__init__()
        self.scene = scene
        self.total_integration_time = total_integration_time
        self.detector_integration_time = detector_integration_time
        self.modulation_period = modulation_period
        self.solar_ecliptic_latitude = solar_ecliptic_latitude
        self.nulling_baseline = nulling_baseline
        self.aperture_diameter = aperture_diameter
        self.nulling_baseline_min = nulling_baseline_min
        self.nulling_baseline_max = nulling_baseline_max
        self.spectral_resolving_power = spectral_resolving_power
        self.wavelength_min = wavelength_min
        self.wavelength_max = wavelength_max
        self.throughput = throughput
        self.quantum_efficiency = quantum_efficiency
        self.instrumental_noise = instrumental_noise
        self.template_fov_rad = template_fov_rad
        self.grid_size = grid_size
        self.seed = seed
        self.device = device

        self._instrument = self._create_instrument()
        self._observation = self._create_observation()
        self._pipeline = None

    def _create_instrument(self) -> Instrument:
        """Create the instrument and set the perturbations.

        Returns
        -------
        Instrument
            The instrument object.
        """
        if self.instrumental_noise == InstrumentalNoise.OPTIMISTIC:
            amplitude_perturbation = PowerLawPSDPerturbation(coefficient=1, rms=0.1 * u.percent)
            phase_perturbation = PowerLawPSDPerturbation(coefficient=1, rms=1.5 * u.nm, chromatic=True)
            polarization_perturbation = PowerLawPSDPerturbation(coefficient=1, rms=0.001 * u.rad)

        elif self.instrumental_noise == InstrumentalNoise.PESSIMISTIC:
            amplitude_perturbation = PowerLawPSDPerturbation(coefficient=1, rms=1 * u.percent)
            phase_perturbation = PowerLawPSDPerturbation(coefficient=1, rms=15 * u.nm, chromatic=True)
            polarization_perturbation = PowerLawPSDPerturbation(coefficient=1, rms=0.01 * u.rad)

        elif self.instrumental_noise == InstrumentalNoise.NONE:
            amplitude_perturbation = None
            phase_perturbation = None
            polarization_perturbation = None

        return Instrument(
            array_configuration_matrix=XArrayConfiguration.acm,
            complex_amplitude_transfer_matrix=DoubleBracewell.catm,
            kernels=DoubleBracewell.kernels,
            aperture_diameter=self.aperture_diameter,
            nulling_baseline_min=self.nulling_baseline_min,
            nulling_baseline_max=self.nulling_baseline_max,
            spectral_resolving_power=self.spectral_resolving_power,
            wavelength_min=self.wavelength_min,
            wavelength_max=self.wavelength_max,
            wavelength_bands_boundaries=[],
            throughput=self.throughput,
            quantum_efficiency=self.quantum_efficiency,
            amplitude_perturbation=amplitude_perturbation,
            phase_perturbation=phase_perturbation,
            polarization_perturbation=polarization_perturbation,
        )

    def _create_observation(self) -> Observation:
        """Create the observation.

        Returns
        -------
        Observation
            The observation object.
        """
        detector_integration_time = self.detector_integration_time \
            if self.detector_integration_time is not None \
            else self.total_integration_time / 200
        modulation_period = self.total_integration_time if self.modulation_period is None else self.modulation_period

        return Observation(
            total_integration_time=self.total_integration_time,
            detector_integration_time=detector_integration_time,
            modulation_period=modulation_period,
            solar_ecliptic_latitude='0 deg',
            nulling_baseline=self.nulling_baseline
        )

    def extract_sed(self, units: Union[str, Quantity] = 'ph/s/m3') -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        module = MLSEDEstimationModule(
            n_setup_in='setup',
            n_data_in='data_white',
            n_template_in='temp_white',
            n_transformation_in='zca',
            n_planets_in='planets_init',
            n_planets_out='planets_ml',
        )
        self._pipeline.add_module(module)
        self._pipeline.run()

        r_planets_ml = self._pipeline.get_resource('planets_ml')
        sed = r_planets_ml.collection[0].sed
        std = r_planets_ml.collection[0].std
        cov = r_planets_ml.collection[0].cov[:-2, :-2]

        units = u.Unit(units) if isinstance(units, str) else units

        sed = convert_spectral_units(
            sed,
            self.get_wavelength_bin_centers(),
            units_in='ph/s/m3',
            units_out=units,
            wavelength_units='m'
        )
        std = convert_spectral_units(
            std,
            self.get_wavelength_bin_centers(),
            units_in='ph/s/m3',
            units_out=units,
            wavelength_units='m'
        )

        cov = convert_spectral_units(
            cov,
            self.get_wavelength_bin_centers(),
            units_in=(u.ph / u.s / u.m ** 3) ** 2,
            units_out=units ** 2,
            wavelength_units='m'
        )

        return sed, std, cov

    def get_detection_significance(self) -> float:
        module = NeymanPearsonTestModule(
            n_setup_in="setup",
            n_data_in='data_white',
            n_transformation_in='zca',
            n_planets_true_in='planets_init',
            n_planets_est_in='planets_ml',
            n_test_out='test_np',
            n_image_out='imag_np',
            pfa=2.87e-7,
            pdet=0.9
        )
        self._pipeline.add_module(module)
        self._pipeline.run()

        xtx = self._pipeline.get_resource('test_np').model_length_xtx

        return np.sqrt(xtx)

    def get_input_sed(self, units: Union[str, Quantity] = 'ph/s/m3') -> np.ndarray:
        sed_ph_s_m3 = self.scene.planets[0].spectral_energy_distribution.cpu().numpy()[:, 0, 0]

        sed_converted = convert_spectral_units(
            sed_ph_s_m3,
            self.get_wavelength_bin_centers(),
            units_in='ph/s/m3',
            units_out=units,
            wavelength_units='m'
        )

        return sed_converted

    def get_matched_filter(self) -> np.ndarray:
        module = MatchedFilterModule(n_data_in='data_white', n_template_in='temp_white', n_image_out='imag_corr')
        self._pipeline.add_module(module)
        self._pipeline.run()

        return self._pipeline.get_resource('imag_corr').get_image(as_numpy=True)

    def get_wavelength_bin_centers(self) -> np.ndarray:
        return self._instrument.wavelength_bin_centers.cpu().numpy()

    def run(self):
        self._pipeline = Pipeline(device=self.device, seed=self.seed, grid_size=self.grid_size)

        module = SetupModule(
            n_setup_out='setup',
            n_planets_out='planets_init',
            scene=self.scene,
            instrument=self._instrument,
            observation=self._observation
        )
        self._pipeline.add_module(module)

        module = DataGenerationModule(n_setup_in='setup', n_data_out='data')
        self._pipeline.add_module(module)

        module = TemplateGenerationModule(n_setup_in='setup', n_template_out='temp', fov=self.template_fov_rad)
        self._pipeline.add_module(module)

        module = ZCAWhiteningModule(
            n_setup_in='setup',
            n_data_in='data',
            n_template_in='temp',
            n_data_out='data_white',
            n_template_out='temp_white',
            n_transformation_out='zca',
            diagonal_only=False
        )
        self._pipeline.add_module(module)

        self._pipeline.run()
