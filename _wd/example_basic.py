from pathlib import Path

from lifesimmc.core.modules.generating.data_generation_module import DataGenerationModule
from lifesimmc.core.modules.generating.template_generation_module import TemplateGenerationModule
from lifesimmc.core.modules.loading.config_loader_module import ConfigLoaderModule
from lifesimmc.core.modules.processing.analytical_mle_module import AnalyticalMLEModule
from lifesimmc.core.modules.processing.covariance_calculation_module import CovarianceCalculationModule
from lifesimmc.core.modules.processing.mcmc_module import MCMCModule
from lifesimmc.core.modules.processing.numerical_mle_module import NumericalMLEModule
from lifesimmc.core.modules.processing.whitening_module import WhiteningModule
from lifesimmc.core.modules.testing.energy_detector_test_module import EnergyDetectorTestModule
from lifesimmc.core.modules.testing.neyman_pearson_test_module import NeymanPearsonTestModule
from lifesimmc.core.pipeline import Pipeline

config_file_path = Path("config.py")

# Create pipeline
pipeline = Pipeline(gpu=4)

# Load configuration
module = ConfigLoaderModule(r_config_out='conf', config_file_path=config_file_path)
pipeline.add_module(module)

# Generate data
module = DataGenerationModule(
    r_config_in='conf',
    r_data_out='data',
    r_spectrum_out='speci',
    write_to_fits=False,
    create_copy=False
)
pipeline.add_module(module)

# Generate templates
module = TemplateGenerationModule(
    r_config_in='conf',
    r_template_out='temp',
    write_to_fits=False,
    create_copy=False
)
pipeline.add_module(module)

# Load templates
# module = TemplateLoadingModule(
#     template_directory=Path('/home/huberph/lifesimmc/_wd/templates_40x40'))
# pipeline.add_module(module)

# Calculate covariance of data
module = CovarianceCalculationModule(config_in='conf', cov_out='cov')
pipeline.add_module(module)

# Whiten data and templates
module = WhiteningModule(
    r_config_in='conf',
    r_cov_in='cov',
    r_data_in='data',
    r_template_in='temp',
    r_data_out='dataw',
    r_template_out='tempw'
)
pipeline.add_module(module)

# Estimate flux using analytical MLE
module = AnalyticalMLEModule(
    r_config_in='conf',
    r_data_in='dataw',
    r_template_in='tempw',
    r_image_out='imag',
    r_spectrum_out='speca'
)
pipeline.add_module(module)

# Estimate flux using numerical MLE
module = NumericalMLEModule(
    r_config_in='conf',
    r_data_in='dataw',
    r_cov_in='cov',
    r_spectrum_out='specn'
)
# pipeline.add_module(module)

# Estimate flux using MCMC # TODO: BB, Full, init positions
module = MCMCModule(
    r_config_in='conf',
    r_data_in='dataw',
    r_cov_in='cov',
    r_spectrum_in='speci',
    r_spectrum_out='specm'
)
# pipeline.add_module(module)

# Perform energy detector test
module = EnergyDetectorTestModule(r_data_in='dataw', r_test_out='teste', pfa=0.05)
pipeline.add_module(module)

# Perform Neyman-Pearson test
module = NeymanPearsonTestModule(
    r_config_in='conf',
    r_cov_in='cov',
    r_data_in='dataw',
    r_spectrum_in='speci',
    r_test_out='testn',
    pfa=0.05
)
pipeline.add_module(module)

# Run pipeline
pipeline.run()
