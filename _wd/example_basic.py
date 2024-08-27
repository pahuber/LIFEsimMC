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
module = ConfigLoaderModule(n_config_out='conf', config_file_path=config_file_path)
pipeline.add_module(module)

# Generate data
module = DataGenerationModule(
    n_config_in='conf',
    n_data_out='data',
    n_flux_out='speci',
    write_to_fits=False,
    create_copy=False
)
pipeline.add_module(module)

# Generate templates
module = TemplateGenerationModule(
    n_config_in='conf',
    n_template_out='temp',
    write_to_fits=False,
    create_copy=False
)
pipeline.add_module(module)

# Load templates
# module = TemplateLoadingModule(
#     template_directory=Path('/home/huberph/lifesimmc/_wd/templates_40x40'))
# pipeline.add_module(module)

# Calculate covariance of data
module = CovarianceCalculationModule(n_config_in='conf', n_cov_out='cov')
pipeline.add_module(module)

# Whiten data and templates
module = WhiteningModule(
    n_config_in='conf',
    n_cov_in='cov',
    n_data_in='data',
    n_template_in='temp',
    n_data_out='dataw',
    n_template_out='tempw'
)
pipeline.add_module(module)

# Estimate flux using analytical MLE
module = AnalyticalMLEModule(
    n_config_in='conf',
    n_data_in='dataw',
    n_template_in='tempw',
    n_image_out='imag',
    n_flux_out='speca'
)
pipeline.add_module(module)

# Estimate flux using numerical MLE
module = NumericalMLEModule(
    n_config_in='conf',
    n_data_in='dataw',
    n_cov_in='cov',
    n_flux_out='specn'
)
# pipeline.add_module(module)

# Estimate flux using MCMC # TODO: BB, Full, init positions
module = MCMCModule(
    n_config_in='conf',
    n_data_in='dataw',
    n_cov_in='cov',
    n_flux_in='speci',
    n_flux_out='specm'
)
# pipeline.add_module(module)

# Perform energy detector test
module = EnergyDetectorTestModule(n_data_in='dataw', n_test_out='teste', pfa=0.05)
pipeline.add_module(module)

# Perform Neyman-Pearson test
module = NeymanPearsonTestModule(
    n_config_in='conf',
    n_cov_in='cov',
    n_data_in='dataw',
    n_flux_in='speci',
    n_test_out='testn',
    pfa=0.05
)
pipeline.add_module(module)

# Run pipeline
pipeline.run()
