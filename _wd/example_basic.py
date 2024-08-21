from pathlib import Path

from lifesimmc.core.modules.generating.data_generation_module import DataGenerationModule
from lifesimmc.core.modules.generating.template_generation_module import TemplateGenerationModule
from lifesimmc.core.modules.loading.config_loader_module import ConfigLoaderModule
from lifesimmc.core.modules.processing.analytical_mle_module import AnalyticalMLEModule
from lifesimmc.core.modules.processing.covariance_calculation_module import CovarianceCalculationModule
from lifesimmc.core.modules.processing.mcmc_extraction_module import MCMCModule
from lifesimmc.core.modules.processing.numerical_mle_module import NumericalMLEModule
from lifesimmc.core.modules.processing.whitening_module import WhiteningModule
from lifesimmc.core.modules.testing.energy_detector_test_module import EnergyDetectorTestModule
from lifesimmc.core.pipeline import Pipeline

config_file_path = Path("config.py")

# Create pipeline
pipeline = Pipeline(gpu=4)

# Load configuration
module = ConfigLoaderModule(config_out='conf', config_file_path=config_file_path)
pipeline.add_module(module)

# Generate data
module = DataGenerationModule(
    config_in='conf',
    data_out='data',
    spectrum_out='speci',
    write_to_fits=False,
    create_copy=False
)
pipeline.add_module(module)

# Generate templates
module = TemplateGenerationModule(
    config_in='conf',
    template_out='temp',
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
    cov_in='cov',
    config_in='conf',
    data_in='data',
    template_in='temp',
    data_out='dataw',
    template_out='tempw'
)
pipeline.add_module(module)

# Estimate flux using analytical MLE
module = AnalyticalMLEModule(
    config_in='conf',
    data_in='dataw',
    template_in='tempw',
    image_out='imag',
    spectrum_out='speca'
)
pipeline.add_module(module)

# Estimate flux using numerical MLE
module = NumericalMLEModule(
    config_in='conf',
    data_in='dataw',
    cov_in='cov',
    spectrum_out='specn'
)
# pipeline.add_module(module)

# Estimate flux using MCMC # TODO: BB, Full, init positions
module = MCMCModule(
    config_in='conf',
    data_in='dataw',
    cov_in='cov',
    spectrum_in='speci',
    spectrum_out='specm'
)
# pipeline.add_module(module)

# Perform energy detector test
module = EnergyDetectorTestModule(data_in='dataw', test_out='teste', pfa=0.05)
pipeline.add_module(module)

# Extract fluxes using MLM
# module = MLDetectionModule3()  # grid numerical ML full flux + np test, semiuseful
# module = MLDetectionModule2_BB()  # cov subtraction, energy detector, np test, num ML estimation bb
# module = MLDetectionModuleCov()  # cov subtraction, grid search, analytical ML full flux + NP test
# module = MLDetectionModule2()  # cov subtraction, energy detector, np test, num ML estimation full flux
# pipeline.add_module(module)
#
# # Get MCMC flux estimation
# module = MCMCBBFittingModule(gpus=(4,))
# # pipeline.add_module(module)
#
# module = MCMCExtractionModule(gpus=(4,))
# # pipeline.add_module(module)
#
# # Blackbody fitting
# module = SpectrumFittingModule()
# pipeline.add_module(module)

# Run pipeline
pipeline.run()

########################################################################################################################

# Plot raw photometry data
# plt.imshow(pipeline._context.data[0], cmap='Greys')
# # plt.colorbar()
# plt.xlabel('Time Step')
# plt.ylabel('Wavelength Channel')
# plt.colorbar()
# plt.show()
#
# # Plot cost function
# center_x, center_y = pipeline._context.extractions[0].cost_function[0].shape[1] / 2, \
#                      pipeline._context.extractions[0].cost_function[0].shape[0] / 2
# plt.imshow(pipeline._context.extractions[0].cost_function[0], cmap='magma')
# plt.plot(center_x - 0.5, center_y - 0.5, marker='*', markersize=20, color='white')
# plt.colorbar()
# plt.axis('off')
# plt.show()
#
# # # Plot flux
# plt.plot(pipeline._context.extractions[0].flux[0])
# plt.xlabel('Wavelength (um)')
# plt.ylabel('Flux (W/m^2/um)')
# plt.show()
