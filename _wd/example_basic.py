from pathlib import Path

from lifesimmc.core.data_gen.data_generation_module import DataGenerationModule
from lifesimmc.core.loading.config_loader_module import ConfigLoaderModule
from lifesimmc.core.pipeline import Pipeline
from lifesimmc.core.processing.mcmc_extraction_module import MCMCExtractionModule
from lifesimmc.core.processing.mcmc_extraction_module_0 import MCMCBBFittingModule
from lifesimmc.core.processing.ml_detection_2 import MLDetectionModule2
from lifesimmc.core.processing.ml_detection_module_cov import MLDetectionModuleCov
from lifesimmc.core.processing.polynomial_subtraction_module import PolynomialSubtractionModule
from lifesimmc.core.processing.spectrum_fitting_module import SpectrumFittingModule

config_file_path = Path("config.py")

# Create pipeline
pipeline = Pipeline()

# Load configuration
module = ConfigLoaderModule(name='config', config_file_path=config_file_path)
pipeline.add_module(module)

# Generate data
module = DataGenerationModule(name='data_gen', gpu=4, write_to_fits=False, create_copy=False)
pipeline.add_module(module)

# Generate templates
# module = TemplateGenerationModule(gpu=4, write_to_fits=True, create_copy=True)
# pipeline.add_module(module)

# Load templates
# module = TemplateLoadingModule(
#     template_directory=Path('/home/huberph/lifesimmc/_wd/templates_40x40'))
# pipeline.add_module(module)

# Subtract polynomial spectral fit
module = PolynomialSubtractionModule()
# pipeline.add_module(module)

# Extract fluxes using MLM
# module = MLDetectionModule3()  # grid numerical ML full flux + np test, semiuseful
# module = MLDetectionModule2_BB()  # cov subtraction, energy detector, np test, num ML estimation bb
module = MLDetectionModuleCov()  # cov subtraction, grid search, analytical ML full flux + NP test
module = MLDetectionModule2()  # cov subtraction, energy detector, np test, num ML estimation full flux
pipeline.add_module(module)

# Get MCMC flux estimation
module = MCMCBBFittingModule(gpus=(4,))
# pipeline.add_module(module)

module = MCMCExtractionModule(gpus=(4,))
# pipeline.add_module(module)

# Blackbody fitting
module = SpectrumFittingModule()
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
