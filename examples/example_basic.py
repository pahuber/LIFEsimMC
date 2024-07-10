from pathlib import Path

from matplotlib import pyplot as plt

from lifesimmc.core.data_gen.data_generation_module import DataGenerationModule
from lifesimmc.core.data_gen.template_generation_module import TemplateGenerationModule
from lifesimmc.core.loading.config_loader_module import ConfigLoaderModule
from lifesimmc.core.loading.scene_loader_module import SceneLoaderModule
from lifesimmc.core.loading.template_loading_module import TemplateLoadingModule
from lifesimmc.core.pipeline import Pipeline
from lifesimmc.core.processing.mlm_exctraction_module import MLMExtractionModule
from lifesimmc.core.processing.polynomial_subtraction_module import PolynomialSubtractionModule

config_file_path = Path("config.yaml")
exoplanetary_system_file_path = Path("exoplanetary_system.yaml")

# Create pipeline
pipeline = Pipeline()

# Load configuration
module = ConfigLoaderModule(config_file_path=config_file_path)
pipeline.add_module(module)

# Load scene
module = SceneLoaderModule(exoplanetary_system_file_path=exoplanetary_system_file_path, spectrum_files=None)
pipeline.add_module(module)

# Generate data
module = DataGenerationModule(gpus=None, write_to_fits=False, create_copy=False)
pipeline.add_module(module)

# Generate templates
module = TemplateGenerationModule(gpus=None, write_to_fits=True, create_copy=True)
pipeline.add_module(module)

# Load templates
module = TemplateLoadingModule(
    template_directory=Path('path'))
# pipeline.add_module(module)

# Subtract polynomial spectral fit
module = PolynomialSubtractionModule()
pipeline.add_module(module)

# Extract fluxes using MLM
module = MLMExtractionModule()
pipeline.add_module(module)

# Run pipeline
pipeline.run()

########################################################################################################################

# Plot raw photometry data
plt.imshow(pipeline._context.data[0], cmap='Greys')
# plt.colorbar()
plt.xlabel('Time Step')
plt.ylabel('Wavelength Channel')
plt.show()

# Plot cost function
center_x, center_y = pipeline._context.extractions[0].cost_function[0].shape[1] / 2, \
                     pipeline._context.extractions[0].cost_function[0].shape[0] / 2
plt.imshow(pipeline._context.extractions[0].cost_function[0], cmap='magma')
plt.plot(center_x - 0.5, center_y - 0.5, marker='*', markersize=20, color='white')
plt.colorbar()
plt.axis('off')
plt.show()

# # Plot flux
plt.plot(pipeline._context.extractions[0].flux[0])
plt.xlabel('Wavelength (um)')
plt.ylabel('Flux (W/m^2/um)')
plt.show()
