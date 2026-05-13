from phringe.util.grid import get_meshgrid

from lifesimmc.core.modules.base_module import BaseModule
from lifesimmc.core.resources.base_resource import BaseResource
from lifesimmc.core.resources.template_resource import TemplateResource


class TemplateGenerationModule(BaseModule):
    """Class representation of the template generation module to generate templates of planets with unit flux.

    Parameters
    ----------
    n_setup_in : str
        The name of the input configuration resource
    n_template_out : str
        The name of the output template resource collection
    fov : float
        The field of view for which to generate the templates in radians
    """

    def __init__(self, n_setup_in: str, n_template_out: str, fov: float):
        """Constructor method.

        Parameters
        ----------
        n_setup_in : str
            The name of the input configuration resource
        n_template_out : str
            The name of the output template resource collection
        fov : float
            The field of view for which to generate the templates in radians
        """
        super().__init__()
        self.n_setup_in = n_setup_in
        self.n_template_out = n_template_out
        self.fov = fov

    def run(self, pipeline_resources: list[BaseResource]) -> tuple[TemplateResource]:
        """Generate templates for a planet at each point in the grid.

        Parameters
        ----------
        pipeline_resources : list[BaseResource]
            List of resources to be used in the module.

        Returns
        -------
        TemplateResource
            The generated template resource.
        """
        print('Generating templates...')

        r_setup_in = self.get_resource_from_name(self.n_setup_in)

        response = r_setup_in.phringe.get_instrument_response(
            fov=self.fov,
            kernels=True,
            perturbations=False
        )

        model_counts = (
                response
                * r_setup_in.phringe._observation.detector_integration_time
                * r_setup_in.phringe.get_wavelength_bin_widths()[None, :, None, None, None]
        )

        r_template_out = TemplateResource(
            name=self.n_template_out,
            grid_coordinates=get_meshgrid(self.fov, self.grid_size, self.device),
        )
        r_template_out.set_data(model_counts)

        print('Done')
        return r_template_out,
