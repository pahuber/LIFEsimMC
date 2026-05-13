from rich.console import Console

from lifesimmc.core.modules.base_module import BaseModule
from lifesimmc.core.resources.base_resource import BaseResource
from lifesimmc.core.resources.data_resource import DataResource


class DataGenerationModule(BaseModule):
    """Class representation of the data generation module.

        Attributes
        ----------
        n_setup_in : str
            The name of the input setup resource.
        n_data_out : str
            The name of the output data resource.
        kernels : bool
            Whether to use kernels or not.
    """

    def __init__(self, n_setup_in: str, n_data_out: str, kernels: bool = True):
        """Constructor method.

        Parameters
        ----------
        n_setup_in : str
            The name of the input configuration resource.
        n_data_out : str
            The name of the output data resource.
        kernels : bool
            Whether to use kernels or not.
        """
        super().__init__()
        self.n_setup_in = n_setup_in
        self.n_data_out = n_data_out
        self.kernels = kernels

    def run(self, pipeline_resources: list[BaseResource]) -> tuple[DataResource]:
        """Use PHRINGE to generate synthetic data.

        Parameters
        ----------
        pipeline_resources : list[BaseResource]
            List of resources to be used in the module.

        Returns
        -------
        tuple[DataResource, PlanetParamsResource]
            Tuple containing the output data resource and planet parameters resource.
        """
        console = Console()

        with console.status("Generating synthetic data...", spinner="dots"):
            r_config_in = self.get_resource_from_name(name=self.n_setup_in)
            r_data_out = DataResource(name=self.n_data_out)

            counts = r_config_in.phringe.get_counts(kernels=self.kernels)
            r_data_out.set_data(counts)

        print('Done')
        return r_data_out,
