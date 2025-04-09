from typing import Union

from lifesimmc.core.modules.base_module import BaseModule
from lifesimmc.core.resources.base_resource import BaseResource


class BaseTransformationModule(BaseModule):
    """Class representation of the base transformation module.

    Parameters
    ----------

    n_data_in : str
        The name of the input data resource.
    n_template_in : str
        The name of the input template resource.
    n_data_out : str
        The name of the output data resource.
    n_template_out : str
        The name of the output template resource.
    n_transformation_out : str
        The name of the output transformation resource.
    """

    def __init__(
            self,
            n_data_in: str = None,
            n_template_in: str = None,
            n_data_out: str = None,
            n_template_out: str = None,
            n_transformation_out: str = None,
    ):
        """Constructor method.

        Parameters
        ----------
        n_data_in : str
            The name of the input data resource.
        n_template_in : str
            The name of the input template resource.
        n_data_out : str
            The name of the output data resource.
        n_template_out : str
            The name of the output template resource.
        n_transformation_out : str
            The name of the output transformation resource.
        """
        super().__init__()
        self.n_data_in = n_data_in
        self.n_template_in = n_template_in
        self.n_data_out = n_data_out
        self.n_template_out = n_template_out
        self.n_transformation_out = n_transformation_out

    def apply(self, resources: list[BaseResource]) -> Union[None, BaseResource, tuple]:
        """Apply the module.

        Parameters
        ----------
        resources : list[BaseResource]
            The resources to apply the module to.

        Returns
        -------
        Union[None, BaseResource, tuple]
            The resource or tuple of resources.
        """
        pass
