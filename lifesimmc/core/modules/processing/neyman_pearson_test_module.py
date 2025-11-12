from typing import Union

import numpy as np
from scipy.stats import norm

from lifesimmc.core.modules.base_module import BaseModule
from lifesimmc.core.resources.base_resource import BaseResource
from lifesimmc.core.resources.test_resource import TestResource
from lifesimmc.util.resources import get_transformations_from_resource_name


class NeymanPearsonTestModule(BaseModule):
    """Class representation of a Neyman-Pearson test module.

    Parameters
    ----------
    n_setup_in : str
        Name of the input configuration resource.
    n_data_in : str
        Name of the input data resource.
    n_planet_params_est_in : str
        Name of the input planet parameters resource.
    n_transformation_in : Union[str, tuple[str]]
        Name of the input transformation resource.
    n_test_out : str
        Name of the output test resource.
    pfa : float
        Probability of false alarm.
    """

    def __init__(
            self,
            n_setup_in: str,
            n_data_in: str,
            n_planet_params_est_in: str,
            n_planet_params_true_in: str,
            n_test_out: str,
            pfa: float,
            n_transformation_in: Union[str, tuple[str], None] = None,
            n_templates_in: str = None,
            n_image_out: str = None,
    ):
        """Constructor method.

        Parameters
        ----------
        n_setup_in : str
            Name of the input configuration resource.
        n_data_in : str
            Name of the input data resource.
        n_planet_params_est_in : str
            Name of the input planet parameters resource.
        n_transformation_in : Union[str, tuple[str]]
            Name of the input transformation resource.
        n_test_out : str
            Name of the output test resource.
        pfa : float
            Probability of false alarm.
        """
        self.n_data_in = n_data_in
        self.n_test_out = n_test_out
        self.pfa = pfa
        self.n_config_in = n_setup_in
        self.n_planet_params_est_in = n_planet_params_est_in
        self.n_planet_params_true_in = n_planet_params_true_in
        self.n_transformation_in = n_transformation_in
        self.n_image_out = n_image_out
        self.pfa = pfa

    def apply(self, resources: list[BaseResource]) -> TestResource:
        """Apply the Neyman-Pearson test.

        Parameters
        ----------
        resources : list[BaseResource]
            The resources to apply the module to.

        Returns
        -------
        TestResource
            The test resource.
        """
        print("Performing Neyman-Pearson test...")

        # Extract all inputs
        r_config_in = self.get_resource_from_name(self.n_config_in) if self.n_config_in is not None else None
        transformations = get_transformations_from_resource_name(self, self.n_transformation_in)
        r_planet_params_est_in = self.get_resource_from_name(
            self.n_planet_params_est_in) if self.n_planet_params_est_in is not None else None
        r_planet_params_true_in = self.get_resource_from_name(self.n_planet_params_true_in)

        # Prepare data
        data = self.get_resource_from_name(self.n_data_in).get_data()
        dataf = data.flatten()
        ndim = dataf.numel()
        dataf = dataf.cpu().numpy()

        # TODO: handle mutiple planets
        flux_est = r_planet_params_est_in.params[0].sed.cpu().numpy()
        # TODO: Handle orbital motion
        posx_est = r_planet_params_est_in.params[0].pos_x
        posy_est = r_planet_params_est_in.params[0].pos_y

        # True model_est
        flux_true = r_planet_params_true_in.params[0].sed.cpu().numpy()
        posx_true = r_planet_params_true_in.params[0].pos_x
        posy_true = r_planet_params_true_in.params[0].pos_y

        model_est = r_config_in.phringe.get_model_counts(
            spectral_energy_distribution=flux_est,
            x_position=posx_est,
            y_position=posy_est,
            kernels=True
        )
        model_true = r_config_in.phringe.get_model_counts(
            spectral_energy_distribution=flux_true,
            x_position=posx_true,
            y_position=posy_true,
            kernels=True
        )
        for transf in transformations:
            model_est = transf(model_est)
            model_true = transf(model_true)
        modelf_est = model_est.flatten()
        modelf_true = model_true.flatten()

        # Get test under H1 (planet present) and H0 (planet absent)
        test_h1 = (dataf @ modelf_est)
        data_h0 = dataf - modelf_true
        test_h0 = (data_h0 @ modelf_est)
        xtx = modelf_true @ modelf_true
        xsi = np.sqrt(xtx) * norm.ppf(1 - self.pfa)
        pdet = 1 - norm.cdf((xsi - xtx) / np.sqrt(xtx))
        p = norm.sf(test_h1 / np.sqrt(xtx))

        r_test_out = TestResource(
            name=self.n_test_out,
            test_statistic_h1=test_h1,
            test_statistic_h0=test_h0,
            threshold_xsi=xsi,
            model_length_xtx=xtx,
            dimensions=ndim,
            detection_probability=pdet,
            probability_false_alarm=self.pfa,
            p_value=p,
        )

        print('Done')
        return r_test_out
