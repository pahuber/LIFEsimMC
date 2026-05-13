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
    n_planets_est_in : str
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
            n_planets_est_in: str,
            n_planets_true_in: str,
            n_test_out: str,
            pfa: float,
            pdet: float,
            n_transformation_in: Union[str, tuple[str], None] = None,
            n_image_out: str = None,
    ):
        """Constructor method.

        Parameters
        ----------
        n_setup_in : str
            Name of the input configuration resource.
        n_data_in : str
            Name of the input data resource.
        n_planets_est_in : str
            Name of the input planet parameters resource.
        n_transformation_in : Union[str, tuple[str]]
            Name of the input transformation resource.
        n_test_out : str
            Name of the output test resource.
        pfa : float
            Probability of false alarm.
        """
        self.n_setup_in = n_setup_in
        self.n_data_in = n_data_in
        self.n_test_out = n_test_out
        self.pfa = pfa
        self.n_planets_est_in = n_planets_est_in
        self.n_planets_true_in = n_planets_true_in
        self.n_transformation_in = n_transformation_in
        self.n_image_out = n_image_out
        self.pfa = pfa
        self.pdet = pdet

    def run(self, pipeline_resources: list[BaseResource]) -> tuple[TestResource]:
        """Apply the Neyman-Pearson test.

        Parameters
        ----------
        pipeline_resources : list[BaseResource]
            The resources to apply the module to.

        Returns
        -------
        TestResource
            The test resource.
        """
        print("Performing Neyman-Pearson test...")

        r_setup_in = self.get_resource_from_name(self.n_setup_in) if self.n_setup_in is not None else None
        transformations = get_transformations_from_resource_name(self, self.n_transformation_in)
        r_planets_est_in = self.get_resource_from_name(
            self.n_planets_est_in) if self.n_planets_est_in is not None else None
        r_planet_params_true_in = self.get_resource_from_name(self.n_planets_true_in)

        # Prepare data
        data_in = self.get_resource_from_name(self.n_data_in).get_data()
        data_in_flat = data_in.flatten()
        ndim = data_in_flat.numel()
        data_in_flat = data_in_flat.cpu().numpy()

        # TODO: handle mutiple planets
        flux_est = r_planets_est_in.collection[0].sed

        # TODO: Handle orbital motion
        posx_est = r_planets_est_in.collection[0].pos_x
        posy_est = r_planets_est_in.collection[0].pos_y

        # True model_est
        flux_true = r_planet_params_true_in.collection[0].planet.spectral_energy_distribution.cpu().numpy()[:, 0, 0]
        posx_true = r_planet_params_true_in.collection[0].planet.sky_coordinates[0, 0, 0, 0, 0].item()
        posy_true = r_planet_params_true_in.collection[0].planet.sky_coordinates[1, 0, 0, 0, 0].item()

        # model_est = r_setup_in.phringe.get_model_counts(
        #     spectral_energy_distribution=flux_est,
        #     x_position=posx_est,
        #     y_position=posy_est,
        #     kernels=True
        # )
        model_true = r_setup_in.phringe.get_model_counts(
            spectral_energy_distribution=flux_true,
            x_position=posx_true,
            y_position=posy_true,
            kernels=True
        )

        for transf in transformations:
            # model_est = transf(model_est)
            model_true = transf(model_true)

        # modelf_est = model_est.flatten()
        modelf_true = model_true.flatten()
        modelf_est = model_true.flatten()

        # Get test under H1 (planet present) and H0 (planet absent)
        test_h1 = (data_in_flat @ modelf_est)
        data_h0 = data_in_flat - modelf_true
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
        return r_test_out,
