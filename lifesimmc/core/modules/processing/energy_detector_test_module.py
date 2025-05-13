from typing import Union

from scipy.stats import ncx2, chi2

from lifesimmc.core.modules.base_module import BaseModule
from lifesimmc.core.resources.base_resource import BaseResource
from lifesimmc.core.resources.test_resource import TestResource
from lifesimmc.util.resources import get_transformations_from_resource_name


class EnergyDetectorTestModule(BaseModule):
    """Class representation of an energy detector test module.

    Parameters
    ----------
    n_setup_in : str
        Name of the input configuration resource.
    n_data_in : str
        Name of the input data resource.
    n_planet_params_in : str
        Name of the input planet parameters resource.
    n_transformation_in : str or tuple[str]
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
            n_planet_params_in: str,
            n_test_out: str,
            pfa: float,
            n_transformation_in: Union[str, tuple[str], None] = None,
    ):
        """Constructor method.

        Parameters
        ----------
        n_setup_in : str
            Name of the input configuration resource.
        n_data_in : str
            Name of the input data resource.
        n_planet_params_in : str
            Name of the input planet parameters resource.
        n_transformation_in : str or tuple[str]
            Name of the input transformation resource.
        """
        self.n_data_in = n_data_in
        self.n_test_out = n_test_out
        self.pfa = pfa
        self.n_config_in = n_setup_in
        self.n_planet_params_in = n_planet_params_in
        self.n_transformation_in = n_transformation_in

    def apply(self, resources: list[BaseResource]) -> TestResource:
        """Apply the energy detector test.

        Parameters
        ----------
        resources : list[BaseResource]
            List of resources.

        Returns
        -------
        TestResource
            The test resource.
        """
        print("Performing energy detector test...")

        # Extract all inputs
        r_config_in = self.get_resource_from_name(self.n_config_in) if self.n_config_in is not None else None
        transformations = get_transformations_from_resource_name(self, self.n_transformation_in)
        r_planet_params_in = self.get_resource_from_name(
            self.n_planet_params_in) if self.n_planet_params_in is not None else None

        times = r_config_in.phringe.get_time_steps().cpu().numpy()
        wavelengths = r_config_in.phringe.get_wavelength_bin_centers().cpu().numpy()
        wavelength_bin_widths = r_config_in.phringe.get_wavelength_bin_widths().cpu().numpy()

        # Prepare data
        data = self.get_resource_from_name(self.n_data_in).get_data()
        dataf = data.flatten()
        ndim = dataf.numel()
        dataf = dataf.cpu().numpy()

        # Get test results under H1 (planet is present)
        test_h1 = (dataf.T @ dataf)
        xsi = ncx2.ppf(1 - self.pfa, df=ndim, nc=0)

        # TODO: handle mutiple planets
        flux = r_planet_params_in.params[0].sed.cpu().numpy()

        # TODO: Handle orbital motion
        posx = r_planet_params_in.params[0].pos_x
        posy = r_planet_params_in.params[0].pos_y

        model = r_config_in.phringe._get_template_diff_counts(
            times,
            wavelengths,
            wavelength_bin_widths,
            flux,
            posx,
            posy
        )
        for transf in transformations:
            model = transf(model)
        modelf = model.flatten()

        # Get test result under H0 (planet is not present)
        data_h0 = dataf - modelf
        test_h0 = (data_h0.T @ data_h0)
        xtx = (modelf @ modelf)
        pdet = ncx2.sf(xsi, df=ndim, nc=xtx)
        p = chi2.sf(test_h1, df=ndim)

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
