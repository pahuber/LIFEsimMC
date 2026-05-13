import numpy as np
import torch

from lifesimmc.core.modules.base_module import BaseModule
from lifesimmc.core.resources.base_resource import BaseResource
from lifesimmc.core.resources.image_resource import ImageResource


class RGBImageModule(BaseModule):
    """Class representation of a RGB image module.

    Parameters
    ----------
    n_setup_in : str
        Name of the input configuration resource.
    n_data_in : str
        Name of the input data resource.
    n_template_in : str
        Name of the input template resource.
    n_image_out : str
        Name of the output image resource.
    metric : int, optional
        Metric to use for the image generation. 0 for correlation, 1 for MLE. Default is 0.
    """

    def __init__(
            self,
            n_setup_in: str,
            n_data_in: str,
            n_template_in: str,
            n_image_out: str,
            metric: int = 0,
    ):
        """Constructor method.

        Parameters
        ----------
        n_setup_in : str
            Name of the input configuration resource.
        n_data_in : str
            Name of the input data resource.
        n_template_in : str
            Name of the input template resource.
        n_image_out : str
            Name of the output image resource.
        metric : int, optional
            Metric to use for the image generation. 0 for correlation, 1 for MLE. Default is 0.
        """
        super().__init__()
        self.n_config_in = n_setup_in
        self.n_data_in = n_data_in
        self.n_template_in = n_template_in
        self.n_image_out = n_image_out
        self.metric = metric

    def run(self, pipeline_resources: list[BaseResource]) -> ImageResource:
        """Apply the RGB image module to generate a false color RGB image of the scene.

        Parameters
        ----------
        pipeline_resources : list[BaseResource]
            List of resources to apply the module to.

        Returns
        -------
        ImageResource
            The generated RGB image resource.
        """
        print('Generating RGB image...')

        def stretch_frame(rgb, lo, hi,
                          stretch="asinh",
                          asinh_Q=10.0,
                          gamma=2.2,
                          eps=1e-12):
            """
            rgb: (H, W, 3) numpy array
            returns: stretched rgb in [0,1]
            """

            rgb = np.nan_to_num(rgb, nan=0.0, posinf=0.0, neginf=0.0)
            rgb = np.clip(rgb, 0.0, None)

            # luminance
            I = np.mean(rgb, axis=-1)

            # clip + normalise
            I = np.clip(I, lo, hi)
            I = (I - lo) / (hi - lo + eps)

            # stretch
            if stretch == "asinh":
                I = np.arcsinh(asinh_Q * I) / np.arcsinh(asinh_Q)
            elif stretch == "gamma":
                I = I ** (1.0 / gamma)
            elif stretch == "linear":
                pass
            else:
                raise ValueError(f"Unknown stretch '{stretch}'")

            # colour-preserving rescale
            scale = I / (np.mean(rgb, axis=-1) + eps)
            rgb_out = rgb * scale[..., None]

            return np.clip(rgb_out, 0.0, 1.0)

        r_config_in = self.get_resource_from_name(self.n_config_in)
        r_templates_in = self.get_resource_from_name(self.n_template_in)
        data_in = self.get_resource_from_name(self.n_data_in).get_data()
        template_data = r_templates_in.get_data()

        frac_red = 0.3
        frac_blue = 0.6
        wl = r_config_in.phringe.get_wavelength_bin_centers().cpu().numpy()
        wl_min = wl[0]
        wl_max = wl[-1]

        wl_1_2 = wl_min + (wl_max - wl_min) * frac_red
        wl_2_3 = wl_min + (wl_max - wl_min) * frac_blue
        i_min = 0
        i_max = len(wl) - 1
        i_1_2 = min(range(len(wl)), key=lambda i: abs(wl[i] - wl_1_2))
        i_2_3 = min(range(len(wl)), key=lambda i: abs(wl[i] - wl_2_3))

        if self.metric == 0:
            datab = data_in[:, i_min:i_1_2, :]
            datag = data_in[:, i_1_2:i_2_3, :]
            datar = data_in[:, i_2_3:i_max, :]

            template_datab = template_data[:, i_min:i_1_2, :, :, :]
            template_datag = template_data[:, i_1_2:i_2_3, :, :, :]
            template_datar = template_data[:, i_2_3:i_max, :, :, :]

            yb = datab.flatten()
            yg = datag.flatten()
            yr = datar.flatten()

            xb = template_datab.reshape(
                -1,
                template_datab.shape[-1],
                template_datab.shape[-1]
            )
            xg = template_datag.reshape(
                -1,
                template_datag.shape[-1],
                template_datag.shape[-1]
            )
            xr = template_datar.reshape(
                -1,
                template_datar.shape[-1],
                template_datar.shape[-1]
            )

            imageb = (
                    torch.einsum('i,ijk->jk', yb, xb)
                    / torch.sqrt(torch.einsum('i, i->', yb, yb))
                    / torch.sqrt(torch.einsum('ijk,ijk->', xb, xb))
            )
            imageg = (
                    torch.einsum('i,ijk->jk', yg, xg)
                    / torch.sqrt(torch.einsum('i, i->', yg, yg))
                    / torch.sqrt(torch.einsum('ijk,ijk->', xg, xg))
            )
            imager = (
                    torch.einsum('i,ijk->jk', yr, xr)
                    / torch.sqrt(torch.einsum('i, i->', yr, yr))
                    / torch.sqrt(torch.einsum('ijk,ijk->', xr, xr))
            )
            image = np.zeros(
                (
                    3,
                    self.grid_size,
                    self.grid_size
                )
            )
            image[2] = imageb.cpu().numpy()
            image[1] = imageg.cpu().numpy()
            image[0] = imager.cpu().numpy()


        elif self.metric == 1:

            # Flatten data along differential outputs and times axes
            data_in = data_in.permute(0, 2, 1)
            data_in = data_in.reshape((-1,) + data_in.shape[2:])
            template_data = template_data.permute(0, 2, 1, 3, 4)
            template_data = template_data.reshape((-1,) + template_data.shape[2:])

            # Calculate matrix C according to equation B.2
            data_variance = torch.var(data_in, axis=0)
            sum = torch.sum(torch.einsum('ij, ijkl->ijkl', data_in, template_data), axis=0)
            vector_c = torch.einsum('ijk, i->ijk', sum, 1 / data_variance)

            # Calculate matrix B according to equation B.3
            sum = torch.sum(template_data ** 2, axis=0)
            vector_b = torch.nan_to_num(torch.einsum('ijk, i->ijk', sum, 1 / data_variance), 1)

            # Create diagonal matrix B
            b_shape = vector_b.shape
            eye = torch.eye(b_shape[0], device=self.device).unsqueeze(-1).unsqueeze(-1)
            x_diag = vector_b.unsqueeze(1)
            matrix_b = eye * x_diag

            # Calculate the optimum flux according to equation B.6 and set positivity constraint
            b_perm = matrix_b.permute(2, 3, 0, 1)
            b_inv = torch.linalg.inv(b_perm).permute(2, 3, 0, 1)

            optimum_flux = torch.einsum('ijkl, jkl->ikl', b_inv, vector_c)

            # Calculate the cost function according to equation B.8
            optimum_flux = torch.where(optimum_flux >= 0, optimum_flux, 0)
            cost_function = optimum_flux * vector_c

            # Map wavelengths
            image = np.zeros(
                (
                    3,
                    self.grid_size,
                    self.grid_size
                )
            )

            image[2] = torch.sum(torch.nan_to_num(cost_function[i_min:i_1_2], 0), dim=0).cpu().numpy()
            image[1] = torch.sum(torch.nan_to_num(cost_function[i_1_2:i_2_3], 0), dim=0).cpu().numpy()
            image[0] = torch.sum(torch.nan_to_num(cost_function[i_2_3:i_max], 0), dim=0).cpu().numpy()

        rgb_image = np.stack([image[0], image[1], image[2]], axis=-1)

        # Stretch frame
        lo = 0
        hi = 0.01
        # print(hi)
        STRETCH = "asinh"
        ASINH_Q = 4.0
        GAMMA = 2.2
        # rgb_image = stretch_frame(rgb_image, lo, hi, STRETCH, ASINH_Q, GAMMA)
        # print(rgb_image.max())
        # rgb_image = rgb_image / 0.01  # rgb_image.max()

        r_image_out = ImageResource(self.n_image_out)
        r_image_out.set_image(torch.tensor(rgb_image))

        print('Done')
        return r_image_out
