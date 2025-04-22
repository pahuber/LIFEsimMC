.. _when_to_use:

When To Use LIFEsimMC
=====================
`LIFEsimMC` is an alternative to `LIFEsim <https://lifesim.readthedocs.io/en/latest/>`_ (aka LIFEsim 1), the first simulator
for the LIFE mission. The main reason to develop `LIFEsimMC` was the need to include the effects of instrumental
(instability) noise in performance assessments of LIFE and thereby enable the next steps in the trade studies and requirement derivation
process. As such, `LIFEsimMC` can generally be used if instrumental noise or architecture comparisons are of interest.

More specifically, we recommend to use

* **LIFEsimMC** to simulate **noisy spectra** from single-epoch observations with LIFE and
* **LIFEsim** for **yield simulations**, as `LIFEsimMC` is currently not able to run yields.

Features of LIFEsimMC
---------------------
The following features have been addressed in the development of `LIFEsimMC`:

* | **Instrumental Noise:**
  | `LIFEsimMC` is able to simulate instrumental instability noise by explicitly calculating the temporal evolution of the instrument response. At each time step, it samples the instability noise from its underlying distributions and calculates the resulting instrument response.
* | **Flexible Instrument Architectures:**
  | `LIFEsimMC` is able to simulate different instrument architectures (array configurations and beam combiners) through the use of symbolic mathematics.
* | **Proper Signal Extraction:**
  | `LIFEsimMC` enables a "proper" signal extraction in the sense that planetary signals (spectra) are `extracted` based on a matched filter approach from the full synthetic data, which containing contributions from planets and noise.
* | **Built-in Processing and Analysis Tools:**
  | `LIFEsimMC` includes a suite of built-in processing and analysis tools that facilitate the interpretation and visualization of the extracted planetary signals.
* | **Computational Resources:**
  | The explicit calculations of each time step in `LIFEsimMC` requires significant computational resources. `LIFEsimMC` has been optimized to run on GPUs, which makes calculations much faster, is, however, also able to run on ordinary CPUs.




LIFEsimMC vs. LIFEsim
---------------------
The following table gives highlights the differences between `LIFEsimMC` and `LIFEsim` regarding the features listed above.

.. list-table:: LIFEsimMC vs. LIFEsim
   :widths: 34 33 33
   :header-rows: 1

   * - Feature
     - LIFEsimMC
     - LIFEsim
   * - **Noise Sources**
     - Astrophysical + Instrumental
     - Astrophysical Only
   * - **Instrument Response**
     - Time-Dependent (MC sampling)
     - No Temporal Evolution
   * - **Instrument Architectures**
     - Any
     - X-Array Double Bracewell Nuller Only
   * - **Signal Extraction / Estimation**
     - Matched Filter-Based (Planetary Signal Extraction From Full Data)
     - Photon Statistics-Based (Separate Calculation of Planetary Signal and Noise)
   * - **Functionality**
     - E.g. Correlation Maps, Spectrum + Full Covariance Estimates, Statistical Hypothesis (Detection) Tests, etc.
     - Spectrum + Standard Error Estimates (SNR), Yield Estimates
   * - **Computational Resources**
     - Computationally Expensive, Runs on CPUs + GPUs
     - Computationally Inexpensive, Runs on CPUs
