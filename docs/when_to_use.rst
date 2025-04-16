.. _when_to_use:

When To Use LIFEsimMC
=====================

LIFEsimMC vs. LIFEsim
---------------------

`LIFEsimMC` is an alternative to `LIFEsim (1) <https://lifesim.readthedocs.io/en/latest/>`_, the first simulator
for the LIFE mission. The main reason to develop `LIFEsimMC` was the need to include the effects of instrumental
(instability) noise in the performance assessments of LIFE. The following list gives an overview of the key differences
between `LIFEsimMC` and `LIFEsim`:

* | **Instrumental Noise:**
  | `LIFEsimMC` is able to simulate instability noise by calculating the instrument response at each time step, whereas `LIFEsim` only calculates the response of an ideal instrument without explicitly considering its temporal evolution.
* | **Flexible Instrument Architectures:**
  | `LIFEsimMC` is able to simulate different instrument architectures (array configurations and beam combiners) through the use of symbolic mathematics, while `LIFEsim` has been specifically developed for the Emma-X architecture.
* | **Proper Signal Extraction:**
  | `LIFEsimMC` enables a proper signal extraction in the sense that planetary signals (spectra) are extracted from the full synthetic data containing contributions from planets and noise. On the other hand, `LIFEsim` is based on photon statistics and separately calculates contributions from planets and noise sources for e.g. its signal-to-noise estimates.
* | **Built-in Processing and Analysis Tools:**
  | `LIFEsimMC` includes a suite of built-in processing and analysis tools,

While both tools simulate the observations that LIFE would make, `LIFEsimMC` is built around the
issue of instrumental (instability) noise, which is not considered in `LFIEsim`.


