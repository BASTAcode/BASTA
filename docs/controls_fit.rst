.. _controls_fit:

Fitting controls
================

In the following, an overview of the all the controls related to what will be fitted
with BASTA is given. The first group controls which (types of) parameters will be fitted,
while the remainder primarily are related to different specific fitting types. How the
different groups relates to the specific fitting types are described in the :ref:`examples<examples>`.

.. _fitparams:

Fit parameters
--------------
.. code-block:: python

    define_fit["fitparams"] = ("Teff", "FeH", "logg")

The observed parameters to be fitted to the grid of models. For spectroscopic and/or global
asteroseismic parameters (:ref:`example <example_global>`) these must be from the list of
parameters known by BASTA, :meth:`constants.parameters`.

For special fitting cases, the following keys can be included in ``"fitparams"``:

* ``"freqs"``: :ref:`Individual frequencies <example_freqs_individual>` if they are provided (see :ref:`controls_fit_freqparams`)
* ``"r012"``: :ref:`Frequency ratios <example_freqs_ratios>` of any available combination defined in :meth:`constants.freqtypes`, for example ``"r01"``, ``"r02"``, ``"r10"``, and ``"r102"``. Can either be provided by user, or will automatically be derived from provided individual frequencies.
* ``"e012"``: :ref:`Epsilon differences <example_freqs_epsdiff>` derived from the individual frequencies. Can be the individual sequences, ``"e01"``, ``"e02"``, or the combined ``"e012"`` sequence.
* ``"parallax"``: :ref:`Distance/parallax <example_parallax>` through apparent magnitudes (see :ref:`controls_fit_parallax`).
* ``"glitches"``: :ref:`Frequency glitches <example_freqs_glitches>` either provided by the user (see :ref:`controls_fit_freqparams`), or derived from the individual frequencies (see :ref:`controls_fit_glitches`). Can be combined with frequency ratios, whereby their cross-covariance is derived and included, by using a ratio key preceded by ``g``, as e.g. ``"gr012"`` or ``"gr10"``.

Priors
------
.. code-block:: python

    define_fit["priors"] = {"Teff": {"sigmacut": "3"}, "FeH": {"abstol": "0.5"}}


Solar scaling
-------------
.. code-block:: python

    define_fit["solarmodel"] = True



.. code-block:: python

    define_fit["sundnu"] = 135.1
    define_fit["sunnumax"] = 3090.0



Isochrones
----------
.. code-block:: python

    define_fit["odea"] = (0, 0, 0, 0)


.. _controls_fit_freqparams:

Individual frequency parameters
-------------------------------
.. code-block:: python

    define_fit["freqparams"] = {
        "freqpath": "data/freqs",
        "fcor": "BG14",
        "correlations": False,
        "dnufrac": 0.15,
        "dnuprior": True,
        "seismicweight": "1/N",
        "N": None,
        "dnubias": 0,
        "dnufit_in_ratios": False,
        "nrealizations": 10000,
        "threepoint": False,
        "readglitchfile": False,
    }

.. _controls_fit_parallax:

Distance/parallax
-----------------
.. code-block:: python

    define_fit["filters"] = ("Mj_2MASS", "Mh_2MASS", "Mk_2MASS")
    define_fit["dustframe"] = "icrs"

:meth:`constants.parameters`

.. _controls_fit_glitches:

Frequency glitches
------------------
.. code-block:: python

    define_fit["glitchparams"] = {
        "method": "Freq",
        "npoly_params": 5,
        "nderiv": 3,
        "tol_grad": 1e-3,
        "regu_param": 7,
        "nguesses": 200,
    }
