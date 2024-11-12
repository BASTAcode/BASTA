.. _controls_fit:

Fitting controls
****************

In the following, an overview of the all the controls related to what will be fitted
with BASTA is given, which corresponds to the fitting controls block in the :py:func:`define_input`
function in the ``create_inputfile.py`` scripts. To see the recommended/default usage of these
controls for specific fitting cases, see the :ref:`examples <examples>` section, or the provided
example scripts ``BASTA/examples/xmlinput/create_inputfile_*.py`` (obtainable with `BASTAexamples`).

The first group controls which (types of) parameters will be fitted,
while the remainder primarily are related to different specific fitting types. How the
different groups relates to the specific fitting types are described in the :ref:`examples<examples>`.
function in the ``create_inputfile.py`` scripts.


.. _controls_fit_fitparams:

Fit parameters
==============
.. code-block:: python

    define_fit["fitparams"] = ("Teff", "FeH", "logg")

The observed parameters to be fitted to the grid of models. For spectroscopic and/or global
asteroseismic parameters (:ref:`example <example_global>`) these must be from the list of
parameters known by BASTA, listed :ref:`here <controls_params>`.

For special fitting cases, the following keys can be included in ``"fitparams"``:

* ``"freqs"``: :ref:`Individual frequencies <example_freqs_individual>` if they are provided (see :ref:`controls_fit_freqparams`)
* ``"r012"``: :ref:`Frequency ratios <example_freqs_ratios>` of any available combination defined in :meth:`constants.freqtypes`, for example ``"r01"``, ``"r02"``, ``"r10"``, and ``"r102"``. Can either be provided by user, or will automatically be derived from provided individual frequencies.
* ``"e012"``: :ref:`Epsilon differences <example_freqs_epsdiff>` derived from the individual frequencies. Can be the individual sequences, ``"e01"``, ``"e02"``, or the combined ``"e012"`` sequence.
* ``"parallax"``: :ref:`Distance/parallax <example_parallax>` through apparent magnitudes (see :ref:`controls_fit_parallax`).
* ``"glitches"``: :ref:`Frequency glitches <example_freqs_glitches>` either provided by the user (see :ref:`controls_fit_freqparams`), or derived from the individual frequencies (see :ref:`controls_fit_glitches`). Can be combined with frequency ratios, whereby their cross-covariance is derived and included, by using a ratio key preceded by ``g``, as e.g. ``"gr012"`` or ``"gr10"``.

.. _controls_fit_priors:

Priors and weights
==================
.. code-block:: python

    define_fit["priors"] = {"IMF": "salpeter1955", "Teff": {"sigmacut": "3"},
                            "FeH": {"abstol": "0.5"},

Used to include statistical and flat priors.

Statistical priors (as introduced in :ref:`methods_stats_bayes`) are set using special keywords.
The full list of statistical priors can be found in :meth:`priors`. Currently, it is only
possible to include an `Intial Mass Function` (IMF), which is set with the key ``"IMF"``, and
can be any one of :meth:`"baldrygkazebrook2003" <priors.baldrygkazebrook2003>`,
:meth:`"chabrier2003" <priors.chabrier2003>`, :meth:`"kennicutt1994" <priors.kennicut1994>`,
:meth:`"kroupa2001" <priors.kroupa2001>`, :meth:`"millerscalo1979" <priors.millerscalo1979>`,
:meth:`"salpeter1955" <priors.salpeter1955>`, or :meth:`"scalo1998" <priors.scalo1998>`.

The flat priors are set in any quantity in the grid, to limit the part of the grid considered
in the statistical inference/cut out models, mostly to save computation time. It is set by the key of the
parameter (from the :ref:`parameter list <controls_params>`), and a dictionary defining how it should be
applied, depending on the set keys:

* ``"sigmacut"``: Only possible for fitted parameters. Will cut out models if the values of the set parameter deviate by this number times the inputted error. For the above, if :math:`\sigma_{T_\text{eff}}=75\,\text{K}`, models can only deviate :math:`3\sigma_{T_\text{eff}}=215\,\text{K}` from the observed :math:`T_\text{eff}`.
* ``"abstol"``: Only possible for inputted parameters. Will only consider models within this `absolute tolerance` around the observed value. For the above, only models with a :math:`[\text{Fe/H}]` within :math:`0.25\,\text{dex}` above or below the observed value are considered.
* ``"min"``: Possible for all parameters in the grid. Will only consider models if the model value is above this `minimum` value.
* ``"max"``: Possible for all parameters in the grid. Will only consider models if the model value is below this `maximum` value.

.. code-block:: python

    define_fit["bayweights"] = False

A key functionality of BASTA is to use so-called :ref:`Bayesian weights <methods_stats_bayes>`,
which take the sampling of the grid into account. These will also accommodate the different
evolutionary speed of stars in different phases. It is **not recommended to disbable** the
use of weights, but can be done for testing or debugging grids.

Solar scaling
=============
.. code-block:: python

    define_fit["solarmodel"] = True

Switch to enable/disable solar scaling of asteroseismic variables. This is preferable
to do, in order to alleviate discrepancies between the assumed solar value for the model
versus the observations. It is for this reason that the values of :math:`\Delta\nu` and
:math:`\nu_\text{max}` are in solar units in default BASTA grids.

.. code-block:: python

    define_fit["sundnu"] = 135.1
    define_fit["sunnumax"] = 3090.0

Used to set the assumed solar values of :math:`\Delta\nu` and :math:`\nu_\text{max}`
of the observations. By default, BASTA uses the values from the
`SYD pipeline <https://arxiv.org/abs/2108.00582>`_, as given here.

Isochrones
==========
.. code-block:: python

    define_fit["odea"] = (0, 0, 0, 0)

If the grid containing `BaSTI iscohrones <http://basti-iac.oa-abruzzo.inaf.it/>`_ is used,
the user has to select which `science case`, the selection microphysics was used for the calculation
of the isochrones, to fit to. These are defined using the ``odea`` tuple, which stands for

* ``o`` - Overshoot: Value used for the convective overshooting efficiency, disabled if 0.
* ``d`` - Diffusion: Whether atomic diffusion of elements is treated, 0 for disabled, 1 for enabled.
* ``e`` - Mass-loss (Reimers eta): Effectiveness of the applied mass-loss, disabled if 0.
* ``a`` - Alphas enhancement: The alpha elements abundance :math:`[\alpha/\text{Fe}]`.

The grid is continuously updated as science cases become available. The science cases
currently available in the grid are

.. code-block:: python

    define_fit["odea"] = (0,   0, 0,   0)
    define_fit["odea"] = (0.2, 0, 0,   0)
    define_fit["odea"] = (0.2, 0, 0.3, 0)
    define_fit["odea"] = (0.2, 1, 0.3, 0)
    define_fit["odea"] = (0.2, 1, 0.3, 0.4)


.. _controls_fit_freqparams:

Individual frequency parameters
===============================
.. code-block:: python

    define_fit["freqparams"] = {
        "freqpath": "data/freqs",
        "fcor": "BG14",
        "bexp": 0,
        "correlations": False,
        "excludemodes": None,
        "dnufrac": 0.15,
        "dnuprior": True,
        "seismicweight": "1/N",
        "N": None,
        "dof": None,
        "dnubias": 0,
        "dnufit_in_ratios": False,
        "nrealizations": 10000,
        "threepoint": False,
        "readglitchfile": False,
    }

Controls related to the treatment of individual frequencies across all methods utilizing these.
All are not necessary, as they usually have appropriate default values, or are only related to
specific :ref:`fitting cases <controls_fit_fitparams>`. To see what is usually necessary for each case,
see the :ref:`examples <examples>`.

The control options are:

* ``freqpath`` (*str*): **Mandatory** location of the directory containing the ``xml`` files with the individual frequencies of each star. These are generated from ASCII format using the :meth:`fileio.freqs_ascii_to_xml` routine, as shown in this :ref:`example <example_freqs>`.
* ``fcor`` (*str*): The formulation of the frequency correction applied to the model frequencies when fitting to account for the asteroseismic surface effect. Options are :meth:`"HK08" <freq_fit.HK08>`, :meth:`"BG14" <freq_fit.BG14>`, :meth:`"cubicBG14" <freq_fit.cubicBG14>` (default), or ``"None"`` to disable the correction.
* ``bexp`` (*float*): Exponent to be used in the :meth:`"HK08" <freq_fit.HK08>` surface correction. It is therefore only necessary to define when using this formulation.
* ``correlations`` (*bool*): Toggle for including correlations between individual frequencies, or their derived parameters (and enable correlation maps of these to be plotted, see :ref:`frequency plots <controls_outplots_freqplots>`). ``True``, however ``False`` by default) changes with :ref:`fitting case <controls_fit_fitparams>` as follows:

   * Individual frequencies: The correlations must be provided by the user in the input ``xml`` along with the frequencies themselves (also converted from ASCII to ``xml`` using :meth:`fileio.freqs_ascii_to_xml`).
   * Ratios/epsilon differences: If provided in the input ``xml`` these will be used. If not provided, they will be determined through Mone-Carlo sampling. *Note:* If no correlations are assumed, but no error on the ratios/epsilon differences have been provided, the error will be sampled through Monte-Carlo sampling, but the correlations discarded.

* ``excludemodes`` (*str or dict*): Path to file containing frequency modes to exclude from the fit. If a string pointing to a single file is provided, the modes within will be excluded from all stars being fitted, see :ref:`description of method <methods_freqs_exclude>`. Provide a string pointing to a single file to exclude the same modes from all stars, or a dictionary with ``starid`` of stars as keys, and the string pointing to the specific file as value.
* ``dnufrac`` (*float*): Fraction of the inputted :math:`\Delta\nu` used to constrain the interval wherein the lowest :math:`\ell =0` frequency between the model and observed frequencies must match to be considered in the fit, see :ref:`method section <methods_freqs_dnufrac>`.
* ``dnuprior`` (*bool*): Enable automatic prior on :math:`\Delta\nu` (default ``True``). This is used before the ``dnufrac`` to speed up the fit, as this is a less restrictive prior but computationally cheaper than the ``dnufrac`` prior.
* ``seismicweight`` (*str*): The method by which the contribution to the :math:`\chi^2` term from individual frequencies (or their derived quantities) is weighted/scaled, which is customary in order to let the classical observables impact the posterior. With the number of frequencies/derived quantities being ``N``, the available methods are ``"1/N"`` (default) whereby the contribution is divided by the number of frequencies/quantities, ``"1/1"`` for no weighting/scaling, or ``"1/N-dof"`` to include an estimate of the degrees-of-freedom (``dof``).
* ``N`` (*int*): Manually define/overwrite the number to use in the weighting of the :math:`\chi^2` value from individual frequencies/derived quantities. When set to the default (``None``), it will be automatically determined as the number of frequencies/quantities.
* ``dof`` (*int*): The degrees-of-freedom to use in the weighting of the :math:`\chi^2` value from individual frequencies/derived quantities, if the method ``"1/N-dof"`` is set for the ``seismicweight`` control option.
* ``dnubias`` (*float*): Bias value to add to the error of :math:`\Delta\nu` automatically determined from the individual frequencies using a :meth:`weighted fit <freq_fit.compute_dnu_wfit>`. The total error is determined as :math:`\sigma_{\Delta\nu} = \sqrt{\sigma_\text{fit}^2 + \sigma_\text{bias}^2}`. Default is 0.
* ``dnufit_in_ratios`` (*bool*): Toggle to include :math:`\Delta\nu` in the :math:`\chi^2` value when fitting ratios. The model value is determined through a :meth:`weighted fit <freq_fit.compute_dnu_wfit>` of the surface-corrected model frequencies, as determined using the method set by the ``fcor`` control option above. Default is ``False``, which disables the feature.
* ``nrealizations`` (*int*): When Monte-Carlo sampling the errors and correlations of quantities derived from individual frequencies (ratios, epsilon differences and frequency glitches), this is the number of realizations of the frequencies that are used to derive these. Default is 10000. When fitting individual frequencies, but plotting a derived quantity, for which sampling is necessary, the default is instead reduced to 2000.
* ``threepoint`` (*bool*): Toggle between the three- and five-point formulation of the small frequency differences used to construct the :math:`r_{01}` and :math:`r_{10}` sequences. Default is ``False``, whereby the five-point formulation is used.
* ``readglitchfile`` (*str*): Toggle to look for an input file containing precomputed frequency glitches, when these are utilized in BASTA. Default is ``False``. If ``True``, the input file must be an ``hdf5`` file, named the same as the star, and following the structure of the output from `GlitchPy <https://github.com/kuldeepv89/GlitchPy>`_. If this is read, the options used for the method by which the observed glitches have been computed is also used for the method for computing the frequency glitches of the models, whereby the frequency glitches :ref:`control group <controls_fit_glitches>` is ignored.


.. _controls_fit_parallax:

Distance/parallax
=================
.. code-block:: python

    define_fit["filters"] = ("Mj_2MASS", "Mh_2MASS", "Mk_2MASS")
    define_fit["dustframe"] = "icrs"

Controls for the fitting of :ref:`distances/parallaxes <methods_general_distance>` in BASTA,
see :ref:`example <example_parallax>`. The module is enabled by including ``"parallax"`` in
the :ref:`list of fitting parameters <controls_fit_fitparams>`, while this block defines how this
parallax/distance is fitted. The filters tuple determines what filters from the input should
be fitted, whereby these must be provided in the :ref:`input parameters <controls_io_paramfile>`.
The full list of filters are found in the :meth:`parameter list <controls_params>`
which are provided along with associated :meth:`reddening law coeffiecients <constants.extinction>`
for the following photometric systems, for the following photometric systems.

.. list-table::
    :header-rows: 1

    * - Name
      - Key
      - Reference
    * - Johnson/Cousins
      - ``"JC"``
      -
    * - SAGE
      - ``"SAGE"``
      -
    * - 2MASS
      - ``"2MASS"``
      -
    * - GAIA
      - ``"GAIA"``
      -
    * - JWST-NIRCam
      - ``"JWST"``
      -
    * - Sloan Digital Sky Survey
      - ``"SLOAN"``
      -
    * - Strömgren
      - ``"STROMGREN"``
      -
    * - VISTA
      - ``"VISTA"``
      -
    * - HST-WFC2
      - ``"WFC2"``
      -
    * - HST-ACS
      - ``"ACS"``
      -
    * - HST-WFC3
      - ``"WFC3"``
      -
    * - DECam
      - ``"DECAM"``
      -
    * - Skymapper
      - ``"SKYMAPPER"``
      -
    * - Kepler band
      - ``"KEPLER"``
      -
    * - TESS band
      - ``"TESS"``
      -
    * - TYCHO
      - ``"TYCHO"``
      -


The ``dustframe`` is used to indicate the coordinate system used to define the position
of the star. These are used to look up the colour excess :math:`E(B-V)` for the given
line of sight from an extinction/dustmap (`Green et al. 2015/2018 <http://argonaut.skymaps.info/.>`_).
The coordinates associated with the given coordinate system must thus be provided in the
:ref:`inpuit parameters <controls_io_paramfile>`. The possible coordinate systems and
corresponding coordinates are:

.. list-table::
    :header-rows: 1

    * - Dustframe key
      - Description
      - Coordinate keys
      - Description
    * - ``"icrs"``
      - International Celestial Reference System
      - ``"RA"``, ``"DEC"``
      - Right ascension, Declination
    * - ``"galactic"``
      - Galactic coordinates
      - ``"lon"``, ``"lat"``
      - Longitude, Lattitude



.. _controls_fit_glitches:

Frequency glitches
==================
.. code-block:: python

    define_fit["glitchparams"] = {
        "method": "Freq",
        "npoly_params": 5,
        "nderiv": 3,
        "tol_grad": 1e-3,
        "regu_param": 7,
        "nguesses": 200,
    }

When fitting/using frequency glitches with BASTA, these controls define the method, and coefficients
within said method, used when deriving the glitch parameters (see the :ref:`example <example_freqs_glitches>`).
The methods are detailed in `Verma et al. 2022 <https://arxiv.org/abs/2207.00235>`_, appendix A.
The controls are, in summary:

* ``method`` (*str*): The individual frequency information from which the glitch parameters are derived. If set to ``Freq`` they are derived directly from the individual frequencies, while for ``SecDif`` they are derived from the second differences of frequencies, which are defined as :math:`\delta^2\nu_{n,\ell}=\nu_{n-1,\ell}-2\nu_{n,\ell}+\nu_{n+1,\ell}`.
* ``npoly_params`` (*int*): Number of parameters in the smooth frequency component. The default is 5, recommended for the ``Freq`` method, while 3 is recommended for the ``SecDif`` method.
* ``nderiv`` (*int*): Order of derivative used in the regularization. The default is 3, recommended for the ``Freq`` method, while 1 is recommended for the ``SecDif`` method.
* ``tol_grad`` (*float*): Tolerance used for determination of gradients. The default is :math:`10^{-3}`. It is typically recommended being between :math:`10^{-2}` and :math:`10^{-5}` depending on the quality of the data and the applied method.
* ``regu_param`` (*int*): Regularization parameters. The default is 7, recommended for the ``Freq`` method, while 1000 is recommended for the ``SecDif`` method.
* ``nguesses`` (*int*): Number of initial guesses in the search for the global minimum. The default is 200.
