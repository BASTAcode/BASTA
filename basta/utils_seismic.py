"""
Auxiliary functions for frequency analysis
"""
from math import frexp
import os
from copy import deepcopy
from distutils.util import strtobool
from tqdm import tqdm

import numpy as np
from scipy.interpolate import CubicSpline

from basta import freq_fit
from basta import fileio as fio
from basta.constants import sydsun as sydc
from basta.constants import freqtypes
from basta.plot_seismic import epsilon_diff_and_correlation


def combined_ratios(r02, r01, r10):
    """
    Routine to combine r02, r01 and r10 ratios to produce ordered ratios r010,
    r012 and r102

    Parameters
    ----------
    r02 : array
        radial orders, r02 ratios,
        scratch for uncertainties (to be calculated), frequencies
    r01 : array
        radial orders, r01 ratios,
        scratch for uncertainties (to be calculated), frequencies
    r10 : array
        radial orders, r10 ratios,
        scratch for uncertainties (to be calculated), frequencies

    Returns
    -------
    r010 : array
        radial orders, r010 ratios,
        scratch for uncertainties (to be calculated), frequencies
    r012 : array
        radial orders, r012 ratios,
        scratch for uncertainties (to be calculated), frequencies
    r102 : array
        radial orders, r102 ratios,
        scratch for uncertainties (to be calculated), frequencies
    """

    # Number of ratios
    n02 = r02.shape[0]
    n01 = r01.shape[0]
    n10 = r10.shape[0]
    n010 = n01 + n10
    n012 = n01 + n02
    n102 = n10 + n02

    # R010 (R01 followed by R10)
    r010 = np.zeros((n010, 4))
    r010[0:n01, :] = r01[:, :]
    r010[n01 : n01 + n10, 0] = r10[:, 0] + 0.1
    r010[n01 : n01 + n10, 1:4] = r10[:, 1:4]
    r010 = r010[r010[:, 0].argsort()]
    r010[:, 0] = np.round(r010[:, 0])

    # R012 (R01 followed by R02)
    r012 = np.zeros((n012, 4))
    r012[0:n01, :] = r01[:, :]
    r012[n01 : n01 + n02, 0] = r02[:, 0] + 0.1
    r012[n01 : n01 + n02, 1:4] = r02[:, 1:4]
    r012 = r012[r012[:, 0].argsort()]
    r012[:, 0] = np.round(r012[:, 0])

    # R102 (R10 followed by R02)
    r102 = np.zeros((n102, 4))
    r102[0:n10, :] = r10[:, :]
    r102[n10 : n10 + n02, 0] = r02[:, 0] + 0.1
    r102[n10 : n10 + n02, 1:4] = r02[:, 1:4]
    r102 = r102[r102[:, 0].argsort()]
    r102[:, 0] = np.round(r102[:, 0])

    return r010, r012, r102


def ratio_and_cov(freq, rtype="R012", nrealizations=10000, threepoint=False):
    """
    Routine to compute ratios of a given type and the corresponding covariance
    matrix

    Parameters
    ----------
    freq : array
        Harmonic degrees, radial orders, frequencies, uncertainties
    rtype : str
        Ratio type (one of R02, R01, R10, R010, R012, R102)
    nrealizations : integer, optional
        Number of realizations used in covariance calculation
    threepoint : bool
        If True, use three point definition of r01 and r10 ratios instead
        of default five point definition.

    Returns
    -------
    obsR : array
        radial orders, corresponding ratios, uncertainties, frequencies
    covR : array
        corresponding covariance matrix
    """

    # Observed ratios
    obsR02, obsR01, obsR10 = freq_fit.ratios(freq, threepoint=threepoint)
    obsR010, obsR012, obsR102 = combined_ratios(obsR02, obsR01, obsR10)

    # Number of ratios of type 'rtype'
    if rtype == "R02":
        nr = obsR02.shape[0]
        obsR = obsR02
    elif rtype == "R01":
        nr = obsR01.shape[0]
        obsR = obsR01
    elif rtype == "R10":
        nr = obsR10.shape[0]
        obsR = obsR10
    elif rtype == "R010":
        nr = obsR010.shape[0]
        obsR = obsR010
    elif rtype == "R012":
        nr = obsR012.shape[0]
        obsR = obsR012
    elif rtype == "R102":
        nr = obsR102.shape[0]
        obsR = obsR102
    else:
        raise ValueError("Invalid ratio type!")

    # Compute and store different realizations
    ratio = np.zeros((nrealizations, nr))
    perturb_freq = deepcopy(freq)
    for i in range(nrealizations):
        perturb_freq[:]["freq"] = np.random.normal(freq[:]["freq"], freq[:]["err"])
        tmp_r02, tmp_r01, tmp_r10 = freq_fit.ratios(perturb_freq, threepoint=threepoint)
        tmp_r010, tmp_r012, tmp_r102 = combined_ratios(tmp_r02, tmp_r01, tmp_r10)
        if rtype == "R02":
            ratio[i, :] = tmp_r02[:, 1]
        elif rtype == "R01":
            ratio[i, :] = tmp_r01[:, 1]
        elif rtype == "R10":
            ratio[i, :] = tmp_r10[:, 1]
        elif rtype == "R010":
            ratio[i, :] = tmp_r010[:, 1]
        elif rtype == "R012":
            ratio[i, :] = tmp_r012[:, 1]
        elif rtype == "R102":
            ratio[i, :] = tmp_r102[:, 1]

    # Compute the covariance matrix and test the convergence
    n = int(round(nrealizations / 2))
    cov = np.cov(ratio[0:n, :], rowvar=False)
    covR = np.cov(ratio, rowvar=False)
    fnorm = np.linalg.norm(covR - cov) / nr**2
    if fnorm > 1.0e-6:
        print("Frobenius norm %e > 1.e-6" % (fnorm))
        print("Warning: covariance failed to converge!")

    # Compute the uncertainties on ratios using covariance matrix
    obsR[:, 2] = np.sqrt(np.diag(covR))

    # Compute inverse of the covariance matrix
    # icovR = np.linalg.pinv(covR, rcond=1e-8)

    return obsR, covR  # , icovR


def solar_scaling(Grid, inputparams, diffusion=None):
    """
    Transform quantities to solar units based on the assumed solar values. Grids use
    solar units for numax and for dnu's based on scaling relations. The input values
    (given in microHz) are converted into solar units to match the grid.

    Secondly, if a solar model is found in the grid, scale input dnu's according to the
    value of this model. This scaling will be reversed before outputting results and
    making plots.

    Parameters
    ----------
    Grid : hdf5 object
        The already loaded grid, containing the tracks/isochrones.

    inputparams : tuple
        Tuple of strings with the variables to be fitted.

    diffusion : None or int, optional
        Selection of solar model with/without diffusion for e.g. the BaSTI isochrones.
        None signals grids with only one available solar model. For isochrones (with
        odea), a value of 0 signals no diffusion; 1 is diffusion.

    Returns
    -------
    inputparams : tuple
        Modified version of `inputparams` with the added scaled values.
    """
    print("\nTransforming solar-based asteroseismic quantities:", flush=True)

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # BLOCK 1: Conversion into solar units
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    print("* Converting to solar units ...")

    # Check for solar values, if not set then use default
    dnusun = inputparams.get("dnusun", sydc.SUNdnu)
    numsun = inputparams.get("numsun", sydc.SUNnumax)

    # ------------------------------
    # TASK 1.1: Conversion of parameters
    # ------------------------------
    fitparams = inputparams.get("fitparams")
    fitpar_convert = [
        par for par in fitparams if (par.startswith("dnu") or par.startswith("numax"))
    ]
    for param in fitpar_convert:
        # dnufit is given in muHz in the grid since it is not based on scaling relations
        if param in ["dnufit", "dnufitMos12"]:
            continue

        # Apply the correct conversion
        oldval = fitparams[param][0]
        if param.startswith("numax"):
            convert_factor = numsun
        else:
            convert_factor = dnusun
        fitparams[param] = [p / convert_factor for p in fitparams[param]]

        # Print conversion for reference
        print(
            "  - {0} converted from {1:.2f} microHz to {2:.6f} solar units".format(
                param, oldval, fitparams[param][0]
            ),
            "(solar value: {0:2f} microHz)".format(convert_factor),
        )

    # ------------------------------
    # TASK 1.2: Conversion of limits
    # ------------------------------
    # Duplicates the approach above)
    limits = inputparams.get("limits")
    limits_convert = [
        par for par in limits if (par.startswith("dnu") or par.startswith("numax"))
    ]
    for param in limits_convert:
        if param in ["dnufit", "dnufitMos12"]:
            continue

        if param.startswith("numax"):
            convert_factor = numsun
        else:
            convert_factor = dnusun
        limits[param] = [p / convert_factor for p in limits[param]]

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # BLOCK 2: Scaling dnu to the solar value in the grid
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # ------------------------------
    # TASK 2.1: Get dnu of solar model
    # ------------------------------
    try:
        avail_models = list(Grid["solar_models"])
    except KeyError:
        avail_models = []

    # Read the user-set flag: Should the scaling be activated?
    try:
        solarmodel = strtobool(inputparams.get("solarmodel", ""))
    except ValueError:
        print(
            "Warning: Invalid value given for activation of solar scaling!",
            "Must be True/False! Will now assume False ...",
        )
        solarmodel = False

    if solarmodel and len(avail_models) > 0:
        # For isochrones, the diffusion is specified and names hardwired!
        if diffusion is not None:
            if diffusion == 0:
                sunmodname = "bastisun_new"
            else:
                sunmodname = "bastisun_new_diff"
        else:
            if len(avail_models) == 1:
                sunmodname = avail_models[0]
            else:
                raise NotImplementedError("More than one solar model found in grid!")

        # Get all solar model dnu values
        sunmodpath = os.path.join("solar_models", sunmodname)
        sunmoddnu = {
            param: Grid.get(os.path.join(sunmodpath, param))[()]
            for param in Grid[sunmodpath]
            if param.startswith("dnu")
        }
        print(
            "* Scaling dnu to the solar model in the grid (path: {0}) ...".format(
                sunmodpath
            )
        )
    elif len(avail_models) == 0:
        print("* No solar model found!  -->  Dnu will not be scaled.")
        sunmoddnu = {}
    else:
        print("* Solar model scaling not activated!  -->  Dnu will not be scaled.")
        sunmoddnu = {}

    # ------------------------------
    # TASK 2.2: Apply the scaling
    # ------------------------------
    dnu_scales = {}
    for dnu in sunmoddnu:
        if dnu in fitparams:
            if dnu in ["dnufit", "dnufitMos12"]:
                # Scaling factor is DNU_SUN_GRID / DNU_SUN_OBSERVED
                dnu_rescal = sunmoddnu[dnu] / dnusun
                print(
                    "  - {0} scaled by {1:.4f} from {2:.2f} to {3:.2f} microHz".format(
                        dnu,
                        dnu_rescal,
                        fitparams[dnu][0],
                        fitparams[dnu][0] * dnu_rescal,
                    ),
                    "(grid Sun: {0:.2f} microHz, real Sun: {1:.2f} microHz)".format(
                        sunmoddnu[dnu], dnusun
                    ),
                )
            else:
                # Using the scaling relations on the solar model in the grid generally
                # yields DNU_SUN_SCALING_GRID != DNU_SUN_SCALING_OBS . The exact value
                # of the solar model DNU from scaling relations is stored in the grid
                # in solar units (a number close to 1, but not exactly 1). This number
                # is used as the scaling factor of solar-unit input dnu's
                dnu_rescal = sunmoddnu[dnu]
                print(
                    "  - {0} scaled by a factor {1:.8f} according to the".format(
                        dnu, dnu_rescal
                    ),
                    "grid-solar-model value from scaling relations",
                )
            print("    (Note: Will be scaled back before outputting results!)")
            fitparams[dnu] = [(dnu_rescal) * p for p in fitparams[dnu]]
            dnu_scales[dnu] = dnu_rescal
    inputparams["dnu_scales"] = dnu_scales

    print("Done!")
    return inputparams


def prepare_obs(inputparams, verbose=False):
    """
    Prepare frequencies and ratios for fitting

    Parameters
    ----------
    inputparams : dict
        Inputparameters for BASTA
    verbose : bool
        Flag that if True adds extra text to the log (for developers).

    Returns
    -------
    obskey : array
        Array containing the angular degrees and radial orders of obs
    obs : array
        Array containing the modes in the observed data
    numax : float
        numax as found in `inputparams`.
    dnufrac : float
        The allowed fraction of the large frequency separation that defines
        when the l=0 mode in the model is close enough to the lowest l=0 mode
        in the observed set that the model can be considered.
    datos : array
        Individual frequencies, uncertainties, and combinations read
        directly from the observational input files
    covinv : array
        Covariances between individual frequencies and frequency ratios read
        from the observational input files.
    fcor : strgs
        Type of surface correction (see :func:'freq_fit.py').
    obsintervals : array
        Array containing the endpoints of the intervals used in the frequency
        fitting routine in :func:'freq_fit.calc_join'.
        As it is the same in all iterations for the observed frequencies,
        this is computed in util.prepare_obs once and given as an argument
        in order to save time and memory.
    dnudata : float
        Large frequency separation obtained by fitting the radial mode observed
        frequencies (like dnufit, but for the data). Used for fitting ratios.
    dnudata_err : float
        Uncertainty on dnudata
    """
    print("\nPreparing asteroseismic input ...")

    fitfreqs = inputparams.get("fitfreqs")
    (freqxml, glhtxt, correlations, bexp, rt, seisw, threepoint) = fitfreqs

    # Get frequency correction method
    fcor = inputparams.get("fcor", "BG14")
    if fcor not in ["None", "HK08", "BG14", "cubicBG14"]:
        raise ValueError(
            'ERROR: fcor must be either "None", "HK08", "BG14" or "cubicBG14"'
        )

    # Get numax
    if inputparams.get("numax", False) is False:
        numaxerr = (
            "ERROR: numax must be specified when fitting individual"
            + " frequencies or ratios!"
        )
        raise ValueError(numaxerr)
    numax = inputparams.get("numax")  # *numsun #NOTE SOLAR UNITS

    # Just check if 'dnufit' is specified, will be used otherwhere
    if inputparams.get("dnufit", False) is False:
        raise ValueError("ERROR: We need a deltanu value!")

    # Read dnu-constraint value
    dnufrac = inputparams.get("dnufrac", 0.15)

    if "freqs" in rt and correlations:
        getfreqcovar = True
    else:
        getfreqcovar = False

    # Get the nottrustedfile for excempted modes
    nottrustedfile = inputparams.get("nottrustedfile")

    # Check if it is unnecesarry to compute ratios
    getratios = False
    freqplots = inputparams.get("freqplots")
    if any(x in freqtypes.rtypes for x in rt):
        getratios = True
    elif len(freqplots):
        if any([freqplots[0] == True, "ratios" in freqplots]):
            getratios = True

    # Check if it is unnecesarry to compute epsilon differences
    getepsdiff = False
    if any(x in freqtypes.epsdiff for x in rt):
        getepsdiff = True
    elif len(freqplots):
        if freqplots[0] == True:
            getepsdiff = True
        elif any(x.endswith("epsdiff") for x in freqplots):
            getepsdiff = True

    if getratios:
        print(
            "Frequency ratios required for fitting and/or plotting. This may take",
            "a little while...",
        )

    # Load or calculate ratios (requires numax)
    # --> datos and cov are 3-tuples with 010, 02 and freqs
    datos, cov, obskey, obs, dnudata, dnudata_err = fio.read_rt(
        freqxml,
        glhtxt,
        rt,
        numax,
        getratios,
        getepsdiff,
        getfreqcovar,
        threepoint=threepoint,
        nottrustedfile=nottrustedfile,
        verbose=verbose,
    )

    if not correlations and "freqs" in rt:
        cov = list(cov)
        cov[2] = np.identity(cov[2].shape[0]) * np.diagonal(cov[2])
        cov = tuple(cov)
    elif not correlations and any(
        x in [*freqtypes.rtypes, *freqtypes.epsdiff] for x in rt
    ):
        cov = list(cov)
        for i in range(len(cov)):
            if cov[i] is None:
                continue
            cov[i] = np.identity(cov[i].shape[0]) * np.diagonal(cov[i])
        cov = tuple(cov)

    # Computing inverse of covariance matrices...
    covinv = []
    for i in range(len(cov)):
        if cov[i] is not None:
            covinv.append(np.linalg.pinv(cov[i], rcond=1e-8))
        else:
            covinv.append(None)

    # Compute the intervals used in frequency fitting
    if "freqs" in rt:
        obsintervals = freq_fit.make_intervals(obs, obskey, dnu=inputparams["dnufit"])
    else:
        obsintervals = None

    # Return 'dnudata' as well, which is determined in a similar way as
    # 'dnufit', but based on the observed frequencies of the data (where as
    # 'dnufit' is from model frequencies in the grid)
    print("Done!")
    return (
        obskey,
        obs,
        numax,
        dnufrac,
        datos,
        covinv,
        fcor,
        obsintervals,
        dnudata,
        dnudata_err,
    )


def get_givenl(l, osc, osckey):
    """
    Returns frequencies, radial orders, and inertias or uncertainties of a
    given angular degree l.

    Parameters
    ----------
    l : int
        Angular degree between l=0 and l=2
    osc : array
        An array containing the individual frequencies in [0,:] and the
        inertias/uncertainties in [1, :].
        This can also be a joined array between observations and model modes
        as long as the sorting corresponds to the sorting in osckey.
    osckey : array
        An array containing the n and l of the values in osc.
        They are ordered in the same way.

    Returns
    -------
    osckey_givenl : array
        An array containing the n and l of the values in osc for the input l.
    osc_givenl : array
        An array containing the individual frequencies, inertias for the
        input l.
    """
    givenl = osckey[0, :] == l
    return osckey[:, givenl], osc[:, givenl]


def transform_obj_array(objarr):
    """
    Transform a two-column nested array with dtype=object into a corresponding
    2D array with a dtype matching the indivial columns.

    Useful for reading frequency information from the BASTA HDF5 files stored
    in variable sized arrays.
    """
    return np.vstack([objarr[0], objarr[1]])


def calculate_epsilon(n, l, f, dnu):
    """
    Calculates epsilon by fitting that and dnu to the White et al. 2012,
    equation (1)

    Parameters
    ----------
    n: array
        Radial order of the modes
    l : array
        Angular degree of the modes
    f : array
        Frequency of the modes
    dnu : float or None
        The large frequency parameter.
        If none is provided, a dnu will be estimated from a fit to l=0 modes.

    Returns
    -------
    epsilon : float
        epsilon parameter
    """
    # Fit a dnu, if none is provided
    if not isinstance(dnu, float):
        dnucoeff = np.polyfit(range(len(f[l == 0])), f[l == 0], 1)
        dnu = dnucoeff[0]

    # The l=0 and l=l mode for each frequency for small separation
    f0 = np.array([f[l == 0][n[l == 0] == fn][0] for fn in n])
    fl = np.array([f[l == fl][n[l == fl] == fn][0] for fn, fl in zip(n, l)])

    # Fitting to: nu_nl + deltanu_0l - dnufit (n + l/2) = dnufit * epsilon
    xvalues = range(len(f))
    yvalues = f + (f0 - fl) - dnu * (n + l / 2)
    fitcoeff = np.polyfit(xvalues, yvalues, 0)
    return fitcoeff[0] / dnu


def check_epsilon_of_freqs(freqs, starid, dnu, quiet=False):
    """
    Calculates the offset epsilon of the frequencies, to check if
    the radial order has been labeled correctly.
    Value range of epsilon from White et al. 2012

    Parameters
    ----------
    freqs : dict
        Dictionary of frequencies and their modes, as read in read_fre
    starid : str
        Unique identifier of star, for printing with the alert
    dnu : float or None
        The large frequency parameter.
    quiet : bool, optional
        Toggle to silence the output (useful for running batches)

    Returns
    -------
    ncor : float
        Correction to the ordering of modes.
    """
    epsilon_limits = [0.6, 1.8]

    # Read in the data
    n = freqs["order"]
    l = freqs["degree"]
    f = freqs["frequency"]

    # Can only do this for orders with l=0 modes
    filt = np.isin(n, n[l == 0])
    n = n[filt]
    l = l[filt]
    f = f[filt]

    epsilon = calculate_epsilon(n, l, f, dnu)
    eps_status = lambda eps: eps >= epsilon_limits[0] and eps <= epsilon_limits[1]

    ncor = 0
    if not eps_status(epsilon):
        if not quiet:
            print(
                "\nStar {:s} has an odd epsilon".format(starid),
                "value of {:.1f},".format(epsilon),
            )
        while not eps_status(epsilon):
            if epsilon > epsilon_limits[1]:
                ncor += 1
            else:
                ncor -= 1
            epsilon = calculate_epsilon(n + ncor, l, f, dnu)

        if not quiet:
            print(
                "Correction of n-order by {:d}".format(ncor),
                "gives epsilon value of {:.1f}.".format(epsilon),
            )
        return ncor
    else:
        if not quiet:
            print(
                "Star {:s} has an".format(starid), "epsilon of: {:.1f}.".format(epsilon)
            )
        return 0


def scale_by_inertia(osckey, osc):
    """
    This function outputs the scaled sizes of the modes scaled inversly by the
    normalized inertia of the mode.

    Parameters
    ----------
    osckey : array
        Mode identification of the modes
    osc : array
        Frequencies and either inertia in the case of modelled modes or
        uncertainties in the case of observed modes.

    Returns
    -------
    s : list
        List containing 3 array, one per angular degree l.
        The arrays contain the sizes of the modes scaled by inertia.
    """
    s = []
    el0min = np.min(osc[1, :])
    _, oscl0 = get_givenl(l=0, osckey=osckey, osc=osc)
    _, oscl1 = get_givenl(l=1, osckey=osckey, osc=osc)
    _, oscl2 = get_givenl(l=2, osckey=osckey, osc=osc)

    if len(oscl0) != 0:
        s0 = [10 * (1 / (np.log10(2 * n / (el0min)))) ** 2 for n in oscl0[1, :]]
        s.append(np.asarray(s0))
    if len(oscl1) != 0:
        s1 = [10 * (1 / (np.log10(2 * n / (el0min)))) ** 2 for n in oscl1[1, :]]
        s.append(np.asarray(s1))
    if len(oscl2) != 0:
        s2 = [10 * (1 / (np.log10(2 * n / (el0min)))) ** 2 for n in oscl2[1, :]]
        s.append(np.asarray(s2))
    return s


def compute_epsilon_diff_and_cov(
    osckey, osc, osccov, avgdnu, seq="e012", nrealisations=20000
):
    """
    Compute epsilon differences and covariances.

    From Roxburgh 2016:
     - Eq. 1 -> Epsilon(n,l)
     - Eq. 4 -> EpsilonDifference(l=0,l={1,2})

    Epsilon differences are independent of surface phase shift/outer
    layers when the epsilons are evaluated at the same frequency. It
    therefore relies on splining from epsilons at the observed frequencies
    of the given degree and order to the frequency of the compared/subtracted
    epsilon. See function `compute_epsilon_diff' for further clarification.

    For MonteCarlo sampling of the covariances, it is replicated from the
    covariance determination of frequency ratios in BASTA, (sec 4.1.3 of
    Aguirre Børsen-Koch et al. 2022). A number of realisations of the
    epsilon differences are drawn from random Gaussian distributions of the
    individual frequencies within their uncertainty.

    TODO: Analytic covariance

    Parameters
    ----------
    osckey : array
        Array containing the angular degrees and radial orders of the modes
    osc : array
        Array containing the modes (and inertias)
    osccov : array
        Covariances between individual frequencies
    avgdnu : float
        Average value of the large frequency separation
    seq : str
        Similar to ratios, what sequence of epsilon differences to be computed.
        Can be 01, 02 or 012 for a combination of the two first.
    nrealisations : int or bool
        If int: number of realisations used for MC-sampling the covariances
        If bool: Whether to use MC (True) or analytic (False) deternubation
        of covariances. If True, nrealisations of 20,000 is used.

    Returns
    -------
    epsilon : array
        Array containing the modes in the observed data
    covinv : array
        Covariances. The inverse.
    """

    # Check covariance determination input
    MonteCarlo = False
    if type(nrealisations) in [int, float]:
        if nrealisations > 0:
            MonteCarlo = True
            nrealisations = int(nrealisations)
    elif type(nrealisations) == bool:
        if nrealisations == True:
            MonteCarlo
            nrealisations = 20000

    # Possible input parameters
    extrapolation = False
    nsort = True
    DEBUG = False

    # Remove modes outside of l=0 range
    if not extrapolation:
        indall = osckey[0, :] > -1
        ind0 = osckey[0, :] == 0
        ind12 = osckey[0, :] > 0
        mask = np.logical_and(
            osc[0, ind12] < max(osc[0, ind0]), osc[0, ind12] > min(osc[0, ind0])
        )
        indall[ind12] = mask
        osc = osc[:, indall]
        osckey = osckey[:, indall]

    if not MonteCarlo:
        # Epsilon is computed analytically from the frequency information
        epsilon = np.zeros(osc.shape[1])
        epsilon_err = np.zeros(osc.shape[1])
        epsilon_cov = np.zeros(osccov.shape)
        if DEBUG:
            print(
                "DEBUG: {:>2}{:>4}{:>9}{:>6}{:>7}{:>8}".format(
                    "l", "n", "nu", "e_nu", "eps", "e_eps"
                )
            )
        for i, freq in enumerate(osc[0, :]):
            ll, nn = osckey[:, i]
            epsilon[i] = freq / avgdnu - nn - ll / 2
            epsilon_cov[i, i] = (
                osccov[i, i] / avgdnu**2
            )  # Error propagation, variance
            epsilon_err[i] = (
                np.sqrt(osccov[i, i]) / avgdnu
            )  # One of the two is redundant
            if DEBUG:
                print(
                    "DEBUG: {:2}{:4}{:9.2f}{:6.2f}{:7.3f}{:8.4f}".format(
                        ll,
                        nn,
                        freq,
                        np.sqrt(osccov[i, i]),
                        epsilon[i],
                        np.sqrt(epsilon_cov[i, i]),
                    )
                )

        # Determination of the epsilon differences by interpolation
        eps0 = epsilon[osckey[0, :] == 0]
        nu0 = osc[0, osckey[0, :] == 0]
        eps1 = epsilon[osckey[0, :] == 1]
        nu1 = osc[0, osckey[0, :] == 1]
        eps2 = epsilon[osckey[0, :] == 2]
        nu2 = osc[0, osckey[0, :] == 2]

        eps0_intpol = CubicSpline(nu0, eps0)
        eps0_at_nu1 = eps0_intpol(nu1)
        eps0_at_nu2 = eps0_intpol(nu2)

        delta_eps01 = eps0_at_nu1 - eps1
        delta_eps02 = eps0_at_nu2 - eps2

    else:
        # Compute epsilon and store them with the information
        eps_diff = freq_fit.compute_epsilon_diff(
            osckey, osc, avgdnu, seq=seq, nsorting=nsort
        )
        pbar = tqdm(total=nrealisations, desc="Sampling covariances", ascii=True)
        # Compute and store different realizations
        eps_reals = np.zeros((nrealisations, eps_diff.shape[1]))
        perturb_osc = deepcopy(osc)
        for i in range(nrealisations):
            pbar.update(1)
            perturb_osc[0][:] = np.random.normal(osc[0][:], osc[1][:])
            perturb_eps = freq_fit.compute_epsilon_diff(
                osckey, perturb_osc, avgdnu, seq=seq, nsorting=nsort
            )
            eps_reals[i, :] = perturb_eps[0]
        pbar.close()

        # Compute the covariance matrix and test the convergence
        n = int(round(nrealisations / 2))
        cov = np.cov(eps_reals[:n, :], rowvar=False)
        covDeps = np.cov(eps_reals, rowvar=False)
        fnorm = np.linalg.norm(covDeps - cov) / eps_diff.shape[1] ** 2
        if fnorm > 1.0e-6:
            print("Frobenius norm {0} > 1e-6".format(fnorm))
            print("Warning: Covariance failed to converge")

        if DEBUG:
            nsort2 = not nsort
            eps2 = freq_fit.compute_epsilon_diff(
                osckey, osc, avgdnu, seq=seq, nsorting=nsort2
            )

            pbar = tqdm(total=nrealisations, desc="Sampling covariances", ascii=True)
            eps_reals = np.zeros((nrealisations, eps2.shape[1]))
            perturb_osc = deepcopy(osc)
            for i in range(nrealisations):
                pbar.update(1)
                perturb_osc[0][:] = np.random.normal(osc[0][:], osc[1][:])
                perturb_eps = freq_fit.compute_epsilon_diff(
                    osckey, perturb_osc, avgdnu, seq=seq, nsorting=nsort2
                )
                eps_reals[i, :] = perturb_eps[0]
            pbar.close()

            # Compute the covariance matrix and test the convergence
            n = int(round(nrealisations / 2))
            cov = np.cov(eps_reals[:n, :], rowvar=False)
            covD2 = np.cov(eps_reals, rowvar=False)

            epsilon_diff_and_correlation(
                eps_diff, eps2, covDeps, covD2, osc, osckey, avgdnu
            )

    return eps_diff, covDeps
