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


def prepare_obs(inputparams, verbose=False, debug=False):
    """
    Prepare frequencies and ratios for fitting

    Parameters
    ----------
    inputparams : dict
        Inputparameters for BASTA
    verbose : bool, optional
        Flag that if True adds extra text to the log (for developers).
    debug : bool, optional
        Activate additional output for debugging (for developers)

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
    fcor : strgs
        Type of surface correction (see :func:'freq_fit.py').
    obsfreqdata : dict
        Requested frequency-dependent data such as glitches, ratios, and
        epsilon difference. It also contains the covariance matrix and its
        inverse of the individual frequency modes.
        The keys correspond to the science case, e.g. `r01a, `glitch`, or
        `e012`.
        Inside each case, you find the data (`data`), the covariance matrix
        (`cov`), and its inverse (`covinv`).
    obsfreqmeta : dict
        The requested information about which frequency products to fit or
        plot, unpacked for easier access later.
    obsintervals : array
        Array containing the endpoints of the intervals used in the frequency
        fitting routine in :func:'freq_fit.calc_join'.
        As it is the same in all iterations for the observed frequencies,
        this is computed in util.prepare_obs once and given as an argument
        in order to save time and memory.
    """
    print("\nPreparing asteroseismic input ...")

    fitfreqs = inputparams.get("fitfreqs")
    (
        freqfilename,
        glitchfilename,
        correlations,
        bexp,
        freqfits,
        seisw,
        threepoint,
        readratios,
    ) = fitfreqs

    # Get frequency correction method
    fcor = inputparams.get("fcor", "BG14")
    if fcor not in ["None", *freqtypes.surfeffcorrs]:
        raise ValueError(
            f'ERROR: fcor must be either "None" or in {freqtypes.surfeffcorrs}'
        )

    # Get numax
    if inputparams.get("numax", False) is False:
        numaxerr = (
            "ERROR: numax must be specified when fitting individual"
            + " frequencies or ratios!"
        )
        raise ValueError(numaxerr)
    numax = inputparams.get("numax")  # *numsun in solar units

    # Just check if 'dnufit' is specified, will be used otherwhere
    if inputparams.get("dnufit", False) is False:
        raise ValueError("ERROR: We need a deltanu value!")

    # Read dnu-constraint value
    dnufrac = inputparams.get("dnufrac", 0.15)

    if "freqs" in freqfits and correlations:
        getfreqcovar = True
    else:
        getfreqcovar = False

    # Get the nottrustedfile for excempted modes
    nottrustedfile = inputparams.get("nottrustedfile")

    freqplots = inputparams.get("freqplots")

    # Load or compute frequency-dependent products
    obskey, obs, obsfreqdata, obsfreqmeta = fio.read_allseismic(
        freqfilename,
        glitchfilename,
        freqfits,
        numax,
        freqplots,
        getfreqcovar=getfreqcovar,
        nottrustedfile=nottrustedfile,
        threepoint=threepoint,
        readratios=readratios,
        verbose=verbose,
        debug=debug,
    )

    # Compute the intervals used in frequency fitting
    if any([x in [*freqtypes.freqs, *freqtypes.rtypes] for x in freqfits]):
        obsintervals = freq_fit.make_intervals(obs, obskey, dnu=inputparams["dnufit"])
    else:
        obsintervals = None

    print("Done!")
    return (
        obskey,
        obs,
        numax,
        dnufrac,
        fcor,
        obsfreqdata,
        obsfreqmeta,
        obsintervals,
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


def compute_cov_from_mc(nr, osckey, osc, fittype, args, nrealisations=10000):
    """
    Compute covariance matrix (and its inverse) using Monte Carlo realisations.

    Parameters
    ----------
    nr : int
        Size of covariance matrix
    osckey : array
        Harmonic degrees, radial orders and radial orders of frequencies.
    osc : array
        Frequencies and their error, following the structure of obs.
    fittype : str
        Which sequence to determine, see `constants.freqtypes.rtypes` and
        `constants.freqtypes.epsdiff` for possible sequences.
    args : dict
        Set of arguments to pass on to the function that computes the fitting
        sequences, i.e. `freq_fit.compute_ratioseqs` and
        `freq_fit.compute_epsilondiffseqs`.
    nrealisations : int
        Number of realisations of the sampling for the computation of the
        covariance matrix.
    """
    # Determine the function used to sample the corresponding sequence type
    if fittype in freqtypes.rtypes:
        seqs_function = freq_fit.compute_ratioseqs
    elif fittype in freqtypes.epsdiff:
        seqs_function = freq_fit.compute_epsilondiffseqs
    else:
        raise NotImplementedError(
            "Science case for covariance matrix is not implemented"
        )

    # Compute different perturbed realisations (Monte Carlo) for covariances
    nvalues = np.zeros((nrealisations, nr))
    perturb_osc = deepcopy(osc)
    for i in tqdm(
        range(nrealisations), desc=f"Sampling {fittype} covariances", ascii=True
    ):
        perturb_osc[0, :] = np.random.normal(osc[0, :], osc[1, :])
        tmp = seqs_function(
            osckey,
            perturb_osc,
            sequence=fittype,
            **args,
        )
        nvalues[i, :] = tmp[0]

    # Derive covariance matrix from MC-realisations and test convergence
    n = int(round(nrealisations / 2))
    cov = np.cov(nvalues[:n, :], rowvar=False)
    n_cov = np.cov(nvalues, rowvar=False)
    fnorm = np.linalg.norm(n_cov - cov) / (nr**2)

    if fnorm > 1.0e-6:
        print(f"Frobenius norm {fnorm} > 1e-6")
        print("Warning: Covariance failed to converge")

    n_covinv = np.linalg.pinv(n_cov, rcond=1e-8)

    return n_cov, n_covinv
