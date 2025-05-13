"""
Auxiliary functions for frequency analysis
"""

import os
from copy import deepcopy
from typing import Any, Literal

import h5py  # type: ignore[import]
import numpy as np
from sklearn import covariance as skcov  # type: ignore[import]
from tqdm import tqdm

from basta import core, freq_fit, glitch_fit
from basta import fileio as fio
from basta import utils_general as util
from basta.constants import freqtypes


def extract_solar_model_dnu(
    Grid: h5py.File,
    gridinfo: util.GridInfo,
    flag_solarmodel: bool,
) -> tuple[str, dict[str, float]]:
    """
    Extracts dnu values from the selected solar model in the grid.

    Parameters
    ----------
    Grid : h5py.File
        Opened grid file containing solar models.
    gridinfo : GridInfo
        Info about grid configuration, including diffusion model selection.
    flag_solarmodel : bool
        Whether to apply solar model-based scaling

    Returns
    -------
    (sunmodpath, sunmoddnu) : tuple
        Path to solar model in grid, and extracted dnu dictionary.
    """
    try:
        models = list(Grid["solar_models"])
    except KeyError:
        print("! No solar models found in grid.")
        return "", {}

    if not flag_solarmodel or not models:
        print("* Solar model scaling is disabled.")
        return "", {}

    if gridinfo.get("difsolarmodel") is not None:
        sunmodname = (
            "bastisun_new_diff" if gridinfo["difsolarmodel"] else "bastisun_new"
        )
    elif len(models) == 1:
        sunmodname = models[0]
    else:
        raise NotImplementedError("More than one solar model found in grid!")

    sunmodpath = os.path.join("solar_models", sunmodname)
    print(f"* Using solar model '{sunmodname}' for dnu scaling.")

    sunmoddnu = {
        name: Grid[sunmodpath][name][()]
        for name in Grid[sunmodpath]
        if name.startswith("dnu")
    }

    return sunmodpath, sunmoddnu


def solar_scaling(
    Grid: h5py.File,
    globalseismicparams: core.GlobalSeismicParameters,
    inferencesettings: core.InferenceSettings,
    outputoptions: core.OutputOptions,
    gridinfo: util.GridInfo,
) -> None:
    """
    Convert certain seismic quantities to solar units based on the assumed solar values.
    Update the GlobalSeismicParameters object with solar-scaled values.

    Grids use solar units for numax and for dnu's based on scaling relations.
    The input values (given in microHz) are converted into solar units to match the grid.

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
    print("\nScaling solar-based asteroseismic quantities:", flush=True)

    # Extract solar constants
    solarvalues = inferencesettings.solarvalues
    print(
        f"* Solar references: dnu = {solarvalues['dnu']} µHz, numax = {solarvalues['numax']} µHz"
    )

    scalefactors = {}
    seismicparams = outputoptions.asciiparams + list(globalseismicparams.params.keys())

    already_scaled = ["dnufit", "dnufitMos12"]
    for key in seismicparams:
        if key.startswith("numax"):
            factor = 1 / solarvalues["numax"]
        elif key.startswith("dnu") and key not in already_scaled:
            factor = 1 / solarvalues["dnu"]
        elif key in already_scaled:
            factor = 1
        else:
            continue

        scalefactors[key] = factor
        if key not in already_scaled:
            if outputoptions.verbose:
                print(
                    f"  - {key}: {globalseismicparams.get_original(key)[0]:.2f} → {globalseismicparams.get_original(key)[0]*factor:.6f} (solar units)"
                )

    # Read the user-set flag: Should the scaling be activated?
    try:
        flag_solarmodel: bool = bool(util.strtobool(inferencesettings.solarmodel))
    except ValueError:
        print(
            "Warning: Invalid value given for activation of solar scaling!",
            "Must be True/False! Will now assume False ...",
        )
        flag_solarmodel = False

    # Extract solar model dnu values (or skip)
    sunmodpath, sunmoddnu = extract_solar_model_dnu(Grid, gridinfo, flag_solarmodel)
    if not sunmoddnu:
        print("* No solar model scaling applied.")

    # Apply scaling using solar model values
    print(f"* Applying solar model scaling from: {sunmodpath}.")
    for key in scalefactors:
        if key in sunmoddnu:
            if key in already_scaled:
                # Scaling factor is DNU_SUN_GRID / DNU_SUN_OBSERVED
                scalefactors[key] *= sunmoddnu[key] / solarvalues["dnu"]
            else:
                # Using the scaling relations on the solar model in the grid generally
                # yields DNU_SUN_SCALING_GRID != DNU_SUN_SCALING_OBS.
                # The exact value of the solar model DNU from scaling relations
                # is stored in the grid in solar units (a number close to 1,
                # but not exactly 1).
                # This number is used as the scaling factor of solar-unit input dnu's
                scalefactors[key] *= sunmoddnu[key]

    globalseismicparams.set_scalefactor(scalefactors)

    for key in globalseismicparams.params.keys():
        orig = globalseismicparams.get_original(key)[0]
        scaled = globalseismicparams.get_scaled(key)[0]
        if outputoptions.verbose:
            if orig != scaled:
                print(f"  - {key}: {orig:.2f} → {scaled:.6f} (solar units)")
        if key in already_scaled:
            print(
                f"  - {key} scaled by {scalefactors[key]:.4f} from {orig:.2f} to {scaled:.2f} µHz"
            )
            print(
                f"    (grid Sun: {sunmoddnu[key]:.2f} µHz, real Sun: {solarvalues['dnu']:.2f} µHz)"
            )
            print(f"    (Note: {key} will be scaled back before outputting results!)")


def get_givenl(
    l: int, osc: np.ndarray, osckey: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
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

    def eps_status(eps):
        return eps >= epsilon_limits[0] and eps <= epsilon_limits[1]

    ncor = 0
    if not eps_status(epsilon):
        if not quiet:
            print(
                f"\nStar {starid:s} has an odd epsilon",
                f"value of {epsilon:.1f},",
            )
        while not eps_status(epsilon):
            if epsilon > epsilon_limits[1]:
                ncor += 1
            else:
                ncor -= 1
            epsilon = calculate_epsilon(n + ncor, l, f, dnu)

        if not quiet:
            print(
                f"Correction of n-order by {ncor:d}",
                f"gives epsilon value of {epsilon:.1f}.",
            )
        return ncor
    if not quiet:
        print(f"Star {starid:s} has an", f"epsilon of: {epsilon:.1f}.")
    return 0


def scale_by_inertia(
    modes: core.ModelFrequencies | core.JoinedModes,
) -> list[np.ndarray]:
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
    el0min = np.amin(modes.inertias)

    oscl0 = modes.of_angular_degree(0)
    oscl1 = modes.of_angular_degree(1)
    oscl2 = modes.of_angular_degree(2)

    if len(oscl0) > 0:
        s0 = [10 * (1 / (np.log10(2 * n / (el0min)))) ** 2 for n in oscl0["inertia"]]
        s.append(np.asarray(s0))
    if len(oscl1) > 0:
        s1 = [10 * (1 / (np.log10(2 * n / (el0min)))) ** 2 for n in oscl1["inertia"]]
        s.append(np.asarray(s1))
    if len(oscl2) > 0:
        s2 = [10 * (1 / (np.log10(2 * n / (el0min)))) ** 2 for n in oscl2["inertia"]]
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
    elif fittype in freqtypes.glitches:
        seqs_function = glitch_fit.compute_glitchseqs
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

    nfailed = np.sum(np.isnan(nvalues[:, -1]))
    if nfailed / nrealisations > 0.3:
        print(f"Warning: {nfailed} of {nrealisations} failed")

    # Filter out bad failed iterations
    nvalues = nvalues[~np.isnan(nvalues).any(axis=1), :]

    # Derive covariance matrix from MC-realisations and test convergence
    n = round((nrealisations - nfailed) / 2)
    tmpcov = skcov.MinCovDet().fit(nvalues[:n, :]).covariance_
    fullcov = skcov.MinCovDet().fit(nvalues).covariance_

    # Test the convergence (change in standard deviations below a relative tolerance)
    rdif = np.amax(
        np.abs(
            np.divide(
                np.sqrt(np.diag(tmpcov)) - np.sqrt(np.diag(fullcov)),
                np.sqrt(np.diag(fullcov)),
            )
        )
    )

    if rdif > 0.1:
        print("Warning: Covariance failed to converge!")
        print(f"Maximum relative difference = {rdif:.2e} (>0.1)")

    # Glitch parameters are more robnustly determined as median of realizations
    if fittype in freqtypes.glitches:
        # Simply overwrite values in tmp with median values
        tmp[0, :] = np.median(nvalues, axis=0)
        return tmp, fullcov
    return fullcov


def extend_modjoin(joinkey, join, modkey, mod):
    """
    Re-determines modkey and mod for an extended range of model frequencies.
    Needed for constructing ratios that are interpolated at observed
    frequencies, to avoid extrapolation.
    For each degree l it finds a mode n lower and higher, and appends these.

    Parameters
    ----------
    joinkey : array
        Joined frequency identification keys of observations and model.
    join : array
        Joined frequencies of observations and model.
    modkey : array
        All frequency identification keys of the model.
    mod : array
        All frequencies of the model.

    Returns
    -------
    key : array
        The extended array of frequency identification keys.
    osc : array
        The extended array of frequencies.
    """
    key = deepcopy(joinkey[:2])
    osc = deepcopy(join[:2])
    # We need additional of each degree
    for ll in set(key[0, :]):
        modkey_gl, mod_gl = get_givenl(l=ll, osc=mod, osckey=modkey)
        key_gl, _ = get_givenl(l=ll, osc=osc, osckey=key)
        # Check we can extend
        if min(modkey_gl[1]) >= min(key_gl[1]) or max(modkey_gl[1]) <= max(key_gl[1]):
            return None, None
        # Find and append one below and one above
        for target in [min(key_gl[1]) - 1, max(key_gl[1]) + 1]:
            ind = np.where(modkey_gl[1] == target)[0]
            key = np.hstack((key, modkey_gl[:, ind]))
            osc = np.hstack((osc, mod_gl[:, ind]))
    # Sort by l then n
    mask = np.lexsort((key[1], key[0]))
    key = key[:, mask]
    osc = osc[:, mask]
    return key, osc
