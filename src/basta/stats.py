"""
Key statistics functions
"""

import math
import os
from dataclasses import dataclass
from typing import Any, NamedTuple

import numpy as np
from scipy.interpolate import CubicSpline, interp1d  # type: ignore[import]
from scipy.ndimage.filters import gaussian_filter1d  # type: ignore[import]

from basta import constants, core, freq_fit, glitch_fit, remtor, surfacecorrections
from basta import utils_seismic as su
from basta.constants import freqtypes, statdata


@dataclass(frozen=True)
class Trackstats:
    index: np.ndarray
    logPDF: np.ndarray
    chi2: np.ndarray

    @property
    def bayw(self) -> float:
        raise Exception("attempt to use bayw when not debugging")

    @property
    def magw(self) -> float:
        raise Exception("attempt to use magw when not debugging")

    @property
    def IMFw(self) -> float:
        raise Exception("attempt to use IMFw when not debugging")


@dataclass(frozen=True)
class priorlogPDF:
    index: np.ndarray
    logPDF: np.ndarray
    chi2: np.ndarray
    bayw: float
    magw: float
    IMFw: float


def _hist_bin_fd(data: np.ndarray) -> float:
    """
    The Freedman-Diaconis histogram bin estimator.

    The Freedman-Diaconis rule uses interquartile range (IQR) to
    estimate binwidth. It is considered a variation of the Scott rule
    with more robustness as the IQR is less affected by outliers than
    the standard deviation. However, the IQR depends on fewer points
    than the standard deviation, so it is less accurate, especially for
    long tailed distributions.

    If the IQR is 0, this function returns 0 for the bin width.
    Binwidth is inversely proportional to the cube root of data size
    (asymptotically optimal).

    Parameters
    ----------
    data : array_like
        Input data that is to be histogrammed, trimmed to range. May not
        be empty.

    Returns
    -------
    h : An estimate of the optimal bin width for the given data.
    """
    data = np.asarray(data)
    iqr = np.subtract(*np.percentile(data, [75, 25]))
    if iqr == 0:
        return 0.0
    return 2.0 * iqr / np.cbrt(len(data))


def _weight(N: int, seisw: dict) -> int:
    """
    Determine weighting scheme dependent on given method

    Parameters
    ----------
    N : int
        Number of data points
    seisw : dict
        User defined control of the weights, by "1/1", "1/N" or
        "1/N-dof", and user defined "dof" and "N" if given

    Returns
    -------
    w : int
        Normalising factor
    """
    # Overwrite N if specified by the user
    N = int(seisw.get("N", N))

    # Select weight case
    weight_scheme = seisw.get("weight")
    if weight_scheme not in ["1/1", "1/N", "1/N-dof"]:
        raise ValueError(
            f"Invalid weight scheme: '{weight_scheme}'. Expected '1/1', '1/N', or '1/N-dof'."
        )

    if weight_scheme == "1/1":
        w = 1
    elif weight_scheme == "1/N":
        w = N
    elif weight_scheme == "1/N-dof":
        dof = int(seisw.get("dof", 0))
        w = N - dof

    return w


def chi2_astero(
    libitem,
    ind,
    star: core.Star,
    inferencesettings: core.InferenceSettings,
    outputoptions: core.OutputOptions,
    warnings=True,
    shapewarn=0,
    debug=False,
    verbose=False,
):
    """
    `chi2_astero` calculates the chi2 contributions from the asteroseismic
    fitting e.g., the frequency, ratio, or glitch fitting.

    Parameters
    ----------
    obskey : array
        Array containing the angular degrees and radial orders of obs
    obs : array
        Array containing the modes in the observed data
    obsfreqmeta : dict
        The requested information about which frequency products to fit or
        plot, unpacked for easier access later.
    obsfreqdata : dict
        Combinations of frequencies read from the frequency input files.
        Contains ratios, glitches, epsilon differences and covariance matrices.
    obsintervals : array
        Array containing the endpoints of the intervals used in the frequency
        fitting routine in :func:'freq_fit.calc_join'.
        As it is the same in all iterations for the observed frequencies,
        this is computed in su.prepare_obs once and given as an argument
        in order to save time and memory.
    libitem : hdf5 group
        Contains the entire track of models being processed.
    ind : int
        The index in libitem corresponding to the current model being
        processed.
    fitfreqs : dict
        Contains all user inputted frequency fitting options.
    warnings : bool
        If True, print something when it fails.
    shapewarn : int
        If a mismatch in array dimensions of the fitted parameters, or range
        of frequencies is encountered, this is set to corresponding integer
        in order to warn the user at the end of the run.
    debug : str
        Flag for print control.
    verbose : str
        Flag for print control.

    Returns
    -------
    chi2rut : float
        The calculated chi2 value for the given model to the observed data.
        If the calculated value is less than zero, chi2rut will be np.inf.
    warnings : bool
        See 'warnings' above.
    """

    # Additional parameters calculated during fitting
    addpars = {"surfacecorrected_dnu": None, "glitchparams": None}

    # Unpack model frequencies
    model_modes = core.make_model_modes_from_ln_freqinertia(
        libitem["osckey"][ind], libitem["osc"][ind]
    )

    # Determine which model modes correspond to observed modes
    assert star.modes is not None
    joinedmodes = freq_fit.calc_join(star.modes, model_modes)
    if joinedmodes is None:
        chi2rut = np.inf
        return chi2rut, warnings, shapewarn, 0

    # Apply surface correction
    corrected_joinedmodes, _ = surfacecorrections.apply_surfacecorrection(
        joinedmodes=joinedmodes, star=star
    )

    # Initialize chi2 value
    chi2rut = 0.0

    if inferencesettings.has_frequencies:
        # The frequency correction moved up before the ratios fitting!
        # --> If fitting frequencies, just add the already calculated things
        assert isinstance(corrected_joinedmodes, core.JoinedModes)
        nmodes = len(corrected_joinedmodes.model_frequencies)
        x = (
            corrected_joinedmodes.model_frequencies
            - corrected_joinedmodes.observed_frequencies
        )
        w = _weight(nmodes, star.modes.seismicweights)
        if x.shape[0] == star.modes.inverse_covariance.shape[0]:
            chi2rut += (x.T.dot(star.modes.inverse_covariance).dot(x)) / w
        else:
            shapewarn = 1
            chi2rut = np.inf

        if ~np.isfinite(chi2rut):
            chi2rut = np.inf
            if outputoptions.debug and outputoptions.verbose:
                print("DEBUG: Computed non-finite chi2, setting chi2 to inf")
        elif chi2rut < 0:
            chi2rut = np.inf
            if outputoptions.debug and outputoptions.verbose:
                print("DEBUG: chi2 less than zero, setting chi2 to inf")

    # Add large frequency separation term (using corrected frequencies!)
    # --> Equivalent to 'dnufit', but using the frequencies *after*
    #     applying the surface correction.
    # --> Compared to the observed value, which is 'dnudata'.
    if inferencesettings.fit_surfacecorrected_dnu:
        # Read observed dnu
        dnudata, dnudata_err = star.globalseismicparams.get_scaled("dnufit")

        # Compute surface corrected dnu
        surfacecorrected_dnu, _ = freq_fit.compute_dnufit(
            modes=joinedmodes, numax=star.globalseismicparams.get_scaled("numax")[0]
        )

        chi2rut += ((dnudata - surfacecorrected_dnu) / dnudata_err) ** 2

    # TODO(Amalie) Fix ratios
    # Add the chi-square terms for ratios
    if inferencesettings.has_ratios:
        if not all(joinkeys[1, joinkeys[0, :] < 3] == joinkeys[2, joinkeys[0, :] < 3]):
            chi2rut = np.inf
            return chi2rut, warnings, shapewarn, 0

        # Add frequency ratios terms
        for ratiotype in inferencesettings.fitparams:
            if ratiotype not in constants.freqtypes.rtypes:
                continue
            # Interpolate model ratios to observed frequencies
            if fitfreqs["interp_ratios"]:
                # Get all available model ratios
                broadratio = freq_fit.compute_ratioseqs(
                    modkey,
                    mod,
                    ratiotype,
                    threepoint=fitfreqs["threepoint"],
                )
                # modratio = star.ratios[ratiotype].values.copy()

                star.ratios[ratiotype]
                # Seperate and interpolate within the separate r01, r10 and r02 sequences
                for rtype in set(modratio[2, :]):
                    obsmask = modratio[2, :] == rtype
                    modmask = broadratio[2, :] == rtype
                    # Check we have the range to interpolate
                    if (
                        modratio[1, obsmask][0] < broadratio[1, modmask][0]
                        or modratio[1, obsmask][-1] > broadratio[1, modmask][-1]
                    ):
                        chi2rut = np.inf
                        shapewarn = 2
                        return chi2rut, warnings, shapewarn, 0
                    intfunc = interp1d(
                        broadratio[1, modmask], broadratio[0, modmask], kind="linear"
                    )
                    modratio[0, obsmask] = intfunc(modratio[1, obsmask])

            else:
                modratio = freq_fit.compute_ratioseqs(
                    joinkeys, join, ratiotype, threepoint=fitfreqs["threepoint"]
                )

            # Calculate chi square with chosen asteroseismic weight
            x = star.ratios[ratiotype].values
            obsfreqdata[ratiotype]["data"][0, :] - modratio[0, :]
            w = _weight(len(x), fitfreqs["seismicweights"])
            covinv = obsfreqdata[ratiotype]["covinv"]
            if x.shape[0] == covinv.shape[0]:
                chi2rut += (x.T.dot(covinv).dot(x)) / w
            else:
                shapewarn = 1
                chi2rut = np.inf

    # Add contribution from glitches
    if inferencesettings.has_glitches:
        # Obtain glitch sequence to be fitted
        glitchtype = obsfreqmeta["glitch"]["fit"][0]
        # Compute surface corrected dnu, if not already computed
        if inferencesettings.fit_surfacecorrected_dnu:
            surfacecorrected_dnu, _ = freq_fit.compute_dnufit(
                joinkeys, corjoin, star.globalseismicparams.get_scaled("numax")
            )

        # Assign acoustic depts for glitch search
        ac_depths = {
            "tauHe": libitem["tauhe"][ind],
            "dtauHe": 100.0,
            "tauCZ": libitem["taubcz"][ind],
            "dtauCZ": 200.0,
        }

        # If interpolating in ratios (default), we need to do an extra step
        if fitfreqs["interp_ratios"] and "r" in glitchtype:
            # Get all ratios of model
            broadglitches = glitch_fit.compute_glitchseqs(
                joinkeys,
                join,
                glitchtype,
                surfacecorrected_dnu,
                fitfreqs,
                ac_depths,
                debug,
            )
            # Construct output array
            modglitches = obsfreqdata[glitchtype]["data"].copy()
            modglitches[0, -3:] = broadglitches[0, -3:]

            # Separate and interpolate within the separate r01, r10 and r02 sequences
            # rtype = {7, 8, 9} is glitch parameters, can't interpolate those
            for rtype in set(modglitches[2, :]) - {7.0, 8.0, 9.0}:
                joinmask = modglitches[2, :] == rtype
                broadmask = broadglitches[2, :] == rtype
                # Check we are in range
                if (
                    modglitches[1, joinmask][0] < broadglitches[1, broadmask][0]
                    or modglitches[1, joinmask][-1] > broadglitches[1, broadmask][-1]
                ):
                    chi2rut = np.inf
                    shapewarn = 2
                    return chi2rut, warnings, shapewarn, addpars
                intfunc = interp1d(
                    broadglitches[1, broadmask],
                    broadglitches[0, broadmask],
                    kind="linear",
                )
                modglitches[0, joinmask] = intfunc(modglitches[1, joinmask])

        else:
            # Get model glitch sequence
            modglitches = glitch_fit.compute_glitchseqs(
                joinkeys,
                join,
                glitchtype,
                surfacecorrected_dnu,
                fitfreqs,
                ac_depths,
                debug,
            )

        # Calculate chi square difference
        x = obsfreqdata[glitchtype]["data"][0, :] - modglitches[0, :]
        w = _weight(len(x), fitfreqs["seismicweights"])
        covinv = obsfreqdata[glitchtype]["covinv"]
        if x.shape[0] == covinv.shape[0] and not any(np.isnan(x)):
            chi2rut += (x.T.dot(covinv).dot(x)) / w
        else:
            shapewarn = 1
            chi2rut = np.inf

        # Store the determined glitch parameters for outputting
        addpars["glitchparams"] = modglitches[0, -3:]

    if inferencesettings.has_epsilondifferences:
        epsdifftype = list(set(fitfreqs["fittypes"]).intersection(freqtypes.epsdiff))[0]
        obsepsdiff = obsfreqdata[epsdifftype]["data"]
        # Purge model freqs of unused modes
        l_available = [int(ll) for ll in obsepsdiff[2]]
        index = np.zeros(mod.shape[1], dtype=bool)
        for ll in [0, *l_available]:
            index |= modkey[0] == ll
        mod = mod[:, index]
        modkey = modkey[:, index]

        # Compute epsilon differences of the model
        # --> (0: deps, 1: nu, 2: l, 3: n)
        modepsdiff = freq_fit.compute_epsilondiffseqs(
            modkey,
            mod,
            libitem["dnufit"][ind],
            sequence=epsdifftype,
            nsorting=fitfreqs["nsorting"],
        )

        # Mixed modes results in negative differences. Flag using nans
        mask = np.where(modepsdiff[0, :] < 0)[0]
        modepsdiff[0, mask] = np.nan

        # Interpolate model epsdiff to the frequencies of the observations
        evalepsdiff = np.zeros(obsepsdiff.shape[1])
        evalepsdiff[:] = np.nan
        for ll in l_available:
            indobs = obsepsdiff[2] == ll
            indmod = modepsdiff[2] == ll
            nans = np.isnan(modepsdiff[0][indmod])
            if sum(~nans) > 1:
                spline = CubicSpline(
                    modepsdiff[1][indmod][~nans],
                    modepsdiff[0][indmod][~nans],
                    extrapolate=False,
                )
                evalepsdiff[indobs] = spline(obsepsdiff[1][indobs])

        # Compute chi^2 of epsilon contribution
        chi2rut = 0.0
        x = obsepsdiff[0] - evalepsdiff
        w = _weight(len(evalepsdiff), fitfreqs["seismicweights"])
        covinv = obsfreqdata[epsdifftype]["covinv"]
        chi2rut += (x.T.dot(covinv).dot(x)) / w

        # Check extreme values
        if any(np.isnan(evalepsdiff)):
            chi2rut = np.inf
            shapewarn = 3
        elif ~np.isfinite(chi2rut) or chi2rut < 0:
            chi2rut = np.inf

    # Store surfacecorrected_dnu for output
    if inferencesettings.fit_surfacecorrected_dnu:
        addpars["surfacecorrected_dnu"] = surfacecorrected_dnu

    return chi2rut, warnings, shapewarn, addpars


def find_best_model(selectedmodels, score_key: str, reducer: callable):
    """
    Generic helper to find the best model based on a score key.

    Parameters
    ----------
    selectedmodels : dict
        Dictionary mapping model path to stats.
    score_key : str
        Attribute name (e.g., 'logPDF', 'chi2') to use for comparison.
    reducer : callable
        np.argmax or np.argmin.

    Returns
    -------
    best_path : str
        Path to the model with the best score.
    best_index : int
        Index within the model with the best score.
    best_score : float
        Best score value.
    """
    best_score = -np.inf if reducer is np.argmax else np.inf
    best_path, best_index = None, None

    for path, trackstats in selectedmodels.items():
        scores = getattr(trackstats, score_key)
        i = reducer(scores)
        score = scores[i]

        if (reducer is np.argmax and score > best_score) or (
            reducer is np.argmin and score < best_score
        ):
            best_score = score
            best_path = path
            best_index = trackstats.index.nonzero()[0][i]

    if best_path is None:
        raise ValueError(f"All {score_key} values are invalid.")

    return best_path, best_index, best_score


def most_likely(selectedmodels):
    """
    Find the index of the model with the highest probability

    Parameters
    ----------
    selectedmodels : dict
        Contains information on all models with a non-zero likelihood.

    Returns
    -------
    maxPDF_path : str
        The path to the model with the highest probability.
    maxPDF_ind : int
        Index of the model with the highest probability.
    """
    return find_best_model(selectedmodels, "logPDF", np.argmax)[:2]


def lowest_chi2(selectedmodels):
    """
    Find the index of the model with the lowest chi2 (only gaussian fitting parameters)

    Parameters
    ----------
    selectedmodels : dict
        Contains information on all models with a non-zero likelihood.

    Returns
    -------
    minchi2_path : str
        Path to the model with the lowest chi2.
    minchi2_ind : int
        Index of the model with the lowest chi2.
    """
    return find_best_model(selectedmodels, "chi2", np.argmin)[:2]


def print_model_info(Grid, path, index, star, outputoptions, label, score):
    remtor._header(f"{label} model:")
    remtor._bullet(f"Score: {score}", level=0)
    remtor._bullet(f"Grid-index: {path}[{index}], with parameters:", level=0)

    if "name" in Grid[path]:
        remtor._bullet(f"Name: {Grid[path + '/name'][index].decode('utf-8')}", level=1)

    for param in outputoptions.asciiparams:
        if param == "distance":
            continue
        paramval = Grid[os.path.join(path, param)][index]

        if param.startswith("dnu") or param.startswith("numax"):
            scale = star.globalseismicparams.get_scalefactor(param)
            scaleval = paramval / scale
            scaleprt = f"(after rescaling: {scaleval:12.6f})"
        else:
            scaleprt = ""
        remtor._bullet(f"{param:10}: {paramval:12.6f} {scaleprt}", level=1)


def get_highest_likelihood(
    Grid,
    selectedmodels,
    star: core.Star,
    inferencesettings: core.InferenceSettings,
    outputoptions: core.OutputOptions,
) -> tuple[str, str]:
    """
    Find highest likelihood model and print info.

    Parameters
    ----------
    Grid : hdf5 object
        The already loaded grid, containing the tracks/isochrones.
    selectedmodels : dict
        Contains information on all models with a non-zero likelihood.
    inputparams : dict
        The standard bundle of all fitting information


    Returns
    -------
    maxPDF_path : str
        Path to model with maximum likelihood.
    maxPDF_ind : int
        Index of model with maximum likelihood.
    """
    path, index, score = find_best_model(selectedmodels, "logPDF", np.argmax)
    print_model_info(
        Grid, path, index, star, outputoptions, "Highest likelihood", score
    )
    return path, index


def get_lowest_chi2(
    Grid,
    selectedmodels,
    star: core.Star,
    inferencesettings: core.InferenceSettings,
    outputoptions: core.OutputOptions,
) -> tuple[str, str]:
    """
    Find model with lowest chi2 value and print info.

    Parameters
    ----------
    Grid : hdf5 object
        The already loaded grid, containing the tracks/isochrones.
    selectedmodels : dict
        Contains information on all models with a non-zero likelihood.
    inputparams : dict
        The standard bundle of all fitting information

    Returns
    -------
    minchi2_path : str
        Path to model with lowest chi2
    minchi2_ind : int
        Index of model with lowest chi2
    """
    path, index, score = find_best_model(selectedmodels, "chi2", np.argmin)
    print_model_info(Grid, path, index, star, outputoptions, "Lowest chi2", score)
    return path, index


def chi_for_plot(selectedmodels):
    """
    FOR VALIDATION PLOTTING!

    Could this be combined with the functions above in a wrapper?

    Return chi2 value of HLM and LCM.
    """
    _, _, maxPDFchi2 = find_best_model(selectedmodels, "logPDF", np.argmax)
    _, _, minchi2 = find_best_model(selectedmodels, "chi2", np.argmin)
    return maxPDFchi2, minchi2


def quantile_1D(data, weights, quantile):
    """
    Compute the weighted quantile of a 1D numpy array.
    The function is borrowed from the Python package wquantiles

    Parameters
    ----------
    data : ndarray
        Input array (one dimension).
    weights : ndarray
        Array with the weights of the same size of `data`.
    quantile : float
        Quantile to compute. It must have a value between 0 and 1.

    Returns
    -------
    result : float
        Weighted quantile value
    """
    if len(data) == 0:
        raise ValueError("Input data array is empty.")
    if len(data) != len(weights):
        raise ValueError("Data and weights must have the same length.")
    if not (0 <= quantile <= 1):
        raise ValueError("Quantile must be between 0 and 1.")

    sorted_indices = np.argsort(data)
    data_sorted = data[sorted_indices]
    weights_sorted = weights[sorted_indices]

    cum_weights = np.cumsum(weights_sorted)
    normalized_cum_weights = (cum_weights - 0.5 * weights_sorted) / cum_weights[-1]

    result = np.interp(quantile, normalized_cum_weights, data_sorted)
    if np.isnan(result):
        raise RuntimeError("NaN encountered in quantile_1D.")

    return result


def posterior(x, nonzeroprop, sampled_indices, nsigma=0.25):
    """
    Compute posterior of x as a weighted histogram of x with weights y.
    The bin width is determined by the Freedman Diaconis Estimator
    The histogram is smoothed using a Gaussian kernel with a width
    of nsigma*sigma

    Parameters
    ----------
    x : array
        samples
    y : array
        likelihood of samples
    nonzeroprop : array
        indices of x with non-zero likelihood
    sampled_indices :
        indices of weighted draws from y, used for bin estimation
    nsigma : float
        fractional standard deviation used for smoothing

    Returns
    -------
    like : function
        interpolated posterior
    """
    samples = x[nonzeroprop][sampled_indices]
    xmin, xmax = np.nanmin(samples), np.nanmax(samples)

    if np.allclose(xmin, xmax, rtol=1e-5):
        xvalues = np.array([xmin, xmax])
        like = np.array([1.0, 1.0])
        return interp1d(xvalues, like, fill_value=0.0, bounds_error=False)

    N = len(samples)
    bin_width = _hist_bin_fd(samples)
    if bin_width <= 0 or not np.isfinite(bin_width):
        nbins = int(np.ceil(np.sqrt(N)))
    else:
        nbins = max(1, int(np.ceil((xmax - xmin) / bin_width)))

    if nbins > N:
        nbins = int(np.ceil(np.sqrt(N)))
    bin_edges = np.linspace(xmin, xmax, nbins + 1)
    bin_centers = bin_edges[:-1] + 0.5 * np.diff(bin_edges)

    hist, _ = np.histogram(samples, bins=bin_edges, density=True)
    sigma = np.std(samples)

    if sigma > 0 and bin_width > 0:
        hist = gaussian_filter1d(hist, nsigma * sigma / bin_width)

    return interp1d(bin_centers, hist, fill_value=0.0, bounds_error=False)


def calc_key_stats(x, centroid, uncert, weights=None):
    """
    Compute statistical summary (centroid + uncertainty) of data.

    Parameters
    ----------
    x : list
        Parameter values of the models with non-zero likelihood.
    centroid : str
        Options for centroid value definition, 'median' or 'mean'.
    uncert : str
        Options for reported uncertainty format, 'quantiles' or
        'std' for standard deviation.
    weights : list, optional
        Weights needed to be applied to x before extracting statistics.

    Returns
    -------
    xcen : float
        Centroid value
    xm : float
        Lower uncertainty (16th percentile or 1-sigma).
    xp : float, None
        Upper uncertainty (84th percentile or None if std).
    """

    x = np.asarray(x)
    if weights is not None:
        weights = np.asarray(weights)

    if uncert == "quantiles":
        if weights is not None:
            xcen = quantile_1D(x, weights, 0.50)
            xm = quantile_1D(x, weights, 0.16)
            xp = quantile_1D(x, weights, 0.84)
        else:
            xm, xcen, xp = np.quantile(x, [0.16, 0.50, 0.84])
    else:
        xm = np.std(x)
        xp = None  # only used for quantile mode

    if centroid == "mean":
        xcen = np.average(x, weights=weights) if weights is not None else np.mean(x)
    elif centroid == "median" and uncert != "quantiles":
        xcen = np.median(x)

    return xcen, xm, xp
