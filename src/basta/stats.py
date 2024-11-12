"""
Key statistics functions
"""

import os
import copy
import math
import collections

import numpy as np
from scipy.interpolate import interp1d, CubicSpline
from scipy.ndimage.filters import gaussian_filter1d

from basta import freq_fit, glitch_fit
from basta import utils_seismic as su
from basta.constants import sydsun as sydc
from basta.constants import freqtypes, statdata

# Define named tuple used in selectedmodels
Trackstats = collections.namedtuple("Trackstats", "index logPDF chi2")
priorlogPDF = collections.namedtuple("Trackstats", "index logPDF chi2 bayw magw IMFw")
Trackdnusurf = collections.namedtuple("Trackdnusurf", "dnusurf")
Trackglitchpar = collections.namedtuple("Trackglitchpar", "AHe dHe tauHe")


def _hist_bin_fd(x: np.array) -> float:
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
    x : array_like
        Input data that is to be histogrammed, trimmed to range. May not
        be empty.

    Returns
    -------
    h : An estimate of the optimal bin width for the given data.
    """
    iqr = np.subtract(*np.percentile(x, [75, 25]))
    return 2.0 * iqr * x.size ** (-1.0 / 3.0)


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
    if seisw["N"]:
        N = int(seisw["N"])

    # Select weight case
    if seisw["weight"] == "1/1":
        w = 1
    elif seisw["weight"] == "1/N":
        w = N
    elif seisw["weight"] == "1/N-dof":
        if not seisw["dof"]:
            dof = 0
        else:
            dof = int(seisw["dof"])
        w = N - dof
    return w


def chi2_astero(
    obskey,
    obs,
    obsfreqmeta,
    obsfreqdata,
    obsintervals,
    libitem,
    ind,
    fitfreqs,
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
    addpars = {"dnusurf": None, "glitchparams": None}

    # Unpack model frequencies
    rawmod = libitem["osc"][ind]
    rawmodkey = libitem["osckey"][ind]
    mod = su.transform_obj_array(rawmod)
    modkey = su.transform_obj_array(rawmodkey)

    # Determine which model modes correspond to observed modes
    joins = freq_fit.calc_join(
        mod=mod, modkey=modkey, obs=obs, obskey=obskey, obsintervals=obsintervals
    )
    if joins is None:
        chi2rut = np.inf
        return chi2rut, warnings, shapewarn, 0
    else:
        joinkeys, join = joins
        nmodes = joinkeys[:, joinkeys[0, :] < 3].shape[1]

    # Apply surface correction
    if fitfreqs["fcor"] == "None":
        corjoin = join
    elif fitfreqs["fcor"] == "HK08":
        corjoin, _ = freq_fit.HK08(
            joinkeys=joinkeys,
            join=join,
            nuref=fitfreqs["numax"],
            bcor=fitfreqs["bexp"],
        )
    elif fitfreqs["fcor"] == "BG14":
        corjoin, _ = freq_fit.BG14(
            joinkeys=joinkeys, join=join, scalnu=fitfreqs["numax"]
        )
    elif fitfreqs["fcor"] == "cubicBG14":
        corjoin, _ = freq_fit.cubicBG14(
            joinkeys=joinkeys, join=join, scalnu=fitfreqs["numax"]
        )
    else:
        print(f'ERROR: fcor must be either "None" or in {freqtypes.surfeffcorrs}')
        return

    # Initialize chi2 value
    chi2rut = 0.0

    if any(x in freqtypes.freqs for x in fitfreqs["fittypes"]):
        # The frequency correction moved up before the ratios fitting!
        # --> If fitting frequencies, just add the already calculated things
        x = corjoin[0, :] - corjoin[2, :]
        w = _weight(len(corjoin[0, :]), fitfreqs["seismicweights"])
        covinv = obsfreqdata["freqs"]["covinv"]
        if x.shape[0] == covinv.shape[0]:
            chi2rut += (x.T.dot(covinv).dot(x)) / w
        else:
            shapewarn = 1
            chi2rut = np.inf

        if ~np.isfinite(chi2rut):
            chi2rut = np.inf
            if debug and verbose:
                print("DEBUG: Computed non-finite chi2, setting chi2 to inf")
        elif chi2rut < 0:
            chi2rut = np.inf
            if debug and verbose:
                print("DEBUG: chi2 less than zero, setting chi2 to inf")

    # Add large frequency separation term (using corrected frequencies!)
    # --> Equivalent to 'dnufit', but using the frequencies *after*
    #     applying the surface correction.
    # --> Compared to the observed value, which is 'dnudata'.
    if fitfreqs["dnufit_in_ratios"]:
        # Read observed dnu
        dnudata = obsfreqdata["freqs"]["dnudata"]
        dnudata_err = obsfreqdata["freqs"]["dnudata_err"]

        # Compute surface corrected dnu
        dnusurf, _ = freq_fit.compute_dnu_wfit(joinkeys, corjoin, fitfreqs["numax"])

        chi2rut += ((dnudata - dnusurf) / dnudata_err) ** 2

    # Add the chi-square terms for ratios
    if any(x in freqtypes.rtypes for x in fitfreqs["fittypes"]):
        if not all(joinkeys[1, joinkeys[0, :] < 3] == joinkeys[2, joinkeys[0, :] < 3]):
            chi2rut = np.inf
            return chi2rut, warnings, shapewarn, 0

        # Add frequency ratios terms
        ratiotype = obsfreqmeta["ratios"]["fit"][0]

        # Interpolate model ratios to observed frequencies
        if fitfreqs["interp_ratios"]:
            # Get all available model ratios
            broadratio = freq_fit.compute_ratioseqs(
                modkey,
                mod,
                ratiotype,
                threepoint=fitfreqs["threepoint"],
            )
            modratio = copy.deepcopy(obsfreqdata[ratiotype]["data"])

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
        x = obsfreqdata[ratiotype]["data"][0, :] - modratio[0, :]
        w = _weight(len(x), fitfreqs["seismicweights"])
        covinv = obsfreqdata[ratiotype]["covinv"]
        if x.shape[0] == covinv.shape[0]:
            chi2rut += (x.T.dot(covinv).dot(x)) / w
        else:
            shapewarn = 1
            chi2rut = np.inf

    # Add contribution from glitches
    if any(x in freqtypes.glitches for x in fitfreqs["fittypes"]):
        # Obtain glitch sequence to be fitted
        glitchtype = obsfreqmeta["glitch"]["fit"][0]
        # Compute surface corrected dnu, if not already computed
        if not fitfreqs["dnufit_in_ratios"]:
            dnusurf, _ = freq_fit.compute_dnu_wfit(joinkeys, corjoin, fitfreqs["numax"])

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
                dnusurf,
                fitfreqs,
                ac_depths,
                debug,
            )
            # Construct output array
            modglitches = copy.deepcopy(obsfreqdata[glitchtype]["data"])
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
                dnusurf,
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

    if any([x in freqtypes.epsdiff for x in fitfreqs["fittypes"]]):
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

    # Store dnusurf for output
    if fitfreqs["dnufit_in_ratios"]:
        addpars["dnusurf"] = dnusurf

    return chi2rut, warnings, shapewarn, addpars


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
    maxPDF = -np.inf
    for path, trackstats in selectedmodels.items():
        i = np.argmax(trackstats.logPDF)
        if trackstats.logPDF[i] > maxPDF:
            maxPDF = trackstats.logPDF[i]
            maxPDF_path = path
            maxPDF_ind = trackstats.index.nonzero()[0][i]
    if maxPDF == -np.inf:
        print("The logPDF are all -np.inf")
    return maxPDF_path, maxPDF_ind


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
    minchi2 = np.inf
    for path, trackstats in selectedmodels.items():
        i = np.argmin(trackstats.chi2)
        if trackstats.chi2[i] < minchi2:
            minchi2 = trackstats.chi2[i]
            minchi2_path = path
            minchi2_ind = trackstats.index.nonzero()[0][i]
    return minchi2_path, minchi2_ind


def chi_for_plot(selectedmodels):
    """
    FOR VALIDATION PLOTTING!

    Could this be combined with the functions above in a wrapper?

    Return chi2 value of HLM and LCM.
    """
    # Get highest likelihood model (HLM)
    maxPDF = -np.inf
    minchi2 = np.inf
    for _, trackstats in selectedmodels.items():
        i = np.argmax(trackstats.logPDF)
        j = np.argmin(trackstats.chi2)
        if trackstats.logPDF[i] > maxPDF:
            maxPDF = trackstats.logPDF[i]
            maxPDFchi2 = trackstats.chi2[i]
        if trackstats.chi2[j] < minchi2:
            minchi2 = trackstats.chi2[j]

    return maxPDFchi2, minchi2


def get_highest_likelihood(Grid, selectedmodels, inputparams):
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
    print("\nHighest likelihood model:")
    maxPDF_path, maxPDF_ind = most_likely(selectedmodels)
    print(
        "* Weighted, non-normalized log-probability:",
        np.max(selectedmodels[maxPDF_path].logPDF),
    )
    print("* Grid-index: {0}[{1}], with parameters:".format(maxPDF_path, maxPDF_ind))

    # Print name if it exists
    if "name" in Grid[maxPDF_path]:
        print("  - Name:", Grid[maxPDF_path + "/name"][maxPDF_ind].decode("utf-8"))

    # Print parameters
    outparams = inputparams["asciiparams"]
    dnu_scales = inputparams.get("dnu_scales", {})
    for param in outparams:
        if param == "distance":
            continue
        paramval = Grid[os.path.join(maxPDF_path, param)][maxPDF_ind]

        # Handle the scaled asteroseismic parameters
        if param.startswith("dnu") and param not in ["dnufit", "dnufitMos12"]:
            dnu_rescal = dnu_scales.get(param, 1.00)
            scaleval = paramval * inputparams.get("dnusun", sydc.SUNdnu) / dnu_rescal
        elif param.startswith("numax"):
            scaleval = paramval * inputparams.get("numsun", sydc.SUNnumax)
        elif param in ["dnufit", "dnufitMos12"]:
            scaleval = paramval / dnu_scales.get(param, 1.00)
        else:
            scaleval = None

        if scaleval:
            scaleprt = f"(after rescaling: {scaleval:12.6f})"
        else:
            scaleprt = ""
        print("  - {0:10}: {1:12.6f} {2}".format(param, paramval, scaleprt))
    return maxPDF_path, maxPDF_ind


def get_lowest_chi2(Grid, selectedmodels, inputparams):
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
    print("\nLowest chi2 model:")
    minchi2_path, minchi2_ind = lowest_chi2(selectedmodels)
    print("* chi2:", np.min(selectedmodels[minchi2_path].chi2))
    print("* Grid-index: {0}[{1}], with parameters:".format(minchi2_path, minchi2_ind))

    # Print name if it exists
    if "name" in Grid[minchi2_path]:
        print("  - Name:", Grid[minchi2_path + "/name"][minchi2_ind].decode("utf-8"))

    # Print parameters
    outparams = inputparams["asciiparams"]
    dnu_scales = inputparams.get("dnu_scales", {})
    for param in outparams:
        if param == "distance":
            continue
        paramval = Grid[os.path.join(minchi2_path, param)][minchi2_ind]

        # Handle the scaled asteroseismic parameters
        if param.startswith("dnu") and param not in ["dnufit", "dnufitMos12"]:
            dnu_rescal = dnu_scales.get(param, 1.00)
            scaleval = paramval * inputparams.get("dnusun", sydc.SUNdnu) / dnu_rescal
        elif param.startswith("numax"):
            scaleval = paramval * inputparams.get("numsun", sydc.SUNnumax)
        elif param in ["dnufit", "dnufitMos12"]:
            scaleval = paramval / dnu_scales.get(param, 1.00)
        else:
            scaleval = None

        if scaleval:
            scaleprt = f"(after rescaling: {scaleval:12.6f})"
        else:
            scaleprt = ""
        print("  - {0:10}: {1:12.6f} {2}".format(param, paramval, scaleprt))

    return minchi2_path, minchi2_ind


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
        The output value.
    """
    # Sort the data
    ind_sorted = np.argsort(data)
    sorted_data = data[ind_sorted]
    sorted_weights = weights[ind_sorted]

    # Compute the auxiliary arrays
    Sn = np.cumsum(sorted_weights)
    Pn = (Sn - 0.5 * sorted_weights) / np.sum(sorted_weights)

    result = np.interp(quantile, Pn, sorted_data)
    assert not np.any(np.isnan(result)), "NaN encounted in quantile_1D."

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

    # Check if all entries of x are equal
    if np.all(x == x[0]) or math.isclose(xmin, xmax, rel_tol=1e-5):
        xvalues = np.array([x[0], x[0]])
        like = np.array([1.0, 1.0])
        like = interp1d(xvalues, like, fill_value=0, bounds_error=False)
        return like

    # Compute bin width and number of bins
    N = len(samples)
    bin_width = _hist_bin_fd(samples)
    if math.isclose(bin_width, 0, rel_tol=1e-5):
        nbins = int(np.ceil(np.sqrt(N)))
    else:
        nbins = int(np.ceil((xmax - xmin) / bin_width)) + 1
    if nbins > N:
        nbins = int(np.ceil(np.sqrt(N)))
    bin_edges = np.linspace(xmin, xmax, nbins)
    xvalues = bin_edges[:-1] + np.diff(bin_edges) / 2.0
    sigma = np.std(samples)
    like = np.histogram(samples, bins=bin_edges, density=True)[0]
    if sigma > 0 and bin_width > 0:
        like = gaussian_filter1d(like, (nsigma * sigma / bin_width))
    like = interp1d(xvalues, like, fill_value=0, bounds_error=False)
    return like


def calc_key_stats(x, centroid, uncert, weights=None):
    """
    Calculate and report the wanted format of centroid value and
    uncertainties.

    Parameters
    ----------
    x : list
        Parameter values of the models with non-zero likelihood.
    centroid : str
        Options for centroid value definition, 'median' or 'mean'.
    uncert : str
        Options for reported uncertainty format, 'quantiles' or
        'std' for standard deviation.
    weights : list
        Weights needed to be applied to x before extracting statistics.

    Returns
    -------
    xcen : float
        Centroid value
    xm : float
        16'th percentile if quantile selected, 1 sigma for standard
        deviation.
    xp : float, None
        84'th percentile if quantile selected, None for standard
        deviation.
    """

    xp = None

    # Handling af all different combinations of input
    if uncert == "quantiles" and not type(weights) == type(None):
        xcen, xm, xp = quantile_1D(x, weights, statdata.quantiles)
    elif uncert == "quantiles":
        xcen, xm, xp = np.quantile(x, statdata.quantiles)
    else:
        xm = np.std(x)
    if centroid == "mean" and not type(weights) == type(None):
        xcen = np.average(x, weights=weights)
    elif centroid == "mean":
        xcen = np.mean(x)
    elif uncert != "quantiles":
        xcen = np.median(x)
    return xcen, xm, xp
