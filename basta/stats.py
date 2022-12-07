"""
Key statistics functions
"""
import copy
import math
import collections

import numpy as np
from numpy.lib.histograms import _hist_bin_fd
from scipy.interpolate import interp1d, CubicSpline
from scipy.ndimage.filters import gaussian_filter1d

from basta import freq_fit, glitch
from basta import utils_seismic as su
from basta.constants import freqtypes

# Define named tuple used in selectedmodels
Trackstats = collections.namedtuple("Trackstats", "index logPDF chi2")
priorlogPDF = collections.namedtuple("Trackstats", "index logPDF chi2 bayw magw IMFw")


def _weight(N, seisw):
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
    shapewarn=False,
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
    shapewarn : bool
        If a mismatch in array dimensions of the fitted parameters is
        encountered, this is set to True in order to warn the user at the end
        of the run.
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

    # Unpack model frequencies
    rawmod = libitem["osc"][ind]
    rawmodkey = libitem["osckey"][ind]
    mod = su.transform_obj_array(rawmod)
    modkey = su.transform_obj_array(rawmodkey)

    if any(
        x in [*freqtypes.freqs, *freqtypes.rtypes, *freqtypes.glitches]
        for x in fitfreqs["fittypes"]
    ):
        # If more observed modes than model modes are in one bin, move on
        joins = freq_fit.calc_join(
            mod=mod, modkey=modkey, obs=obs, obskey=obskey, obsintervals=obsintervals
        )
        if joins is None:
            chi2rut = np.inf
            return chi2rut, warnings, shapewarn
        else:
            joinkeys, join = joins
            nmodes = joinkeys[:, joinkeys[0, :] < 3].shape[1]

    # Apply surface correction
    if any(x in [*freqtypes.freqs, *freqtypes.rtypes] for x in fitfreqs["fittypes"]):
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
            shapewarn = True
            chi2rut = np.inf

        if ~np.isfinite(chi2rut):
            chi2rut = np.inf
            if debug and verbose:
                print("DEBUG: Computed non-finite chi2, setting chi2 to inf")
        elif chi2rut < 0:
            chi2rut = np.inf
            if debug and verbose:
                print("DEBUG: chi2 less than zero, setting chi2 to inf")

    # Add the chi-square terms for ratios
    if any(x in freqtypes.rtypes for x in fitfreqs["fittypes"]):
        if not all(joinkeys[1, joinkeys[0, :] < 3] == joinkeys[2, joinkeys[0, :] < 3]):
            chi2rut = np.inf
            return chi2rut, warnings, shapewarn

        # Add large frequency separation term (using corrected frequencies!)
        # --> Equivalent to 'dnufit', but using the frequencies *after*
        #     applying the surface correction.
        # --> Compared to the observed value, which is 'dnudata'.
        if fitfreqs["dnufit_in_ratios"]:
            # Read observed dnu
            dnudata = obsfreqdata["freqs"]["dnudata"]
            dnudata_err = obsfreqdata["freqs"]["dnudata_err"]

            FWHM_sigma = 2.0 * np.sqrt(2.0 * np.log(2.0))
            yfitdnu = corjoin[0, joinkeys[0, :] == 0]
            xfitdnu = np.arange(0, len(yfitdnu))
            wfitdnu = np.exp(
                -1.0
                * np.power(yfitdnu - fitfreqs["numax"], 2)
                / (2 * np.power(0.25 * fitfreqs["numax"] / FWHM_sigma, 2.0))
            )
            fitcoef = np.polyfit(xfitdnu, yfitdnu, 1, w=np.sqrt(wfitdnu))
            dnusurf = fitcoef[0]
            chi2rut += ((dnudata - dnusurf) / dnudata_err) ** 2

        # Add frequency ratios terms
        ratiotype = obsfreqmeta["ratios"]["fit"][0]

        # Interpolate model ratios to observed frequencies
        if fitfreqs["interp_ratios"]:
            # Get extended frequency modes to provide interpolation range of ratios
            broad_key, broad_osc = su.extend_modjoin(joinkeys, join, modkey, mod)
            if broad_key is None:
                shapewarn = True
                chi2rut = np.inf
                return chi2rut, warnings, shapewarn
            # Get model ratios
            broadratio = freq_fit.compute_ratioseqs(
                broad_key,
                broad_osc,
                ratiotype,
                threepoint=fitfreqs["threepoint"],
            )
            modratio = copy.deepcopy(obsfreqdata[ratiotype]["data"])
            # Seperate and interpolate within the separate r01, r10 and r02 sequences
            for rtype in set(obsfreqdata[ratiotype]["data"][2, :]):
                obsmask = obsfreqdata[ratiotype]["data"][2, :] == rtype
                modmask = broadratio[2, :] == rtype
                intfunc = interp1d(
                    broadratio[1, modmask], broadratio[0, modmask], kind="linear"
                )
                modratio[0, obsmask] = intfunc(
                    obsfreqdata[ratiotype]["data"][1, obsmask]
                )

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
            shapewarn = True
            chi2rut = np.inf

    # Add contribution from glitches
    if any(x in freqtypes.glitches for x in fitfreqs["fittypes"]):
        # Read model glitch parameters
        tau0 = libitem["tau0"][ind]
        tauhe = libitem["tauhe"][ind]
        taubcz = libitem["taubcz"][ind]

        if nmodes > 200:
            raise NotImplementedError("> 200 modes to fit!")
        freq = np.zeros((200, 4), dtype=float)

        # l and n
        maxl = 3
        freq[0:nmodes, 0] = joinkeys[0, joinkeys[0, :] < maxl]
        freq[0:nmodes, 1] = joinkeys[1, joinkeys[0, :] < maxl]

        # Model frequency and observational uncertainty used for weighing
        freq[0:nmodes, 2] = join[0, joinkeys[0, :] < maxl]
        freq[0:nmodes, 3] = join[3, joinkeys[0, :] < maxl]
        nu1 = np.amin(join[2, joinkeys[0, :] < maxl])
        nu2 = np.amax(join[2, joinkeys[0, :] < maxl])

        glhParams, nerr = glitch.glh_params(freq, nmodes, nu1, nu2, tau0, tauhe, taubcz)
        if nerr == 0:
            x = obsfreqdata["glitches"]["data"] - glhParams
            covinv = obsfreqdata["glitches"]["covinv"]
            w = _weight(len(x), fitfreqs["seismicweights"])
            if x.shape[0] == covinv.shape[0]:
                chi2rut += (x.T.dot(covinv).dot(x)) / w
            else:
                shapewarn = True
                chirut = np.inf
        else:
            chi2rut = np.inf
            return chi2rut, warnings, shapewarn

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

        # Interpolate model epsdiff to the frequencies of the observations
        evalepsdiff = np.zeros(obsepsdiff.shape[1])
        for ll in l_available:
            indobs = obsepsdiff[2] == ll
            indmod = modepsdiff[2] == ll
            spline = CubicSpline(modepsdiff[1][indmod], modepsdiff[0][indmod])
            evalepsdiff[indobs] = spline(obsepsdiff[1][indobs])

        # Compute chi^2 of epsilon contribution
        chi2rut = 0.0
        x = obsepsdiff[0] - evalepsdiff
        w = _weight(len(evalepsdiff), fitfreqs["seismicweights"])
        covinv = obsfreqdata[epsdifftype]["covinv"]
        chi2rut += (x.T.dot(covinv).dot(x)) / w

        # Check extreme values
        if ~np.isfinite(chi2rut):
            chi2rut = np.inf
        elif chi2rut < 0:
            chi2rut = np.inf

    return chi2rut, warnings, shapewarn


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


def get_highest_likelihood(Grid, selectedmodels, outparams):
    """
    Find highest likelihood model and print info.

    Parameters
    ----------
    Grid : hdf5 object
        The already loaded grid, containing the tracks/isochrones.
    selectedmodels : dict
        Contains information on all models with a non-zero likelihood.
    outparams : dict
        Dict containing the wanted output of the run.

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
    for param in outparams:
        if param == "distance":
            continue
        print(
            "  - {0:10}: {1:12.6f}".format(
                param, Grid[maxPDF_path + "/" + param][maxPDF_ind]
            )
        )
    return maxPDF_path, maxPDF_ind


def get_lowest_chi2(Grid, selectedmodels, outparams):
    """
    Find model with lowest chi2 value and print info.

    Parameters
    ----------
    Grid : hdf5 object
        The already loaded grid, containing the tracks/isochrones.
    selectedmodels : dict
        Contains information on all models with a non-zero likelihood.
    outparams : dict
        Dict containing the wanted output of the run.

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
    for param in outparams:
        if param == "distance":
            continue
        print(
            "  - {0:10}: {1:12.6f}".format(
                param, Grid[minchi2_path + "/" + param][minchi2_ind]
            )
        )
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
    bin_width = _hist_bin_fd(samples, None)
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
    # Definition of Bayesian 16, 50, and 84 percentiles
    qs = [0.5, 0.158655, 0.841345]
    xp = None

    # Handling af all different combinations of input
    if uncert == "quantiles" and not type(weights) == type(None):
        xcen, xm, xp = quantile_1D(x, weights, qs)
    elif uncert == "quantiles":
        xcen, xm, xp = np.quantile(x, qs)
    else:
        xm = np.std(x)
    if centroid == "mean" and not type(weights) == type(None):
        xcen = np.average(x, weights=weights)
    elif centroid == "mean":
        xcen = np.mean(x)
    elif uncert != "quantiles":
        xcen = np.median(x)
    return xcen, xm, xp
