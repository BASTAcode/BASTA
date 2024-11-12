"""
Fitting of frequencies and frequency-dependent properties.
"""

import numpy as np
import itertools
from sklearn import linear_model
from scipy.optimize import minimize
from scipy.interpolate import CubicSpline

from basta import utils_seismic as su


"""
Individual frequencies
"""


def compute_dnu_wfit(obskey, obs, numax):
    """
    Compute large frequency separation weighted around numax, the same way as dnufit.
    Coefficients based on White et al. 2011.

    Parameters
    ----------
    obskey : array
        Array containing the angular degrees and radial orders of obs
    obs : array
        Individual frequencies and uncertainties.
    numax : scalar
        Frequency of maximum power

    Returns
    -------
    dnu : scalar
        Large frequency separation obtained by fitting the radial mode observed
        frequencies.
    dnu_err : scalar
        Uncertainty on dnudata.
    """

    FWHM_sigma = 2.0 * np.sqrt(2.0 * np.log(2.0))
    yfitdnu = obs[0, obskey[0, :] == 0]
    xfitdnu = np.arange(0, len(yfitdnu))
    xfitdnu = obskey[1, obskey[0, :] == 0]
    wfitdnu = np.exp(
        -1.0
        * np.power(yfitdnu - numax, 2)
        / (2 * np.power(0.25 * numax / FWHM_sigma, 2.0))
    )
    fitcoef, fitcov = np.polyfit(xfitdnu, yfitdnu, 1, w=np.sqrt(wfitdnu), cov=True)
    dnu, dnu_err = fitcoef[0], np.sqrt(fitcov[0, 0])

    return dnu, dnu_err


def make_intervals(osc, osckey, dnu=None):
    """
    This function computes the interval bins used in the frequency
    mapping in :func:`freqfit.calc_join()`.

    Parameters
    ----------
    osc : array
        Array containing either model or observed modes.
    osckey : array
        Array containing the angular degrees and radial orders of osc
    dnu : float, optional
        Large frequency separation in microhertz.
        If left None, the large frequency separation will be computed as
        the median difference between consecutive l=0 modes.

    Returns
    -------
    intervals : array
        Array containing the endpoints of the intervals used in the frequency
        fitting routine in :func:`freq_fit.calc_join``.
    """
    # Get l=0 modes
    osckeyl0, oscl0 = su.get_givenl(l=0, osc=osc, osckey=osckey)
    fl0 = oscl0[0, :]
    nl0 = osckeyl0[1, :]

    if dnu is None:
        dnu = np.median(np.diff(fl0))

    # limit is a fugde factor that determines the size in frequency of a gap
    limit = 1.9
    difffl0 = np.diff(fl0) > (limit * dnu)

    # Fix gaps in l=0 by generating 'fake' l=0 used in the intervals
    while np.any(difffl0):
        print("Warning: gap in l=0 detected!")
        i = np.nonzero(difffl0)[0][0]
        fl0 = np.insert(fl0, i + 1, fl0[i] + dnu)
        difffl0 = np.diff(fl0) > (limit * dnu)

    # Make the binning
    intervals = np.arange(nl0[0] + 1) * -dnu + fl0[0]
    intervals = intervals[::-1]
    intervals = np.concatenate((intervals, fl0[1:-1]))
    upper = np.arange(np.amax(osckey[1, :]) + 2 - len(intervals)) * dnu + fl0[-1]
    intervals = np.concatenate((intervals, upper))
    return intervals


def calc_join(mod, modkey, obs, obskey, obsintervals=None, dnu=None):
    """
    This functions maps the observed modes to the model modes.

    Parameters
    ----------
    mod : array
        Array containing the modes in the model.
    modkey : array
        Array containing the angular degrees and radial orders of mod
    obs : array
        Array containing the modes in the observed data
    obskey : array
        Array containing the angular degrees and radial orders of obs
    obsintervals : array, optional
        Array containing the endpoints of the intervals used in the frequency
        fitting routine in freq_fit.calc_join.
        As it is the same in all iterations for the observed frequencies,
        when computing chi2, this is computed in su.prepare_obs once
        and given as an argument in order to save time.
        If obsintervals is None, it will be computed.
    dnu : float, optional
        Large frequency separation in microhertz used for the computation
        of `obsintervals`.
        If left None, the large frequency separation will be computed as
        the median difference between consecutive l=0 modes.

    Returns
    -------
    joins : list or None
        List containing joinkeys and join.

        * joinkeys : int array
            Array containing the mapping of model mode to the observed mode.
            Model mode (l=joinkeys[i, 0], n=joinkeys[i, 1])
            is mapped to observed mode (l=joinkeys[i, 0], n=joinkeys[i, 2]).

        * join : array
            Array containing the model frequency, model inertia, observed
            frequency, uncertainty in observed frequency for a pair of mapped
            modes.
            Model mode (l=joinkeys[i, 0], n=joinkeys[i, 1]) has frequency
            join[i, 0], inertia join[i, 1].
            Observed mode (l=joinkeys[i, 0], n=joinkeys[i, 2]) has frequency
            join[i, 2] and uncertainty join[i, 3].
    """
    # Move obsintervals out to optimize (as it is the same every time)
    if obsintervals is None:
        obsintervals = make_intervals(osc=obs, osckey=obskey, dnu=dnu)
    modintervals = make_intervals(osc=mod, osckey=modkey)

    # Initialise
    join = []
    joinkeys = []

    # Count the number of observed and modelled modes in each bin
    for l in [0, 1, 2]:
        obskey_givenl, obs_givenl = su.get_givenl(l=l, osc=obs, osckey=obskey)
        modkey_givenl, mod_givenl = su.get_givenl(l=l, osc=mod, osckey=modkey)
        nobs = obskey_givenl[1, :]
        nmod = modkey_givenl[1, :]
        fobs, eobs = obs_givenl
        fmod, emod = mod_givenl

        minlength = min(len(obsintervals), len(modintervals))
        for i in range(minlength - 1):
            ofilter = (obsintervals[i] <= fobs) & (fobs < obsintervals[i + 1])
            mfilter = (modintervals[i] <= fmod) & (fmod < modintervals[i + 1])

            msum = np.sum(mfilter)
            osum = np.sum(ofilter)
            if osum > 0:  # & (msum > 0):
                if msum == osum:
                    pass
                elif msum > osum:
                    # Look at the inertia at k=osum+1
                    emodsort = np.sort(emod[mfilter])
                    ke = emodsort[osum]
                    # Divide the modes into three area: sure, maybe & discard
                    # This is done by arbitarily choosing that
                    # everything with inertias below ke/10 should be matched,
                    # everything with inertias higher than 10*ke are not,
                    # and a subset of the in-between is selected based on the
                    # the distance in frequency space.
                    avedist = (emodsort[osum] - emodsort[0]) / (osum + 1)
                    sure = (emod < (ke / 10)) & (mfilter)
                    maybe = (
                        (emod < np.amax((ke + (avedist), emodsort[0] * 10)))
                        & (~sure)
                        & (mfilter)
                    )
                    maybe = maybe.nonzero()[0]
                    sure = sure.nonzero()[0]
                    bestchi2 = np.inf
                    solution = None
                    if len(sure) == osum:
                        solution = sure
                    else:
                        # Here we check all subsets of maybe.
                        # First, find the proper offset
                        if osum == 1:
                            offset = offsets[
                                (np.abs(matched_l0s[2, :] - fmod[mfilter][0])).argmin()
                            ]
                        else:
                            offset = 0
                        for ss in itertools.combinations(maybe, osum - len(sure)):
                            subset = np.sort(np.concatenate((ss, sure)))
                            fmodsubset = fmod[subset]
                            # offs = [(np.abs(matched_l0s[2, :] - f)).argmin()
                            #    for f in fmodsubset]
                            chi2 = np.sum(np.abs((fmodsubset - offset - fobs[ofilter])))
                            if chi2 < bestchi2:
                                bestchi2 = chi2
                                solution = subset
                    assert solution is not None
                    mfilter = np.zeros(len(mfilter), dtype=bool)
                    mfilter[solution] = True
                elif msum < osum:
                    # If more observed modes than model, move on to next model
                    return None

                joinkeys.append(
                    np.transpose(
                        [l * np.ones(osum, dtype=int), nmod[mfilter], nobs[ofilter]]
                    )
                )
                join.append(
                    np.transpose(
                        [fmod[mfilter], emod[mfilter], fobs[ofilter], eobs[ofilter]]
                    )
                )
        if l == 0:
            # Compute the offset due to the surface effect to use in the case
            # of mixed modes for the l=1 and l=2 matching.
            same_ns = np.transpose(np.concatenate(joinkeys))[1, :]
            matched_l0s = np.transpose(np.concatenate(join))
            # Compute offset
            # modmask = [mode in same_ns for mode in modkey_givenl[1, :]]
            # obsmask = [mode in same_ns for mode in obskey_givenl[1, :]]
            # offsets = mod_givenl[0, modmask] - obs_givenl[0, obsmask]
            offsets = matched_l0s[0, :] - matched_l0s[2, :]
    if len(join) != 0:
        joins = [
            np.transpose(np.concatenate(joinkeys)),
            np.transpose(np.concatenate(join)),
        ]
        return joins
    else:
        return None


def HK08(joinkeys, join, nuref, bcor):
    """
    Kjeldsen frequency correction

    Correcting stellar oscillation frequencies for near-surface effects
    following the approach in Hans Kjeldsen, Timothy R. Bedding, and Jørgen
    Christensen-Dalsgaard. "Correcting stellar oscillation frequencies for
    near-surface effects." `The Astrophysical Journal Letters 683.2 (2008):
    L175.`

    The correction has the form:

        .. math:: r\\nu{corr} - \\nu{model} = \\frac{a}{Q}\\left(\\frac{\\nu{model}}{\\nu_{max}}\\right)^b .

    Parameters
    ----------
    joinkeys : int array
        Array containing the mapping of model mode to the observed mode.
        Model mode (l=joinkeys[i, 0], n=joinkeys[i, 1])
        is mapped to observed mode (l=joinkeys[i, 0], n=joinkeys[i, 2]).
    join : array
        Array containing the model frequency, model inertia, observed
        frequency, uncertainty in observed frequency for a pair of mapped
        modes.
        Model mode (l=joinkeys[i, 0], n=joinkeys[i, 1]) has frequency
        join[i, 0], inertia join[i, 1].
        Observed mode (l=joinkeys[i, 0], n=joinkeys[i, 2]) has frequency
        join[i, 2] and uncertainty join[i, 3].
    nuref : float
        The reference frequency used in the Kjeldsen correction. Normally the
        reference frequency used is numax of the observed star or numax of the
        Sun, but it shouldn't make a difference.
    bcor : float
        The exponent in the Kjeldsen correction.

    Returns
    -------
    cormod : array
        Array containing the corrected model frequencies and the unchanged
        mode inertias in the same format as the osc-arrays.
    coeffs : array
        Array containing the coefficients in the found correction.
    """
    # Unpacking for readability
    # If we do not have many modes but many mixed modes: only use l=0
    _, joinl0 = su.get_givenl(l=0, osc=join, osckey=joinkeys)
    f_modell0 = joinl0[0, :]
    e_modell0 = joinl0[1, :]
    f_obsl0 = joinl0[2, :]

    f_model = join[0, :]
    e_model = join[1, :]
    f_obs = join[2, :]
    e_obs = join[3, :]

    # Interpolate inertia to l=0 inertia at same frequency
    interp_inertia = np.interp(f_model, f_modell0, e_modell0)

    # Calculate Q's
    q_nl = e_model / interp_inertia

    # Compute quantities used in the Kjelden et al. 2008 paper
    dnuavD = np.mean(np.diff(f_obsl0))
    dnuavM = np.mean(np.diff(f_modell0))

    rpar = (bcor - 1) / (bcor * (np.mean(f_model) / np.mean(f_obs)) - dnuavM / dnuavD)
    acor = (
        len(f_obs)
        * (np.mean(f_obs) - rpar * np.mean(f_model))
        / np.sum(((f_obs / nuref) ** bcor) / q_nl)
    )
    coeffs = [rpar, acor, bcor]

    # Apply the frequency correction
    corosc = apply_HK08(modkey=joinkeys, mod=join, coeffs=coeffs, scalnu=nuref)
    corjoin = np.copy(join)
    corjoin[0, :] = np.asarray(corosc)

    return corjoin, coeffs


def apply_HK08(modkey, mod, coeffs, scalnu):
    """
    Applies the HK08 frequency correction to frequency for a given set
    of coefficients

    Parameters
    ----------
    modkey : array
        Array containing the angular degrees and radial orders of mod.
    mod : array
        Array containing the modes in the model.
    coeffs : list
        List containing [rpar, acor, bcor] used in the HK08 correction.
    scalnu : float
        A scaling frequency used purely to non-dimensionalize the frequencies.

    Returns
    -------
    corosc : array
        Array containing the corrected model frequencies.
    """
    # Unpack
    f_model = mod[0, :]
    e_model = mod[1, :]

    modkeyl0, modl0 = su.get_givenl(l=0, osc=mod, osckey=modkey)
    f_modell0 = modl0[0, :]
    e_modell0 = modl0[1, :]

    # Interpolate inertia to l=0 inertia at same frequency
    interp_inertia = np.interp(f_model, f_modell0, e_modell0)

    # Calculate Q's
    q_nl = e_model / interp_inertia

    corosc = (f_model / coeffs[0]) + (coeffs[1] / (coeffs[0] * q_nl)) * (
        (f_model / scalnu) ** coeffs[2]
    )
    return corosc


def cubicBG14(joinkeys, join, scalnu, method="l1", onlyl0=False):
    """
    Ball & Gizon frequency correction

    Correcting stellar oscillation frequencies for near-surface effects.
    This routine follows the approach from
    Warrick H. Ball and Laurent Gizon.
    "A new correction of stellar oscillation frequencies for near-surface effects."
    Astronomy & Astrophysics 568 (2014): A123.

    The correction has the form:

        .. math:: d\\nu = \\frac{b \\left(\\frac{\\nu}{\\nu_{scal}}\\right)^3}{I}

    where :math:`b` are the found parameters, :math:`\\nu_{scal}` is a scaling frequency used to non-dimensionalize the frequencies :math:`\\nu`, and :math:`I` is the mode inertia for each mode.

    Parameters
    ----------
    joinkeys : int array
        Array containing the mapping of model mode to the observed mode.
        Model mode (l=joinkeys[i, 0], n=joinkeys[i, 1])
        is mapped to observed mode (l=joinkeys[i, 0], n=joinkeys[i, 2]).
    join : array
        Array containing the model frequency, model inertia, observed
        frequency, uncertainty in observed frequency for a pair of mapped
        modes.
        Model mode (l=joinkeys[i, 0], n=joinkeys[i, 1]) has frequency
        join[i, 0], inertia join[i, 1].
        Observed mode (l=joinkeys[i, 0], n=joinkeys[i, 2]) has frequency
        join[i, 2] and uncertainty join[i, 3].
    scalnu : float
        A scaling frequency used purely to non-dimensionalize the frequencies.
    method : string, optional
        The name of the fitting method used.
        It can be 'ransac' (default), 'l1', or 'l2'.
    onlyl0 : bool
        Flag that if True only computes the correction based on the l=0 modes.
        This can be usefull if not many modes have been observed.

    Returns
    -------
    cormod : array
        Array containing the corrected model frequencies and the unchanged
        mode inertias in the same format as the osc-arrays.
    coeffs : array
        Array containing the coefficients a and b to the found correction.
    """
    # Unpacking for readability
    # If we do not have many modes but many mixed modes: only use l=0
    if onlyl0:
        _, joinl0 = su.get_givenl(l=0, osc=join, osckey=joinkeys)

        f_model = joinl0[0, :]
        f_inertia = joinl0[1, :]
        f_obs = joinl0[2, :]
        f_unc = joinl0[3, :]
    else:
        f_model = join[0, :]
        f_inertia = join[1, :]
        f_obs = join[2, :]
        f_unc = join[3, :]

    # Initialize vector y and matrix X
    nmodes = len(f_obs)
    y = np.zeros((nmodes, 1))
    matX = np.zeros((nmodes, 1))

    # Filling in y and X
    y[:, 0] = (f_obs - f_model) / f_unc
    matX[:, 0] = np.power((f_model / scalnu), 3) / (f_inertia * f_unc)

    # Try for the l1-norm minimization
    def l1(params, data, labels):
        prediction = np.dot(data, params.reshape(-1, 1))
        dist = prediction - labels
        return (np.abs(dist)).sum()

    def l2(params, data, labels):
        prediction = np.dot(data, params.reshape(-1, 1))
        dist = prediction - labels
        return (dist**2).sum()

    if method == "ransac":
        np.random.seed(5)
        lr = linear_model.LinearRegression(fit_intercept=False)
        ransac = linear_model.RANSACRegressor(lr)
        ransac.fit(matX, y)
        coeffs = ransac.estimator_.coef_.reshape(-1, 1)
        coeffs = np.asarray([coeffs[0][0]])
    elif method == "l1":
        initial_params = np.zeros(1)
        res = minimize(l1, initial_params, method="Nelder-Mead", args=(matX, y))
        coeffs = res.x
        coeffs = np.asarray([coeffs[0]])
    elif method == "l2":
        # Calculation of coefficients using Penrose-Moore coefficients
        # XTXinv = np.linalg.inv(np.dot(matX.T, matX))
        # XTy = np.dot(matX.T, y)
        # coeffs = np.dot(XTXinv, XTy)
        # Or shorter (and more efficient):
        l2res = np.linalg.lstsq(matX, y)[0]
        coeffs = np.array([l2res[0][0]])

    # Apply the frequency correction
    corjoin = np.copy(join)
    corosc = apply_cubicBG14(modkey=joinkeys, mod=join, coeffs=coeffs, scalnu=scalnu)
    corjoin[0, :] = corosc

    return corjoin, coeffs


def apply_cubicBG14(modkey, mod, coeffs, scalnu):
    """
    Applies the BG14 cubic frequency correction to frequency for a given set
    of coefficients

    Parameters
    ----------
    modkey : array
        Array containing the angular degrees and radial orders of mod.
    mod : array
        Array containing the modes in the model.
    coeffs : list
        List containing the two coefficients used in the BG14 correction.
    scalnu : float
        A scaling frequency used purely to non-dimensionalize the frequencies.

    Returns
    -------
    corosc : array
        Array containing the corrected model frequencies.
    """
    corosc = []
    # The correction is applied to all model modes with non-zero inertia
    for l in [0, 1, 2]:
        lmask = modkey[0, :] == l
        df = (coeffs[0] * np.power((mod[0, lmask] / scalnu), 3)) / (mod[1, lmask])
        corosc.append(mod[0, lmask] + df)
    corosc = np.asarray(np.concatenate(corosc))
    return corosc


def BG14(joinkeys, join, scalnu, method="l1", onlyl0=False):
    """
    Ball & Gizon frequency correction

    Correcting stellar oscillation frequencies for near-surface effects.
    This routine follows the approach from Warrick H. Ball and Laurent Gizon. "A new correction of stellar oscillation frequencies for near-surface effects." Astronomy & Astrophysics 568 (2014): A123.

    The correction has the form:

        .. math:: d\\nu = \\frac{a \\left(\\frac{\\nu}{\\nu_{scal}}\\right)^{-1} + b \\left(\\frac{\\nu}{\\nu_{scal}}\\right)^3}{I}

    where :math:`a` and :math:`b` are the found parameters, :math:`\\nu_{scal}` is a scaling frequency used to non-dimensionalize the frequencies :math:`\\nu`, and :math:`I` is the mode inertia for each mode.

    Parameters
    ----------
    joinkeys : int array
        Array containing the mapping of model mode to the observed mode.
        Model mode (l=joinkeys[i, 0], n=joinkeys[i, 1])
        is mapped to observed mode (l=joinkeys[i, 0], n=joinkeys[i, 2]).
    join : array
        Array containing the model frequency, model inertia, observed
        frequency, uncertainty in observed frequency for a pair of mapped
        modes.
        Model mode (l=joinkeys[i, 0], n=joinkeys[i, 1]) has frequency
        join[i, 0], inertia join[i, 1].
        Observed mode (l=joinkeys[i, 0], n=joinkeys[i, 2]) has frequency
        join[i, 2] and uncertainty join[i, 3].
    scalnu : float
        A scaling frequency used purely to non-dimensionalize the frequencies.
    method : string, optional
        The name of the fitting method used.
        It can be 'ransac' (default), 'l1', or 'l2'.
    onlyl0 : bool
        Flag that if True only computes the correction based on the l=0 modes.
        This can be usefull if not many modes have been observed.

    Returns
    -------
    cormod : array
        Array containing the corrected model frequencies and the unchanged
        mode inertias in the same format as the osc-arrays.
    coeffs : array
        Array containing the coefficients a and b to the found correction.
    """
    # Unpacking for readability
    # If we do not have many modes but many mixed modes: only use l=0
    if onlyl0:
        _, joinl0 = su.get_givenl(l=0, osc=join, osckey=joinkeys)

        f_model = joinl0[0, :]
        f_inertia = joinl0[1, :]
        f_obs = joinl0[2, :]
        f_unc = joinl0[3, :]
    else:
        f_model = join[0, :]
        f_inertia = join[1, :]
        f_obs = join[2, :]
        f_unc = join[3, :]

    # Initialize vector y and matrix X
    nmodes = len(f_obs)
    y = np.zeros((nmodes, 1))
    matX = np.zeros((nmodes, 2))

    # Filling in y and X
    y[:, 0] = (f_obs - f_model) / f_unc
    matX[:, 0] = np.power((f_model / scalnu), -1) / (f_inertia * f_unc)
    matX[:, 1] = np.power((f_model / scalnu), 3) / (f_inertia * f_unc)

    # Try for the l1-norm minimization
    def l1(params, data, labels):
        prediction = np.dot(data, params.reshape(-1, 1))
        dist = prediction - labels
        return (np.abs(dist)).sum()

    def l2(params, data, labels):
        prediction = np.dot(data, params.reshape(-1, 1))
        dist = prediction - labels
        return (dist**2).sum()

    if method == "ransac":
        np.random.seed(5)
        lr = linear_model.LinearRegression(fit_intercept=False)
        ransac = linear_model.RANSACRegressor(lr)
        ransac.fit(matX, y)
        coeffs = ransac.estimator_.coef_.reshape(-1, 1)
        coeffs = np.asarray([coeffs[0][0], coeffs[1][0]])
    elif method == "l1":
        initial_params = np.zeros(2)
        res = minimize(l1, initial_params, method="Nelder-Mead", args=(matX, y))
        coeffs = res.x
        coeffs = np.asarray([coeffs[0], coeffs[1]])
    elif method == "l2":
        # Calculation of coefficients using Penrose-Moore coefficients
        # XTXinv = np.linalg.inv(np.dot(matX.T, matX))
        # XTy = np.dot(matX.T, y)
        # coeffs = np.dot(XTXinv, XTy)
        # Or shorter (and more efficient):
        l2res = np.linalg.lstsq(matX, y)[0]
        coeffs = np.array([l2res[0][0], l2res[1][0]])

    # Apply the frequency correction
    corjoin = np.copy(join)
    corosc = apply_BG14(modkey=joinkeys, mod=join, coeffs=coeffs, scalnu=scalnu)
    corjoin[0, :] = corosc

    return corjoin, coeffs


def apply_BG14(modkey, mod, coeffs, scalnu):
    """
    Applies the BG14 frequency correction to frequency for a given set
    of coefficients

    Parameters
    ----------
    modkey : array
        Array containing the angular degrees and radial orders of mod.
    mod : array
        Array containing the modes in the model.
    coeffs : list
        List containing the two coefficients used in the BG14 correction.
    scalnu : float
        A scaling frequency used purely to non-dimensionalize the frequencies.

    Returns
    -------
    corosc : array
        Array containing the corrected model frequencies.
    """
    corosc = []
    # The correction is applied to all model modes with non-zero inertia
    for l in [0, 1, 2]:
        lmask = modkey[0, :] == l
        df = (
            coeffs[0] * np.power((mod[0, lmask] / scalnu), -1)
            + coeffs[1] * np.power((mod[0, lmask] / scalnu), 3)
        ) / (mod[1, lmask])
        corosc.append(mod[0, lmask] + df)
    corosc = np.asarray(np.concatenate(corosc))
    return corosc


"""
Frequency ratios
"""


def compute_ratios(obskey, obs, ratiotype, nrealisations=10000, threepoint=False):
    """
    Routine to compute the ratios r02, r01 and r10 from oscillation
    frequencies, and return the desired ratio sequence, both individual
    and combined sequences, along with their covariance matrix and its
    inverse.

    Developers note: We currently do not store identifying information
    of the sequence origin in the combined sequences. We rely on them
    being constructed/sorted identically when computed.

    Parameters
    ----------
    obskey : array
        Harmonic degrees, radial orders and radial orders of frequencies.
    obs : array
        Frequencies and their error, following the structure of obs.
    ratiotype : str
        Which ratio sequence to determine, see constants.freqtypes.rtypes
        for possible sequences.
    nrealizations : int
        Number of realizations of the sampling for the computation of the
        covariance matrix.
    threepoint : bool
        If True, use three point definition of r01 and r10 ratios
        instead of default five point definition.

    Returns
    -------
    ratio : array
        Ratio requested from `ratiotype`.
    ratio_cov : array
        Covariance matrix of the requested ratio.
    """
    ratio = compute_ratioseqs(obskey, obs, ratiotype, threepoint=threepoint)

    # Check for valid return
    if ratio is None:
        return None

    ratio_cov = su.compute_cov_from_mc(
        ratio.shape[1],
        obskey,
        obs,
        ratiotype,
        args={"threepoint": threepoint},
        nrealisations=nrealisations,
    )
    return ratio, ratio_cov


def compute_ratioseqs(obskey, obs, sequence, threepoint=False):
    """
    Routine to compute the ratios r02, r01 and r10 from oscillation
    frequencies, and return the desired ratio sequence, both individual
    and combined sequences.

    Developers note: We currently do not store identifying information
    of the sequence origin in the combined sequences. We rely on them
    being constructed/sorted identically when computed.

    Parameters
    ----------
    obskey : array
        Harmonic degrees, radial orders and radial orders of frequencies
    obs : array
        Frequencies and their error, following the structure of obs
    sequence : str
        Which ratio sequence to determine, see constants.freqtypes.rtypes
        for possible sequences.
    threepoint : bool
        If True, use three point definition of r01 and r10 ratios
        instead of default five point definition.

    Returns
    -------
    ratio : array
        Ratio requested from `sequence`. First index correspond to:
        0 - Frequency ratios
        1 - Defining/corresponding frequency
        2 - Identifying integer (r01: 1, r02: 2, r10: 10)
        3 - Identifying radial order n
    """
    r01, r10, r02 = True, True, True

    f0 = obs[0, obskey[0, :] == 0]
    n0 = obskey[1, obskey[0, :] == 0]
    f1 = obs[0, obskey[0, :] == 1]
    n1 = obskey[1, obskey[0, :] == 1]
    f2 = obs[0, obskey[0, :] == 2]
    n2 = obskey[1, obskey[0, :] == 2]

    if (len(f0) == 0) or (len(f0) != (n0[-1] - n0[0] + 1)):
        r01 = None
        r10 = None

    if (len(f1) == 0) or (len(f1) != (n1[-1] - n1[0] + 1)):
        r01 = None
        r10 = None

    if (len(f2) == 0) or (len(f2) != (n2[-1] - n2[0] + 1)):
        r02 = None

    # Two-point frequency ratio R02
    # -----------------------------
    if r02 and sequence in ["r02", "r012", "r102"]:
        lowest_n0 = (n0[0] - 1, n1[0], n2[0])
        l0 = lowest_n0.index(max(lowest_n0))

        # Find lowest indices for l = 0, 1, and 2
        if l0 == 0:
            i00 = 0
            i01 = n0[0] - n1[0] - 1
            i02 = n0[0] - n2[0] - 1
        elif l0 == 1:
            i00 = n1[0] - n0[0] + 1
            i01 = 0
            i02 = n1[0] - n2[0]
        elif l0 == 2:
            i00 = n2[0] - n0[0] + 1
            i01 = n2[0] - n1[0]
            i02 = 0

        # Number of r02s
        nn = (n0[-1], n1[-1], n2[-1] + 1)
        ln = nn.index(min(nn))
        if ln == 0:
            nr02 = n0[-1] - n0[i00] + 1
        elif ln == 1:
            nr02 = n1[-1] - n1[i01]
        elif ln == 2:
            nr02 = n2[-1] - n2[i02] + 1

        # R02
        r02 = np.zeros((4, nr02))
        r02[2, :] = 2
        for i in range(nr02):
            r02[3, i] = n0[i00 + i]
            r02[1, i] = f0[i00 + i]
            r02[0, i] = f0[i00 + i] - f2[i02 + i]
            r02[0, i] /= f1[i01 + i + 1] - f1[i01 + i]

    # Five-point frequency ratio R01
    # ------------------------------
    if not threepoint:
        if r01 and sequence in ["r01", "r012", "r010"]:
            # Find lowest indices for l = 0 and 1
            if n0[0] >= n1[0]:
                i00 = 0
                i01 = n0[0] - n1[0]
            else:
                i00 = n1[0] - n0[0]
                i01 = 0

            # Number of r01s
            if n0[-1] - 1 >= n1[-1]:
                nr01 = n1[-1] - n1[i01]
            else:
                nr01 = n0[-1] - n0[i00] - 1

            # R01
            r01 = np.zeros((4, nr01))
            r01[2, :] = 1
            for i in range(nr01):
                r01[3, i] = n0[i00 + i + 1]
                r01[1, i] = f0[i00 + i + 1]
                r01[0, i] = f0[i00 + i] + 6.0 * f0[i00 + i + 1] + f0[i00 + i + 2]
                r01[0, i] -= 4.0 * (f1[i01 + i + 1] + f1[i01 + i])
                r01[0, i] /= 8.0 * (f1[i01 + i + 1] - f1[i01 + i])

        if r10 and sequence in ["r10", "r102", "r010"]:
            # Find lowest indices for l = 0 and 1
            if n0[0] - 1 >= n1[0]:
                i00 = 0
                i01 = n0[0] - n1[0] - 1
            else:
                i00 = n1[0] - n0[0] + 1
                i01 = 0

            # Number of r10s
            if n0[-1] >= n1[-1]:
                nr10 = n1[-1] - n1[i01] - 1
            else:
                nr10 = n0[-1] - n0[i00]

            # R10
            r10 = np.zeros((4, nr10))
            r10[2, :] = 10
            for i in range(nr10):
                r10[3, i] = n1[i01 + i + 1]
                r10[1, i] = f1[i01 + i + 1]
                r10[0, i] = f1[i01 + i] + 6.0 * f1[i01 + i + 1] + f1[i01 + i + 2]
                r10[0, i] -= 4.0 * (f0[i00 + i + 1] + f0[i00 + i])
                r10[0, i] /= -8.0 * (f0[i00 + i + 1] - f0[i00 + i])

    # Three-point frequency ratios
    # ----------------------------
    else:
        if r01 and sequence in ["r01", "r012", "r010"]:
            # Find lowest indices for l = 0 and 1
            # i01 point to one n-value lower than i00
            if n0[0] - 1 >= n1[0]:
                i00 = 0
                i01 = n0[0] - n1[0] - 1
            else:
                i00 = n1[0] - n0[0] + 1
                i01 = 0

            # Number of r01s
            if n0[-1] >= n1[-1]:
                nr01 = n1[-1] - n1[i01]
            else:
                nr01 = n0[-1] - n0[i00] + 1

            # R01
            r01 = np.zeros((4, nr01))
            r01[2, :] = 1
            for i in range(nr01):
                r01[3, i] = n0[i00 + i]
                r01[1, i] = f0[i00 + i]
                r01[0, i] = f0[i00 + i]
                r01[0, i] -= (f1[i01 + i + 1] + f1[i01 + i]) / 2.0
                r01[0, i] /= f1[i01 + i + 1] - f1[i01 + i]

        elif r10 and sequence in ["r10", "r102", "r010"]:
            # Find lowest indices for l = 0 and 1
            if n0[0] >= n1[0]:
                i00 = 0
                i01 = n0[0] - n1[0]
            else:
                i00 = n1[0] - n0[0]
                i01 = 0

            # Number of r10s
            if n0[-1] >= n1[-1]:
                nr10 = n1[-1] - n1[i01]
            else:
                nr10 = n0[-1] - n0[i00]

            # R10
            r10 = np.zeros((4, nr10))
            r10[2, :] = 10
            for i in range(nr10):
                r10[3, i] = n1[i01 + i]
                r10[1, i] = f1[i01 + i]
                r10[0, i] = f1[i01 + i]
                r10[0, i] -= (f0[i00 + i] + f0[i00 + i + 1]) / 2.0
                r10[0, i] /= f0[i00 + i] - f0[i00 + i + 1]

    if sequence == "r02":
        return r02

    elif sequence == "r01":
        return r01

    elif sequence == "r10":
        return r10

    elif sequence == "r012":
        if r01 is None or r02 is None:
            return None
        # R012 (R01 followed by R02) ordered by n (R01 first for identical n)
        mask = np.argsort(np.append(r01[3, :], r02[3, :] + 0.1))
        r012 = np.hstack((r01, r02))[:, mask]
        return r012

    elif sequence == "r102":
        if r10 is None or r02 is None:
            return None
        # R102 (R10 followed by R02) ordered by n (R10 first for identical n)
        mask = np.argsort(np.append(r10[3, :], r02[3, :] + 0.1))
        r102 = np.hstack((r10, r02))[:, mask]
        return r102

    elif sequence == "r010":
        if r01 is None or r10 is None:
            return None
        # R010 (R01 followed by R10) ordered by n (R01 first for identical n)
        mask = np.argsort(np.append(r01[3, :], r10[3, :] + 0.1))
        r010 = np.hstack((r01, r10))[:, mask]
        return r010


"""
Epsilon difference fitting
"""


def compute_epsilondiff(
    osckey,
    osc,
    avgdnu,
    sequence="e012",
    nsorting=True,
    extrapolation=False,
    nrealisations=20000,
    debug=False,
):
    """
    Compute epsilon differences and covariances.

    From Roxburgh 2016:
    * Eq. 1: Epsilon(n,l)
    * Eq. 4: EpsilonDifference(l=0,l=(1,2))

    Epsilon differences are independent of surface phase shift/outer
    layers when the epsilons are evaluated at the same frequency. It
    therefore relies on splining from epsilons at the observed frequencies
    of the given degree and order to the frequency of the compared/subtracted
    epsilon. See function "compute_epsilondiffseqs" for further clarification.

    For MonteCarlo sampling of the covariances, it is replicated from the
    covariance determination of frequency ratios in BASTA, (sec 4.1.3 of
    Aguirre Børsen-Koch et al. 2022). A number of realisations of the
    epsilon differences are drawn from random Gaussian distributions of the
    individual frequencies within their uncertainty.

    Parameters
    ----------
    osckey : array
        Array containing the angular degrees and radial orders of the modes.
    osc : array
        Array containing the modes (and inertias).
    avgdnu : float
        Average value of the large frequency separation.
    sequence : str, optional
        Similar to ratios, what sequence of epsilon differences to be computed.
        Can be e01, e02 or e012 for a combination of the two first.
    nsorting : bool, optional
        If True (default), the sequences are sorted by n-value of the frequencies. If
        False, the entire 01 sequence is followed by the 02 sequence.
    extrapolation : bool, optional
        If False (default), modes outside the range of the l=0 modes are discarded to
        avoid extrapolation.
    nrealisations : int or bool, optional
        If int: number of realisations used for MC-sampling the covariances
        If bool: Whether to use MC (True) or analytic (False) deternubation
        of covariances. If True, nrealisations of 20,000 is used.
    debug : bool, optional
        Print additional output and make plots for debugging (incl. a plot of the
        correlation matrix).

    Returns
    -------
    epsdiff : array
        Array containing the modes in the observed data.
    epsdiff_cov : array
        Covariances matrix.
    """

    # Remove modes outside of l=0 range
    if not extrapolation:
        indall = osckey[0, :] > -1
        ind0 = osckey[0, :] == 0
        ind12 = osckey[0, :] > 0
        mask = np.logical_and(
            osc[0, ind12] < max(osc[0, ind0]), osc[0, ind12] > min(osc[0, ind0])
        )
        indall[ind12] = mask
        if debug and any(mask):
            print(
                "The following modes have been skipped from epsilon differences to avoid extrapolation:"
            )
            for f, (l, n) in zip(osc[0, ~indall], osckey[:, ~indall].T):
                print(" - (l,n,f) = ({0}, {1:02d}, {2:.3f})".format(l, n, f))

        osc = osc[:, indall]
        osckey = osckey[:, indall]

    epsdiff = compute_epsilondiffseqs(
        osckey, osc, avgdnu, sequence=sequence, nsorting=nsorting
    )
    epsdiff_cov = su.compute_cov_from_mc(
        epsdiff.shape[1],
        osckey,
        osc,
        fittype=sequence,
        args={"avgdnu": avgdnu, "nsorting": nsorting},
        nrealisations=nrealisations,
    )

    return epsdiff, epsdiff_cov


def compute_epsilondiffseqs(
    osckey,
    osc,
    avgdnu,
    sequence,
    nsorting=True,
):
    """
    Computed epsilon differences, based on Roxburgh 2016 (eq. 1 and 4)

    Epsilons E of frequency v with order n and degree l is determined as:
    E(n,l) = E(v(n,l)) = v(n,l)/dnu - n - l/2

    From this, an epsilon is determined for each original frequncy. These
    are not independent on the surface layers, but their differences
    between different degrees are, if evaluated at the same frequency.
    Therefore, the epsilon differences dE of e.g. E(n,l=0) and E(n,l=2),
    dE(02) is determined from interpolating/splining the l=0 epsilon sequence
    SE0 and evaluating it at v(n,l=2), and subtracting the corresponding
    E(n,l=2). Therefore, the epsilon difference can be summarised as
    dE(0l) = SE0(v(n,l)) - E(n,l)

    Parameters
    ----------
    osckey : array
        Array containing the angular degrees and radial orders of the modes
    osc : array
        Array containing the modes (and inertias)
    avgdnu : float
        Average large frequency separation
    sequence : str
        Similar to ratios, what sequence of epsilon differences to be computed.
        Can be 01, 02 or 012 for a combination of the two first.
    nsorting : bool
        If True (default), the sequences are sorted by n-value of the frequencies.
        If False, the entire 01 sequence is followed by the 02 sequence.

    Returns
    -------
    deps : array
        Array containing epsilon differences. First index correpsonds to:
        0 - Epsilon differences
        1 - Indentifying frequencies
        2 - Identifying degree l
        3 - Radial degree n of identifying l={1,2} mode
    """

    # Select the sequence(s) to use
    if sequence == "e012":
        l_used = [1, 2]
    elif sequence == "e02":
        l_used = [2]
    elif sequence == "e01":
        l_used = [1]
    else:
        raise KeyError("Undefined epsilon difference sequence requested!")

    # Epsilon is computed analytically from the frequency information
    epsilon = np.zeros(osc.shape[1])

    for i, freq in enumerate(osc[0, :]):
        ll, nn = osckey[:, i]
        epsilon[i] = freq / avgdnu - nn - ll / 2

    # Setup base l=0 interpolater object
    nu0 = osc[0, osckey[0, :] == 0]
    eps0 = epsilon[osckey[0, :] == 0]
    eps0_intpol = CubicSpline(nu0, eps0)

    # Compute the epsilon differences of the selected sequence(s)
    nmodes = sum([sum(osckey[0] == ll) for ll in l_used])
    deps = np.zeros((4, nmodes))
    Niter = 0
    for ll in l_used:
        # Extract freq and epsilon for l=ll modes
        nul = osc[0, osckey[0] == ll]
        epsl = epsilon[osckey[0] == ll]

        # Evaluate epsilon(l=0) at nu(l=ll)
        eps0_at_nul = eps0_intpol(nul)

        # Difference
        diff_eps0l = eps0_at_nul - epsl

        # Store 0: difference, 1: freq, 2: l, 3: n
        deps[0, Niter : Niter + len(diff_eps0l)] = diff_eps0l
        deps[1, Niter : Niter + len(diff_eps0l)] = nul
        deps[2, Niter : Niter + len(diff_eps0l)] = ll
        deps[3, Niter : Niter + len(diff_eps0l)] = osckey[1][osckey[0] == ll]

        Niter += len(diff_eps0l)

    # Sort according to n if flagged (ensure l=1 before l=2 with 0.1)
    if nsorting:
        mask = np.argsort(deps[3, :] + deps[2, :] * 0.1)
        deps = deps[:, mask]
    return deps
