"""
This module contains the possible surface effect corrections that can be specified and used in BASTA.
"""

import h5py  # type: ignore[import]
import numpy as np
from scipy.optimize import minimize  # type: ignore[import]

from basta import core


SURFACECORRECTIONS = {}


def register_surfacecorrection(fn):
    SURFACECORRECTIONS[fn.__name__] = fn
    return fn


@register_surfacecorrection
def KBC08(joinkeys, join, nuref, bcor):
    """
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
        The exponent in the KBC08 correction.

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
    corosc = apply_KBC08(modkey=joinkeys, mod=join, coeffs=coeffs, scalnu=nuref)
    corjoin = np.copy(join)
    corjoin[0, :] = np.asarray(corosc)

    return corjoin, coeffs


@register_surfacecorrection
def cubic_term_BG14(joinkeys, join, scalnu, method="l1", onlyl0=False):
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
    corosc = apply_cubic_term_BG14(
        modkey=joinkeys, mod=join, coeffs=coeffs, scalnu=scalnu
    )
    corjoin[0, :] = corosc

    return corjoin, coeffs


@register_surfacecorrection
def two_term_BG14(joinkeys, join, scalnu, method="l1", onlyl0=False):
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
    corosc = apply_two_term_BG14(
        modkey=joinkeys, mod=join, coeffs=coeffs, scalnu=scalnu
    )
    corjoin[0, :] = corosc

    return corjoin, coeffs


def apply_KBC08(modkey, mod, coeffs, scalnu):
    """
    Applies the KBC08 frequency correction to frequency for a given set
    of coefficients

    Parameters
    ----------
    modkey : array
        Array containing the angular degrees and radial orders of mod.
    mod : array
        Array containing the modes in the model.
    coeffs : list
        List containing [rpar, acor, bcor] used in the KBC08 correction.
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


def apply_cubic_term_BG14(modkey, mod, coeffs, scalnu):
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


def apply_two_term_BG14(modkey, mod, coeffs, scalnu):
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


def apply_surfacecorrection(
    coeffs: np.ndarray | None, star: core.Star, model_modes: core.ModelFrequencies
) -> np.ndarray:
    assert star.modes is not None
    if star.modes.surfacecorrection is None:
        return model_modes
    if coeffs is None:
        return model_modes

    assert star.modes.surfacecorrection in SURFACECORRECTIONS

    if star.modes.surfacecorrection.get("KBC08") is not None:
        corosc = apply_KBC08(
            modkey=modkey,
            mod=mod,
            coeffs=coeffs,
            scalnu=star.globalseismicparams.get_scaled("numax")[0],
        )
    elif star.modes.surfacecorrection.get("two_term_BG14") is not None:
        corosc = apply_two_term_BG14(
            modkey=modkey,
            mod=mod,
            coeffs=coeffs,
            scalnu=star.globalseismicparams.get_scaled("numax")[0],
        )
    elif star.modes.surfacecorrection.get("cubic_term_BG14") is not None:
        corosc = apply_cubic_term_BG14(
            modkey=modkey,
            mod=mod,
            coeffs=coeffs,
            scalnu=star.globalseismicparams.get_scaled("numax")[0],
        )
    cormod[0, :] = corosc
    return cormod
