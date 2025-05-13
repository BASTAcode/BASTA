"""
This module contains the possible surface effect corrections that can be specified and used in BASTA.
"""

import h5py  # type: ignore[import]
import numpy as np
from scipy.optimize import minimize  # type: ignore[import]
from sklearn import linear_model  # type: ignore[import]

from basta import core


SURFACECORRECTIONS = {}


def register_surfacecorrection(fn):
    SURFACECORRECTIONS[fn.__name__] = fn
    return fn


def _update_modes(
    corrected_frequencies: np.ndarray, modes: core.JoinedModes | core.ModelFrequencies
):
    corrected_data = modes.data.copy()
    if isinstance(modes, core.JoinedModes):
        corrected_data["model_frequency"] = corrected_frequencies
        return core.JoinedModes(data=corrected_data)
    else:
        corrected_data["frequency"] = corrected_frequencies
        return core.ModelFrequencies(data=corrected_data)


def _l1(params, data, labels):
    prediction = np.dot(data, params.reshape(-1, 1))
    dist = prediction - labels
    return (np.abs(dist)).sum()


@register_surfacecorrection
def KBC08(
    joinedmodes: core.JoinedModes, nuref: float, bcor: float
) -> tuple[np.ndarray, np.ndarray]:
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
    joinedmodes : ModelFrequencies or JoinedModes
        Mode data containing frequencies and inertias.
    nuref : float
        The reference frequency used in the KBC08 correction.
        Typically, this is the numax of the observed star.
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
    joinl0 = joinedmodes.of_angular_degree(0)
    f_modell0 = joinl0["model_frequency"]
    e_modell0 = joinl0["inertia"]
    f_obsl0 = joinl0["observed_frequency"]

    f_model = joinedmodes.model_frequencies
    e_model = joinedmodes.inertias
    f_obs = joinedmodes.observed_frequencies

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
    coeffs = np.asarray([rpar, acor, bcor])

    # Apply the frequency correction
    corrected_frequencies = apply_KBC08(modes=joinedmodes, coeffs=coeffs, scalnu=nuref)
    assert corrected_frequencies is not None
    corrected_joinedmodes = _update_modes(
        corrected_frequencies=corrected_frequencies, modes=joinedmodes
    )

    return corrected_joinedmodes, coeffs


@register_surfacecorrection
def cubic_term_BG14(
    joinedmodes: core.JoinedModes,
    scalnu: float,
    method: str = "l1",
    flag_useonlyradial: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Ball & Gizon frequency correction

    Correcting stellar oscillation frequencies for near-surface effects.
    This routine follows the approach from Warrick H. Ball and Laurent Gizon.
    "A new correction of stellar oscillation frequencies for near-surface effects."
    Astronomy & Astrophysics 568 (2014): A123.

    The correction has the form:

        .. math:: d\\nu = \\frac{b \\left(\\frac{\\nu}{\\nu_{scal}}\\right)^3}{I}

    where :math:`b` are the found parameters, :math:`\\nu_{scal}` is a scaling frequency used to non-dimensionalize the frequencies :math:`\\nu`, and :math:`I` is the mode inertia for each mode.

    Parameters
    ----------
    joinedmodes : ModelFrequencies or JoinedModes
        Mode data containing frequencies and inertias.
    scalnu : float
        A scaling frequency used purely to non-dimensionalize the frequencies.
    method : string, optional
        The name of the fitting method used.
        It can be 'l1' (default), 'ransac', or 'l2'.
    flag_useonlyradial : bool
        Flag that if True only computes the correction based on the l=0 modes.
        This can be usefull if not many modes have been observed.

    Returns
    -------
    corrected_frequencies : np.ndarray
        Corrected mode frequencies
    coeffs: np.ndarray
        Coefficients for the computed surface effect correction
    """
    if flag_useonlyradial:
        joinl0 = joinedmodes.of_angular_degree(0)
        f_model = joinl0["model_frequency"]
        f_inertia = joinl0["inertia"]
        f_obs = joinl0["observed_frequency"]
        f_unc = joinl0["error"]
    else:
        f_model = joinedmodes.model_frequencies
        f_inertia = joinedmodes.inertias
        f_obs = joinedmodes.observed_frequencies
        f_unc = joinedmodes.observed_error

    # Initialize vector y and matrix X
    nmodes = len(f_obs)
    y = np.zeros((nmodes, 1))
    matX = np.zeros((nmodes, 1))

    # Filling in y and X
    y[:, 0] = (f_obs - f_model) / f_unc
    matX[:, 0] = np.power((f_model / scalnu), 3) / (f_inertia * f_unc)

    # Try for the l1-norm minimization

    assert method in ["ransac", "l1", "l2"]
    if method == "ransac":
        np.random.seed(5)
        lr = linear_model.LinearRegression(fit_intercept=False)
        ransac = linear_model.RANSACRegressor(lr)
        ransac.fit(matX, y)
        coeffs = ransac.estimator_.coef_.reshape(-1, 1)
        coeffs = np.asarray([coeffs[0][0]])
    elif method == "l1":
        initial_params = np.zeros(1)
        res = minimize(_l1, initial_params, method="Nelder-Mead", args=(matX, y))
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
    corrected_frequencies = apply_cubic_term_BG14(
        modes=joinedmodes, coeffs=coeffs, scalnu=scalnu
    )
    corrected_joinedmodes = _update_modes(
        corrected_frequencies=corrected_frequencies, modes=joinedmodes
    )

    return corrected_joinedmodes, coeffs


@register_surfacecorrection
def two_term_BG14(
    joinedmodes: core.JoinedModes,
    scalnu: float,
    method: str = "l1",
    flag_useonlyradial=False,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Ball & Gizon frequency correction

    Correcting stellar oscillation frequencies for near-surface effects.
    This routine follows the approach from Warrick H. Ball and Laurent Gizon. "A new correction of stellar oscillation frequencies for near-surface effects." Astronomy & Astrophysics 568 (2014): A123.

    The correction has the form:

        .. math:: d\\nu = \\frac{a \\left(\\frac{\\nu}{\\nu_{scal}}\\right)^{-1} + b \\left(\\frac{\\nu}{\\nu_{scal}}\\right)^3}{I}

    where :math:`a` and :math:`b` are the found parameters, :math:`\\nu_{scal}` is a scaling frequency used to non-dimensionalize the frequencies :math:`\\nu`, and :math:`I` is the mode inertia for each mode.

    Parameters
    ----------
    joinedmodes : ModelFrequencies or JoinedModes
        Mode data containing frequencies and inertias.
    scalnu : float
        A scaling frequency used purely to non-dimensionalize the frequencies.
    method : string, optional
        The name of the fitting method used.
        It can be 'l1' (default), 'ransac', or 'l2'.
    flag_useonlyradial : bool
        Flag that if True only computes the correction based on the l=0 modes.
        This can be usefull if not many modes have been observed.

    Returns
    -------
    corrected_frequencies : np.ndarray
        Corrected mode frequencies
    coeffs: np.ndarray
        Coefficients for the computed surface effect correction
    """
    if flag_useonlyradial:
        joinl0 = joinedmodes.of_angular_degree(0)
        f_model = joinl0["model_frequency"]
        f_inertia = joinl0["inertia"]
        f_obs = joinl0["observed_frequency"]
        f_unc = joinl0["error"]
    else:
        f_model = joinedmodes.model_frequencies
        f_inertia = joinedmodes.inertias
        f_obs = joinedmodes.observed_frequencies
        f_unc = joinedmodes.observed_error

    # Initialize vector y and matrix X
    nmodes = len(f_obs)
    y = np.zeros((nmodes, 1))
    matX = np.zeros((nmodes, 2))

    # Filling in y and X
    y[:, 0] = (f_obs - f_model) / f_unc
    matX[:, 0] = np.power((f_model / scalnu), -1) / (f_inertia * f_unc)
    matX[:, 1] = np.power((f_model / scalnu), 3) / (f_inertia * f_unc)

    assert method in ["ransac", "l1", "l2"]
    if method == "ransac":
        np.random.seed(5)
        lr = linear_model.LinearRegression(fit_intercept=False)
        ransac = linear_model.RANSACRegressor(lr)
        ransac.fit(matX, y)
        coeffs = ransac.estimator_.coef_.reshape(-1, 1)
        coeffs = np.asarray([coeffs[0][0], coeffs[1][0]])
    elif method == "l1":
        initial_params = np.zeros(2)
        res = minimize(_l1, initial_params, method="Nelder-Mead", args=(matX, y))
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
    corrected_frequencies = apply_two_term_BG14(
        modes=joinedmodes, coeffs=coeffs, scalnu=scalnu
    )
    corrected_joinedmodes = _update_modes(
        corrected_frequencies=corrected_frequencies, modes=joinedmodes
    )

    return corrected_joinedmodes, coeffs


def apply_KBC08(
    modes: core.ModelFrequencies | core.JoinedModes, coeffs: np.ndarray, scalnu: float
) -> np.ndarray | None:
    """
    Applies the KBC08 frequency correction to the mode frequencies.

    Parameters
    ----------
    modes : ModelFrequencies or JoinedModes
        Mode data containing frequencies and inertias.
    coeffs : ndarray
        Coefficients for the power-law model. Expected: [a, b, c]
        - a: normalization
        - b: Q scaling coefficient
        - c: power-law exponent
    scalnu : float
        A scaling frequency used purely to non-dimensionalize the frequencies.

    Returns
    -------
    np.ndarray
        Corrected frequencies.
    """
    if isinstance(modes, core.ModelFrequencies):
        model_frequencies = modes.frequencies
        model_inertia = modes.inertias
        radial_modes = modes.of_angular_degree(0)
        radial_model_frequencies = radial_modes["frequency"]
        radial_model_inertias = radial_modes["inertia"]
    elif isinstance(modes, core.JoinedModes):
        model_frequencies = modes.model_frequencies
        model_inertia = modes.inertias
        radial_modes = modes.of_angular_degree(0)
        radial_model_frequencies = radial_modes["model_frequency"]
        radial_model_inertias = radial_modes["inertia"]
    else:
        raise TypeError("Unsupported mode type")

    # Interpolate inertia to l=0 inertia at same frequency
    interp_inertia = np.interp(
        model_frequencies, radial_model_frequencies, radial_model_inertias
    )

    # Calculate Q's
    q_nl = model_inertia / interp_inertia

    corrected_frequencies = (model_frequencies / coeffs[0]) + (
        coeffs[1] / (coeffs[0] * q_nl)
    ) * np.power(model_frequencies / scalnu, coeffs[2])
    return corrected_frequencies


def apply_cubic_term_BG14(
    modes: core.ModelFrequencies | core.JoinedModes, coeffs: np.ndarray, scalnu: float
) -> np.ndarray:
    """
    Applies the BG14 cubic frequency correction to mode frequencies.

    Parameters
    ----------
    modes : ModelFrequencies or JoinedModes
        Mode data containing frequencies and inertias.
    coeffs : ndarray
        Correction coefficients. Expected: [a, b], corresponding to inverse and cubic terms.
    scalnu : float
        Scaling factor used purely to non-dimensionalize the frequencies.

    Returns
    -------
    np.ndarray
        Corrected frequencies.
    """
    if isinstance(modes, core.ModelFrequencies):
        model_frequencies = modes.frequencies
        model_inertia = modes.inertias
    elif isinstance(modes, core.JoinedModes):
        model_frequencies = modes.model_frequencies
        model_inertia = modes.inertias
    else:
        raise TypeError("Unsupported mode type")

    corrected_freqs: list[np.ndarray] = []

    # TODO(Amalie) pi-modes only
    for l in np.unique(modes.l):
        modes_givenl = modes.of_angular_degree(l)
        if len(modes_givenl) < 1:
            continue

        freq = modes_givenl["frequency"]
        inertia = modes_givenl["inertia"]
        df = (coeffs[0] * np.power(freq / scalnu, 3)) / inertia
        corrected_freqs.append(freq + df)

    if not corrected_freqs:
        return np.array([])

    return np.concatenate(corrected_freqs)


def apply_two_term_BG14(
    modes: core.ModelFrequencies | core.JoinedModes, coeffs: np.ndarray, scalnu: float
) -> np.ndarray:
    """
    Applies the BG14 two-term frequency correction to mode frequency.

    Parameters
    ----------
    modes : ModelFrequencies or JoinedModes
        Mode data containing frequencies and inertias.
    coeffs : ndarray
        Correction coefficients. Expected: [a, b], corresponding to inverse and cubic terms.
    scalnu : float
        Scaling factor used purely to non-dimensionalize the frequencies.

    Returns
    -------
    np.ndarray
        Corrected frequencies.
    """
    if isinstance(modes, core.ModelFrequencies):
        model_frequencies = modes.frequencies
        model_inertia = modes.inertias
    elif isinstance(modes, core.JoinedModes):
        model_frequencies = modes.model_frequencies
        model_inertia = modes.inertias
    else:
        raise TypeError("Unsupported mode type")

    corrected_freqs: list[np.ndarray] = []

    # TODO(Amalie) pi-modes only
    for l in np.unique(modes.l):
        modes_givenl = modes.of_angular_degree(l)
        if len(modes_givenl) < 1:
            continue
        freq = modes_givenl["frequency"]
        inertia = modes_givenl["inertia"]

        df = (
            coeffs[0] * np.power(freq / scalnu, -1)
            + coeffs[1] * np.power(freq / scalnu, 3)
        ) / inertia

        corrected_freqs.append(freq + df)
    if not corrected_freqs:
        return np.array([])

    return np.concatenate(corrected_freqs)


def apply_surfacecorrection(
    joinedmodes: core.JoinedModes,
    star: core.Star,
) -> tuple[np.ndarray, np.ndarray] | tuple[core.JoinedModes, None]:
    """
    Compute a specified surfacecorrection for the mode frequencies
    """
    assert star.modes is not None
    if star.modes.surfacecorrection is None:
        return joinedmodes, None

    assert star.modes.surfacecorrection in SURFACECORRECTIONS

    if star.modes.surfacecorrection.get("KBC08") is not None:
        corrected_joinedmodes, coeffs = KBC08(
            joinedmodes=joinedmodes,
            nuref=star.globalseismicparams.get_scaled("numax")[0],
            bcor=star.modes.surfacecorrection["KBC08"]["bexp"],
        )
    elif star.modes.surfacecorrection.get("two_term_BG14") is not None:
        corrected_joinedmodes, coeffs = two_term_BG14(
            joinedmodes=joinedmodes,
            scalnu=star.globalseismicparams.get_scaled("numax")[0],
        )
    elif star.modes.surfacecorrection.get("cubic_term_BG14") is not None:
        corrected_joinedmodes, coeffs = cubic_term_BG14(
            joinedmodes=joinedmodes,
            scalnu=star.globalseismicparams.get_scaled("numax")[0],
        )
    return corrected_joinedmodes, coeffs


def apply_surfacecorrection_coefficients(
    coeffs: np.ndarray | None,
    star: core.Star,
    modes: core.ModelFrequencies | core.JoinedModes,
) -> np.ndarray | core.ModelFrequencies | core.JoinedModes:
    """
    Apply a specified surface effect correction to the mode frequencies given the coefficients for the specified type of surface effect correction.

    """
    assert star.modes is not None
    if star.modes.surfacecorrection is None:
        return modes
    if coeffs is None:
        return modes

    assert star.modes.surfacecorrection in SURFACECORRECTIONS

    if star.modes.surfacecorrection.get("KBC08") is not None:
        corrected_frequencies = apply_KBC08(
            modes=modes,
            coeffs=coeffs,
            scalnu=star.globalseismicparams.get_scaled("numax")[0],
        )
    elif star.modes.surfacecorrection.get("two_term_BG14") is not None:
        corrected_frequencies = apply_two_term_BG14(
            modes=modes,
            coeffs=coeffs,
            scalnu=star.globalseismicparams.get_scaled("numax")[0],
        )
    elif star.modes.surfacecorrection.get("cubic_term_BG14") is not None:
        corrected_frequencies = apply_cubic_term_BG14(
            modes=modes,
            coeffs=coeffs,
            scalnu=star.globalseismicparams.get_scaled("numax")[0],
        )

    assert corrected_frequencies is not None
    corrected_modes = _update_modes(
        corrected_frequencies=corrected_frequencies, modes=modes
    )
    return corrected_modes
