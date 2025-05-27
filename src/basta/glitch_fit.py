"""
Auxiliary functions for glitch fitting
"""

from typing import TypedDict

import numpy as np

from basta import core, freq_fit
from basta import utils_seismic as su

try:
    from basta.glitch_fq import fit_fq  # type: ignore[import]
    from basta.glitch_sd import fit_sd  # type: ignore[import]
    from basta.icov_sd import icov_sd  # type: ignore[import]
    from basta.sd import sd  # type: ignore[import]

    GLITCH_AVAIL = True
except ImportError:
    GLITCH_AVAIL = False


def compute_observed_glitches(
    modes: core.StarModes,
    sequence: str,
    dnu: float,
    inferencesettings: core.InferenceSettings,
    kwargs: dict,
) -> tuple[np.ndarray, np.ndarray] | None:
    """
    Routine to compute glitch parameters (and ratios) with full covariance
    matrix using MC sampling.

    Parameters
    ----------
    osckey : numpy array
        Spherical degrees and radial orders of the frequencies to be used.
    osc : numpy array
        Frequencies and corresponding uncertainties.
    sequence : str
        Glitch sequence to be computed, see constants.freqtypes.glitches.
    dnu : float
        Value of large frequency separation to be used in the computation.
    fitfreqs : dict
        Dictionary containing frequency fitting options/controls.

    Returns
    -------
    glitchseq : numpy array
        Computed glitch parameters (and ratios) as median values from MC
        sampling
    glitchseq_cov : numpy array
        Determined covariance matrix of glitch parameters (and ratios)
    """

    # Get length of sequence
    if sequence == "glitches":
        # Purely the three glitch parameters
        sequence_length = 3
    else:
        # Ratios and glitch parameters
        ratios = freq_fit.compute_ratio_sequences(
            modes=modes.modes,
            sequence=sequence[1:],
            threepoint=inferencesettings.kwargs_ratios.get("threepoint", False),
        )
        if ratios is None:
            return None
        sequence_length = len(ratios["ratio"]) + 3

    # Call routine for sampling covariance
    glitchseq, glitchseq_cov = su.compute_glitch_covariances(
        nr=sequence_length,
        dnu=dnu,
        modes=modes.modes,
        sequence=sequence,
        inferencesettings=inferencesettings,
        **kwargs,
    )

    return glitchseq, glitchseq_cov


class AcDepths(TypedDict):
    tauHe: float
    dtauHe: float
    tauCZ: float
    dtauCZ: float


def compute_sequence_of_glitches(
    modes: core.ObservedFrequencies | core.JoinedModes,
    sequence: str,
    dnu: float,
    inferencesettings: core.InferenceSettings,
    ac_depths: AcDepths | None = None,
    debug: bool = False,
) -> np.ndarray:
    """
    Routine to compute glitch parameters of given frequencies, based
    on the given glitchmethod options.

    Parameters
    ----------
    osckey : numpy array
        Spherical degrees and radial orders of the frequencies to be used.
    osc : numpy array
        Frequencies and corresponding uncertainties.
    sequence : str
        Glitch sequence to be computed, see constants.freqtypes.glitches.
    dnu : float
        Value of large frequency separation to be used in the computation.
    ac_depts : bool or dict
        Acoustic depths used to search for glitch signatures. If not provided as a
        dict, they will be calculated as a simple estimate.

    Returns
    -------
    glitchseq : numpy array
        Determined glitch parameters (and ratios) from the provided frequencies.
        If computation failed, the glitch parameters will be NaNs.
    """

    # Check compilation of external FORTRAN routines
    if not GLITCH_AVAIL:
        raise ModuleNotFoundError(
            "Unable to import glitch modules, see installation guide for compiling them"
        )
    assert modes is not None

    # Setup array, make similar to ratios
    glitchseq = np.empty((4, 3)) * np.nan

    # Acoustic radius and acoustic depths of the glitches
    acousticRadius = 5.0e5 / dnu
    # If not inputted, use standard assumptions:
    if ac_depths is None:
        ac_depths = {
            "tauHe": 0.17 * acousticRadius + 18.0,
            "dtauHe": 0.05 * acousticRadius,
            "tauCZ": 0.34 * acousticRadius + 929.0,
            "dtauCZ": 0.10 * acousticRadius,
        }

    # Reformat frequencies for input to glitchmethod and filter out l=3
    mask = modes.data["l"] < 3
    if isinstance(modes, core.JoinedModes):
        frequency_column = "model_frequency"
        n_column = "model_n"
        error_column = "obs_error"
    else:
        frequency_column = "frequency"
        n_column = "n"
        error_column = "error"

    freqs = np.empty(((len(modes.data["l"][mask])), 4))
    freqs[:, 0] = modes.data["l"][mask]  # osckey[0, osckey[0, :] < 3]
    freqs[:, 1] = modes.data[n_column][mask]  # osckey[1, osckey[0, :] < 3]
    freqs[:, 2] = modes.data[frequency_column][mask]  # osc[0, osckey[0, :] < 3]
    freqs[:, 3] = modes.data[error_column][mask]

    # Number of n values for each l
    num_of_n = np.array(
        [sum(modes.data["l"] == given_l) for given_l in set(modes.data["l"])]
    )
    glitchmethod = inferencesettings.kwargs_glitches.get("glitchmethod", "freq")
    npoly_params = inferencesettings.kwargs_glitches.get("npoly_params", 5)
    nderiv = inferencesettings.kwargs_glitches.get("nderiv", 3)
    tol_grad = inferencesettings.kwargs_glitches.get("tol_grad", 1e-3)
    regu_param = inferencesettings.kwargs_glitches.get("regu_param", 7)
    nguesses = inferencesettings.kwargs_glitches.get("nguesses", 200)

    if glitchmethod == "freq":
        param, _, _, ier = fit_fq(
            freqs,
            num_of_n,
            acousticRadius,
            ac_depths["tauHe"],
            ac_depths["dtauHe"],
            ac_depths["tauCZ"],
            ac_depths["dtauCZ"],
            npoly_fq=npoly_params,
            total_num_of_param_fq=len(num_of_n) * npoly_params + 7,
            nderiv_fq=nderiv,
            tol_grad_fq=tol_grad,
            regu_param_fq=regu_param,
            num_guess=nguesses,
        )
    elif glitchmethod == "second_differences":
        freq_sd = sd(freqs, num_of_n, icov_sd.shape[0])
        param, _, _, ier = fit_sd(
            freq_sd,
            icov_sd,
            acousticRadius,
            ac_depths["tauHe"],
            ac_depths["dtauHe"],
            ac_depths["tauCZ"],
            ac_depths["dtauCZ"],
            npoly_sd=npoly_params,
            total_num_of_param_sd=npoly_params + 7,
            nderiv_sd=nderiv,
            tol_grad_sd=tol_grad,
            regu_param_sd=regu_param,
            num_guess=nguesses,
        )
    else:
        raise KeyError(f"Invalid glitch-fitting method {glitchmethod} requested!")
    # If failed, don't overwrite NaNS in output
    if ier == 0:
        # Determine average amplitudes
        _, AHe = _average_amplitudes(
            param,
            fmin=np.amin(freqs[:, 2]),
            fmax=np.amax(freqs[:, 2]),
            dnu=dnu,
            glitchmethod=glitchmethod,
        )
        # Restructure glitch parameters
        glitchseq[0, :] = [AHe, param[-3], param[-2]]
        glitchseq[2, :] = [7, 8, 9]
    # If only glitches, return these
    if sequence == "glitches":
        return glitchseq
    # Compute ratio sequence
    ratios = freq_fit.compute_ratio_sequences(
        modes=modes,
        sequence=sequence[1:],
        threepoint=inferencesettings.kwargs_ratios.get("threepoint", False),
    )
    assert ratios is not None

    # Stack arrays and return full sequence
    glitchseq = np.hstack((ratios, glitchseq))

    return glitchseq


def _average_amplitudes(param, fmin, fmax, dnu=None, glitchmethod="freq"):
    """
    Compute average amplitude of He and CZ signature

    Parameters
    ----------
    param : array
        Fitted parameters
    fmin : float
        Lower limit on frequency used in averaging (muHz)
    fmax : float
        Upper limit on frequency used in averaging (muHz)
    dnu : float
        An estimate of the large frequency separation, only
        necessary for method "second_differences"

    Returns
    -------
    Acz : float
        Average amplitude of CZ signature (muHz)
    Ahe : float
        Average amplitude of He signature (muHz)
    """

    # Check dnu is available for Second Deifferences method
    if glitchmethod.lower() == "second_differences" and dnu is None:
        raise ValueError(
            "An estimate of dnu is necessary for the second_differences method!"
        )

    n0 = len(param) - 7

    # Amplitude of CZ signature
    Acz = param[n0] / (fmin * fmax)

    # Amplitude of He signature
    fminhz = 1.0e-6 * fmin
    fmaxhz = 1.0e-6 * fmax
    Ahe = (
        param[n0 + 3]
        * (
            np.exp(-8.0 * np.pi**2 * fminhz**2 * param[n0 + 4] ** 2)
            - np.exp(-8.0 * np.pi**2 * fmaxhz**2 * param[n0 + 4] ** 2)
        )
        / (16.0 * np.pi**2 * 1.0e-12 * (fmax - fmin) * param[n0 + 4] ** 2)
    )

    # Scale amplitudes from Freq to SecDif
    if glitchmethod.lower() == "second_differences":
        Acz /= (2.0 * np.sin(2.0 * np.pi * dnu * 1.0e-6 * param[n0 + 1])) ** 2
        Ahe /= (2.0 * np.sin(2.0 * np.pi * dnu * 1.0e-6 * param[n0 + 5])) ** 2

    return Acz, Ahe
