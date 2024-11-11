"""
Auxiliary functions for glitch fitting
"""

import numpy as np

from basta import freq_fit
from basta import utils_seismic as su

try:
    from basta.sd import sd
    from basta.icov_sd import icov_sd
    from basta.glitch_fq import fit_fq
    from basta.glitch_sd import fit_sd

    GLITCH_AVAIL = True
except:
    GLITCH_AVAIL = False


def compute_observed_glitches(
    osckey: np.array,
    osc: np.array,
    sequence: str,
    dnu: float,
    fitfreqs: dict,
    debug=False,
) -> tuple[np.array, np.array]:
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
        ratios = freq_fit.compute_ratioseqs(
            osckey, osc, sequence[1:], fitfreqs["threepoint"]
        )
        sequence_length = ratios.shape[1] + 3

    # Call routine for sampling covariance
    glitchseq, glitchseq_cov = su.compute_cov_from_mc(
        sequence_length,
        osckey,
        osc,
        sequence,
        args={"dnu": dnu, "fitfreqs": fitfreqs},
        nrealisations=fitfreqs["nrealizations"],
    )

    return glitchseq, glitchseq_cov


def compute_glitchseqs(
    osckey: np.array,
    osc: np.array,
    sequence: str,
    dnu: float,
    fitfreqs: dict,
    ac_depths: bool = False,
    debug: bool = False,
) -> np.array:
    """
    Routine to compute glitch parameters of given frequencies, based
    on the given method options.

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
            "Unable to import glitch modules, check "
            + "installation guide for compilation of these!"
        )

    # Setup array, make similar to ratios
    glitchseq = np.empty((4, 3)) * np.nan

    # Acoustic radius and acoustic depths of the glitches
    acousticRadius = 5.0e5 / dnu
    # If not inputted, use standard assumptions:
    if not ac_depths:
        ac_depths = {
            "tauHe": 0.17 * acousticRadius + 18.0,
            "dtauHe": 0.05 * acousticRadius,
            "tauCZ": 0.34 * acousticRadius + 929.0,
            "dtauCZ": 0.10 * acousticRadius,
        }

    # Reformat frequencies for input to methods and filter out l=3
    freqs = np.zeros((len(osckey[0, osckey[0, :] < 3]), 4))
    freqs[:, 0] = osckey[0, osckey[0, :] < 3]
    freqs[:, 1] = osckey[1, osckey[0, :] < 3]
    freqs[:, 2] = osc[0, osckey[0, :] < 3]
    # If model frequencies from join, use error of observed frequencies
    if osc.shape[0] > 2:
        freqs[:, 3] = osc[3, osckey[0, :] < 3]
    else:
        freqs[:, 3] = osc[1, osckey[0, :] < 3]

    # Number of n values for each l
    num_of_n = np.array([sum(osckey[0, :] == ll) for ll in set(osckey[0])])
    if fitfreqs["glitchmethod"].lower() == "freq":
        nparams = len(num_of_n) * fitfreqs["npoly_params"] + 7
        param, _, _, ier = fit_fq(
            freqs,
            num_of_n,
            acousticRadius,
            ac_depths["tauHe"],
            ac_depths["dtauHe"],
            ac_depths["tauCZ"],
            ac_depths["dtauCZ"],
            npoly_fq=fitfreqs["npoly_params"],
            total_num_of_param_fq=nparams,
            nderiv_fq=fitfreqs["nderiv"],
            tol_grad_fq=fitfreqs["tol_grad"],
            regu_param_fq=fitfreqs["regu_param"],
            num_guess=fitfreqs["nguesses"],
        )
    elif fitfreqs["glitchmethod"].lower() == "secdif":
        freq_sd = sd(freqs, num_of_n, icov_sd.shape[0])
        param, _, _, ier = fit_sd(
            freq_sd,
            icov_sd,
            acousticRadius,
            ac_depths["tauHe"],
            ac_depths["dtauHe"],
            ac_depths["tauCZ"],
            ac_depths["dtauCZ"],
            npoly_sd=fitfreqs["npoly_params"],
            total_num_of_param_sd=fitfreqs["npoly_params"] + 7,
            nderiv_sd=fitfreqs["nderiv"],
            tol_grad_sd=fitfreqs["tol_grad"],
            regu_param_sd=fitfreqs["regu_param"],
            num_guess=fitfreqs["nguesses"],
        )
    else:
        raise KeyError(
            f"Invalid glitch-fitting method {fitfreqs['glitchmethod']} requested!"
        )
    # If failed, don't overwrite NaNS in output
    if ier == 0:
        # Determine average amplitudes
        _, AHe = _average_amplitudes(
            param,
            fmin=np.amin(freqs[:, 2]),
            fmax=np.amax(freqs[:, 2]),
            dnu=dnu,
            method=fitfreqs["glitchmethod"],
        )
        # Restructure glitch parameters
        glitchseq[0, :] = [AHe, param[-3], param[-2]]
        glitchseq[2, :] = [7, 8, 9]
    # If only glitches, return these
    if sequence == "glitches":
        return glitchseq
    else:
        # Compute ratio sequence
        ratios = freq_fit.compute_ratioseqs(
            osckey, osc, sequence[1:], fitfreqs["threepoint"]
        )

        # Stack arrays and return full sequence
        glitchseq = np.hstack((ratios, glitchseq))
        return glitchseq


def _average_amplitudes(param, fmin, fmax, dnu=None, method="Freq"):
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
        necessary for method "SecDif"

    Returns
    -------
    Acz : float
        Average amplitude of CZ signature (muHz)
    Ahe : float
        Average amplitude of He signature (muHz)
    """

    # Check dnu is available for Second Deifferences method
    if method.lower() == "secdif" and dnu is None:
        raise ValueError("An estimate of dnu is necessary for the SecDif method!")

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
    if method.lower() == "secdif":
        Acz /= (2.0 * np.sin(2.0 * np.pi * dnu * 1.0e-6 * param[n0 + 1])) ** 2
        Ahe /= (2.0 * np.sin(2.0 * np.pi * dnu * 1.0e-6 * param[n0 + 5])) ** 2

    return Acz, Ahe
