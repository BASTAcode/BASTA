"""
This module contains functions that prints to the logfile.
"""

import os
import sys
import time
from collections.abc import Sequence
from io import IOBase
from pathlib import Path
from typing import IO, Literal, NamedTuple, TypedDict, Any

import h5py  # type: ignore[import]
import numpy as np
import scipy.linalg  # type: ignore[import]

from basta import core, constants
from basta import utils_general as util
from basta import fileio as fio
from basta import utils_seismic as su
from basta.__about__ import __version__


def prt_center(text: str, llen: int) -> None:
    """
    Prints a centered line

    Parameters
    ----------
    text : str
        The text string to print
    llen : int
        Length of the line
    """
    sp = int((llen - len(text)) / 2) * " "
    print(f"{sp}{text}{sp}")


def _header(title: str) -> None:
    print(f"\n{title}")


def _bullet(text: str, level: int = 1, indent: int = 1) -> None:
    if level == 0:
        symbol = "*"
        indent = 0
    elif level == 1:
        symbol = "-"
        indent = 1
    elif level == 2:
        symbol = "+"
        indent = 2
    print("  " * indent + f"{symbol} {text}")


def _banner(
    text: str | None = None, lines: list[str] | None = None, pad: int = 3
) -> None:
    if lines is None:
        assert text is not None
        lines = text.split("\n")
    content_width = max(len(line) for line in lines)
    full_width = content_width + pad * 2
    border = "*" * (full_width + 4)

    print(border)
    empty_line = f"**{' ' * full_width}**"
    print(empty_line)
    for line in lines:
        centered = line.center(full_width)
        print(f"**{centered}**")
    print(empty_line)
    print(border)


def print_bastaheader(
    t0: time.struct_time, seed: int, llen: int = 88, developermode: bool = False
) -> None:
    """
    Prints the header of the BASTA run

    Parameters
    ----------
    llen : int
        Length of the line, default=88
    t0 : time.struct_time
        Local time of the beginning of the BASTA run
    seed : int
        Seed for BASTA run
    developermode : bool, optional
        Activate experimental features (for developers)
    """
    print(llen * "=")
    prt_center("BASTA", llen)
    prt_center("The BAyesian STellar Algorithm", llen)
    print()
    prt_center(f"Version {__version__}", llen)
    print()
    prt_center("(c) 2025, The BASTA Team", llen)
    prt_center("https://github.com/BASTAcode/BASTA", llen)
    print(llen * "=")
    print(f"\nRun started on {time.strftime('%Y-%m-%d %H:%M:%S', t0)}.\n")
    if developermode:
        print("DEVELOPERMODE ACTIVATED: Experimental features activated!\n")
    print(f"Random numbers initialised with seed: {seed}")


def print_gridinfo(
    inferencesettings: core.InferenceSettings, header: util.GridHeader
) -> None:
    print(
        f"\n* Using the grid '{inferencesettings.gridfile}' of type '{header['gridtype']}'."
    )
    print(
        f"  - Grid built with BASTA version {header['version']}, timestamp: {header['time']}."
    )


def print_targetinformation(starid: str) -> None:
    _header(f"Fitting star id: {starid}.")


def print_fitparams(star: core.Star, inferencesettings: core.InferenceSettings) -> None:
    _header("Fitting information:")
    _bullet("Fitting parameters with values and uncertainties:", level=0)

    for param in inferencesettings.fitparams:
        if param in star.classicalparams.params:
            val, err = star.classicalparams.params[param]
        elif param in star.globalseismicparams.params:
            val, err = star.globalseismicparams.get_scaled(param)
        elif param in ["parallax"]:
            continue
        elif param in constants.freqtypes.alltypes:
            continue
        else:
            print(f"  [#TODO DEBUG] Unknown parameter: {param}")
            continue

        if param in ["numax", "dnuSer", "dnuscal", "dnuAsf"]:
            label = f"{param} (solar units)"
        else:
            label = param
        # TODO(Amalie) If constants.parameters.params was a dict, unit could be added here.
        _bullet(f"{label}: {pretty_param(value=val, error=err)}", level=1)


def print_seismic(
    inferencesettings: core.InferenceSettings,
    inputstar: core.InputStar,  # obskey: np.ndarray, obs: np.ndarray
) -> None:
    if not inferencesettings.has_any_seismic_case:
        return

    _header("Seismic Fitting Information")

    strmap = {True: "Yes", False: "No"}

    if inferencesettings.has_frequencies:
        _bullet("Fitting of individual frequencies activated!", level=0)
    elif inferencesettings.has_ratios:
        types = [
            fit
            for fit in inferencesettings.fitparams
            if fit in constants.freqtypes.rtypes
        ]
        _bullet(f"Fitting of frequency ratios {', '.join(types)} activated!", level=0)
        if "r010" in inferencesettings.fitparams:
            _bullet("WARNING: r01 and r10 together → overfitting risk!", level=1)
    elif inferencesettings.has_glitches:
        types = [
            fit
            for fit in inferencesettings.fitparams
            if fit in constants.freqtypes.glitches
        ]
        _bullet(f"Glitch fitting: {types}", level=0)
        # TODO(Amalie) Do we use these parameters? How? Where?
        for key in [
            "glitchmethod",
            "npoly_params",
            "nderiv",
            "tol_grad",
            "regu_param",
            "nguesses",
        ]:
            continue
    elif inferencesettings.has_epsilondifferences:
        types = [
            fit
            for fit in inferencesettings.fitparams
            if fit in constants.freqtypes.epsdiff
        ]
        _bullet(f"Epsilon differences fitting: {types}", level=0)

    _header("Frequency Fitting Configuration")
    _bullet(
        f"Frequency file: {os.path.join(inputstar.freqpath, inputstar.freqfile)}",
        level=1,
    )

    if inputstar.correlations is not None:
        if inputstar.correlations:
            _bullet(
                "Given correlations between frequencies are being taken into account",
                level=1,
            )
    # TODO(Amalie) these modes should be translated in get_input to something like ls in star
    if inputstar.nottrustedfile is not None:
        _bullet("File with frequencies to ignore: {inputstar.nottrustedfile}", level=1)
    if inputstar.excludemodes is not None:
        _bullet("File with frequencies to ignore: {inputstar.excludemodes}", level=1)
    if inputstar.onlyradial is not None:
        if inputstar.onlyradial:
            _bullet("Only radial frequencies will be used!", level=1)

    _bullet(
        f"Speed-up prior on dnu: {strmap[bool(inferencesettings.dnuprior)]}", level=1
    )
    # TODO(Amalie) When it is easy to get the anchor frequency, rewrite
    """
    _bullet(
        f"Constraining lowest l=0 (n={obskey[1, 0]}): f={obs[0, 0]:.3f} ± {obs[1, 0]:.3f} µHz "
        f"within {inferencesettings.boxpriors['dnufrac'].kwargs['dnufit']*100:.2f}% of dnu"  # ({star.limits['dnufit'][1] - star.limits['dnufit'][0]} µHz)"
    )
    """

    if inputstar.surfacecorrection is not None:
        surfcorr = list(inputstar.surfacecorrection.keys())[0]
        _bullet(f"Surface correction: {surfcorr}", level=1)
        if inputstar.surfacecorrection[surfcorr]:
            _bullet(
                f"Power law exponent: b = {inputstar.surfacecorrection[surfcorr]['bexp']}",
                level=2,
            )

    if inferencesettings.fit_surfacecorrected_dnu:
        _bullet(
            f"Surface-effect corrected large frequency separation added as fitting constraint",
            level=1,
        )
    if inputstar.interp_ratios:
        _bullet("#TODO(Amalie) interp_ratios", level=1)
    if inputstar.threepoint:
        _bullet("#TODO(Amalie) threepoint", level=1)

    for param in ["dnufit", "numax"]:
        if inputstar.globalseismicparams.get_scaled(param):
            val, err = inputstar.globalseismicparams.get_scaled(param)
            _bullet(f"{param}: {pretty_param(value=val, error=err)} µHz", level=1)

    weights = inferencesettings.seismicweights
    weightinfo = f"{weights['weight']}"
    if weights.get("dof"):
        weightinfo += f" | dof = {weights['dof']}"
    if weights.get("N"):
        weightinfo += f" | N = {weights['N']}"
    _bullet(f"Weighting scheme: {weightinfo}", level=1)


def pretty_param(
    param: core.Fitparam | None = None,
    value: float | None = None,
    error: float | None = None,
):
    if param is not None:
        value, error = param
    assert value is not None
    assert error is not None

    # Convert error to string with repr to preserve formatting
    err_str = f"{error:.10f}".rstrip("0")
    if "." in err_str:
        precision = len(err_str.split(".")[-1])
    else:
        precision = 0

    fmt = f".{precision}f"
    return f"{format(value, fmt)} ± {format(error, fmt)}"


def print_distances(star: core.Star, outputoptions: core.OutputOptions) -> None:
    if not star.distanceparams.magnitudes:
        return

    _header("Distance and Parallax Information")

    is_parallax = bool(star.distanceparams.params.get("parallax"))
    is_distance = "distance" in outputoptions.asciiparams

    # TODO(Amalie) It needs to be clearer what the difference between 'parallax' and 'distance' in fitparams means
    if is_parallax and is_distance:
        _bullet("Parallax fitting and distance inference activated!", level=0)
    elif is_parallax:
        _bullet("Parallax fitting activated!", level=0)
    elif is_distance:
        _bullet("Distance inference activated!", level=0)

    frame = star.distanceparams.coordinates.get("frame", "").lower()
    coords = star.distanceparams.coordinates
    if frame == "icrs":
        _bullet(f"RA = {coords['RA']}, DEC = {coords['DEC']}", level=1)
    elif frame == "galactic":
        _bullet(f"lat = {coords['lat']}, lon = {coords['lon']}", level=1)

    if is_parallax:
        _bullet(
            f"Parallax: {pretty_param(param=star.distanceparams.params['parallax'])}",
            level=1,
        )

    _bullet("Magnitudes:", level=1)
    for filt, (m, m_err) in star.distanceparams.magnitudes.items():
        _bullet(f"{filt}: {pretty_param(value=m, error=m_err)}", level=2)

    if (
        isinstance(star.distanceparams.EBV, (list, tuple))
        and len(star.distanceparams.EBV) > 1
    ):
        _bullet(f"EBV: {star.distanceparams.EBV[1]} (uniform)")


def print_additional(star: core.Star) -> None:
    if star.phase is not None:
        _bullet(f"Evolutionary phase: {star.phase}", level=1)

    """
    _header("Additional Parameters")
    #TODO(Amalie) When is this run?
    ignored = {
        "asciioutput", "asciioutput_dist", "distanceparams", "dnufit", "dnufrac",
        "erroutput", "fcor", "fitfreqs", "fitparams", "limits", "magnitudes",
        "excludemodes", "numax", "warnoutput"
    }

    for ip in sorted(star.classicalparams.params):
        if ip not in ignored:
            _bullet(f"{ip}: {star.classicalparams.params[ip]}", level=0)
    """


def print_weights(bayweights: tuple[str, ...] | None, gridtype: str) -> None:
    _header("Weights and Priors")
    if not bayweights:
        _bullet("No Bayesian weights applied", level=0)
        return

    if "isochrones" in gridtype.lower():
        gtname, dwname = "isochrones", "mass"
    elif "tracks" in gridtype.lower():
        gtname, dwname = "tracks", "age"
    else:
        gtname, dwname = "unknown", "?"

    _bullet("Bayesian weights:", level=0)
    _bullet(f"Along {gtname}: {dwname}", level=1)
    _bullet(
        f"Between {gtname}: {', '.join(q.split('_')[0] for q in bayweights)}", level=1
    )


def print_priors(inferencesettings: core.InferenceSettings) -> None:
    imf = inferencesettings.imf
    if imf is not None or imf != "":
        _bullet(f"Initial mass function: {imf}", level=0)
    else:
        _bullet("No initial mass function applied.", level=0)

    boxpriors = inferencesettings.boxpriors
    if not boxpriors:
        return

    empty_priors = sorted(
        [
            lim
            for lim, entry in boxpriors.items()
            if lim not in ["gridcut", "dnufrac"] and not entry.kwargs
        ]
    )
    if empty_priors:
        _bullet("Additional set priors:", level=0)
        for lim in empty_priors:
            _bullet(lim, level=1)

    constrained = sorted(
        (lim, k, v)
        for lim, entry in boxpriors.items()
        if lim not in ["gridcut", "dnufrac", "anchormode"] and entry.kwargs
        for k, v in entry.kwargs.items()
    )

    if constrained:
        _bullet("Flat, constrained priors:", level=0)
        strmap = {
            "abstol": "Symmetric bound around observed value of half-width",
            "nsigma": "Symmetric bound around observed value of half-width (in sigma)",
            "min": "Absolute lower bound:",
            "max": "Absolute upper bound:",
        }
        for lim, k, v in constrained:
            _bullet(f"{lim}: {strmap[k]} {v}", level=1)


class Logger:
    """
    Class used to redefine stdout to terminal and an output file.

    Parameters
    ----------
    outfilename : str
        Absolute path to an output file
    """

    # Credit: http://stackoverflow.com/a/14906787
    def __init__(self, outfilename: Path) -> None:
        self.terminal = sys.stdout
        self.log = open(outfilename, "a")

    def write(self, message: str) -> None:
        self.terminal.write(message)
        self.log.write(message)

    def flush(self) -> None:
        self.terminal.flush()
        self.log.flush()


def print_param(
    param, xmed, xstdm, xstdp, uncert="quantiles", centroid="median"
) -> None:
    """
    Pretty-print of output parameter to log and console.

    Parameters
    ----------
    param : str
        Name of parameter
    xmed : float
        Centroid value (median or mean)
    xstdm : float
        Lower bound uncertainty, or symmetric unceartainty
    xstdp : float
        Upper bound uncertainty, if not symmetric. Unused if uncert is std.
    uncert : str, optional
        Type of reported uncertainty, "quantiles" or "std"
    centroid : str, optional
        Type of reported uncertainty, "median" or "mean"

    Returns
    -------
    None
    """
    # Formats made to accomodate longest possible parameter name ("E(B-V)(joint)")
    print(f"{centroid:9}  {param:13} :  {xmed:12.6f}")
    if uncert == "quantiles":
        print("{:9}  {:13} :  {:12.6f}".format("err_plus", param, xstdp - xmed))
        print("{:9}  {:13} :  {:12.6f}".format("err_minus", param, xmed - xstdm))
    else:
        print("{:9}  {:13} :  {:12.6f}".format("stdev", param, xstdm))
    print("-----------------------------------------------------")
