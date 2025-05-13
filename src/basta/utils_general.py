"""
This module contains general purpose functions that are utilized throughout BASTA.
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

from basta import core, constants, distances, errors, freq_fit, glitch_fit, utils_priors
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


class GridHeader(TypedDict):
    gridtype: str
    version: str
    time: str
    is_interpolated: bool


class GridInfo(TypedDict):
    entryname: str
    defaultpath: str
    # TODO(Amalie) this could have a better name
    difsolarmodel: int | None


class BayesianWeights(NamedTuple):
    weight_keys: tuple[str, ...]
    along_weight: str


def read_grid_header(Grid) -> GridHeader:
    """
    Reads the essential metadata from a stellar grid HDF5 file header.
    """
    try:
        gridtype = Grid["header/library_type"][()]
        gridver = Grid["header/version"][()]
        gridtime = Grid["header/buildtime"][()]

        if isinstance(gridtype, bytes):
            gridtype = gridtype.decode("utf-8")
            gridver = gridver.decode("utf-8")
            gridtime = gridtime.decode("utf-8")

    except KeyError:
        raise SystemExit(
            "Error: Some information is missing in the header of the grid!\n"
            "Required: header/library_type, header/version, header/buildtime."
        )

    try:
        Grid["header/interpolation_time"][()]
        is_interpolated = True
    except KeyError:
        is_interpolated = False

    return {
        "gridtype": gridtype,
        "version": gridver,
        "time": gridtime,
        "is_interpolated": is_interpolated,
    }


def print_gridinfo(gridfile: str, header: GridHeader) -> None:
    print(f"\n* Using the grid '{gridfile}' of type '{header['gridtype']}'.")
    print(
        f"  - Grid built with BASTA version {header['version']}, timestamp: {header['time']}."
    )


def extract_gridid(Grid) -> tuple[float, float, float, float] | bool:
    """
    Extracts model parameters needed to build isochrone paths.
    Returns False if missing.
    """
    try:
        ove = Grid["header/ove"][()]
        dif = Grid["header/dif"][()]
        eta = Grid["header/eta"][()]
        alphaFe = Grid["header/alphaFe"][()]
        return (ove, dif, eta, alphaFe)
    except KeyError:
        return False


def check_gridtype(
    gridtype: str,
    gridid: tuple[float, float, float, float] | bool = False,
) -> GridInfo:
    """
    Constructs the appropriate file path based on the grid type.
    """
    gridtype = gridtype.lower()

    if "tracks" in gridtype:
        return {"entryname": "tracks", "defaultpath": "grid/", "difsolarmodel": None}

    if "isochrones" in gridtype:
        if not gridid or not isinstance(gridid, Sequence) or len(gridid) != 4:
            raise errors.GridTypeError(
                "Missing or invalid `gridid`. Expected tuple of (ove, dif, eta, alphaFe)."
            )

        ove, dif, eta, alphaFe = gridid
        path = f"ove={ove:.4f}/dif={dif:.4f}/eta={eta:.4f}/alphaFe={alphaFe:.4f}/"
        return {
            "entryname": "isochrones",
            "defaultpath": path,
            "difsolarmodel": int(dif),
        }

    raise errors.GridTypeError(
        f"Gridtype '{gridtype}' not supported. Must be 'tracks' or 'isochrones'."
    )


def get_grid(
    inferencesettings: core.InferenceSettings,
) -> tuple[h5py.File, GridHeader, GridInfo]:
    """
    Convenience wrapper to extract all required metadata from a grid.
    """
    Grid = h5py.File(inferencesettings.gridfile, "r")
    header = read_grid_header(Grid)
    gridid = extract_gridid(Grid)
    print_gridinfo(gridfile=inferencesettings.gridfile, header=header)
    return Grid, header, check_gridtype(header["gridtype"], gridid)


def print_targetinformation(starid: str) -> None:
    _header(f"Fitting star id: {starid}.")


def read_bayesianweights(
    Grid, gridtype: str, optional: bool = False
) -> BayesianWeights | tuple[None, None]:
    """
    Reads Bayesian weights and determines relevant dimensions.

    Parameters
    ----------
    Grid: HDF5 file handle.
        The grid of stellar models
    gridtype: str
        Grid type, e.g. 'isochrones' or 'tracks'.
    optional: bool
        If True, returns (None, None) when weights are missing.

    Returns:
        A `BayesianWeights` namedtuple or `(None, None)` if optional and not found.
    """
    try:
        raw_weights = Grid["header/active_weights"]
        grid_weights = [x.decode("utf-8") for x in list(raw_weights)]
    except KeyError:
        if optional:
            return (None, None)
        raise errors.MissingBayesianWeightsError(
            "Bayesian weights requested, but none specified in grid file."
        )

    bayweights = tuple(f"{x}_weight" for x in grid_weights)

    gridtype = gridtype.lower()
    if "isochrones" in gridtype:
        dweight = "dmass"
    elif "tracks" in gridtype:
        dweight = "dage"
    else:
        raise errors.GridTypeError(f"Unknown gridtype '{gridtype}'.")

    return BayesianWeights(weight_keys=bayweights, along_weight=dweight)


def prepare_distancefitting(
    star: core.InputStar,
    inferencesettings: core.InferenceSettings,
    filepaths: core.FilePaths,
    outputoptions: core.OutputOptions,
) -> core.AbsoluteMagnitudes:
    # TODO DEPRECATED
    # Add magnitudes and colors to fitparams if fitting distance
    absolutmagnitudes = distances.add_absolute_magnitudes(
        star=star,
        filepaths=filepaths,
        inferencesettings=inferencesettings,
        outputoptions=outputoptions,
    )
    return absolutmagnitudes[0]
    """
    # TODO(Amalie): Why? I think we need a better overview of what is being fitted than this
    # If keyword present, add individual filters
    if "distance" in allparams:
        allparams = list(
            np.unique(allparams + list(star.distanceparams.magnitudes.keys()))
        )
        allparams.remove("distance")
    return absolutmagnitudes, allparams
    """


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


def list_metallicities(
    grid: h5py.File, gridinfo: GridInfo, inferencesettings: core.InferenceSettings
) -> np.ndarray:
    """
    Get a list of metallicities in the grid that we loop over

    Parameters
    ----------
    Grid : hdf5 object
        The already loaded grid, containing the tracks/isochrones.
    defaultpath : str
        Path in Grid
    inputparams : dict
        Dictionary of all controls and input.
    limits : dict
        Dict of flat priors used in run.

    Returns
    -------
    metal : list
        List of possible metalliticies that should be looped over in
        `bastamain`.
    """
    if "grid" in gridinfo["defaultpath"]:
        return np.asarray([0])

    # Collect metallicities available in grid
    metallicity_strings = [
        name for name, _ in grid[gridinfo["defaultpath"]].items() if "=" in name
    ]
    metallist = [float(name[4:]) for name in metallicity_strings]
    metallicities = np.array(metallist)

    # TODO(Amalie) Redo this when limits is determined
    priors = inferencesettings.boxpriors
    metal_name = "MeH" if "MeH" in priors else "FeH"

    if metal_name in list(priors.keys()):
        limits = priors[metal_name].limits
        if limits is not None:
            lower, upper = limits
            metallicities = metallicities[
                (metallicities >= lower) & (metallicities <= upper)
            ]

    return metallicities


def unique_unsort(params):
    """
    As we want to check for unique elements to not copy elements, but retain the
    order they were given in, we have to do this, until numpy implements an 'unsort'
    key to numpy.unique...

    Parameters
    ----------
    params : list
        List of parameters

    Returns
    -------
    params : list
        List of unique params, retaining order
    """
    indexes = np.unique(params, return_index=True)[1]
    return [params[index] for index in sorted(indexes)]


def compare_output_to_input(
    star: core.Star,
    absolutemagnitudes: core.AbsoluteMagnitudes,
    runfiles: core.RunFiles,
    inferencesettings: core.InferenceSettings,
    hout,
    out,
    hout_dist,
    out_dist,
    uncert="qunatiles",
    sigmacut=1,
):
    """
    This function compares the outputted value of all fitting parameters
    to the input that was fitted.

    If one or more fitting parameters deviates more than 'sigmacut' number
    of the effective symmetric uncertainty away from their input parameter,
    a warning is printed and 'starid' is appended to the .warn-file.

    Parameters
    ----------
    starid : str
        Unique identifier of current target.
    inputparms : dict
        Dict containing input from xml-file.
    hout : list
        List of column headers for output
    out : list
        List of output values for the columns given in `hout`.
    uncert : str
        Type of reported uncertainty to use for comparison.
    sigmacut : float, optional
        Number of standard deviation used for determining when to issue
        a warning.

    Returns
    -------
    comparewarn : bool
        Flag to determine whether or not a warning was raised.
    """
    if runfiles.warnoutput is None:
        return False
    fitparams = star.classicalparams.params
    warnfile = runfiles.warnoutput
    comparewarn = False
    ps = []
    sigmas = []

    for p in fitparams:
        if p in hout:
            idx = np.nonzero([p == xout for xout in hout])[0][0]
            xin, xinerr = fitparams[p]
            if uncert == "quantiles":
                outerr = (out[idx + 1] + out[idx + 2]) / 2
            else:
                outerr = out[idx + 1]
            serr = np.sqrt(outerr**2 + xinerr**2)
            sigma = np.abs(out[idx] - xin) / serr
            bigdiff = sigma >= sigmacut
            if bigdiff:
                comparewarn = True
                ps.append(p)
                sigmas.append(sigma)

    if len(absolutemagnitudes["magnitudes"].keys()) > 0:
        for m in list(absolutemagnitudes["magnitudes"].keys()):
            mdist = "M_" + m
            if mdist in hout_dist:
                idx = np.nonzero([x == mdist for x in hout_dist])[0][0]
                priorM = absolutemagnitudes["magnitudes"][m]["median"]
                priorerrp = absolutemagnitudes["magnitudes"][m]["errp"]
                priorerrm = absolutemagnitudes["magnitudes"][m]["errm"]
                if uncert == "quantiles":
                    outerr = (out_dist[idx + 1] + out_dist[idx + 2]) / 2
                else:
                    outerr = out_dist[idx + 1]
                serr = np.sqrt(((priorerrp + priorerrm) / 2) ** 2 + outerr**2)
                sigma = np.abs(out_dist[idx] - priorM) / serr
                bigdiff = sigma >= sigmacut
                if bigdiff:
                    comparewarn = True
                    ps.append(mdist)
                    sigmas.append(sigma)

    if comparewarn:
        print(f"A >{sigmacut} sigma difference was found between input and output of")
        print(ps)
        print("with sigma differences of")
        print(sigmas)
        if isinstance(warnfile, IOBase):
            warnfile.write(f"{star.starid}\t{ps}\t{sigmas}\n")
        else:
            with open(warnfile, "a") as wf:
                wf.write(f"{star.starid}\t{ps}\t{sigmas}\n")

    return comparewarn


def inflog(x):
    "np.log(x), but where x=0 returns -inf without a warning"
    with np.errstate(divide="ignore"):
        return np.log(x)


def add_out(hout, out, par, x, xm, xp, uncert):
    """
    Add entries in out list, according to the wanted uncertainty.

    Parameters
    ----------
    hout : list
        Names in header
    out : list
        Parameter values
    par : str
        Parameter name
    x : float
        Centroid value
    xm : float
        Lower bound uncertainty, or symmetric uncertainty
    xp : float, None
        Upper bound uncertainty if not symmetric uncertainty (None for symmetric)
    uncert : str
        Type of reported uncertainty, "quantiles" or "std"

    Returns
    -------
    hout : list
        Header list with added names
    out : list
        Parameter list with added entries
    """
    if uncert == "quantiles":
        hout += [par, par + "_errp", par + "_errm"]
        out += [x, xp - x, x - xm]
    else:
        hout += [par, par + "_err"]
        out += [x, xm]
    return hout, out


def get_parameter_values(parameter, Grid, selectedmodels, noofind):
    """
    Get parameter values from grid

    Parameters
    ----------
    parameter : str
        Grid, hdf5 object
    selectedmodels :
        models to return
    noofind :
        number of parameter values

    Returns
    -------
    x_all : array
        parameter values
    """
    x_all = np.zeros(noofind)
    i = 0
    for modelpath in selectedmodels:
        N = len(selectedmodels[modelpath].logPDF)
        try:
            x_all[i : i + N] = selectedmodels[modelpath].paramvalues[parameter]
        except Exception:
            x_all[i : i + N] = Grid[modelpath + "/" + parameter][
                selectedmodels[modelpath].index
            ]
        i += N
    return x_all


def printparam(
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


def strtobool(val: str | Literal[0, 1]) -> Literal[0, 1]:
    """
    Convert a string representation of truth to true (1) or false (0).
    True values are 'y', 'yes', 't', 'true', 'on', and '1'.
    False values are 'n', 'no', 'f', 'false', 'off', and '0'.

    Raises ValueError if 'val' is anything else.

    Parameter
    ---------
    val: str
        String value to be converted into a boolean.
    """
    if not isinstance(val, str):
        return val
    val = val.lower()
    if val in ("y", "yes", "t", "true", "on", "1"):
        return 1
    if val in ("n", "no", "f", "false", "off", "0"):
        return 0
    raise ValueError(f"invalid truth value {val!r}")


def flush_all(*files: IO | None) -> None:
    """Flush multiple file buffers to ensure data is written."""
    for f in files:
        if f:
            f.flush()


def h5py_to_array(xs) -> np.ndarray:
    """
    Copy vector/dataset from an HDF5 file to a NumPy array

    Parameters
    ----------
    xs : h5py_dataset
       The input dataset read by h5py from an HDF5 object

    Returns
    -------
    res : array_like
        Copy of the dataset as NumPy array
    """
    res = np.empty(shape=xs.shape, dtype=xs.dtype)
    res[:] = xs[:]
    return res


def compute_group_names(
    gridinfo: GridInfo, metallicities: np.ndarray
) -> dict[float, str]:
    if "grid" in gridinfo["defaultpath"]:
        return {feh: gridinfo["defaultpath"] + "tracks/" for feh in metallicities}
    return {feh: f"{gridinfo['defaultpath']}FeH={feh:.4f}/" for feh in metallicities}


def add_bias_to_dnuerror(globalseismicparams, inputstar, dnu="dnufit"):
    if inputstar.dnubias == 0.0:
        return
    dnu_value, dnu_error = globalseismicparams.params[dnu]
    new_dnu_error = np.sqrt(dnu_error**2.0 + inputstar.dnubias**2.0)
    print(
        f"Added a given systematic increase of uncertainty in dnu of {inputstar.dnubias}"
    )
    print(f"Increases uncertainty from {dnu_error:.3f} µHz to {new_dnu_error:.3f} µHz")
    globalseismicparams.params[dnu][1] = new_dnu_error


def compute_inverse_covariancematrix(covariance: np.ndarray, inputstar: core.InputStar):
    if not inputstar.correlations:
        covariance = np.diag(np.diag(covariance))
    return compute_matrix_inverse(covariance)


def compute_matrix_inverse(covariance: np.ndarray) -> np.ndarray:
    try:
        # Compute Cholesky factorization
        c_factor = scipy.linalg.cho_factor(covariance, lower=True, check_finite=False)
        covinv = scipy.linalg.cho_solve(
            c_factor, np.eye(covariance.shape[0]), check_finite=False
        )
    except np.linalg.LinAlgError:
        covinv = np.linalg.pinv(covariance, rcond=1e-8)

    return covinv


def get_modes(
    fit_plot_params: list[str],
    inputstar: core.InputStar,
    globalseismicparams: core.GlobalSeismicParameters,
    obskey: np.ndarray,
    obs: np.ndarray,
    obscov: np.ndarray,
    inferencesettings: core.InferenceSettings,
) -> core.StarModes | None:
    if not any(
        x in [*constants.freqtypes.freqs, *constants.freqtypes.rtypes]
        for x in inferencesettings.fitparams
    ):
        return None

    # TODO this function should not use obskey,obs
    modedata = core.make_star_modes_from_l_n_freq_error(
        obskey[0], obskey[1], obs[0], obs[1]
    )
    if "dnufit" in globalseismicparams.params:
        dnu = globalseismicparams.get_scaled("dnufit")[0]
    elif "numax" in globalseismicparams.params:
        dnu = freq_fit.compute_dnufit(
            data=modedata, numax=globalseismicparams.get_scaled("numax")[0]
        )
    else:
        raise ValueError("Missing dnu")
    obsintervals = freq_fit.make_intervals(data=modedata, dnu=dnu)

    covinv = compute_inverse_covariancematrix(covariance=obscov, inputstar=inputstar)

    modes = core.StarModes(
        modes=modedata,
        surfacecorrection=inputstar.surfacecorrection,
        obsintervals=obsintervals,
        correlations=inputstar.correlations,
        seismicweights=inferencesettings.seismicweights,
        inverse_covariance=covinv,
    )
    return modes


def get_ratios(
    fit_plot_params: list[str],
    inputstar: core.InputStar,
    obskey: np.ndarray,
    obs: np.ndarray,
    inferencesettings: core.InferenceSettings,
):
    if not any(np.isin(constants.freqtypes.rtypes, fit_plot_params)):
        return None

    ratios_dict: dict[str, core.SeismicSignature] = {}
    for ratiotype in constants.freqtypes.rtypes:
        if ratiotype not in fit_plot_params:
            continue

        if inputstar.readratios:
            datos = fio._read_precomputed_ratios_xml(
                filename=inputstar.freqfile,
                ratiotype=ratiotype,
                obskey=obskey,
                obs=obs,
                excludemodes=inputstar.nottrustedfile,
                correlations=bool(inputstar.correlations),
            )
        else:
            datos = freq_fit.compute_ratios(
                obskey, obs, ratiotype, threepoint=inputstar.threepoint
            )

        if datos is None:
            if ratiotype in inferencesettings.fitparams:
                raise ValueError(
                    f"Fitting parameter {ratiotype} could not be computed."
                )
            datos = (None, None)

        covinv = compute_inverse_covariancematrix(
            covariance=datos[1], inputstar=inputstar
        )

        ratios_dict[ratiotype] = core.SeismicSignature(datos[0], covinv)

    return ratios_dict


def get_glitches(
    fit_plot_params: list[str],
    inputstar: core.InputStar,
    globalseismicparams: core.GlobalSeismicParameters,
    obskey: np.ndarray,
    obs: np.ndarray,
    inferencesettings: core.InferenceSettings,
    outputoptions: core.OutputOptions,
) -> dict[str, core.SeismicSignature] | None:
    if not any(np.isin(constants.freqtypes.glitches, fit_plot_params)):
        return None

    glitches_dict: dict[str, core.SeismicSignature] = {}
    for glitchtype in constants.freqtypes.glitches:
        if glitchtype not in fit_plot_params:
            continue

        if inputstar.readratios:
            assert inputstar.glitchfile is not None
            datos = fio._read_precomputed_glitches(
                filename=inputstar.glitchfile,
                type=glitchtype,
            )
        else:
            datos = glitch_fit.compute_observed_glitches(
                osckey=obskey,
                osc=obs,
                sequence=glitchtype,
                dnu=globalseismicparams.get_scaled("dnufit")[0],
                fitfreqs={
                    "threepoint": inputstar.threepoint,
                    "nrealisations": inputstar.nrealizations,
                },
                debug=outputoptions.debug,
            )

        if datos is None:
            if glitchtype in inferencesettings.fitparams:
                raise ValueError(
                    f"Fitting parameter {glitchtype} could not be computed."
                )
            datos = (None, None)

        covinv = compute_inverse_covariancematrix(datos[1], inputstar=inputstar)

        glitches_dict[glitchtype] = core.SeismicSignature(datos[0], covinv)
    return glitches_dict


def get_epsilondifferences(
    fit_plot_params: list[str],
    inputstar: core.InputStar,
    globalseismicparams: core.GlobalSeismicParameters,
    obskey: np.ndarray,
    obs: np.ndarray,
    inferencesettings: core.InferenceSettings,
    outputoptions: core.OutputOptions,
):
    if not any(np.isin(constants.freqtypes.epsdiff, fit_plot_params)):
        return None

    epsilondiff_dict: dict[str, core.SeismicSignature] = {}
    for epsilondifftype in constants.freqtypes.epsdiff:
        if epsilondifftype not in fit_plot_params:
            continue

        nrealisations = (
            inputstar.nrealizations
            if epsilondifftype in inferencesettings.fitparams
            else 2000
        )
        datos = freq_fit.compute_epsilondiff(
            osckey=obskey,
            osc=obs,
            avgdnu=globalseismicparams.get_scaled("dnufit")[0],
            sequence=epsilondifftype,
            nsorting=inputstar.nsorting,
            nrealisations=nrealisations,
            debug=outputoptions.debug,
        )
        if datos is None:
            if epsilondifftype in inferencesettings.fitparams:
                raise ValueError(
                    f"Fitting parameter {epsilondifftype} could not be computed."
                )
            datos = (None, None)

        covinv = compute_inverse_covariancematrix(datos[1], inputstar=inputstar)

        epsilondiff_dict[epsilondifftype] = core.SeismicSignature(datos[0], covinv)

    return epsilondiff_dict


def setup_star(
    inputstar: core.InputStar,
    inferencesettings: core.InferenceSettings,
    filepaths: core.FilePaths,
    outputoptions: core.OutputOptions,
    plotconfig: core.PlotConfig,
) -> core.Star:

    classicalparams = inputstar.classicalparams
    globalseismicparams = inputstar.globalseismicparams
    distanceparams = inputstar.distanceparams

    if globalseismicparams.params:
        assert globalseismicparams.scalefactors is not None

    add_bias_to_dnuerror(globalseismicparams, inputstar)

    modes = ratios = glitches = epsilondifferences = None
    absolutemagnitudes = distancelimits = None

    if inferencesettings.has_any_seismic_case:
        obskey, obs, obscov = fio.read_freq(
            filename=inputstar.freqfile,
            excludemodes=inputstar.excludemodes,
            onlyradial=inputstar.onlyradial,
            covarfre=bool(inputstar.correlations),
        )
        fit_plot_params = np.unique(
            np.asarray(inferencesettings.fitparams + plotconfig.freqplots)
        )

        modes = get_modes(
            fit_plot_params=fit_plot_params,
            inputstar=inputstar,
            globalseismicparams=globalseismicparams,
            obskey=obskey,
            obs=obs,
            obscov=obscov,
            inferencesettings=inferencesettings,
        )
        ratios = get_ratios(
            fit_plot_params=fit_plot_params,
            inputstar=inputstar,
            obskey=obskey,
            obs=obs,
            inferencesettings=inferencesettings,
        )
        glitches = get_glitches(
            fit_plot_params=fit_plot_params,
            inputstar=inputstar,
            globalseismicparams=globalseismicparams,
            obskey=obskey,
            obs=obs,
            inferencesettings=inferencesettings,
            outputoptions=outputoptions,
        )
        epsilondifferences = get_epsilondifferences(
            fit_plot_params=fit_plot_params,
            inputstar=inputstar,
            globalseismicparams=globalseismicparams,
            obskey=obskey,
            obs=obs,
            inferencesettings=inferencesettings,
            outputoptions=outputoptions,
        )

    if inferencesettings.has_distance_case:
        absolutemagnitudes, distancelimits = distances.add_absolute_magnitudes(
            star=inputstar,
            filepaths=filepaths,
            inferencesettings=inferencesettings,
            outputoptions=outputoptions,
        )

    # Translate boxpriors into ranges, depending on the given star
    limits = utils_priors.get_limits(
        inputstar=inputstar,
        inferencesettings=inferencesettings,
        distancelimits=distancelimits,
        modes=modes,
    )

    return core.Star(
        starid=inputstar.starid,
        limits=limits,
        classicalparams=classicalparams,
        globalseismicparams=globalseismicparams,
        distanceparams=distanceparams,
        absolutemagnitudes=absolutemagnitudes,
        modes=modes,
        ratios=ratios,
        glitches=glitches,
        epsilondifferences=epsilondifferences,
    )


def list_phases(star: core.Star) -> list[int]:
    if star.phase is None:
        return []
    if isinstance(star.phase, tuple):
        return [constants.phasemap.pmap[ip] for ip in star.phase]
    else:
        return [constants.phasemap.pmap[star.phase]]


def should_skip_due_to_diffusion(libitem, star: core.Star):
    """
    Check if the current track should be skipped based on given diffusion.
    """
    if any(np.isin(["dif", "diffusion"], list(star.classicalparams.params.keys()))):
        lib_dif = int(round(libitem["dif"][0]))
        star_dif = int(round(float(star.classicalparams.params["dif"][0])))
        return lib_dif != star_dif
    return False


def gridlimits(
    grid: h5py.File,
    gridheader: GridHeader,
    gridinfo: GridInfo,
    inferencesettings: core.InferenceSettings,
    outputoptions: core.OutputOptions,
) -> None:
    """
    Refactor of grid cut section

    Check if any specified limit in prior is in header, and can be used to
    skip computation of models, in order to speed up computation
    """
    limits = list(inferencesettings.boxpriors.keys())

    gridcut = {}

    # Determine header path
    if "tracks" in gridheader["gridtype"]:
        headerpath = "header/"
    elif "isochrones" in gridheader["gridtype"]:
        headerpath = f"header/{gridinfo['defaultpath']}"
        if "FeHini" in limits:
            print("Warning: Dropping prior in FeHini, redundant for isochrones!")
            inferencesettings.boxpriors.pop("FeHini")
    else:
        headerpath = None

    if headerpath:
        header_keys = grid[headerpath].keys()

        # Extract gridcut params
        gridcut_keys = set(header_keys) & set(limits)
        gridcut = {key: limits.pop(key) for key in gridcut_keys}

        if gridcut:
            print("\nCutting in grid based on sampling parameters ('gridcut'):")
            for cutpar, cutval in gridcut.items():
                if cutpar != "dif":
                    print(f"* {cutpar}: {cutval}")

            # Special handling for diffusion switch
            if "dif" in gridcut:
                # Expecting value like [-inf, 0.5] or [0.5, inf]
                switch = np.where(np.array(gridcut["dif"]) == 0.5)[0][0]
                print(
                    f"* Only considering tracks with diffusion turned {'on' if switch == 1 else 'off'}!"
                )
    inferencesettings.boxpriors["gridcut"] = core.PriorEntry(
        kwargs={"gridcut": gridcut},
        limits=None,
    )
