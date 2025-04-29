"""
This module contains general purpose functions that are utilized throughout BASTA.
"""

import sys
import time
from collections.abc import Sequence
from io import IOBase
from pathlib import Path
from typing import IO, Literal, NamedTuple, TypedDict

import h5py  # type: ignore[import]
import numpy as np
import scipy.linalg  # type: ignore[import]

from basta import core, constants, distances, errors, freq_fit, glitch_fit
from basta import fileio as fio
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
    print(f"\nFitting star id: {starid}.")


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
    return absolutmagnitudes
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


def print_fitparams(fitparams: dict) -> None:
    # Print fitparams
    print("\nFitting information:")
    print("* Fitting parameters with values and uncertainties:")
    for fp, fpval in fitparams.items():
        if fp in ["numax", "dnuSer", "dnuscal", "dnuAsf"]:
            fpstr = f"{fp} (solar units)"
        else:
            fpstr = fp
        print(f"  - {fpstr}: {fpval}")


def print_seismic(fitfreqs: dict, obskey: np.ndarray, obs: np.ndarray) -> None:
    # Fitting info: Frequencies
    if not fitfreqs["active"]:
        return
    if "freqs" in fitfreqs["fittypes"]:
        print("* Fitting of individual frequencies activated!")
    elif any(x in constants.freqtypes.rtypes for x in fitfreqs["fittypes"]):
        print(f"* Fitting of frequency ratios {fitfreqs['fittypes']} activated!")
        if "r010" in fitfreqs["fittypes"]:
            print(
                "  - WARNING: Fitting r01 and r10 simultaniously results in overfitting, and is thus not recommended!"
            )
    elif any(x in constants.freqtypes.glitches for x in fitfreqs["fittypes"]):
        print(f"* Fitting of glitches {fitfreqs['fittypes']} activated using:")
        print(f"  - Method: {fitfreqs['glitchmethod']}")
        print(f"  - Parameters in smooth component: {fitfreqs['npoly_params']}")
        print(f"  - Order of derivative: {fitfreqs['nderiv']}")
        print(f"  - Gradient tolerance: {fitfreqs['tol_grad']}")
        print(f"  - Regularization parameter: {fitfreqs['regu_param']}")
        print(f"  - Initial guesses: {fitfreqs['nguesses']}")
        print("* General frequency fitting configuration:")
    elif any(x in constants.freqtypes.epsdiff for x in fitfreqs["fittypes"]):
        print(f"* Fitting of epsilon differences {fitfreqs['fittypes']} activated!")

    # Translate True/False to Yes/No
    strmap = ("No", "Yes")
    print(f"  - Automatic prior on dnu: {strmap[fitfreqs['dnuprior']]}")
    print(
        f"  - Constraining lowest l = 0 (n = {obskey[1, 0]}) with f = {obs[0, 0]:.3f} +/-",
        f"{obs[1, 0]:.3f} muHz to within {fitfreqs['dnufrac'] * 100:.1f} % of dnu ({fitfreqs['dnufrac'] * fitfreqs['dnufit']:.3f} microHz)",
    )
    if fitfreqs["bexp"] is not None:
        bexpstr = f" with b = {fitfreqs['bexp']}"
    else:
        bexpstr = ""
    print(f"  - Correlations: {strmap[fitfreqs['correlations']]}")
    print(f"  - Frequency input data: {fitfreqs['freqfile']}")
    print(
        f"  - Frequency input data (list of ignored modes): {fitfreqs['excludemodes']}"
    )
    print(f"  - Inclusion of dnu in ratios fit: {strmap[fitfreqs['dnufit_in_ratios']]}")
    print(f"  - Interpolation in ratios: {strmap[fitfreqs['interp_ratios']]}")
    print(f"  - Surface effect correction: {fitfreqs['fcor']}{bexpstr}")
    print(f"  - Use alternative ratios (3-point): {strmap[fitfreqs['threepoint']]}")
    if fitfreqs["dnufit_err"]:
        print(
            f"  - Value of dnu: {fitfreqs['dnufit']:.3f} +/- {fitfreqs['dnufit_err']:.3f} microHz"
        )
    else:
        print(f"  - Value of dnu: {fitfreqs['dnufit']:.3f} microHz")
    print(f"  - Value of numax: {fitfreqs['numax']:.3f} microHz")

    weightcomment = ""
    if fitfreqs["seismicweights"]["dof"]:
        weightcomment += f"  |  dof = {fitfreqs['seismicweights']['dof']}"
    if fitfreqs["seismicweights"]["N"]:
        weightcomment += f"  |  N = {fitfreqs['seismicweights']['N']}"
    print(
        f"  - Weighting scheme: {fitfreqs['seismicweights']['weight']}{weightcomment}"
    )


def print_distances(star: core.Star, outputoptions: core.OutputOptions) -> None:
    # Fitting info: Distance
    if len(star.distanceparams.magnitudes) < 1:
        return
    print("")
    if (
        len(star.distanceparams.params["parallax"]) > 0
    ) and "distance" in outputoptions.asciiparams:
        print("* Parallax fitting and distance inference activated!")
    elif len(star.distanceparams.params["parallax"]) > 0:
        print("* Parallax fitting activated!")
    elif "distance" in outputoptions.asciiparams:
        print("* Distance inference activated!")

    if star.distanceparams.coordinates["frame"].lower() == "icrs":
        print(
            f"  - Coordinates (icrs): RA = {star.distanceparams.coordinates['RA']}, DEC = {star.distanceparams.coordinates['DEC']}"
        )
    elif star.distanceparams.coordinates["frame"].lower() == "galactic":
        print(
            f"  - Coordinates (galactic): lat = {star.distanceparams.coordinates['lat']}, lon = {star.distanceparams.coordinates['lon']}"
        )

    if len(star.distanceparams.params["parallax"]) > 0:
        print(f"  - Parallax: {star.distanceparams.params['parallax']}")

    print("  - Filters (magnitude value and uncertainty): ")
    for filt, (m, m_err) in star.distanceparams.magnitudes.items():
        print(f"    + {filt}: [{m}, {m_err}]")

    if len(star.distanceparams.EBV) > 0:
        # TODO(Amalie) is EBV a list of [0, value, 0] or a flat value? should probably be the latter
        print(
            f"  - EBV: {star.distanceparams.EBV[1]} (uniform across all distance samples)"
        )


def print_additional(star: core.Star) -> None:
    if "phase" in star.classical.params.keys():
        print("* Fitting evolutionary phase!")
    # TODO(Amalie) check that this points at the right things
    print("\nAdditional input parameters and settings in alphabetical order:")
    noprint = [
        "asciioutput",
        "asciioutput_dist",
        "distanceparams",
        "dnufit",
        "dnufrac",
        "erroutput",
        "fcor",
        "fitfreqs",
        "fitparams",
        "limits",
        "magnitudes",
        "excludemodes",
        "numax",
    ]
    for ip in sorted(star.classical.params.keys()):
        assert ip not in [
            "warnoutput",
        ]
        if ip not in noprint:
            print(f"* {ip}: {star.classical.params[ip]}")


def print_weights(bayweights: tuple[str, ...] | None, gridtype: str) -> None:
    # Print weights and priors
    print("\nWeights and priors:")
    if bayweights is not None:
        if "isochrones" in gridtype.lower():
            gtname = "isochrones"
            dwname = "mass"
        elif "tracks" in gridtype.lower():
            gtname = "tracks"
            dwname = "age"

        print("* Bayesian weights:")
        print(f"  - Along {gtname}: {dwname}")
        print(
            f"  - Between {gtname}: {', '.join([q.split('_')[0] for q in bayweights])}"
        )
    else:
        print("No Bayesian weights applied")


def print_priors(inferencesettings: core.InferenceSettings) -> None:
    priors = inferencesettings.priors
    if not priors:
        return

    print("* Additional set priors:")

    empty_priors = sorted(
        [lim for lim, entry in priors.items() if lim != "gridcut" and not entry.kwargs]
    )

    for lim in empty_priors:
        print(f"  - {lim}")

    constrained = sorted(
        [
            (lim, k, v)
            for lim, entry in priors.items()
            if lim != "gridcut" and entry.kwargs
            for k, v in entry.kwargs.items()
        ]
    )

    if constrained:
        print("* Flat, constrained priors:")
        for lim, k, v in constrained:
            print(f"  - {lim}: ({k}: {v})")


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

    priors = inferencesettings.priors
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
    fitparams = star.classical.params
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


def normfactor(alphas, ms):
    # Algorithm from App. A in Pflamm-Altenburg & Kroupa (2006)
    # https://ui.adsabs.harvard.edu/abs/2006MNRAS.373..295P/abstract
    ks = np.zeros(len(alphas))
    ks[0] = (1 / ms[1]) ** alphas[0]
    ks[1] = (1 / ms[1]) ** alphas[1]
    if len(ks) == 2:
        return ks
    ks[2] = (ms[2] / ms[1]) ** alphas[1] * (1 / ms[2]) ** alphas[2]
    if len(ks) == 3:
        return ks
    if len(ks) == 4:
        ks[3] = (
            (ms[2] / ms[1]) ** alphas[1]
            * (ms[3] / ms[2]) ** alphas[2]
            * (1 / ms[3]) ** alphas[3]
        )
        return ks
    print("Mistake in normfactor")
    return None


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


def should_skip_track(
    libitem, name: str, noingrid: int, star: core.Star
) -> bool:
    """
    Return True if a track should be skipped based on prior limits.
    """
    param, val = name.split("=")
    if param == "mass":
        param += "ini"

    limits = star.limits
    #TODO(Amalie this does not work
    """
    if limits not is None:
        for limit in limits:
            if limit is not None:
                assert isinstance(limit, tuple(float, float))
                return not (limit[0] <= float(val) <= limit[1])
    """
    return False


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
        cov = np.diag(np.diag(covariance))

    try:
        # Compute Cholesky factorization
        c_factor = scipy.linalg.cho_factor(cov, lower=True, check_finite=False)
        covinv = scipy.linalg.cho_solve(c_factor, np.eye(cov.shape[0]), check_finite=False)
    except np.linalg.LinAlgError:
        covinv = np.linalg.pinv(cov, rcond=1e-8)

    return covinv


def get_frequencies_and_intervals(
    fit_plot_params: list[str],
    inputstar: core.InputStar,
    globalseismicparams: core.GlobalSeismicParameters,
    obskey: np.ndarray,
    obs: np.ndarray,
    inferencesettings: core.InferenceSettings,
):
    if not any(
        x in [*constants.freqtypes.freqs, *constants.freqtypes.rtypes]
        for x in inferencesettings.fitparams
    ):
        return None, None

    if "dnufit" in globalseismicparams.params:
        dnu = globalseismicparams.get_scaled("dnufit")
    elif "numax" in globalseismicparams.params:
        dnu = freq_fit.compute_dnu_wfit(
            obskey=obskey, obs=obs, numax=globalseismicparams.get_scaled("numax")
        )
    else:
        raise ValueError("Missing dnu")

    obsintervals = freq_fit.make_intervals(obs, obskey, dnu=dnu)


    frequencies = core.IndividualFrequencies(
        l=obskey[0, :],
        n=obskey[1, :],
        frequencies=obs[0, :],
        errors=obs[1, :],
        surfacecorrection=inputstar.surfacecorrection,
        obsintervals=obsintervals,
        correlations=inputstar.correlations,
        seismicweights=inputstar.seismicweights,
    )
    return frequencies, obsintervals


def get_ratios(
    fit_plot_params: list[str],
    inputstar: core.InputStar,
    obskey: np.ndarray,
    obs: np.ndarray,
    inferencesettings: core.InferenceSettings,
):
    if not any(np.isin(constants.freqtypes.rtypes, fit_plot_params)):
        return None

    ratios_dict: dict[str, core.Ratio] = {}
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

        covinv = compute_inverse_covariancematrix(covariance=datos[1], inputstar=inputstar)

        ratios_dict[ratiotype] = core.Ratio(ratios=datos[0], inverse_covariance=covinv)

    return core.Ratios(ratios=ratios_dict)


def get_glitches(
    fit_plot_params: list[str],
    inputstar: core.InputStar,
    globalseismicparams: core.GlobalSeismicParameters,
    obskey: np.ndarray,
    obs: np.ndarray,
    inferencesettings: core.InferenceSettings,
    outputoptions: core.OutputOptions,
):
    if not any(np.isin(constants.freqtypes.glitches, fit_plot_params)):
        return None

    glitches_dict: dict[str, core.Glitch] = {}
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
                fitfreqs={"threepoint": inputstar.threepoint, "nrealisations": inputstar.nrealizations,}, 
                debug=outputoptions.debug,
            )

        if datos is None:
            if glitchtype in inferencesettings.fitparams:
                raise ValueError(
                    f"Fitting parameter {glitchtype} could not be computed."
                )
            datos = (None, None)

        covinv = compute_inverse_covariancematrix(datos[1], inputstar=inputstar)

        glitches_dict[glitchtype] = core.Glitch(glitches=datos[0], inverse_covariance=covinv)
    return core.Glitches(glitches=glitches_dict)


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

    epsilondiff_dict: dict[str, core.EpsilonDifference] = {}
    for epsilondifftype in constants.freqtypes.epsdiff:
        if epsilondifftype not in fit_plot_params:
            continue

        nrealisations = (
            inputstar.nrealizations if epsilondifftype in inferencesettings.fitparams else 2000
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

        epsilondiff_dict[epsilondifftype] = core.EpsilonDifference(
            epsilondifferences=datos[0], inverse_covariance=covinv
        )

    return core.EpsilonDifferences(epsilondifferences=epsilondiff_dict)
