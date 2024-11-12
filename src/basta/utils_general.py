"""
General mix of utility functions
"""

import sys
import time
from io import IOBase

import numpy as np
from basta.__about__ import __version__
from basta import distances
from basta.constants import freqtypes


def h5py_to_array(xs) -> np.array:
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
    prt_center("Version {0}".format(__version__), llen)
    print()
    prt_center("(c) 2024, The BASTA Team", llen)
    prt_center("https://github.com/BASTAcode/BASTA", llen)
    print(llen * "=")
    print("\nRun started on {0} . \n".format(time.strftime("%Y-%m-%d %H:%M:%S", t0)))
    if developermode:
        print("RUNNING WITH EXPERIMENTAL FEATURES ACTIVATED!\n")
    print(f"Random numbers initialised with seed: {seed} .")


def check_gridtype(
    gridtype: str,
    allowed_gridtype: list[str] = ["tracks", "isochrones"],
    gridid: str | bool = False,
) -> tuple[str, str, None | int]:
    # Check type of grid (isochrones/tracks) and set default grid path
    gridtype = gridtype.lower()
    if "tracks" in gridtype:
        entryname = "tracks"
        defaultpath = "grid/"
        difsolarmodel = None
    elif "isochrones" in gridtype:
        entryname = "isochrones"
        if gridid:
            difsolarmodel = int(gridid[1])
            defaultpath = f"ove={gridid[0]:.4f}/dif={gridid[1]:.4f}/eta={gridid[2]:.4f}/alphaFe={gridid[3]:.4f}/"
        else:
            print(
                "Unable to construct path for science case."
                + " Probably missing (ove, dif, eta, alphaFe) in input!"
            )
            raise LibraryError

    else:
        raise OSError(
            f"Gridtype {gridtype} not supported, only 'tracks' and 'isochrones'!"
        )
    return entryname, defaultpath, difsolarmodel


def read_grid_header(Grid) -> tuple[str, str, str, bool]:
    try:
        gridtype = Grid["header/library_type"][()]
        gridver = Grid["header/version"][()]
        gridtime = Grid["header/buildtime"][()]

        # Allow for usage of both h5py 2.10.x and 3.x.x
        # --> If things are encoded as bytes, they must be made into standard strings
        if isinstance(gridtype, bytes):
            gridtype = gridtype.decode("utf-8")
            gridver = gridver.decode("utf-8")
            gridtime = gridtime.decode("utf-8")
    except KeyError:
        raise SystemExit(
            "Error: Some information is missing in the header of the grid!\n"
            "Please check the entries in the header! It must include all "
            "of the following:\n * header/library_type\n * header/version "
            "\n * header/buildtime"
        )

    # Check if grid is interpolated
    try:
        Grid["header/interpolation_time"][()]
    except KeyError:
        grid_is_intpol = False
    else:
        grid_is_intpol = True

    assert isinstance(gridtype, str), gridtype
    assert isinstance(gridver, str), gridver
    assert isinstance(gridtime, str), gridtime
    return gridtype, gridver, gridtime, grid_is_intpol


def read_grid_bayweights(Grid, gridtype) -> tuple[tuple[str, ...], str]:
    try:
        grid_weights = [x.decode("utf-8") for x in list(Grid["header/active_weights"])]

    except KeyError:
        print("WARNING: Bayesian weights requested, but none specified in grid file!\n")
        raise
    bayweights = tuple([x + "_weight" for x in grid_weights])

    # Specify the along weight variable, varies due to conceptual difference
    # - Isochrones --> dmass
    # - Tracks     --> dage
    if "isochrones" in gridtype.lower():
        dweight = "dmass"
    elif "tracks" in gridtype.lower():
        dweight = "dage"
    else:
        raise Exception(f"Unknown gridtype {gridtype}")
    return bayweights, dweight


def prepare_distancefitting(
    inputparams: dict, debug: bool, debug_dirpath: str, allparams: list[str]
) -> tuple[dict, list[str]]:
    # Special case if assuming gaussian magnitudes
    if "gaussian_magnitudes" in inputparams:
        use_gaussian_priors = inputparams["gaussian_magnitudes"]
    else:
        use_gaussian_priors = False

    # Add magnitudes and colors to fitparams if fitting distance
    inputparams = distances.add_absolute_magnitudes(
        inputparams,
        debug=debug,
        debug_dirpath=debug_dirpath,
        use_gaussian_priors=use_gaussian_priors,
    )

    # If keyword present, add individual filters
    if "distance" in allparams:
        allparams = list(
            np.unique(allparams + inputparams["distanceparams"]["filters"])
        )
        allparams.remove("distance")
    return inputparams, allparams


def print_fitparams(fitparams: dict) -> None:
    # Print fitparams
    print("\nFitting information:")
    print("* Fitting parameters with values and uncertainties:")
    for fp in fitparams.keys():
        if fp in ["numax", "dnuSer", "dnuscal", "dnuAsf"]:
            fpstr = f"{fp} (solar units)"
        else:
            fpstr = fp
        print(f"  - {fpstr}: {fitparams[fp]}")


def print_seismic(fitfreqs: dict, obskey: np.array, obs: np.array) -> None:
    # Fitting info: Frequencies
    if not fitfreqs["active"]:
        return
    if "freqs" in fitfreqs["fittypes"]:
        print("* Fitting of individual frequencies activated!")
    elif any(x in freqtypes.rtypes for x in fitfreqs["fittypes"]):
        print(f"* Fitting of frequency ratios {fitfreqs['fittypes']} activated!")
        if "r010" in fitfreqs["fittypes"]:
            print(
                "  - WARNING: Fitting r01 and r10 simultaniously results in overfitting, and is thus not recommended!"
            )
    elif any(x in freqtypes.glitches for x in fitfreqs["fittypes"]):
        print(f"* Fitting of glitches {fitfreqs['fittypes']} activated using:")
        print(f"  - Method: {fitfreqs['glitchmethod']}")
        print(f"  - Parameters in smooth component: {fitfreqs['npoly_params']}")
        print(f"  - Order of derivative: {fitfreqs['nderiv']}")
        print(f"  - Gradient tolerance: {fitfreqs['tol_grad']}")
        print(f"  - Regularization parameter: {fitfreqs['regu_param']}")
        print(f"  - Initial guesses: {fitfreqs['nguesses']}")
        print("* General frequency fitting configuration:")
    elif any(x in freqtypes.epsdiff for x in fitfreqs["fittypes"]):
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


def print_distances(distparams, asciiparams) -> None:
    # Fitting info: Distance
    if not distparams:
        return
    if ("parallax" in distparams) and "distance" in asciiparams:
        print("* Parallax fitting and distance inference activated!")
    elif "parallax" in distparams:
        print("* Parallax fitting activated!")
    elif "distance" in asciiparams:
        print("* Distance inference activated!")

    if distparams["dustframe"] == "icrs":
        print(
            f"  - Coordinates (icrs): RA = {distparams['RA']}, DEC = {distparams['DEC']}"
        )
    elif distparams["dustframe"] == "galactic":
        print(
            f"  - Coordinates (galactic): lat = {distparams['lat']}, lon = {distparams['lon']}"
        )

    print("  - Filters (magnitude value and uncertainty): ")
    for filt in distparams["filters"]:
        print(f"    + {filt}: [{distparams['m'][filt]}, {distparams['m_err'][filt]}]")

    if "parallax" in distparams:
        print(f"  - Parallax: {distparams['parallax']}")

    if "EBV" in distparams:
        print(f"  - EBV: {distparams['EBV'][1]} (uniform across all distance samples)")


def print_additional(inputparams) -> None:
    if "phase" in inputparams:
        print("* Fitting evolutionary phase!")
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
        "warnoutput",
    ]
    for ip in sorted(inputparams.keys()):
        if ip not in noprint:
            print(f"* {ip}: {inputparams[ip]}")


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


def print_priors(limits: dict, usepriors: list[str]) -> None:
    print("* Flat, constrained priors and ranges:")
    for lim in limits.keys():
        print(f"  - {lim}: {limits[lim]}")
    print(f"* Additional priors (IMF): {', '.join(usepriors)}")


class Logger(object):
    """
    Class used to redefine stdout to terminal and an output file.

    Parameters
    ----------
    outfilename : str
        Absolute path to an output file
    """

    # Credit: http://stackoverflow.com/a/14906787
    def __init__(self, outfilename):
        self.terminal = sys.stdout
        self.log = open(outfilename + ".log", "a")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        # this flush method is needed for python 3 compatibility.
        # this handles the flush command by doing nothing.
        # you might want to specify some extra behavior here.
        pass


def list_metallicities(Grid, defaultpath, inputparams, limits):
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
    if "grid" in defaultpath:
        metal = range(1)
    else:
        metal = [x for x in Grid[defaultpath].items() if "=" in x[0]]
        for i in range(len(metal)):
            metal[i] = float(metal[i][0][4:])
        metal = np.asarray(metal)

        metal_name = "MeH" if "MeH" in limits else "FeH"
        if metal_name in limits:
            metal = metal[
                (metal >= limits[metal_name][0]) & (metal <= limits[metal_name][1])
            ]
    return metal


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
    starid, inputparams, hout, out, hout_dist, out_dist, uncert="qunatiles", sigmacut=1
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
    if inputparams["warnoutput"] is False:
        return False
    fitparams = inputparams["fitparams"]
    warnfile = inputparams["warnoutput"]
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

    if len(inputparams["magnitudes"]) > 0:
        for m in list(inputparams["distanceparams"]["filters"]):
            mdist = "M_" + m
            if mdist in hout_dist:
                idx = np.nonzero([x == mdist for x in hout_dist])[0][0]
                priorM = inputparams["magnitudes"][m]["median"]
                priorerrp = inputparams["magnitudes"][m]["errp"]
                priorerrm = inputparams["magnitudes"][m]["errm"]
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

        if "distance" in hout_dist:
            idx = np.nonzero([x == "distance_joint" for x in hout_dist])[0][0]
            priordistqs = inputparams["distanceparams"]["priordistance"]
            priorerrm = priordistqs[0] - priordistqs[1]
            priorerrp = priordistqs[2] - priordistqs[0]
            if uncert == "quantiles":
                outerr = (out_dist[idx + 1] + out_dist[idx + 2]) / 2
            else:
                outerr = out_dist[idx + 1]
            serr = np.sqrt(((priorerrp + priorerrm) / 2) ** 2 + outerr**2)
            sigma = np.abs(out_dist[idx] - priordistqs[1]) / serr
            bigdiff = sigma >= sigmacut
            if bigdiff:
                comparewarn = True
                ps.append("distance")
                sigmas.append(sigma)

    if comparewarn:
        print("A >%s sigma difference was found between input and output of" % sigmacut)
        print(ps)
        print("with sigma differences of")
        print(sigmas)
        if isinstance(warnfile, IOBase):
            warnfile.write("{}\t{}\t{}\n".format(starid, ps, sigmas))
        else:
            with open(warnfile, "a") as wf:
                wf.write("{}\t{}\t{}\n".format(starid, ps, sigmas))

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
    else:
        print("Mistake in normfactor")


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


def printparam(param, xmed, xstdm, xstdp, uncert="quantiles", centroid="median"):
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
    print("{0:9}  {1:13} :  {2:12.6f}".format(centroid, param, xmed))
    if uncert == "quantiles":
        print("{0:9}  {1:13} :  {2:12.6f}".format("err_plus", param, xstdp - xmed))
        print("{0:9}  {1:13} :  {2:12.6f}".format("err_minus", param, xmed - xstdm))
    else:
        print("{0:9}  {1:13} :  {2:12.6f}".format("stdev", param, xstdm))
    print("-----------------------------------------------------")


def strtobool(val):
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
    val = val.lower()
    if val in ("y", "yes", "t", "true", "on", "1"):
        return 1
    elif val in ("n", "no", "f", "false", "off", "0"):
        return 0
    else:
        raise ValueError("invalid truth value %r" % (val,))
