"""
Interpolation for BASTA: Main driver and interface routine
"""
import os
import sys
import time

import h5py
import numpy as np

from basta.constants import parameters
from basta.constants import sydsun as sydc
from basta.utils_general import Logger
from basta import interpolation_helpers as ih
from basta import interpolation_across as iac
from basta import interpolation_along as ial
from basta import interpolation_combined as ico


# ======================================================================================
# Helper routines
# ======================================================================================
def _rescale_param(param, value, dnu):
    """
    Rescales the resolution value if a dnu parameter is chosen.

    Parameters
    ----------
    param : str
        Parameter name.
    value : float, array
        Value(s) to be rescaled
    dnu : float
        Solar value of dnu given in input.

    Returns
    -------
    value : float, array
        The rescaled value(s)
    """

    # Only run for dnu params, do nothing for other parameters
    if param.startswith("dnu") and param != "dnufit":
        print(
            "Note: {0} converted to solar units from {1} muHz".format(param, value),
            "assuming dnusun = {0:.2f} muHz".format(dnu),
        )
        value /= dnu
    return value


def _redundancy_print(case, intpol):
    """
    Prints a warning to the user, that a specified resolution is not used.

    Parameters
    ----------
    case : str
        The chosen interpolation case.
    intpol : dict
        Dictionary of all interpolation input.
    """

    red = False
    if case == "along" and "gridresolution" in intpol:
        red = ["Grid", "along"]
    elif case == "across" and "trackresolution" in intpol:
        red = ["Track", "across"]
    prtstr = "Warning: {0}resolution is set for {1} interpolation but is ignored."
    if red:
        print(prtstr.format(red[0], red[1]))


def _unpack_intpol(intpol, dnusun, basepath):
    """
    Unpacks the input dictionary into the case, resolutions and limits.

    Parameters
    ----------
    intpol : dict
        The input interpolation dictionary.
    dnusun : float
        The solar dnu input value for rescaling dnu.
    basepath : str
        TEMPORARY. The base path in the grid. To check whether the unsupported
        across interpolation of isochrones has been requested.

    Returns
    -------
    case : str
        The requested interpolation case
    trackres : dict
        Dictionary containing the parameter set for resolution, and its value.
    gridres : dict
        Dictionary containing the parameter(s) set for resolution and its value(s).
    limits : dict
        The parameters(s) set for limiting the models considered in the input grid.
    """
    # Possible cases
    cases = ["along", "across", "combined", "alongacross"]

    # Initialise what to unpack
    trackres, gridres, limits, alongvar = None, None, {}, None
    try:
        case = intpol["method"]["case"]
        if case not in cases:
            prtstr = "\nThe selected case {0} is not valid for interpolation!".format(
                case
            )
            prtstr += "The allowed cases are: {0}".format(" | ".join(cases))
            raise ValueError(prtstr)
    except KeyError:
        print("\nNo interpolation case specified!")
        raise

    # Give the user a warning that there is redundant input
    _redundancy_print(case, intpol)

    # Unpack and check along track resolution
    if case != "across":
        try:
            trackres = intpol["trackresolution"]
            trackres["value"] = _rescale_param(
                trackres["param"], trackres["value"], dnusun
            )
            print(
                "Required trackresolution in {0}: {1}".format(
                    trackres["param"], trackres["value"]
                )
            )
            if trackres["baseparam"] != "default":
                print(
                    "Base parameter changed from default to '{0}'.".format(
                        trackres["baseparam"]
                    )
                )
        except KeyError:
            print(
                "\nERROR! Trackresolution must be specified for {0} interpolation!".format(
                    case
                )
            )
            raise

    # Unpack and check across track resolution
    if case != "along":
        try:
            gridres = intpol["gridresolution"]
            alongvar = gridres["baseparam"]
            del gridres["baseparam"]
            print("Required gridresolution given by:")
            [print(" ", k, gridres[k]) if gridres[k] != 0 else None for k in gridres]
        except KeyError:
            print(
                "\nERROR! Gridresolution must be specified for {0} interpolation!".format(
                    case
                )
            )
            raise

    # TODO: While no Cartesian formulation of isochrone interpolation
    if gridres is not None and ("grid" not in basepath and "scale" not in gridres):
        prtstr = "\nERROR! Isochrones can currently not be interpolated across with"
        prtstr += " a Cartesian, please specify 'scale' for Sobol interpolation!"
        raise KeyError(prtstr)

    # Unpack limits
    for par in intpol["limits"]:
        val = list(_rescale_param(par, np.array(intpol["limits"][par]), dnusun))
        limits[par] = val
    print("Limiting the parameters:")
    [print(" ", p, limits[p]) for p in limits]

    return case, trackres, gridres, limits, alongvar


# ======================================================================================
# Main processing routine
# ======================================================================================
def interpolate_grid(
    gridname,
    outname,
    inputdict,
    dnusun,
    intpolparams,
    basepath="grid/",
    outbasename="",
    debug=False,
    verbose=False,
):
    """
    Select a part of a BASTA grid based on observational limits. Interpolate all
    quantities in that part and write to a new grid file. Create and interpolate
    new tracks to reach required resolution across tracks/isochrones.
    The grid is saved and closed to be reopened in the main routine.

    Parameters
    ----------
    gridname : str
        Path to grid to process

    outname : str
        Where to write output grid

    inputdict : dict
        Contains the input information for the interpolation routine. These consist of
        the case, the track- and gridresolution and limits.

    dnusun : float
        The solar dnu input value for rescaling dnu.

    intpolparams : list
        List of parameters to be interpolated in across interpolation. Should consist
        of everything from fitparams, cornerparams, and outparams.

    basepath : str, optional
        Path in the grid where the tracks are stored. The default value applies to
        standard grids of tracks. It must be modified for isochrones!

    debug : bool, optional
        Activate debug mode. Will print extra info and create plots.

    verbose : bool, optional
        Print information to console and make simple diagnostic plots. Will be
        automatically set by debug.

    Returns
    -------
    None

    """

    # Pretty print a marker of mode activation
    starsep = int(79 / 2) * "*"
    intnamestr = "Interpolation activated"
    tmpexpstr = "(experimental)"
    print(starsep)
    print("{0}{1}{0}".format(int((79 / 2 - len(intnamestr)) / 2) * " ", intnamestr))
    print("{0}{1}{0}".format(int((79 / 2 - len(tmpexpstr)) / 2) * " ", tmpexpstr))
    print(starsep)

    # Open the grids for reading and writing
    grid = h5py.File(gridname, "r")
    outfile = h5py.File(outname, "w")

    # Unpack the input dictionary
    case, trackresolution, gridresolution, limits, alongvar = _unpack_intpol(
        inputdict, dnusun, basepath
    )
    # Check that the specified base parameter exist
    if alongvar:
        try:
            groupname = list(grid[basepath])[0]
            itemname = list(grid[os.path.join(basepath, groupname)])[0]
            grid[os.path.join(basepath, groupname, itemname, alongvar)]
            intpolparams.append(alongvar)
            print(
                "Using {0} as additional base parameter in across interpolation".format(
                    alongvar
                )
            )
        except KeyError:
            print(
                "Specified base parameter {0} does not exist in the grid!".format(
                    alongvar
                )
            )
            raise

    # Nicknames for resolution in frequency
    freqres = ["freq", "freqs", "frequency", "frequencies", "osc"]
    intpol_freqs = False

    # Check that necessary parameters are in intpolparams
    # First is standard for tracks/isochrones, mostly for Kiel-diagram
    intpolparams += ["Teff", "logg", "massini", "age", "FeH"]
    if "freqs" in intpolparams:
        intpolparams.remove("freqs")
        intpolparams += ["tau0", "tauhe", "taubcz", "dnufit"]
        intpol_freqs = True
    # Determine if individual frequencies should be interpolated
    if case in ["along", "combined", "alongacross"]:
        if trackresolution["param"] in freqres:
            intpol_freqs = True
    if intpol_freqs:
        print("Warning: Interpolation of individual oscillation frequencies requested!")
    intpolparams = list(np.unique(intpolparams))

    # Interpolate according to each of the possible cases
    if case == "along":
        grid, outfile, fail = ial._interpolate_along(
            grid,
            outfile,
            limits,
            trackresolution,
            intpolparams,
            basepath,
            intpol_freqs,
            debug,
            verbose,
        )
        grid, outfile = ih._write_header(grid, outfile, basepath)
        grid.close()
    elif case == "across":
        grid, outfile = ih._write_header(grid, outfile, basepath)
        grid, outfile, fail = iac._interpolate_across(
            grid,
            outfile,
            gridresolution,
            limits,
            intpolparams,
            basepath,
            intpol_freqs,
            alongvar,
            outbasename,
            debug,
            verbose,
        )
        grid.close()
    elif case == "combined":
        grid, outfile = ih._write_header(grid, outfile, basepath)
        ico.interpolate_combined(
            grid,
            outfile,
            limits,
            trackresolution,
            gridresolution,
            intpolparams,
            basepath,
            intpol_freqs,
            False,
            outbasename,
            debug,
        )
    elif case == "acrossalong":
        grid, outfile = ih._write_header(grid, outfile, basepath)
        grid, outfile, fail = iac._interpolate_across(
            grid,
            outfile,
            gridresolution,
            limits,
            intpolparams,
            basepath,
            intpol_freqs,
            alongvar,
            outbasename,
            debug,
            verbose,
        )
        grid.close()
        outfile, outfile, fail = ial._interpolate_along(
            outfile,
            outfile,
            limits,
            trackresolution,
            intpolparams,
            basepath,
            intpol_freqs,
            debug,
            verbose,
        )
    elif case == "alongacross":
        grid, outfile, fail = ial._interpolate_along(
            grid,
            outfile,
            limits,
            trackresolution,
            intpolparams,
            basepath,
            intpol_freqs,
            debug,
            verbose,
        )
        grid, outfile = ih._write_header(grid, outfile, basepath)
        # Grid can be closed now, across will use the along interpolated grid as input
        grid.close()
        if not fail:
            outfile, outfile, fail = iac._interpolate_across(
                outfile,
                outfile,
                gridresolution,
                limits,
                intpolparams,
                basepath,
                intpol_freqs,
                alongvar,
                outbasename,
                debug,
                verbose,
            )

    print("\n*********************\nInterpolation wrap-up\n*********************")

    # Remove interpolated models outside of limits
    print("Removing models outside limits ... ", flush=True)
    if "freqs" in limits:
        del limits["freqs"]
    selectedmodels = ih.get_selectedmodels(outfile, basepath, limits, cut=True)
    for name, index in selectedmodels.items():
        if sum(index) != len(index):
            libname = os.path.join(basepath, name)
            for key in outfile[libname].keys():
                keypath = os.path.join(libname, key)
                if type(outfile[keypath][()]) != np.ndarray:
                    continue
                # If osc or osckey, transform to 2D index array
                elif key in ["osc", "osckey"]:
                    index2d = np.array(np.transpose([index, index]))
                    vec = outfile[keypath][index2d].reshape((-1, 2))
                else:
                    vec = outfile[keypath][index]
                del outfile[keypath]
                outfile[keypath] = vec

    # Write timestamp to outfile and close
    outfile[os.path.join("header", "interpolation_time")] = time.strftime(
        "%Y-%m-%d at %H:%M:%S"
    )
    print(intpolparams)
    print(list(outfile["grid/tracks/track270"]))
    outfile.close()


def perform_interpolation(
    grid, idgrid, intpol, inputparams, debug=False, verbose=False
):
    """
    Setup and call of main interpolation driver.

    Parameters
    ----------
    grid : str
        Name of input gridfile.
    idgrid : tuple
        Science case for isochrones
    intpol : dict
        Interpolation settings.
    inputparams : dict
        BASTA settings extracted from xml.
    debug : bool
        Flag for showing debugging output
    verbose : bool
        Flag for showing additional information while running

    Returns
    -------
    intpolgrid : str
        Name of new interpolated grid to use for fitting
    """

    # Check that it is not already interpolated
    Grid = h5py.File(grid, "r")
    try:
        gridtype = Grid["header/library_type"][()]

        # Allow for usage of both h5py 2.10.x and 3.x.x
        # --> If things are encoded as bytes, they must be made into standard strings
        if isinstance(gridtype, bytes):
            gridtype = gridtype.decode("utf-8")
    except KeyError:
        print("Error: Grid header missing 'library_type'!")
        Grid.close()
        sys.exit(1)
    try:
        intime = Grid["header/interpolation_time"][()]
    except KeyError:
        Grid.close()
        grid_is_intpol = False
    else:
        print(
            "WARNING! The use of interpolation was requested. However,",
            "the input grid was already interpolated at {0}".format(intime),
            "Will *not* calculate another interpolation, moving on...\n",
        )
        Grid.close()
        return grid

    # Names for output files
    intpolgrid = os.path.join(inputparams["output"], intpol["name"]["value"] + ".hdf5")
    outbasename = os.path.join(
        inputparams["output"],
        intpol["name"]["value"] + "_intpolbase." + inputparams["plotfmt"],
    )

    # If requested interpolated grid already exists, move on
    if os.path.exists(intpolgrid):
        print(
            "The requested interpolated grid '{0}' already exists!".format(intpolgrid),
            "No need to re-calculate, moving on...",
        )
        return intpolgrid

    ##############
    # Let's do it!
    ##############

    # Start the log
    outdir = inputparams.get("output")
    logname = os.path.join(outdir, intpol["name"]["value"])
    stdout = sys.stdout
    sys.stdout = Logger(logname)

    print("INFO: Calculating interpolated grid (be patient) ...")

    # Obtain model dnu for rescaling input
    dnusun = inputparams.get("dnusun", sydc.SUNdnu)
    # Get basepath
    if "tracks" in gridtype.lower():
        basepath = "grid/"
    elif "isochrones" in gridtype.lower():
        if idgrid:
            basepath = "ove={0:.4f}/dif={1:.4f}/eta={2:.4f}/alphaFe={3:.4f}/".format(
                idgrid[0], idgrid[1], idgrid[2], idgrid[3]
            )
        else:
            print(
                "Unable to construct path for science case, due to missing",
                "(ove, dif, eta, alphaFe) in input!",
            )
            raise RuntimeError

    # Retrieve list of parameters to be interpolated
    allparams = (
        list(inputparams["fitparams"].keys())
        + inputparams["cornerplots"]
        + inputparams["asciiparams"]
    )
    if any([x in allparams for x in ["distance", "parallax"]]):
        allparams += inputparams["distanceparams"]["filters"]
    intpolparams = list(np.unique(allparams))
    if "distance" in intpolparams:
        intpolparams.remove("distance")
    if "parallax" in intpolparams:
        intpolparams.remove("parallax")
    mask = [True if par in parameters.names else False for par in intpolparams]
    intpolparams = list(np.asarray(intpolparams)[mask])
    if inputparams.get("fitfreqs", False):
        intpolparams.append("freqs")

    # Run the external routine
    it0 = time.localtime()
    interpolate_grid(
        gridname=grid,
        outname=intpolgrid,
        inputdict=intpol,
        dnusun=dnusun,
        intpolparams=intpolparams,
        basepath=basepath,
        outbasename=outbasename,
        debug=debug,
        verbose=verbose,
    )
    it1 = time.localtime()
    dt = time.mktime(it1) - time.mktime(it0)
    print("\nInterpolation done in {0} seconds!".format(dt))

    # CLose the log
    sys.stdout = stdout
    print("Saved log to {0}.log".format(logname))
    print("\nNow fitting using the interpolated grid '{0}'\n\n".format(intpolgrid))
    return intpolgrid
