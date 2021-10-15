"""
Interpolation for BASTA: Across/between tracks
"""
import os
import sys
import copy

import h5py
import numpy as np
from tqdm import tqdm
from scipy import spatial
from scipy import interpolate

from basta import sobol_numbers
from basta import interpolation_helpers as ih
from basta import plot_interp as ip

# ======================================================================================
# Interpolation helper routines
# ======================================================================================
def _check_sobol(grid, res):
    """
    Checks and unpacks whether sobol interpolation has been requested, and if Cartesian
    interpolation has been requested for Sobol grid, which is invalid.

    Parameters
    ----------
    grid : h5py file
        Handle of grid to process
    res : dict
        Dictionary of all the inputted resolution parameters

    Returns
    -------
    sobol : float/bool
        The scale value for across resolution if Sobol interpolation, False if not Sobol

    """
    # Read gridtype from header
    gridtype = grid["header/library_type"][()]

    # Check type and inputted scale resolution
    if "sobol" in gridtype.lower():
        if res["scale"] < 1.0:
            errstr = "For Sobol type grid only an increase in tracks via the 'scale' "
            errstr += "parameter is possible, please enter a value > 1."
            raise KeyError(errstr)
        sobol = res["scale"]
    elif "cartesian" in gridtype.lower():
        if res["scale"] > 1.0:
            sobol = res["scale"]
        else:
            sobol = False
    elif "isochrones" in gridtype.lower():
        if res["scale"] < 1.0:
            errstr = "For isochrone grids only an increase in isochrones  via the "
            errstr += "'scale' parameter is possible, please enter a value > 1."
            raise KeyError(errstr)
        sobol = res["scale"]
    else:
        raise KeyError(
            "Interpolation not possible for grid of type {0}".format(gridtype)
        )

    # Highlight redundant resolution for the user
    if sobol:
        for var in res:
            if (var not in ["scale", "baseparam"]) and res[var] != 0:
                prtstr = "Gridresolution in '{0}' is set but ignored, ".format(var)
                prtstr += "as 'scale' is set for Sobol interpolation."
                print(prtstr)
    return sobol


def _calc_across_points(
    base, baseparams, tri, sobol, outbasename, debug=False, verbose=False
):
    """
    Determine the new points for tracks in the base parameters, for either Cartesian
    or Sobol sampling. For Sobol, it determines a new Sobol sampling which satisfy an
    increase in number of tracks given a scale value. For Cartesian, given a set of
    whole numbers for each interpolation dimension, it will assign equally spaced points
    between the existing points, with possible 'overflow' in the positive direction
    of the parameter.
    Also plots the old vs new base of the interpolation, no plot for dim(base) = 1,
    corner plot for dim(base) > 2.

    Parameters
    ----------
    base : array
        The current base of the grid, formed as (number of tracks, parameters in base).

    baseparams : dict
        Dictionary of the parameters forming the grid, with the required resolution of
        the parameters.

    tri : object
        Triangulation of the base.

    sobol : float/bool
        Scale resolution for Sobol-sampled interpolation, False for Cartesian.

    outbasename : str
        Name of the outputted plot of the base.

    Returns
    -------
    newbase : array
        A base of the new points in the base, same structure as input base.

    trindex : array
        List of simplexes of the new points, for determination of the enveloping
        tracks.

    """

    if not sobol:
        # Cartesian routine. Stores arrays of points to be added and all points
        newbase = None
        wholebase = copy.deepcopy(base)

        # For each interpolation parameter, add the desired number of points
        for i, (par, res) in enumerate(baseparams.items()):
            newpoints = None
            # Unique values of the parameter
            uniq = np.unique(wholebase[:, i])
            # New spacing in parameter
            diff = np.mean(np.diff(uniq)) / (res + 1)
            # For each requested new point, add an offsetted copy of the base
            for j in range(res):
                points = wholebase.copy()
                points[:, i] += diff * (j + 1)
                if type(newpoints) != np.ndarray:
                    newpoints = points
                else:
                    newpoints = np.vstack((newpoints, points))
            # Update the arrays
            wholebase = np.vstack((wholebase, newpoints))
            if type(newbase) != np.ndarray:
                newbase = newpoints
            else:
                newbase = np.vstack((newbase, newpoints))

        # Find all points within triangulation
        mask = tri.find_simplex(newbase)
        newbase = newbase[mask != -1]
        trindex = tri.find_simplex(newbase)

    elif sobol:
        # Check that we increase the number of tracks
        lorgbase = len(base)
        lnewbase = int(lorgbase * sobol)
        assert lnewbase > lorgbase
        ndim = len(baseparams)
        l_trim = 1

        # Try sampling the parameter space, and retry until increase met
        while l_trim / sobol < lorgbase:
            # Extract Sobol sequences
            lnewbase = int(lnewbase * 1.2)
            sob_nums = np.zeros((lnewbase, ndim))
            iseed = 1
            for i in range(lnewbase):
                iseed, sob_nums[i, :] = sobol_numbers.i8_sobol(ndim, iseed)

            # Assign parameter values by sequence
            newbase = []
            for npar in range(ndim):
                Cmin = min(base[:, npar])
                Cmax = max(base[:, npar])
                newbase.append((Cmax - Cmin) * sob_nums[:, npar] + Cmin)

            # Remove points outside subgrid
            newbase = np.asarray(newbase).T
            mask = tri.find_simplex(newbase)
            newbase = newbase[mask != -1]
            l_trim = len(newbase[:, 0])

        # Compute new simplex list
        trindex = tri.find_simplex(newbase)

    # Plot of old vs. new base of subgrid
    if len(baseparams) > 1 and (debug or verbose):
        outname = outbasename.split(".")[-2] + "_all"
        outname += "." + outbasename.split(".")[-1]
        success = ip.base_corner(baseparams, base, newbase, tri, sobol, outname)
        if success:
            print(
                "Initial across interpolation base has been plotted in",
                "figure",
                outname,
            )
    return newbase, trindex


# ======================================================================================
# Interpolation across tracks
# ======================================================================================
def _interpolate_across(
    grid,
    outfile,
    resolution,
    limits,
    intpolparams,
    basepath="grid/",
    intpol_freqs=False,
    along_var="xcen",
    outbasename="",
    debug=False,
    verbose=False,
):
    """
    Interpolates a grid across the tracks, within a box of observational limits.

    Parameters
    ----------
    grid : h5py file
        Handle of grid to process

    outfile : h5py file
        Handle of output grid to write to

    resolution : dict
       Required resolution. Must contain "param" with a valid parameter name from the
       grid and "value" with the desired precision/resolution.

    limits : dict
        Constraints on the selection in the grid. Must be valid parameter names in the
        grid. Example of the form: {'Teff': [5000, 6000], 'FeH': [-0.2, 0.2]}

    intpolparams : list
        List of parameters to be interpolated, avoid interpolating *everything*.

    basepath : str, optional
        Path in the grid where the tracks are stored. The default value applies to
        standard grids of tracks. It must be modified for isochrones!

    intpol_freqs : bool
        Whether or not to interpolate individual oscillation frequencies.

    along_var : str
        User-defined parameter to use as base along the track in interpolation routine.

    outbasename : str
        Name and destionaion of plot for old vs new base of grid.

    debug : bool, optional
        Activate debug mode. Will print extra info upon a failed interpolation of a track.

    verbose : bool, optional
        Print information to console and make simple diagnostic plots. Will be
        automatically set by debug.

    Returns
    -------
    grid : h5py file
        Handle of grid to process

    outfile : h5py file
        Handle of output grid to write to

    fail : bool
        Boolean to indicate whether the routine has failed or succeeded

    """
    print("\n********************\nAcross interpolation\n********************")
    # Parameters possibly in header
    headvars = [
        "tracks",
        "isochs",
        "massini",
        "age",
        "FeHini",
        "MeHini",
        "yini",
        "alphaMLT",
        "ove",
        "gcut",
        "eta",
        "alphaFe",
        "dif",
    ]

    # Determine whether the grid is iscohrones or tracks
    if "track" in grid["header/library_type"][()]:
        isomode = False
        modestr = "track"
        dname = "dage"

        # Determine the number to assign the new tracks
        tracklist = list(grid[basepath + "tracks"].items())
        newnum = max([int(f[0].split("track")[-1]) for f in tracklist]) + 1
        numfmt = len(tracklist[0][0].split("track")[-1])

        # Form basis of varied parameters
        bpars = [par.decode("UTF-8") for par in grid["header/active_weights"]]
        baseparams = {par: resolution[par] for par in bpars}
        const_vars = {}
        for par in headvars:
            if (par not in bpars) and (par in grid["header"]):
                const_vars[par] = grid[os.path.join("header", par)][0]

        # Collect the headvars, as they are constant along the track
        headvars = list(np.unique(list(bpars) + list(const_vars)))
        sobol = _check_sobol(grid, resolution)

    elif "isochrone" in grid["header/library_type"][()]:
        isomode = True
        modestr = "isochrone"
        dname = "dmass"
        newnum = 0

        # Parameters for forming basis
        bpars = [par.decode("UTF-8") for par in grid["header/active_weights"]]
        baseparams = {par: resolution[par] for par in bpars}
        const_vars = {}
        isochhead = os.path.join("header", basepath)
        for par in headvars:
            if (par not in bpars) and (par in grid[isochhead]):
                const_vars[par] = grid[os.path.join(isochhead, par)][0]
        # Only propagate the present parameters
        headvars = list(np.unique(list(bpars) + list(const_vars)))
        sobol = _check_sobol(grid, resolution)

    # Check frequency limits
    if "freqs" in limits:
        freqlims = limits["freqs"]
        del limits["freqs"]

    # Extract tracks/isochrones within user-specified limits
    print("Locating limits and restricting sub-grid ... ", flush=True)
    selectedmodels = ih.get_selectedmodels(grid, basepath, limits, cut=False)

    # If Cartesian method, save tracks/isochrones within limits to new grid
    fail = False
    if grid != outfile and not sobol:
        for name, index in selectedmodels:
            if not isomode:
                index2d = np.array(np.transpose([index, index]))
            if not (any(index) and sum(index) > 2):
                outfile[os.path.join(name, "FeHini_weight")] = -1
            else:
                # Write everything from the old grid to the new in the region
                for key in grid[name].keys():
                    keypath = os.path.join(name, key)
                    if "_weight" in key:
                        outfile[keypath] = grid[keypath][()]
                    elif "osc" in key:
                        if intpol_freqs:
                            outfile[keypath] = grid[keypath][index2d]
                    else:
                        outfile[keypath] = grid[keypath][index]

    # Form the base array for interpolation
    base = np.zeros((len(selectedmodels), len(baseparams)))
    for i, name in enumerate(selectedmodels):
        for j, bpar in enumerate(baseparams):
            parm = grid[basepath + name][bpar][0]
            base[i, j] = parm

    # Determine the base params for new tracks
    print("\nBuilding triangulation ... ", end="", flush=True)
    triangulation = spatial.Delaunay(base)
    new_points, trindex = _calc_across_points(
        base, baseparams, triangulation, sobol, outbasename, debug, verbose
    )
    print("done!")

    # List of tracknames for accessing grid
    tracknames = list(selectedmodels)
    # List to sort out failed tracks/isochrones at the end
    success = np.ones(len(new_points[:, 0]), dtype=bool)

    #############
    # Main loop #
    #############
    numnew = len(new_points)
    print("Interpolating {0} tracks/isochrones ... ".format(numnew))

    # Use a progress bar (with the package tqdm; will write to stderr)
    pbar = tqdm(total=numnew, desc="--> Progress", ascii=True)

    # Use tqdm for progress bar
    for tracknum, (point, tind) in enumerate(zip(new_points, trindex)):
        # Update progress bar in the start of the loop to count skipped tracks
        pbar.update(1)

        # Directory of the track/isochrone
        if not isomode:
            libname = (
                basepath + "tracks/track" + str(int(newnum + tracknum)).zfill(numfmt)
            )
        else:
            FeH = point[bpars.index("FeHini")]
            age = point[bpars.index("age")]
            libname = basepath + "FeH={0:.4f}/age={1:.4f}".format(FeH, age)

        # Form the basis of interpolation, and collect minmax of the along track variable
        ind = triangulation.simplices[tind]
        count = sum([sum(selectedmodels[tracknames[i]]) for i in ind])
        intbase = np.zeros((count, len(bpars) + 1))
        y = np.zeros((count))
        minmax = np.zeros((len(ind), 3))
        ir = 0

        # Loop over the enveloping tracks
        for j, i in enumerate(ind):
            track = tracknames[i]
            bvar = grid[basepath + track][along_var][selectedmodels[track]]
            minmax[j, :] = [min(bvar), max(bvar), abs(np.median(np.diff(bvar)))]
            for k, a in enumerate(list(bvar)):
                intbase[k + ir, : len(base[i])] = base[i]
                intbase[k + ir, -1] = a
            ir += len(bvar)
        minmax = [max(minmax[:, 0]), min(minmax[:, 1]), np.mean(minmax[:, 2])]
        if minmax[0] > minmax[1]:
            warstr = "Warning: Interpolating {0} {1} ".format(
                modestr, newnum + tracknum
            )
            warstr += "was aborted due to no overlap in {0}".format(along_var)
            warstr += " of the enveloping {0}!".format(modestr)
            print(warstr)
            success[tracknum] = False
            outfile[os.path.join(libname, "FeHini_weight")] = -1
            continue

        # Assume equal spacing, but approximately the same number of points
        try:
            Npoints = abs(int(np.ceil((minmax[1] - minmax[0]) / minmax[2])))
        except:
            prtstr = "Choice of base parameter '{:s}' resulted".format(along_var)
            prtstr += " in an error when determining it's variance along the "
            prtstr += "{:s}, consider choosing another.".format(modestr)
            raise ValueError(prtstr)
        # The base along the new track
        newbvar = np.linspace(minmax[0], minmax[1], Npoints)
        newbase = np.ones((len(newbvar), len(bpars) + 1))
        for i, p in enumerate(point):
            newbase[:, i] *= p
        newbase[:, -1] = newbvar
        sub_triangle = spatial.Delaunay(intbase)

        try:
            # Interpolate and write each individual parameter, apart from oscillations
            for key in intpolparams:
                keypath = os.path.join(libname, key)
                # Weights are given a placeholder value
                if "_weight" in key:
                    outfile[keypath] = 1.0
                elif key == along_var:
                    outfile[keypath] = newbase[:, -1]
                elif key == dname:
                    outfile[keypath] = ih.bay_weights(newbase[:, -1])
                elif "name" in key:
                    outfile[keypath] = len(newbase[:, -1]) * [b"interpolated-entry"]
                elif ("osc" in key) or (key in const_vars):
                    continue
                else:
                    ir = 0
                    for j, i in enumerate(ind):
                        track = tracknames[i]
                        yind = selectedmodels[track]
                        y[ir : ir + sum(yind)] = grid[basepath + track][key][yind]
                        ir += sum(yind)
                    intpol = interpolate.LinearNDInterpolator(sub_triangle, y)
                    newparam = intpol(newbase)
                    if any(np.isnan(newparam)):
                        nan = "{0} {1} had NaN value(s)!".format(
                            modestr, newnum + tracknum
                        )
                        raise ValueError(nan)
                    outfile[keypath] = newparam

            # Dealing with oscillations
            if intpol_freqs:
                osc = []
                osckey = []
                sections = [0]
                for i in ind:
                    # Extract the oscillation fequencies and id's
                    track = tracknames[i]
                    for model in np.where(selectedmodels[track])[0]:
                        osc.append(grid[basepath + track]["osc"][model])
                        osckey.append(grid[basepath + track]["osckey"][model])
                    sections.append(len(osc))
                newosckey, newosc = ih.interpolate_frequencies(
                    fullosc=osc,
                    fullosckey=osckey,
                    agevec=intbase,
                    newagevec=newbase,
                    sections=sections,
                    freqlims=freqlims,
                    debug=debug,
                    trackid=newnum + tracknum,
                )
                Npoints = len(newosc)
                # Writing variable length arrays to an HDF5 file is a bit tricky,
                # but can be done using datasets with a special datatype.
                # --> Here we follow the approach from BASTA/make_tracks
                dsetosc = outfile.create_dataset(
                    name=os.path.join(libname, "osc"),
                    shape=(Npoints, 2),
                    dtype=h5py.special_dtype(vlen=np.float),
                )
                dsetosckey = outfile.create_dataset(
                    name=os.path.join(libname, "osckey"),
                    shape=(Npoints, 2),
                    dtype=h5py.special_dtype(vlen=np.int),
                )
                for i in range(Npoints):
                    dsetosc[i] = newosc[i]
                    dsetosckey[i] = newosckey[i]

            # Dealing with constants of the track
            outfile[os.path.join(libname, "FeHini_weight")] = 1.0
            for par, parval in zip(baseparams, point):
                keypath = os.path.join(libname, par)
                try:
                    outfile[keypath]
                except:
                    outfile[keypath] = np.ones(len(newbase[:, -1])) * parval
            for par in const_vars:
                keypath = os.path.join(libname, par)
                if par in ["tracks", "isochs"]:
                    continue
                try:
                    outfile[keypath]
                except:
                    outfile[keypath] = np.ones(len(newbase[:, -1])) * const_vars[par]
        except KeyboardInterrupt:
            print("BASTA interpolation stopped manually. Goodbye!")
            sys.exit()
        except:
            # If it fails, delete progress for the track, and just mark it as failed
            try:
                del outfile[libname]
            except:
                None
            success[tracknum] = False
            print("Error:", sys.exc_info()[1])
            outfile[os.path.join(libname, "FeHini_weight")] = -1
            print("Interpolation failed for {0}".format(libname))
            if debug:
                print("Point at:")
                [print(name, value, ", ") for name, value in zip(bpars, point)]
                print("Simplex formed by the {0}s:".format(modestr))
                print(", ".join([tracknames[i] for i in ind]))

    ####################
    # End of main loop #
    ####################
    pbar.close()

    # Plot the new resulting base
    plotted = ip.base_corner(
        baseparams, base, new_points[success], triangulation, sobol, outbasename
    )
    if plotted:
        print("Across interpolation base has been plotted in figure", outbasename)

    # Remove all previous tracks, to conserve sobol homogeniety
    if grid == outfile and sobol:
        for name in tracknames:
            namepath = os.path.join(basepath, name)
            del outfile[namepath]

    # Re-add frequency limits for combined approaches
    if intpol_freqs:
        limits["freqs"] = freqlims

    # Write the new tracks to the header, and recalculate the weights
    outfile = ih._extend_header(outfile, basepath, headvars)
    outfile = ih._recalculate_weights(outfile, basepath, headvars)
    return grid, outfile, fail
