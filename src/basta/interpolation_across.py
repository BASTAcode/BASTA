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

from basta import interpolation_helpers as ih
from basta import plot_interp as ip

from basta.utils_seismic import transform_obj_array

import traceback


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
    # Read gridtype from header | Allow for usage of both h5py 2.10.x and 3.x.x
    # --> If things are encoded as bytes, they must be made into standard strings
    gridtype = grid["header/library_type"][()]
    if isinstance(gridtype, bytes):
        gridtype = gridtype.decode("utf-8")

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
            if (var not in ["scale", "baseparam", "extend", "retrace"]) and res[
                var
            ] != 0:
                prtstr = "Gridresolution in '{0}' is set but ignored, ".format(var)
                prtstr += "as 'scale' is set for Sobol interpolation."
                print(prtstr)
    return sobol


def _calc_cartesian_points(
    base,
    baseparams,
    tri,
    outbasename,
    debug=False,
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

    # Stores arrays of points to be added and all points
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

    # Plot of old vs. new base of subgrid
    if len(baseparams) > 1 and debug:
        outname = outbasename.split(".")[-2] + "_all"
        outname += "." + outbasename.split(".")[-1]
        success = ip.base_corner(baseparams, base, newbase, tri, False, outname)
        if success:
            print(
                "Initial across interpolation base has been plotted in",
                "figure",
                outname,
            )
    return newbase, trindex


def _calc_sobol_points(
    base,
    baseparams,
    tri,
    sobol,
    outbasename,
    debug=False,
):
    """
    Determine the new points for tracks in the base parameters, for Sobol sampling.
    It determines a new Sobol sampling which satisfy an increase in number of tracks,
    given a scale value.

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

    sobol : float
        Scale resolution for Sobol-sampled interpolation

    outbasename : str
        Name of the outputted plot of the base.

    Returns
    -------
    newbase : array
        A base of the new points in the base, same structure as input base.

    trindex : array
        List of simplexes of the new points, for determination of the enveloping
        tracks.

    sob_nums : array
        Sobol numbers used to generate new base, which is needed for determining
        volume weights of the tracks
    """

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
        sob_nums = ih.sobol_wrapper(ndim, lnewbase, 1, debug=debug)

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
    if len(baseparams) > 1 and debug:
        outname = outbasename.split(".")[-2] + "_all"
        outname += "." + outbasename.split(".")[-1]
        success = ip.base_corner(baseparams, base, newbase, tri, sobol, outname)
        if success:
            print(
                "Initial across interpolation base has been plotted in",
                "figure",
                outname,
            )
    return newbase, trindex, sob_nums[mask != -1]


# ======================================================================================
# Interpolation across tracks
# ======================================================================================
def interpolate_across(
    grid,
    outfile,
    resolution,
    selectedmodels,
    intpolparams,
    basepath="grid/",
    intpol_freqs=False,
    along_var="xcen",
    outbasename="",
    debug=False,
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

    selectedmodels : dict
        Dictionary of every track/isochrone with models inside the limits, and the index
        of every model that satisfies this.

    intpolparams : list
        List of parameters to be interpolated, avoid interpolating *everything*.

    basepath : str, optional
        Path in the grid where the tracks are stored. The default value applies to
        standard grids of tracks. It must be modified for isochrones!

    intpol_freqs : list, bool
        List of interpolated frequency interval if frequency interpolation requested.
        False if not interpolating frequencies.

    along_var : str
        User-defined parameter to use as base along the track in interpolation routine.

    outbasename : str
        Name and destionaion of plot for old vs new base of grid.

    debug : bool, optional
        Activate debug mode. Will print extra info upon a failed interpolation of a track.

    Returns
    -------
    None
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

    # Determine whether the grid is iscohrones or tracks (convert to allow all h5py's)
    gridtype = grid["header/library_type"][()]
    if isinstance(gridtype, bytes):
        gridtype = gridtype.decode("utf-8")
    if "track" in gridtype:
        isomode = False
        modestr = "track"
        dname = "dage"

        # Determine the number to assign the new tracks
        tracklist = list(grid[basepath + "tracks"].items())
        newnum = max([int(f[0].split("track")[-1]) for f in tracklist]) + 1
        numfmt = len(tracklist[0][0].split("track")[-1])

        # Form basis of varied parameters
        bpars = [par.decode("UTF-8") for par in grid["header/pars_sampled"]]
        baseparams = {par: resolution[par] for par in bpars}
        const_vars = {}
        for par in headvars:
            if (par not in bpars) and (par in grid["header"]):
                const_vars[par] = grid[os.path.join("header", par)][0]

        # Collect the headvars, as they are constant along the track
        headvars = list(np.unique(list(bpars) + list(const_vars)))
        sobol = _check_sobol(grid, resolution)
        retrace = resolution["retrace"] if "retrace" in resolution else False

    elif "isochrone" in gridtype:
        isomode = True
        modestr = "isochrone"
        dname = "dmass"
        newnum = 0

        # Parameters for forming basis
        bpars = [par.decode("UTF-8") for par in grid["header/pars_sampled"]]
        baseparams = {par: resolution[par] for par in bpars}
        const_vars = {}
        isochhead = os.path.join("header", basepath)
        for par in headvars:
            if (par not in bpars) and (par in grid[isochhead]):
                const_vars[par] = grid[os.path.join(isochhead, par)][0]
        # Only propagate the present parameters
        headvars = list(np.unique(list(bpars) + list(const_vars)))
        sobol = _check_sobol(grid, resolution)

    # Form the base array for interpolation
    base = np.zeros((len(selectedmodels), len(baseparams)))
    for i, name in enumerate(selectedmodels):
        for j, bpar in enumerate(baseparams):
            parm = grid[basepath + name][bpar][0]
            base[i, j] = parm

    # Determine the base params for new tracks
    print("\nBuilding triangulation ... ", end="", flush=True)
    triangulation = spatial.Delaunay(base)
    if sobol:
        new_points, trindex, sobnums = _calc_sobol_points(
            base,
            baseparams,
            triangulation,
            sobol,
            outbasename,
            debug,
        )
    else:
        new_points, trindex = _calc_cartesian_points(
            base,
            baseparams,
            triangulation,
            outbasename,
            debug,
        )
    print("done!")

    # List of tracknames for accessing grid
    tracknames = list(selectedmodels)
    # List to sort out failed tracks/isochrones at the end
    success = np.ones(len(new_points[:, 0]), dtype=bool)

    # Set up for debugging during run
    if debug:
        debugpath = "intpolout"
        if not os.path.exists(debugpath):
            os.mkdir(debugpath)

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
        sections = [0]

        # Names if retracing information requested
        if retrace:
            basenames = np.empty((count), dtype=object)

        # Loop over the enveloping tracks
        for j, i in enumerate(ind):
            track = grid[os.path.join(basepath, tracknames[i])]
            selmod = selectedmodels[tracknames[i]]

            bvar = track[along_var][selmod]
            minmax[j, :] = [min(bvar), max(bvar), abs(np.median(np.diff(bvar)))]
            for k, a in enumerate(list(bvar)):
                intbase[k + ir, : len(base[i])] = base[i]
                intbase[k + ir, -1] = a
            if retrace:
                basenames[ir : ir + len(bvar)] = track["name"][selmod]
            ir += len(bvar)
            sections.append(ir)

        minmax = [max(minmax[:, 0]), min(minmax[:, 1])]
        if minmax[0] > minmax[1]:
            warstr = "Warning: Interpolating {0} {1} ".format(
                modestr, newnum + tracknum
            )
            warstr += "was aborted due to no overlap in {0}".format(along_var)
            warstr += " of the enveloping {0}!".format(modestr)
            print(warstr)
            success[tracknum] = False
            outfile[os.path.join(libname, "IntStatus")] = -1
            continue

        # Get base for new track, mimicing enveloping tracks
        try:
            newbvar = ih.calc_along_points(intbase, sections, minmax, point)
        except:
            warstr = "Choice of base parameter '{:s}' resulted".format(along_var)
            warstr += " in an error when determining it's variance along the track."
            raise ValueError(warstr)

        # The base along the new track
        newbase = np.ones((len(newbvar), len(bpars) + 1))
        for i, p in enumerate(point):
            newbase[:, i] *= p
        newbase[:, -1] = newbvar

        # Create triangulation for re-use for each parameter
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
                elif "name" in key:
                    outfile[keypath] = len(newbase[:, -1]) * [b"interpolated-entry"]
                elif ("osc" in key) or (key in const_vars):
                    continue
                else:
                    # Collect values from enveloping tracks
                    for j, i in enumerate(ind):
                        track = tracknames[i]
                        yind = selectedmodels[track]
                        y[sections[j] : sections[j + 1]] = grid[basepath + track][key][
                            yind
                        ]

                    # Interpolate, check for NaNs
                    newparam = ih.interpolation_wrapper(
                        sub_triangle, y, newbase, along=False
                    )

                    if any(np.isnan(newparam)):
                        nan = "{0} {1} had NaN value(s)!".format(
                            modestr, newnum + tracknum
                        )
                        raise ValueError(nan)

                    # Write to new grid
                    outfile[keypath] = newparam

            # Dealing with oscillations
            if intpol_freqs:
                osc = []
                osckey = []
                for i in ind:
                    # Extract the oscillation fequencies and id's
                    track = grid[os.path.join(basepath, tracknames[i])]
                    for model in np.where(selectedmodels[tracknames[i]])[0]:
                        osc.append(transform_obj_array(track["osc"][model]))
                        osckey.append(transform_obj_array(track["osckey"][model]))
                newosckey, newosc = ih.interpolate_frequencies(
                    fullosc=osc,
                    fullosckey=osckey,
                    sections=sections,
                    triangulation=sub_triangle,
                    newvec=newbase,
                    freqlims=intpol_freqs,
                )
                Npoints = len(newosc)
                # Writing variable length arrays to an HDF5 file is a bit tricky,
                # but can be done using datasets with a special datatype.
                # --> Here we follow the approach from BASTA/make_tracks
                dsetosc = outfile.create_dataset(
                    name=os.path.join(libname, "osc"),
                    shape=(Npoints, 2),
                    dtype=h5py.special_dtype(vlen=float),
                )
                dsetosckey = outfile.create_dataset(
                    name=os.path.join(libname, "osckey"),
                    shape=(Npoints, 2),
                    dtype=h5py.special_dtype(vlen=int),
                )
                for i in range(Npoints):
                    dsetosc[i] = newosc[i]
                    dsetosckey[i] = newosckey[i]

            # Dealing with constants of the track
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

            # Bayesian weight along track
            par = "massfin" if dname == "dmass" else "age"
            parpath = os.path.join(libname, par)
            keypath = os.path.join(libname, dname)
            outfile[keypath] = ih.bay_weights(outfile[parpath])

            # Interpolation source model information
            if retrace:
                isnames = np.empty(
                    (newbase.shape[0], newbase.shape[1] + 1), dtype="S30"
                )
                iscoefs = np.empty((newbase.shape[0], newbase.shape[1] + 1), dtype="f8")
                for s, point in enumerate(newbase):
                    simplex = sub_triangle.find_simplex(point)
                    # Magic math from scipy.Delaunay documentation
                    dif = point - sub_triangle.transform[simplex, -1]
                    dot = sub_triangle.transform[simplex, : len(point)].dot(dif)
                    dists = [*dot, 1 - dot.sum()]

                    # Relating to source models
                    simplices = sub_triangle.simplices[simplex]
                    isnames[s, :] = basenames[simplices]
                    iscoefs[s, :] = dists
                outfile[os.path.join(libname, "isnames")] = isnames
                outfile[os.path.join(libname, "iscoefs")] = iscoefs

            # Successfully interpolated, mark it as such
            outfile[os.path.join(libname, "IntStatus")] = 0

            if debug:
                debugnum = str(int(newnum + tracknum)).zfill(numfmt)
                plotpath = os.path.join(
                    debugpath, "debug_kiel_{0}.png".format(debugnum)
                )
                if not os.path.exists(plotpath):
                    try:
                        tracks = [tracknames[i] for i in ind]
                        selmods = [selectedmodels[t] for t in tracks]
                        ip.across_debug(
                            grid,
                            outfile,
                            basepath,
                            along_var,
                            libname,
                            tracks,
                            selmods,
                            plotpath,
                        )
                        print(
                            "Plotted debug Kiel for {0} {1}".format(modestr, debugnum)
                        )
                    except:
                        print(
                            "Debug plotting failed for {0} {1}".format(
                                modestr, debugnum
                            )
                        )

        except KeyboardInterrupt:
            print("BASTA interpolation stopped manually. Goodbye!")
            sys.exit()
        except Exception as e:
            # If it fails, delete progress for the track, and just mark it as failed
            try:
                del outfile[libname]
            except:
                None
            success[tracknum] = False
            print("Error:", sys.exc_info()[1])
            outfile[os.path.join(libname, "IntStatus")] = -1
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

    # Write the new tracks to the header, and recalculate the weights
    ih.update_header(outfile, basepath, headvars)
    if "volume" in grid["header/active_weights"] or sobol:
        ih.recalculate_weights(
            outfile, basepath, sobnums, extend=resolution["extend"], debug=debug
        )
    else:
        ih.recalculate_param_weights(outfile, basepath)
