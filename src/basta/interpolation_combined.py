"""
Interpolation for BASTA: Combined approach
"""

import os
import sys
import time

import h5py
import numpy as np
import matplotlib.pyplot as plt

from tqdm import tqdm
from scipy import spatial

from basta.utils_seismic import transform_obj_array

from basta import plot_interp as ip
from basta import interpolation_helpers as ih


def _calc_across_points(
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


def interpolate_combined(
    grid,
    outfile,
    selectedmodels,
    trackresolution,
    gridresolution,
    intpolparams,
    basepath="grid/",
    intpol_freqs=False,
    outbasename="",
    debug=False,
):
    """
    Routine for interpolating both across and along tracks, in a combined
    approach. Creates basis for new tracks using along interpolation specifications,
    thus using the the across interpolater to map to an increased number of points
    along the tracks, compared to the original tracks.

    Parameters
    ----------
    grid : h5py file
        Handle of grid to process

    outfile : h5py file
        Handle of output grid to write to

    selectedmodels : dict
        Dictionary of every track/isochrone with models inside the limits, and the index
        of every model that satisfies this.

    trackresolution : dict
       Controls for resolution along the tracks. Must contain "param" with a valid parameter
       name from the grid and "value" with the desired precision/resolution.

    gridresolution : dict
       Controls for resolution across the tracks. Must contain "param" with a valid parameter
       name from the grid and "value" with the desired precision/resolution.

    intpolparams : list
        List of parameters to be interpolated, avoid interpolating *everything*.

    basepath : str, optional
        Path in the grid where the tracks are stored. The default value applies to
        standard grids of tracks. It must be modified for isochrones!

    intpol_freqs : list, bool
        List of interpolated frequency interval if frequency interpolation requested.
        False if not interpolating frequencies.

    outbasename : str
        Name of the outputted plot of the base.

    Returns
    -------
    None
    """

    ###############################
    # BLOCK 0: Unpack and prepare #
    ###############################

    # Ensure Bayesian weight along
    if "grid" in basepath:
        dname = "dage"
        intpolparams = np.unique(np.append(intpolparams, dname))
    else:
        dname = "dmass"
        intpolparams = np.unique(np.append(intpolparams, dname))

    # Read input
    scale = gridresolution["scale"]
    retrace = gridresolution["retrace"] if "retrace" in gridresolution else False
    if trackresolution:
        along_var = trackresolution["baseparam"]
        respar = trackresolution["param"]
        alongintpol = True
    else:
        along_var = gridresolution["baseparam"]
        respar = along_var
        alongintpol = False

    # Read the parameters of the grid
    pars_sampled = [par.decode("UTF-8") for par in grid["header/pars_sampled"]]
    pars_varied = [par.decode("UTF-8") for par in grid["header/pars_variable"]]
    pars_constant = [par.decode("UTF-8") for par in grid["header/pars_constant"]]

    # Collect the parameters
    headvars = list(np.unique(pars_sampled + pars_varied + pars_constant))

    # Determine the number to assign the new tracks
    tracklist = list(grid[os.path.join(basepath, "tracks")].items())
    newnum = max([int(f[0].split("track")[-1]) for f in tracklist]) + 1
    numfmt = max(len(str(newnum)), len(str(int(newnum * scale))))

    # Check we have enough source tracks in the sub-box to form a simplex
    if len(selectedmodels) < len(pars_sampled) + 1:
        warstr = "Sub-box contains only {:d} tracks, while ".format(len(selectedmodels))
        warstr += "{:d} is needed for interpolation in the ".format(
            len(pars_sampled) + 1
        )
        warstr += (
            "parameter space of the input grid. Consider expanding sub-box (limits)."
        )
        raise ValueError(warstr)

    # Form the base array for interpolation
    base = np.zeros((len(selectedmodels), len(pars_sampled)))
    for i, name in enumerate(selectedmodels):
        for j, bpar in enumerate(pars_sampled):
            parm = grid[os.path.join(basepath, name, bpar)][0]
            base[i, j] = parm

    # Determine the base params for new tracks
    print("\nBuilding triangulation ... ", end="", flush=True)
    triangulation = spatial.Delaunay(base)
    new_points, trindex, sobnums = _calc_across_points(
        base,
        pars_sampled,
        triangulation,
        scale,
        outbasename,
        debug,
    )
    print("done!")

    # List of tracknames for accessing grid
    tracknames = list(selectedmodels)
    # List to sort out failed tracks at the end
    success = np.ones(len(new_points[:, 0]), dtype=bool)

    # Determine values constant along track
    const_vals = {}
    for par in pars_constant:
        val = grid[os.path.join(basepath, tracknames[0], par)][0]
        const_vals[par] = val
    # Parameters not sampled, but still vary across tracks
    varied_vals = {}
    for par in pars_varied:
        yvec = np.zeros((len(selectedmodels)))
        for i, name in enumerate(selectedmodels):
            yvec[i] = grid[os.path.join(basepath, name, par)][0]
        newval = ih.interpolation_wrapper(triangulation, yvec, new_points, along=False)
        varied_vals[par] = newval

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
        libname = os.path.join(
            basepath,
            "tracks",
            "track{{:0{0}d}}".format(numfmt).format(int(newnum + tracknum)),
        )

        #############################################################
        # BLOCK 1: Enveloping tracks data collection and along base #
        #############################################################

        # Information to be collected from enveloping tracks
        ind = triangulation.simplices[tind]
        count = sum([sum(selectedmodels[tracknames[i]]) for i in ind])
        intbase = np.zeros((count, len(pars_sampled) + 1))
        envres = np.zeros((count, len(pars_sampled) + 1))
        y = np.zeros((count))
        minmax = np.zeros((len(ind), 2))
        sections = [0]
        ir = 0

        # Names if retracing information requested
        if retrace:
            basenames = np.empty((count), dtype=object)

        # For along frequency resolution, check available l=0 modes
        if respar == "freqs":
            Nl0s = []
            for i in ind:
                track = tracknames[i]
                selmod = selectedmodels[tracknames[i]]
                Nl0s.append(ih.lowest_l0(grid, basepath, track, selmod))
            Nl0 = max(Nl0s)

        # Loop over the enveloping tracks to collect info
        for j, i in enumerate(ind):
            # Unpack the track and selected models of the track
            track = grid[os.path.join(basepath, tracknames[i])]
            selmod = selectedmodels[tracknames[i]]

            # Get the resolution variable from the track
            if respar == "freqs" and alongintpol:
                resvar = ih.get_l0_freqs(track, selmod, Nl0)
            elif alongintpol:
                resvar = track[respar][selmod]
            else:
                resvar = np.zeros(sum(selmod))

            # Base variable of the track
            bvar = track[along_var][selmod]
            minmax[j, :] = [min(bvar), max(bvar)]
            for k, (b, res) in enumerate(zip(list(bvar), list(resvar))):
                intbase[k + ir, : len(base[i])] = base[i]
                intbase[k + ir, -1] = b
                if alongintpol:
                    envres[k + ir, : len(base[i])] = base[i]
                    envres[k + ir, -1] = res
            if retrace:
                basenames[ir : ir + len(bvar)] = track["name"][selmod]
            ir += len(bvar)
            sections.append(ir)

        # Check of overlap from min and max
        minmax = [max(minmax[:, 0]), min(minmax[:, 1])]
        if minmax[0] > minmax[1]:
            warstr = "Warning: Track {0} ".format(newnum + tracknum)
            warstr += "aborted, no overlap in {0}.".format(along_var)
            print(warstr)
            success[tracknum] = False
            outfile[os.path.join(libname, "IntStatus")] = -1
            continue

        # Get base for new track, based on requested along resolution
        try:
            newbvar = ih.calc_along_points(
                intbase, sections, minmax, point, envres, trackresolution["value"]
            )
        except:
            warstr = "Choice of base parameter '{:s}' resulted".format(along_var)
            warstr += " in an error when determining it's variance along the track."
            raise ValueError(warstr)

        #################################
        # BLOCK 2: Actual interpolation #
        #################################

        # The base along the new track
        newbase = np.ones((len(newbvar), len(pars_sampled) + 1))
        for i, p in enumerate(point):
            newbase[:, i] *= p
        newbase[:, -1] = newbvar

        # Create triangulation of tinerpolation base once only
        sub_triangle = spatial.Delaunay(intbase)

        try:
            ################################
            # BLOCK 2a: Classic parameters #
            ################################
            for key in intpolparams:
                keypath = os.path.join(libname, key)
                # Weights are given a placeholder value
                if key == along_var:
                    outfile[keypath] = newbase[:, -1]
                elif "name" in key:
                    outfile[keypath] = newbase.shape[0] * [b"interpolated-entry"]
                elif key == dname:
                    dpath = os.path.join(libname, key[1:])
                    keypath = os.path.join(libname, key)
                    outfile[keypath] = ih.bay_weights(outfile[dpath])
                elif (key in headvars) or ("_weight" in key):
                    continue
                else:
                    # Collect from enveloping
                    for j, i in enumerate(ind):
                        track = os.path.join(basepath, tracknames[i])
                        yind = selectedmodels[tracknames[i]]
                        y[sections[j] : sections[j + 1]] = grid[track][key][yind]

                    # Interpolate, check for NaNs
                    newparam = ih.interpolation_wrapper(
                        sub_triangle, y, newbase, along=False
                    )
                    if any(np.isnan(newparam)):
                        nan = "Track {0} had NaN value(s)!".format(newnum + tracknum)
                        raise ValueError(nan)

                    # Write to new gridfile
                    outfile[keypath] = newparam

            ##################################
            # BLOCK 2b: Frequency parameters #
            ##################################
            if intpol_freqs:
                osc = []
                osckey = []
                for i in ind:
                    # Extract the oscillation fequencies and id's
                    track = tracknames[i]
                    trackosc = grid[basepath + track]["osc"]
                    trackosckey = grid[basepath + track]["osckey"]
                    for model in np.where(selectedmodels[track])[0]:
                        osc.append(transform_obj_array(trackosc[model]))
                        osckey.append(transform_obj_array(trackosckey[model]))

                # Compute new individual frequencies for track
                newosckey, newosc = ih.interpolate_frequencies(
                    osc,
                    osckey,
                    sections,
                    sub_triangle,
                    newbase,
                    freqlims=intpol_freqs,
                )

                # Writing variable length arrays to an HDF5 file is a bit tricky,
                # but can be done using datasets with a special datatype.
                # --> Here we follow the approach from BASTA/make_tracks
                dsetosc = outfile.create_dataset(
                    name=os.path.join(libname, "osc"),
                    shape=(len(newosc), 2),
                    dtype=h5py.special_dtype(vlen=float),
                )
                dsetosckey = outfile.create_dataset(
                    name=os.path.join(libname, "osckey"),
                    shape=(len(newosc), 2),
                    dtype=h5py.special_dtype(vlen=int),
                )
                for i in range(len(newosc)):
                    dsetosc[i] = newosc[i]
                    dsetosckey[i] = newosckey[i]

            #################################################
            # BLOCK 2c: Constant parameters along the track #
            #################################################
            for par, parval in zip(pars_sampled, point):
                keypath = os.path.join(libname, par)
                outfile[keypath] = np.ones(newbase.shape[0]) * parval
            for par in const_vals:
                keypath = os.path.join(libname, par)
                outfile[keypath] = np.ones(newbase.shape[0]) * const_vals[par]
            for par in varied_vals:
                keypath = os.path.join(libname, par)
                vval = varied_vals[par][tracknum]
                outfile[keypath] = np.ones(newbase.shape[0]) * vval

            ####################################################
            # BLOCK 2d: Interpolation source model information #
            ####################################################
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

            # Success! Set status as completed
            outfile[os.path.join(libname, "IntStatus")] = 0

        except KeyboardInterrupt:
            print("Interpolation stopped manually. Goodbye!")
            sys.exit()
        except:
            # If it fails, delete progress for the track, and just mark it as failed
            try:
                del outfile[libname]
            except:
                pass
            success[tracknum] = False
            print("Error:", sys.exc_info()[1])
            outfile[os.path.join(libname, "IntStatus")] = -1
            print("Interpolation failed for track {0}".format(newnum + tracknum))

    ####################
    # End of main loop #
    ####################
    pbar.close()

    ########################################
    # BLOCK 3: Wrap up, header and weights #
    ########################################

    # Plot the new resulting base
    if outbasename:
        plotted = ip.base_corner(
            pars_sampled, base, new_points[success], triangulation, scale, outbasename
        )
    else:
        plotted = False
    if not plotted:
        print("Plotting of full base failed in post")

    # Write the new tracks to the header, and recalculate the weights
    ih.update_header(outfile, basepath, headvars)
    ih.recalculate_weights(
        outfile, basepath, sobnums, gridresolution["extend"], debug=debug
    )
    grid.close()

    #######
    # END #
    #######
