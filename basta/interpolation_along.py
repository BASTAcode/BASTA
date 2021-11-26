"""
Interpolation for BASTA: Along a track
"""
import os
import time

import h5py
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt

from basta import utils_general as gu
from basta import utils_seismic as su
from basta import interpolation_helpers as ih


# ======================================================================================
# Interpolation helper routines
# ======================================================================================
def _calc_npoints_freqs(libitem, index2d, freq_resolution, verbose=False, debug=False):
    """
    Estimate the number of points required for interpolation given a desired frequency
    resolution, by calculating the largest variation of any frequency in a given track.

    Currently it is only based on l=0.

    Parameters
    ----------
    libitem : h5py_group
        A track in the form of an HDF5 group from a BASTA grid

    index2d : array
        Mask for libitem to obtain selected entries (note the shape for osc indexing!)

    freq_resolution : float
       Required frequency resolution in microHertz

    verbose : bool, optional
        Print info.

    extra_debug : bool, optional
        Print extra information on all frequencies. Warning: Huge output.

    Returns
    -------
    Npoints : float
        Estimated number of points required in interpolated track

    """
    if debug:
        print("    l = 0 frequency information for i(nitial) and f(inal) model:")

    # Extract oscillation arrays for first and final point of the selection
    # --> Only for l=0
    fullosc = libitem["osc"][index2d].reshape((-1, 2))
    fullosckey = libitem["osckey"][index2d].reshape((-1, 2))
    osc = []
    osckey = []
    for i in [0, -1]:
        osckeyl0, oscl0 = su.get_givenl(
            l=0,
            osc=su.transform_obj_array(fullosc[i]),
            osckey=su.transform_obj_array(fullosckey[i]),
        )
        osckey.append(osckeyl0[1])
        osc.append(oscl0[0])

    # Calculate difference between first and final frequency of given order
    freqdiffs = []
    for nval, freqstart in zip(osckey[0], osc[0]):
        nmask = osckey[1] == nval
        if any(nmask):
            freqend = osc[1][nmask][0]
            freqdiff = np.abs(freqend - freqstart)
            freqdiffs.append(freqdiff)
            if debug:
                print(
                    "{0}n = {1:2}, freq_i = {2:8.3f}, freq_f = {3:8.3f},".format(
                        4 * " ", nval, freqstart, freqend
                    ),
                    "Delta(f) = {0:7.3f}".format(freqdiff),
                )

    # Obtain quantities in the notation of Aldo Serenelli
    DELTA = max(freqdiffs)
    Npoints = int(DELTA / freq_resolution)
    if debug:
        print("\n    DELTA = {0:6.2f} muHz ==> Npoints = {1:4}".format(DELTA, Npoints))
    elif verbose:
        print("DELTA = {0:6.2f} muHz ==> Npoints = {1:4}".format(DELTA, Npoints))

    return Npoints


def _calc_npoints(libitem, index, resolution, verbose=False, debug=False):
    """
    Estimate the number of points required for interpolation given a desired resolution,
    by calculating the variation in a given track.

    Parameters
    ----------
    libitem : h5py_group
        A track in the form of an HDF5 group from a BASTA grid

    index : array
        Mask for libitem to obtain selected entries

    resolution : dict
       Required resolution. Must contain "param" with a valid parameter name from the
       grid and "value" with the desired precision/resolution.

    verbose : bool, optional
        Print info.

    extra_debug : bool, optional
        Print extra information on all frequencies. Warning: Huge output.

    Returns
    -------
    Npoints : float
        Estimated number of points required in interpolated track

    """
    param = resolution["param"]
    paramvec = libitem[param][index]
    DELTA = np.abs(paramvec[-1] - paramvec[0])
    Npoints = int(DELTA / resolution["value"])

    if debug:
        print(
            "    DELTA({0}) = {1:6.2f} ==> Npoints = {2:4}".format(
                param, DELTA, Npoints
            )
        )
    elif verbose:
        print("DELTA = {0:6.2f} ==> Npoints = {1:4}".format(DELTA, Npoints))

    return Npoints


# ======================================================================================
# Interpolation along tracks
# ======================================================================================
def _interpolate_along(
    grid,
    outfile,
    limits,
    resolution,
    intpolparams,
    basepath="grid/",
    intpol_freqs=False,
    debug=False,
    verbose=False,
):
    """
    Select a part of a BASTA grid based on observational limits. Interpolate all
    quantities in the tracks within that part and write to a new grid file.

    Parameters
    ----------
    grid : h5py file
        Handle of grid to process

    outfile : h5py file
        Handle of output grid to write to

    limits : dict
        Constraints on the selection in the grid. Must be valid parameter names in the
        grid. Example of the form: {'Teff': [5000, 6000], 'FeH': [-0.2, 0.2]}

    resolution : dict
       Required resolution. Must contain "param" with a valid parameter name from the
       grid and "value" with the desired precision/resolution. A special case of "param"
       is "freq" (or "freqs"), in which case it is the required frequency resolution in
       microHertz (this corresponds to the old input "freq_resolution").

    basepath : str, optional
        Path in the grid where the tracks are stored. The default value applies to
        standard grids of tracks. It must be modified for isochrones!

    debug : bool, optional
        Activate debug mode. Will print extra info and create plots of the selection.
        WILL ONLY WORK PROPERLY FOR FREQUENCIES AND GRIDS (NOT DNU OR ISOCHRONES)!

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
    print("\n*******************\nAlong interpolation\n*******************")

    #
    # *** BLOCK 0: Initial preparation ***
    #
    if "grid" in basepath:
        isomode = False
        modestr = "track"
        headvars = [
            "tracks",
            "massini",
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
    else:
        isomode = True
        modestr = "isochrone"
        headvars = [
            "FeH",
            "FeHini",
            "MeH",
            "MeHini",
            "xini",
            "yini",
            "zini",
            "alphaMLT",
            "ove",
            "gcut",
            "eta",
            "alphaFe",
            "dif",
        ]

    overwrite = grid == outfile

    if debug:
        verbose = True
    if verbose:
        # Initialize logging to file (duplicate stdout)
        logdir = "intpollogs"
        if not os.path.exists(logdir):
            os.mkdir(logdir)
        logfile = open(
            os.path.join(
                logdir, "intpol_{0}.txt".format(time.strftime("%Y%m%dT%H%M%S"))
            ),
            "w",
        )

        # Initialise diagnostic plot(s) and print info
        plt.close("all")
        fig1, ax1 = plt.subplots()  # Full grid (Kiel)
        fig2, ax2 = plt.subplots()  # Only selection (Kiel)
        fig3, ax3 = plt.subplots()  # Age/mass information
        print("Interpolating in {0}s with basepath '{1}'".format(modestr, basepath))
        print("Limiting the parameters:\n{0}".format(limits))
        print(
            "Required resolution in {0}: {1}".format(
                resolution["param"], resolution["value"]
            )
        )

    # For tracks, interpolate sampling in age. For isochrones, in mass
    if isomode:
        baseparam = "massfin"
        dname = "dmass"
    else:
        baseparam = "age"
        dname = "dage"

    # Change to desired baseparameter, if requested
    if resolution["baseparam"] != "default":
        baseparam = resolution["baseparam"]

    # Nicknames for resolution in frequency
    freqres = ["freq", "freqs", "frequency", "frequencies", "osc"]

    # Get frequency interpolation limits from limits dict
    if "freqs" in limits:
        freqlims = limits["freqs"]
        del limits["freqs"]

    # Construct selectedmodels
    print("Locating limits and restricting sub-grid ... ", flush=True)
    selectedmodels = ih.get_selectedmodels(grid, basepath, limits, cut=False)

    #
    # *** LOOP THROUGH THE GRID ONE TRACK/ISOCHRONE AT A TIME  ***
    #
    # Before running the actual loop, all tracks/isochrones are counted to better
    # estimate the progress.
    intcount = 0
    for _, tracks in grid[basepath].items():
        intcount += len(tracks.items())

    # Use a progress bar (with the package tqdm; will write to stderr)
    print("\nInterpolating along {0} tracks/isochrones ...".format(intcount))
    pbar = tqdm(total=intcount, desc="--> Progress", ascii=True)

    # Do the actual loop
    trackcounter = 0
    fail = False
    # For tracks, the outer loop is just a single iteration
    # For isochrones, it is the metallicities one at a time
    for gname, tracks in grid[basepath].items():
        if fail:
            break

        for noingrid, (name, libitem) in enumerate(tracks.items()):
            # Update progress bar in the start of the loop to count skipped tracks
            pbar.update(1)

            if libitem["FeHini_weight"][()] < 0:
                continue
            # Make sure the user provides a valid parameter as resolution requirement
            # --> In case of failure, assume the user wanted a dnu-related parameter
            if resolution["param"].lower() not in freqres:
                try:
                    libitem[resolution["param"]]
                except KeyError:
                    print("\nCRITICAL ERROR!")
                    print(
                        "The resolution parameter '{0}'".format(resolution["param"]),
                        "is not found in the grid!",
                    )
                    paramguess = [
                        key for key in libitem.keys() if key.startswith("dnu")
                    ]
                    print("Did you perhaps mean one of these:", paramguess)
                    print("Please provide a valid name! I WILL ABORT NOW...\n")
                    fail = True
                    break

            if verbose:
                pltTeff = gu.h5py_to_array(libitem["Teff"])
                pltlogg = gu.h5py_to_array(libitem["logg"])
                pltbase = gu.h5py_to_array(libitem[baseparam])
                ax1.plot(pltTeff, pltlogg, color="darkgrey", alpha=0.2)
                ax3.plot(pltbase, pltTeff, color="darkgrey", alpha=0.2)

            #
            # *** BLOCK 1: Obtain reduced tracks ***
            #
            # Check which models have parameters within limits to define mask
            if os.path.join(gname, name) in selectedmodels:
                index = selectedmodels[os.path.join(gname, name)]

                # Make special 2D mask (for the frequency arrays). Necessary because h5py
                # does not support 1D bool array indexing for non 1D data!
                if intpol_freqs:
                    index2d = np.array(np.transpose([index, index]))

                #
                # *** BLOCK 2: Define interpolation mesh ***
                #
                # Calc number of points required and make uniform mesh
                if resolution["param"].lower() in freqres:
                    Npoints = _calc_npoints_freqs(
                        libitem=libitem,
                        index2d=index2d,
                        freq_resolution=resolution["value"],
                        verbose=verbose,
                        debug=debug,
                    )
                else:
                    Npoints = _calc_npoints(
                        libitem=libitem,
                        index=index,
                        resolution=resolution,
                        verbose=verbose,
                        debug=debug,
                    )
                if Npoints < sum(index):
                    print(
                        "Stopped interpolation along {0} as the number of points would decrease from {1} to {2}".format(
                            name, sum(index), Npoints
                        )
                    )
                    continue
                # Isochrones: mass | Tracks: age
                basevec = libitem[baseparam][index]
                intpolmesh = np.linspace(
                    start=basevec[0], stop=basevec[-1], num=Npoints
                )
                if debug:
                    print(
                        "{0}Range in {1} = [{2:4.3f}, {3:4.3f}]".format(
                            4 * " ", baseparam, basevec[0], basevec[-1]
                        )
                    )

                #
                # *** BLOCK 3: Interpolate in all quantities but frequencies ***
                #
                # Different cases...
                # --> No need to interpolate INTER-track-weights (which is a scalar)
                # --> INTRA-track weights (for the base parameter) is recalculated
                # --> Name of orignal gong files have no meaning anymore
                # --> Frequency arrays are tricky and are treated seperately
                tmpparam = {}
                for key in libitem.keys():
                    keypath = os.path.join(libitem.name, key)
                    if "_weight" in key:
                        newparam = libitem[key][()]
                    elif key == dname:
                        newparam = ih.bay_weights(intpolmesh)
                    elif "name" in key:
                        newparam = len(intpolmesh) * [b"interpolated-entry"]
                    elif "osc" in key:
                        continue
                    elif key in intpolparams:
                        newparam = ih._interpolation_wrapper(
                            basevec, libitem[key][index], intpolmesh
                        )
                    elif key in headvars:
                        newparam = np.ones(Npoints) * libitem[key][0]
                    else:
                        continue

                    # Delete old entry, write new entry
                    if overwrite:
                        del outfile[keypath]
                    outfile[keypath] = newparam

                    # Storage for plotting the Kiel diagram
                    if key in ["Teff", "logg", "age", "massfin"]:
                        tmpparam[key] = newparam

                #
                # *** BLOCK 4: Interpolate in frequencies ***
                #
                # No frequencies present in isochrones!
                if not isomode and intpol_freqs:
                    fullosc = libitem["osc"][index2d].reshape((-1, 2))
                    fullosckey = libitem["osckey"][index2d].reshape((-1, 2))
                    osckeylist, osclist = ih.interpolate_frequencies(
                        fullosc=fullosc,
                        fullosckey=fullosckey,
                        agevec=basevec,
                        newagevec=intpolmesh,
                        freqlims=freqlims,
                        verbose=verbose,
                        debug=debug,
                        trackid=noingrid + 1,
                    )

                    # Delete the old entries
                    if overwrite:
                        del outfile[os.path.join(libitem.name, "osc")]
                        del outfile[os.path.join(libitem.name, "osckey")]

                    # Writing variable length arrays to an HDF5 file is a bit tricky,
                    # but can be done using datasets with a special datatype.
                    # --> Here we follow the approach from BASTA/make_tracks
                    dsetosc = outfile.create_dataset(
                        name=os.path.join(libitem.name, "osc"),
                        shape=(Npoints, 2),
                        dtype=h5py.special_dtype(vlen=np.float),
                    )
                    dsetosckey = outfile.create_dataset(
                        name=os.path.join(libitem.name, "osckey"),
                        shape=(Npoints, 2),
                        dtype=h5py.special_dtype(vlen=np.int),
                    )
                    for i in range(Npoints):
                        dsetosc[i] = osclist[i]
                        dsetosckey[i] = osckeylist[i]

                trackcounter += 1

            #
            # *** BLOCK 1b: Handle tracks without any models inside selection ***
            #
            else:
                # Flag the empty tracks with a single entry: A negative weight
                # (note: requires at least BASTA v0.28 to be useful)
                if overwrite:
                    del outfile[os.path.join(libitem.name, "FeHini_weight")]
                outfile[os.path.join(libitem.name, "FeHini_weight")] = -1
                if verbose:
                    print()

            # Add information to the diagnostic plots
            if verbose and False:
                pltTeff = gu.h5py_to_array(libitem["Teff"])
                pltlogg = gu.h5py_to_array(libitem["logg"])
                pltbase = gu.h5py_to_array(libitem[baseparam])
                ax1.plot(
                    pltTeff[index],
                    pltlogg[index],
                    color="#482F76",
                    lw=4,
                    alpha=0.8,
                    zorder=2.5,
                )
                ax2.plot(
                    pltTeff[index],
                    pltlogg[index],
                    "x",
                    color="#482F76",
                    alpha=0.4,
                )
                ax3.plot(
                    pltbase[index],
                    pltTeff[index],
                    color="#482F76",
                    lw=4,
                    alpha=0.8,
                    zorder=2.5,
                )
                if any(index):
                    ax2.plot(
                        tmpparam["Teff"],
                        tmpparam["logg"],
                        "-",
                        color="#56B4E9",
                        alpha=0.5,
                    )
            #
            # *** STOP! HERE THE TRACK LOOP IS FINISHED!
            #

    pbar.close()

    # Re-add frequency limits for combined approaches
    if intpol_freqs:
        limits["freqs"] = freqlims

    # Finish debugging plot with some decoration
    if verbose:
        print("\nDone! Finishing diagnostic plots!")
        ax1.set_xlabel("Teff / K")
        ax1.set_ylabel("log g")
        ax1.invert_xaxis()
        ax1.invert_yaxis()
        fig1.savefig("intpol_diagnostic_kiel.pdf", bbox_inches="tight")

        ax2.set_xlabel("Teff / K")
        ax2.set_ylabel("log g")
        ax2.invert_xaxis()
        ax2.invert_yaxis()
        fig2.savefig("intpol_diagnostic_kiel-zoom.pdf", bbox_inches="tight")

        ax3.set_xlabel("Age / Myr" if baseparam == "age" else "Mass / Msun")
        ax3.set_ylabel("Teff / K")
        fig3.savefig("intpol_diagnostic_{0}.pdf".format(baseparam), bbox_inches="tight")

        print("\nIn total {0} {1}(s) interpolated!\n".format(trackcounter, modestr))
        print("Interpolation process finished!")

    return grid, outfile, fail
