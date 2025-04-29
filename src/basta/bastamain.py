"""
Main module for running BASTA analysis
"""

import os
import gc
import sys
import time
from copy import deepcopy

import h5py
import numpy as np
from tqdm import tqdm

from basta import freq_fit, stats, process_output, priors, distances, plot_driver
from basta import utils_seismic as su
from basta import utils_general as util
from basta.__about__ import __version__
from basta import fileio as fio
from basta.constants import freqtypes

# Import matplotlib after other plotting modules for proper setup
# --> Here in main it is only used for clean-up
import matplotlib.pyplot as plt


# Custom exception
class LibraryError(Exception):
    pass


def BASTA(
    starid: str,
    gridfile: str,
    inputparams: dict,
    gridid: bool | tuple = False,
    usebayw: bool = True,
    usepriors: tuple = (None,),
    optionaloutputs: bool = False,
    seed: int | None = None,
    debug: bool = False,
    verbose: bool = False,
    developermode: bool = False,
    validationmode: bool = False,
):
    """
    The BAyesian STellar Algorithm (BASTA).
    (c) 2025, The BASTA Team

    For a description of how to use BASTA, please explore the documentation (https://github.com/BASTAcode/BASTA).
    This function is typically called by :func:'xmltools.run_xml()'

    Parameters
    ----------
    starid : str
        Unique identifier for this target.
    gridfile : str
        Path and name of the hdf5 file containing the isochrones or tracks
        used in the fitting
    inputparams : dict
        Dictionary containing most information needed, e.g. controls, fitparameters,
        output options.
    gridid : bool or tuple
        For isochrones, a tuple containing (overshooting [f],
        diffusion [0 or 1], mass loss [eta], alpha enhancement [0.0 ... 0.4])
        used for selecting a science case / path in the library.
    usebayw : bool or tuple
        If True, bayesian weights are applied in the computation of the
        likelihood. See :func:`interpolation_helpers.bay_weights()` for details.
    usepriors : tuple
        Tuple of strings containing name of priors (e.g., an IMF).
        See :func:`priors` for details.
    optionaloutputs : bool, optional
        If True, saves a 'json' file for each star with the global results and the PDF.
    seed : int, optional
        The seed of randomness
    debug : bool, optional
        Activate additional output for debugging (for developers)
    verbose : bool, optional
        Activate a lot (!) of additional output (for developers)
    developermode : bool, optional
        Activate experimental features (for developers)
    validationmode : bool, optional
        Activate validation mode features (for validation purposes only)
    """
    # Enable legacy printing of NumPy data types
    # --> E.g., print 104.14836386995329 instead of np.float64(104.14836386995329)
    #     and 'Teff' instead of np.str_('Teff') to the .log file
    np.set_printoptions(legacy="1.25")

    # Set output directory and filenames
    t0 = time.localtime()
    outputdir = inputparams.get("output")
    outfilename = os.path.join(outputdir, starid)

    # Start the log
    stdout = sys.stdout
    sys.stdout = util.Logger(outfilename)

    # Pretty printing a header
    util.print_bastaheader(t0=t0, seed=seed, developermode=developermode)

    # Load the desired grid and obtain information from the header
    Grid = h5py.File(gridfile, "r")
    gridtype, gridver, gridtime, grid_is_intpol = util.read_grid_header(Grid)

    # Verbose information on the grid file
    print(f"\nFitting star id: {starid} .")

    print(f"* Using the grid '{gridfile}' of type '{gridtype}'.")
    print(f"  - Grid built with BASTA version {gridver}, timestamp: {gridtime}.")

    entryname, defaultpath, difsolarmodel = util.check_gridtype(gridtype, gridid=gridid)

    # Read available weights if not provided by the user
    bayweights, dweight = (
        util.read_grid_bayweights(Grid, gridtype) if usebayw else (None, None)
    )

    # Get list of parameters
    cornerplots = inputparams["cornerplots"]
    outparams = inputparams["asciiparams"]
    allparams = list(np.unique(cornerplots + outparams))

    inputparams, allparams = util.prepare_distancefitting(
        inputparams=inputparams,
        debug=debug,
        debug_dirpath=outfilename,
        allparams=allparams,
    )

    # Create list of all available input parameters
    fitparams = inputparams.get("fitparams")
    fitfreqs = inputparams["fitfreqs"]
    distparams = inputparams.get("distanceparams", False)
    limits = inputparams.get("limits")

    # Scale dnu and numax using a solar model or default solar values
    inputparams = su.solar_scaling(Grid, inputparams, diffusion=difsolarmodel)

    # Prepare asteroseismic quantities if required
    if fitfreqs["active"]:
        if not all(x in freqtypes.alltypes for x in fitfreqs["fittypes"]):
            print(fitfreqs["fittypes"])
            raise ValueError("Unrecognized frequency fitting parameters!")

        # Obtain/calculate all frequency related quantities
        (
            obskey,
            obs,
            obsfreqdata,
            obsfreqmeta,
            obsintervals,
        ) = su.prepare_obs(inputparams, verbose=verbose, debug=debug)
        # Apply prior on dnufit to mimick the range defined by dnufrac
        if fitfreqs["dnuprior"] and ("dnufit" not in limits):
            dnufit_frac = fitfreqs["dnufrac"] * fitfreqs["dnufit"]
            dnuerr = max(3 * fitfreqs["dnufit_err"], dnufit_frac)
            limits["dnufit"] = [
                fitfreqs["dnufit"] - dnuerr,
                fitfreqs["dnufit"] + dnuerr,
            ]

    # Check if any specified limit in prior is in header, and can be used to
    # skip computation of models, in order to speed up computation
    tracks_headerpath = "header/"
    if "tracks" in gridtype.lower():
        headerpath: str | bool = tracks_headerpath
    elif "isochrones" in gridtype.lower():
        headerpath = tracks_headerpath + defaultpath
        if "FeHini" in limits:
            del limits["FeHini"]
            print("Warning: Dropping prior in FeHini, redundant for isochrones!")
    else:
        headerpath = False

    # Gridcut dictionary containing cutting parameters
    gridcut = {}
    if headerpath:
        keys = Grid[headerpath].keys()
        # Compare keys in header and limits
        for key in keys:
            if key in limits:
                gridcut[key] = limits[key]
                # Remove key from limits, to avoid redundant second check
                del limits[key]

    # Apply the cut on header parameters with a special treatment of diffusion
    if headerpath and gridcut:
        print("\nCutting in grid based on sampling parameters ('gridcut'):")
        noofskips = [0, 0]
        for cpar in gridcut:
            if cpar != "dif":
                print(f"* {cpar}: {gridcut[cpar]}")

        # Diffusion switch printed in a more readable format
        if "dif" in gridcut:
            # As gridcut['dif'] is always either [-inf, 0.5] or [0.5, inf]
            # The location of 0.5 can be used as the switch
            switch = np.where(np.array(gridcut["dif"]) == 0.5)[0][0]
            print(
                "* Only considering tracks with diffusion turned",
                "{:s}!".format(["on", "off"][switch]),
            )

    util.print_fitparams(fitparams=fitparams)
    if fitfreqs["active"]:
        util.print_seismic(fitfreqs=fitfreqs, obskey=obskey, obs=obs)
    util.print_distances(distparams, inputparams["asciiparams"])
    util.print_additional(inputparams)
    util.print_weights(bayweights, gridtype)
    util.print_priors(limits, usepriors)

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Start likelihood computation
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    # Two loop cases for the outer "metal" loop:
    # - For Garstec and MESA grids, the top level contains only one element ("tracks").
    #   Here the outer loop will run only once.
    # - For BaSTI, the top level is a list of metallicities and the outer loop will run
    #   multiple times.
    metal = util.list_metallicities(Grid, defaultpath, inputparams, limits)

    # We assume Garstec grid structure. The path will be updated in the loop for BaSTI
    group_name = defaultpath + "tracks/"

    # Before running the actual loop, all tracks/isochrones are counted to better
    # estimate the progress.
    trackcounter = 0
    for FeH in metal:
        if "grid" not in defaultpath:
            group_name = f"{defaultpath}FeH={FeH:.4f}/"
            assert group_name == defaultpath + "FeH=" + format(FeH, ".4f") + "/"

        group = Grid[group_name]
        trackcounter += len(group.items())

    # Prepare the main loop
    shapewarn = 0
    warn = True
    selectedmodels = {}
    noofind = 0
    noofposind = 0
    # In some cases we need to store quantities computed at runtime
    if fitfreqs["active"] and fitfreqs["dnufit_in_ratios"]:
        dnusurfmodels = {}
    if fitfreqs["active"] and fitfreqs["glitchfit"]:
        glitchmodels = {}

    print(
        f"\n\nComputing likelihood of models in the grid ({trackcounter} {entryname}) ..."
    )

    # Use a progress bar (with the package tqdm; will write to stderr)
    pbar = tqdm(total=trackcounter, desc="--> Progress", ascii=True)
    for FeH in metal:
        if "grid" not in defaultpath:
            group_name = f"{defaultpath}FeH={FeH:.4f}/"

        group = Grid[group_name]
        for noingrid, (name, libitem) in enumerate(group.items()):
            # Update progress bar in the start of the loop to count skipped tracks
            pbar.update(1)

            # For grid with interpolated tracks, skip tracks flagged as empty
            if grid_is_intpol:
                if libitem["IntStatus"][()] < 0:
                    continue

            # Check for diffusion
            if "dif" in inputparams:
                if int(round(libitem["dif"][0])) != int(
                    round(float(inputparams["dif"]))
                ):
                    continue

            # Check if mass or age is in limits to efficiently skip
            if "grid" not in defaultpath:
                param, val = name.split("=")
                if param == "mass":
                    param += "ini"
                if param in limits:
                    # if age or massini is outside limits, skip this iteration
                    if float(val) < limits[param][0] or float(val) > limits[param][1]:
                        continue

            # Check if track should be skipped from cut in initial parameters
            if gridcut:
                noofskips[1] += 1
                docut = False
                for param in gridcut:
                    if "tracks" in gridtype.lower():
                        value = Grid[tracks_headerpath][param][noingrid]
                    elif "isochrones" in gridtype.lower():
                        # For isochrones, metallicity is already cut from the
                        # metal list and lookup of age is simplest and fastest
                        if param == "age":
                            value = float(name[4:])
                    # If value is outside cut limits, skip looking at the rest
                    if not (value >= gridcut[param][0] and value <= gridcut[param][1]):
                        docut = True
                        continue
                # Actually skip this iteration
                if docut:
                    noofskips[0] += 1
                    continue

            # Check which models have parameters within limits
            index = np.ones(len(libitem["age"][:]), dtype=bool)
            for param in limits:
                index &= libitem[param][:] >= limits[param][0]
                index &= libitem[param][:] <= limits[param][1]

            # Check which models have phases as specified
            if "phase" in inputparams:
                # Mapping of verbose input phases to internal numbers
                pmap = {
                    "pre-ms": 1,
                    "solar": 2,
                    "rgb": 3,
                    "flash": 4,
                    "clump": 5,
                    "agb": 6,
                }

                # Fitting multiple phases or just one
                if isinstance(inputparams["phase"], tuple):
                    iphases = [pmap[ip] for ip in inputparams["phase"]]

                    phaseindex = libitem["phase"][:] == iphases[0]
                    for j in range(1, len(iphases)):
                        phaseindex |= libitem["phase"][:] == iphases[j]
                    index &= phaseindex
                else:
                    iphase = pmap[inputparams["phase"]]
                    index &= libitem["phase"][:] == iphase

            # Check which models have l=0, lowest n within tolerance
            if fitfreqs["active"]:
                indexf = np.zeros(len(index), dtype=bool)
                for ind in np.where(index)[0]:
                    rawmod = libitem["osc"][ind]
                    rawmodkey = libitem["osckey"][ind]
                    mod = su.transform_obj_array(rawmod)
                    modkey = su.transform_obj_array(rawmodkey)
                    modkeyl0, modl0 = su.get_givenl(l=0, osc=mod, osckey=modkey)
                    # As mod is ordered (stacked in increasing n and l),
                    # then [0, 0] is the lowest l=0 mode
                    same_n = modkeyl0[1, :] == obskey[1, 0]
                    cl0 = modl0[0, same_n]
                    if len(cl0) > 1:
                        cl0 = cl0[0]

                    # Note to self: This code is pretty hard to read...
                    if (
                        cl0
                        >= (
                            obs[0, 0]
                            - min(
                                (fitfreqs["dnufrac"] / 2 * fitfreqs["dnufit"]),
                                (3 * obs[1, 0]),
                            )
                        )
                    ) and (cl0 - obs[0, 0]) <= (
                        fitfreqs["dnufrac"] * fitfreqs["dnufit"]
                    ):
                        indexf[ind] = True
                index &= indexf

            # If any models are within tolerances, calculate statistics
            if np.any(index):
                chi2 = np.zeros(index.sum())
                paramvalues = {}
                for param in fitparams:
                    paramvals = libitem[param][index]
                    chi2 += (
                        (paramvals - fitparams[param][0]) / fitparams[param][1]
                    ) ** 2.0
                    if param in allparams:
                        paramvalues[param] = paramvals

                # Add parameters not in fitparams
                for param in allparams:
                    if param not in fitparams:
                        paramvalues[param] = libitem[param][index]

                # Frequency (and/or ratio and/or glitch) fitting
                if fitfreqs["active"]:
                    if fitfreqs["dnufit_in_ratios"]:
                        dnusurf = np.zeros(index.sum())
                    if fitfreqs["glitchfit"]:
                        glitchpar = np.zeros((index.sum(), 3))
                    for indd, ind in enumerate(np.where(index)[0]):
                        chi2_freq, warn, shapewarn, addpars = stats.chi2_astero(
                            obskey,
                            obs,
                            obsfreqmeta,
                            obsfreqdata,
                            obsintervals,
                            libitem,
                            ind,
                            fitfreqs,
                            warnings=warn,
                            shapewarn=shapewarn,
                            debug=debug,
                            verbose=verbose,
                        )
                        chi2[indd] += chi2_freq

                        if fitfreqs["dnufit_in_ratios"]:
                            dnusurf[indd] = addpars["dnusurf"]
                        if fitfreqs["glitchfit"]:
                            glitchpar[indd] = addpars["glitchparams"]

                # Bayesian weights (across tracks/isochrones)
                logPDF = 0.0
                if debug:
                    bayw = 0.0
                    magw = 0.0
                    IMFw = 0.0
                if bayweights is not None:
                    for weight in bayweights:
                        logPDF += util.inflog(libitem[weight][()])
                        if debug:
                            bayw += util.inflog(libitem[weight][()])

                    # Within a given track/isochrone; these are called dweights
                    assert dweight is not None
                    logPDF += util.inflog(libitem[dweight][index])
                    if debug:
                        bayw += util.inflog(libitem[dweight][index])

                # Multiply by absolute magnitudes, if present
                for f in inputparams["magnitudes"]:
                    mags = inputparams["magnitudes"][f]["prior"]
                    absmags = libitem[f][index]
                    interp_mags = mags(absmags)

                    logPDF += util.inflog(interp_mags)
                    if debug:
                        magw += util.inflog(interp_mags)

                # Multiply priors into the weight
                for prior in usepriors:
                    logPDF += util.inflog(getattr(priors, prior)(libitem, index))
                    if debug:
                        IMFw += util.inflog(getattr(priors, prior)(libitem, index))

                # Calculate likelihood from weights, priors and chi2
                # PDF = weights * np.exp(-0.5 * chi2)
                logPDF -= 0.5 * chi2
                if debug and verbose:
                    print(
                        "DEBUG: Mass with nonzero likelihood:",
                        libitem["massini"][index][~np.isinf(logPDF)],
                    )

                # Sum the number indexes and nonzero indexes
                noofind += len(logPDF)
                noofposind += np.count_nonzero(~np.isinf(logPDF))
                if debug and verbose:
                    print(
                        f"DEBUG: Index found: {group_name + name}, {~np.isinf(logPDF)}"
                    )

                # Store statistical info
                if debug:
                    selectedmodels[group_name + name] = stats.priorlogPDF(
                        index, logPDF, chi2, bayw, magw, IMFw
                    )
                else:
                    selectedmodels[group_name + name] = stats.Trackstats(
                        index, logPDF, chi2
                    )
                if fitfreqs["active"] and fitfreqs["dnufit_in_ratios"]:
                    dnusurfmodels[group_name + name] = stats.Trackdnusurf(dnusurf)
                if fitfreqs["active"] and fitfreqs["glitchfit"]:
                    glitchmodels[group_name + name] = stats.Trackglitchpar(
                        glitchpar[:, 0],
                        glitchpar[:, 1],
                        glitchpar[:, 2],
                    )
            else:
                if debug and verbose:
                    print(
                        f"DEBUG: Index not found: {group_name + name}, {~np.isinf(logPDF)}"
                    )
        # End loop over isochrones/tracks
        #######################################################################
    # End loop over metals
    ###########################################################################
    pbar.close()
    print(
        f"Done! Computed the likelihood of {str(noofind)} models,",
        f"found {str(noofposind)} models with non-zero likelihood!\n",
    )
    if gridcut:
        print(
            f"(Note: The use of 'gridcut' skipped {noofskips[0]} out of {noofskips[1]} {gridtype})\n"
        )

    # Raise possible warnings
    if shapewarn == 1:
        print(
            "Warning: Found models with fewer frequencies than observed!",
            "These were set to zero likelihood!",
        )
        if "intpol" in gridfile:
            print(
                "This is probably due to the interpolation scheme. Lookup",
                "`interpolate_frequencies` for more details.",
            )
    if shapewarn == 2:
        print(
            "Warning: Models without frequencies overlapping with observed",
            "ignored due to interpolation of ratios being impossible.",
        )
    if shapewarn == 3:
        print(
            "Warning: Models ignored due to phase shift differences being",
            "unapplicable to models with mixed modes.",
        )
    if noofposind == 0:
        fio.no_models(starid, inputparams, "No models found")
        return

    # Print a header to signal the start of the output section in the log
    print("\n*****************************************")
    print("**                                     **")
    print("**   Output and results from the fit   **")
    print("**                                     **")
    print("*****************************************\n")

    # Find and print highest likelihood model info
    maxPDF_path, maxPDF_ind = stats.get_highest_likelihood(
        Grid, selectedmodels, inputparams
    )
    stats.get_lowest_chi2(Grid, selectedmodels, inputparams)

    # Generate posteriors of ascii- and plotparams
    # --> Print posteriors to console and log
    # --> Generate corner plots
    # --> Generate Kiel diagrams
    print("\n\nComputing posterior distributions for the requested output parameters!")
    print("==> Summary statistics printed below ...\n")
    process_output.compute_posterior(
        starid=starid,
        selectedmodels=selectedmodels,
        Grid=Grid,
        inputparams=inputparams,
        outfilename=outfilename,
        gridtype=gridtype,
        debug=debug,
        developermode=developermode,
        validationmode=validationmode,
    )

    # Collect additional output for plotting and saving
    addstats = {}
    if fitfreqs["active"] and fitfreqs["dnufit_in_ratios"]:
        addstats["dnusurf"] = dnusurfmodels
    if fitfreqs["active"] and fitfreqs["glitchfit"]:
        addstats["glitchparams"] = glitchmodels

    # Make frequency-related plots
    freqplots = inputparams.get("freqplots")
    if fitfreqs["active"] and len(freqplots):
        plot_driver.plot_all_seismic(
            freqplots,
            Grid=Grid,
            fitfreqs=fitfreqs,
            obsfreqmeta=obsfreqmeta,
            obsfreqdata=obsfreqdata,
            obskey=obskey,
            obs=obs,
            obsintervals=obsintervals,
            selectedmodels=selectedmodels,
            path=maxPDF_path,
            ind=maxPDF_ind,
            plotfname=outfilename + "_{0}." + inputparams["plotfmt"],
            nameinplot=inputparams["nameinplot"],
            **addstats,
            debug=debug,
        )
    else:
        print(
            "Did not get any frequency file input, skipping ratios and echelle plots."
        )

    # Save dictionary with full statistics
    if optionaloutputs:
        pfname = outfilename + ".json"
        fio.save_selectedmodels(pfname, selectedmodels)
        print(f"Saved dictionary to {pfname}")

    # Print time of completion
    t1 = time.localtime()
    print(
        f"\nFinished on {time.strftime('%Y-%m-%d %H:%M:%S', t1)}",
        f"(runtime {time.mktime(t1) - time.mktime(t0)} s).\n",
    )

    # Save log and recover standard output
    sys.stdout = stdout
    print(f"Saved log to {outfilename}.log")

    # Close grid, close open plots, and try to free memory between multiple runs
    Grid.close()
    plt.close("all")
    gc.collect()
