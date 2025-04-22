"""
The main module of BASTA which functions as the main pipeline.

It handles the flow of input and output from the various modules internal in BASTA.
"""

import sys
import time
from typing import Any

# Import matplotlib after other plotting modules for proper setup
# --> Here in main it is only used for clean-up
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

from basta import core, plot_driver, priors, process_output, stats
from basta import fileio as fio
from basta import utils_general as util
from basta import utils_seismic as su
from basta.constants import freqtypes


def BASTA(
    star: core.Star,
    inferencesettings: core.InferenceSettings,
    filepaths: core.FilePaths,
    runfiles: core.RunFiles,
    outputoptions: core.OutputOptions,
    plotconfig: core.PlotConfig,
) -> None:
    """
    The BAyesian STellar Algorithm (BASTA).
    (c) 2025, The BASTA Team

    For a description of how to use BASTA, please explore the documentation (https://github.com/BASTAcode/BASTA).

    Parameters
    ----------
    star : core.Star
        A data class containing all star-specific inputs, see `core.py`.
    inferencesettings : core.InferenceSettings
        A data class containing all settings related to how the inference
        is done, see `core.py`.
    filepaths : core.FilePaths
        A data class containing all given file paths and methods for resolving and creating directories.
    outputoptions : core.OutputOptions
        A data class containing all options related to optional output
        from BASTA, e.g. additional plots or files, see `core.py`.
    plotconfig : core.PlotConfig
        A data class containing all options related to plots outputted
        from BASTA, , see `core.py`.
    """
    # Use try-finally to ensure that sys.stdout is reverted back
    # even when the run raises an exception.
    stdout = sys.stdout
    sys.stdout = util.Logger(filepaths.logfile)  # type: ignore
    try:
        _bastamain(
            star,
            inferencesettings,
            filepaths,
            runfiles,
            outputoptions,
            plotconfig,
        )
    finally:
        sys.stdout = stdout
        print(f"Saved log to {filepaths.logfile}")
        plt.close("all")


def _bastamain(
    star: core.Star,
    inferencesettings: core.InferenceSettings,
    filepaths: core.FilePaths,
    runfiles: core.RunFiles,
    outputoptions: core.OutputOptions,
    plotconfig: core.PlotConfig,
) -> None:
    #### INITIALISATION ####
    # Enable legacy printing of NumPy data types
    # --> E.g., print 104.14836386995329 instead of np.float64(104.14836386995329)
    #     and 'Teff' instead of np.str_('Teff') to the .log file
    # np.set_printoptions(legacy="1.25")

    # Start the log
    t0 = time.localtime()

    # Pretty printing a header
    util.print_bastaheader(
        t0=t0, seed=inferencesettings.seed, developermode=outputoptions.developermode
    )
    util.print_targetinformation(star)

    # Load the desired grid and obtain information from the header
    Grid, gridheader, gridinfo = util.get_grid(inferencesettings)
    bayweights, dweight = util.read_bayesianweights(
        Grid, gridinfo["entryname"], optional=not inferencesettings.usebayw
    )
    #### END INITIALISATION ####
    #### PREPARE DISTANCE FITTING AND FREQUENCY FITTING, IF REQUIRED ####

    # Get list of parameters
    cornerplots = plotconfig.cornerplots
    outparams = outputoptions.asciiparams
    allparams = list(np.unique(cornerplots + outparams))

    absolutemagnitudes, allparams = util.prepare_distancefitting(
        star=star,
        inferencesettings=inferencesettings,
        filepaths=filepaths,
        outputoptions=outputoptions,
        allparams=allparams,
    )

    # Create list of all available input parameters
    fitparams = star.fitparams
    fitfreqs = star.fitfreqs
    limits = inferencesettings.limits

    # Scale dnu and numax using a solar model or default solar values
    dnu_scales = su.solar_scaling(
        Grid, star=star, inferencesettings=inferencesettings, gridinfo=gridinfo
    )

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
        ) = su.prepare_obs(
            inputparams, verbose=outputoptions.verbose, debug=outputoptions.debug
        )

    #### END PREPARATION ####
    #### APPLY PRIORS ####
    if fitfreqs["active"]:
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
    if "tracks" in gridheader["gridtype"].lower():
        headerpath: str | bool = tracks_headerpath
    elif "isochrones" in gridheader["gridtype"].lower():
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
    util.print_distances(star, outputoptions)
    util.print_additional(star)
    util.print_weights(bayweights, gridheader["gridtype"])
    util.print_priors(inferencesettings)

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Start likelihood computation
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    # Two loop cases for the outer "metal" loop:
    # - For Garstec and MESA grids, the top level contains only one element ("tracks").
    #   Here the outer loop will run only once.
    # - For BaSTI, the top level is a list of metallicities and the outer loop will run
    #   multiple times.
    metal = util.list_metallicities(
        Grid, gridinfo=gridinfo, inferencesettings=inferencesettings
    )

    # We assume Garstec grid structure. The path will be updated in the loop for BaSTI
    group_name = gridinfo["defaultpath"] + "tracks/"

    # Before running the actual loop, all tracks/isochrones are counted to better
    # estimate the progress.
    trackcounter = 0
    for FeH in metal:
        if "grid" not in gridinfo["defaultpath"]:
            group_name = f"{gridinfo['defaultpath']}FeH={FeH:.4f}/"

        group = Grid[group_name]
        trackcounter += len(group.items())

    # Prepare the main loop
    shapewarn = 0
    warn = True
    selectedmodels: dict[str, stats.priorlogPDF | stats.Trackstats] = {}
    noofind = 0
    noofposind = 0
    # In some cases we need to store quantities computed at runtime
    if fitfreqs["active"] and fitfreqs["dnufit_in_ratios"]:
        dnusurfmodels = {}
    if fitfreqs["active"] and fitfreqs["glitchfit"]:
        glitchmodels = {}

    print(
        f"\n\nComputing likelihood of models in the grid ({trackcounter} {gridinfo['entryname']}) ..."
    )

    # Use a progress bar (with the package tqdm; will write to stderr)
    pbar = tqdm(total=trackcounter, desc="--> Progress", ascii=True)
    for FeH in metal:
        # TODO this can be dry'er, make list of group_names outside loop?
        if "grid" not in gridinfo["defaultpath"]:
            group_name = f"{gridinfo['defaultpath']}FeH={FeH:.4f}/"

        group = Grid[group_name]
        for noingrid, (name, libitem) in enumerate(group.items()):
            # Update progress bar in the start of the loop to count skipped tracks
            pbar.update(1)

            # For grid with interpolated tracks, skip tracks flagged as empty
            if gridheader["is_interpolated"]:
                if libitem["IntStatus"][()] < 0:
                    continue

            # Check for diffusion
            # TODO what
            # if "dif" in inputparams:
            #    if int(round(libitem["dif"][0])) != int(
            #        round(float(inputparams["dif"]))
            #    ):
            #        continue

            # TODO we must be able to optimise this
            # Check if mass or age is in limits to efficiently skip
            if "grid" not in gridinfo["defaultpath"]:
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
                    if "tracks" in gridheader["gridtype"].lower():
                        value = Grid[tracks_headerpath][param][noingrid]
                    elif "isochrones" in gridheader["gridtype"].lower():
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
            if "phase" in star.fitparams:
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
                    if param not in ["parallax", "distance"]:
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
                            debug=outputoptions.debug,
                            verbose=outputoptions.verbose,
                        )
                        chi2[indd] += chi2_freq

                        if fitfreqs["dnufit_in_ratios"]:
                            dnusurf[indd] = addpars["dnusurf"]
                        if fitfreqs["glitchfit"]:
                            glitchpar[indd] = addpars["glitchparams"]

                # Bayesian weights (across tracks/isochrones)
                logPDF = 0.0
                if outputoptions.debug:
                    bayw = 0.0
                    magw = 0.0
                    IMFw = 0.0
                if bayweights is not None:
                    for weight in bayweights:
                        logPDF += util.inflog(libitem[weight][()])
                        if outputoptions.debug:
                            bayw += util.inflog(libitem[weight][()])

                    # Within a given track/isochrone; these are called dweights
                    assert dweight is not None
                    logPDF += util.inflog(libitem[dweight][index])
                    if outputoptions.debug:
                        bayw += util.inflog(libitem[dweight][index])

                # Fold with absolute magnitudes, if present
                for f in absolutemagnitudes["magnitudes"].keys():
                    mags = absolutemagnitudes["magnitudes"][f]["prior"]
                    absmags = libitem[f][index]
                    interp_mags = mags(absmags)

                    logPDF += util.inflog(interp_mags)
                    if outputoptions.debug:
                        magw += util.inflog(interp_mags)

                # Multiply priors into the weight
                for prior in inferencesettings.priors or ():
                    logPDF += util.inflog(getattr(priors, prior)(libitem, index))
                    if outputoptions.debug:
                        IMFw += util.inflog(getattr(priors, prior)(libitem, index))

                # Calculate likelihood from weights, priors and chi2
                # PDF = weights * np.exp(-0.5 * chi2)
                logPDFarr = logPDF - 0.5 * chi2
                if outputoptions.debug and outputoptions.verbose:
                    print(
                        "DEBUG: Mass with nonzero likelihood:",
                        libitem["massini"][index][~np.isinf(logPDFarr)],
                    )

                # Sum the number indexes and nonzero indexes
                noofind += len(logPDFarr)
                noofposind += np.count_nonzero(~np.isinf(logPDFarr))
                if outputoptions.debug and outputoptions.verbose:
                    print(
                        f"DEBUG: Index found: {group_name + name}, {~np.isinf(logPDFarr)}"
                    )

                # Store statistical info
                if outputoptions.debug:
                    selectedmodels[group_name + name] = stats.priorlogPDF(
                        index, logPDFarr, chi2, bayw, magw, IMFw
                    )
                else:
                    selectedmodels[group_name + name] = stats.Trackstats(
                        index, logPDFarr, chi2
                    )
                if fitfreqs["active"] and fitfreqs["dnufit_in_ratios"]:
                    dnusurfmodels[group_name + name] = stats.Trackdnusurf(dnusurf)
                if fitfreqs["active"] and fitfreqs["glitchfit"]:
                    glitchmodels[group_name + name] = stats.Trackglitchpar(
                        glitchpar[:, 0],
                        glitchpar[:, 1],
                        glitchpar[:, 2],
                    )
                elif outputoptions.debug and outputoptions.verbose:
                    print(
                        f"DEBUG: Index not found: {group_name + name}, {~np.isinf(logPDFarr)}"
                    )
        # End loop over isochrones/tracks
        #######################################################################
    # End loop over metals
    ###########################################################################
    pbar.close()
    print(
        f"Done! Computed the likelihood of {noofind!s} models,",
        f"found {noofposind!s} models with non-zero likelihood!\n",
    )
    if gridcut:
        print(
            f"(Note: The use of 'gridcut' skipped {noofskips[0]} out of {noofskips[1]} {gridheader['gridtype']})\n"
        )

    # Raise possible warnings
    if shapewarn == 1:
        print(
            "Warning: Found models with fewer frequencies than observed!",
            "These were set to zero likelihood!",
        )
        if "intpol" in inferencesettings.gridfile:
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
        fio.no_models(
            star.starid,
            filepaths,
            runfiles,
            outputoptions,
            list(star.distanceparams.magnitudes.keys()),
            "No models found",
        )
        return

    # Print a header to signal the start of the output section in the log
    print("\n*****************************************")
    print("**                                     **")
    print("**   Output and results from the fit   **")
    print("**                                     **")
    print("*****************************************\n")

    # Find and print highest likelihood model info
    maxPDF_path, maxPDF_ind = stats.get_highest_likelihood(
        Grid,
        selectedmodels,
        dnu_scales=dnu_scales,
        inferencesettings=inferencesettings,
        outputoptions=outputoptions,
    )
    stats.get_lowest_chi2(
        Grid,
        selectedmodels,
        dnu_scales=dnu_scales,
        inferencesettings=inferencesettings,
        outputoptions=outputoptions,
    )

    # Generate posteriors of ascii- and plotparams
    # --> Print posteriors to console and log
    # --> Generate corner plots
    # --> Generate Kiel diagrams
    print("\n\nComputing posterior distributions for the requested output parameters!")
    print("==> Summary statistics printed below ...\n")
    process_output.compute_posterior(
        starid=star.starid,
        selectedmodels=selectedmodels,
        Grid=Grid,
        gridheader=gridheader,
        dnu_scales=dnu_scales,
        absolutemagnitudes=absolutemagnitudes,
        star=star,
        filepaths=filepaths,
        runfiles=runfiles,
        inferencesettings=inferencesettings,
        outputoptions=outputoptions,
        plotconfig=plotconfig,
    )

    # Collect additional output for plotting and saving
    addstats: dict[str, Any] = {}
    if fitfreqs["active"] and fitfreqs["dnufit_in_ratios"]:
        addstats["dnusurf"] = dnusurfmodels
    if fitfreqs["active"] and fitfreqs["glitchfit"]:
        addstats["glitchparams"] = glitchmodels

    # Make frequency-related plots
    if fitfreqs["active"] and len(plotconfig.freqplots):
        plot_driver.plot_all_seismic(
            plotconfig,
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
            plotfname=filepaths.plotfile_template,
            nameinplot=inputparams["nameinplot"],
            **addstats,
            debug=outputoptions.debug,
        )
    else:
        print(
            "Did not get any frequency file input, skipping ratios and echelle plots."
        )

    # Save dictionary with full statistics
    if outputoptions.optionaloutputs:
        fio.save_selectedmodels(filepaths.jsonfile, selectedmodels)
        print(f"Saved dictionary to {filepaths.jsonfile}")

    # Print time of completion
    t1 = time.localtime()
    print(
        f"\nFinished on {time.strftime('%Y-%m-%d %H:%M:%S', t1)}",
        f"(runtime {time.mktime(t1) - time.mktime(t0)} s).\n",
    )

    Grid.close()
