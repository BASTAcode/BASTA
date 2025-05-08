"""
The main module of BASTA which functions as the main pipeline.

It handles the flow of input and output from the various modules internal in BASTA.
"""

import sys
import time
from typing import Any

import numpy as np
from tqdm import tqdm

from basta import (
    core,
    constants,
    distances,
    plot_driver,
    imfs,
    priors,
    process_output,
    stats,
)
from basta import fileio as fio
from basta import utils_general as util
from basta import utils_seismic as su
from basta.constants import freqtypes


def BASTA(
    star: core.InputStar,
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
    # Import matplotlib after other plotting modules for proper setup
    # --> Here in main it is only used for clean-up
    import matplotlib.pyplot as plt

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
    inputstar: core.InputStar,
    inferencesettings: core.InferenceSettings,
    filepaths: core.FilePaths,
    runfiles: core.RunFiles,
    outputoptions: core.OutputOptions,
    plotconfig: core.PlotConfig,
) -> None:
    #### INITIALISATION OF BASTA RUN ####
    t0 = time.localtime()
    util.print_bastaheader(
        t0=t0, seed=inferencesettings.seed, developermode=outputoptions.developermode
    )
    util.print_targetinformation(inputstar.starid)

    ## Load the grid
    Grid, gridheader, gridinfo = util.get_grid(inferencesettings)
    bayweights, dweight = util.read_bayesianweights(
        Grid, gridinfo["entryname"], optional=not inferencesettings.usebayw
    )
    priors.gridlimits(
        grid=Grid,
        gridheader=gridheader,
        gridinfo=gridinfo,
        inferencesettings=inferencesettings,
        outputoptions=outputoptions,
    )

    # Apply solar scaling
    su.solar_scaling(
        Grid,
        globalseismicparams=inputstar.globalseismicparams,
        inferencesettings=inferencesettings,
        gridinfo=gridinfo,
        outputoptions=outputoptions,
    )

    ## Prepare star
    star = util.setup_star(
        inputstar=inputstar,
        inferencesettings=inferencesettings,
        filepaths=filepaths,
        outputoptions=outputoptions,
        plotconfig=plotconfig,
    )

    util.print_fitparams(star=star, inferencesettings=inferencesettings)
    util.print_seismic(
        inferencesettings, inputstar=inputstar
    )  # , obskey=obskey, obs=obs)
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
    metallicities = util.list_metallicities(
        Grid, gridinfo=gridinfo, inferencesettings=inferencesettings
    )

    if star.phase is not None:
        if isinstance(star.phase, tuple):
            iphases = [constants.phasemap.pmap[ip] for ip in star.phase]
        else:
            iphases = [constants.phasemap.pmap[star.phase]]

    group_names = util.compute_group_names(
        gridinfo=gridinfo, metallicities=metallicities
    )

    # Before running the actual loop, all tracks/isochrones are counted to better
    # estimate the progress.
    trackcounter = sum(len(Grid[group_names[feh]].items()) for feh in metallicities)

    # Prepare the main loop
    shapewarn = 0
    warn = True
    selectedmodels: dict[str, stats.priorlogPDF | stats.Trackstats] = {}
    noofind = 0
    noofposind = 0
    noofskips = [0, 0]

    # In some cases we need to store quantities computed at runtime
    # TODO(Amalie) Why do we need this? Is this the right logic?
    if inferencesettings.fit_surfacecorrected_dnu:
        dnusurfmodels = {}
    if inferencesettings.has_glitches:
        glitchmodels = {}

    print(
        f"\n\nComputing likelihood of models in the grid ({trackcounter} {gridinfo['entryname']}) ..."
    )
    if inferencesettings.has_any_seismic_case:
        obskey = np.asarray([star.frequencies.l, star.frequencies.n])
        obs = np.asarray([star.frequencies.frequencies, star.frequencies.errors])

    # Use a progress bar (with the package tqdm; will write to stderr)
    pbar = tqdm(total=trackcounter, desc="--> Progress", ascii=True)
    # TODO(Amalie) can we avoid allparams?
    # allparams = np.unique(np.asarray(inferencesettings.fitparams + plotconfig.freqplots + star.distanceparams.))
    allparams = ["Teff", "dnuSer", "numax", "MeH"]

    for FeH in metallicities:
        group_name = group_names[FeH]
        group = Grid[group_name]

        for noingrid, (name, libitem) in enumerate(group.items()):
            # Update progress bar in the start of the loop to count skipped tracks
            pbar.update(1)

            # Check entire track
            # For grid with interpolated tracks, skip tracks flagged as empty
            if gridheader["is_interpolated"]:
                if libitem["IntStatus"][()] < 0:
                    continue

            # TODO(Amalie) This would be easier to read if it was a prior? function? handled earlier?
            if any(
                np.isin(["dif", "diffusion"], list(star.classicalparams.params.keys()))
            ):
                if int(round(libitem["dif"][0])) != int(
                    round(float(star.classicalparams.params["dif"][0]))
                ):
                    continue

            """
            if "gridcut" in inferencesettings.priors.keys():
                noofskips[1] += 1
                docut = False
                for param in inferencesettings.priors['gridcut'].kwargs.keys():
                    if param == "gridcut":
                        continue
                    if "tracks" in gridheader["gridtype"].lower():
                        value = Grid[tracks_headerpath][param][noingrid]
                    elif "isochrones" in gridheader["gridtype"].lower():
                        # For isochrones, metallicity is already cut from the
                        # metal list and lookup of age is simplest and fastest
                        if param == "age":
                            value = float(name[4:])
                    # If value is outside cut limits, skip looking at the rest
                    if not (value >= inferencesettings.priors['gridcut'].kwargs[param]["limits"][0] and value <= inferencesettings.priors['gridcut'].kwargs[param]["limits"][1]):
                        docut = True
                        continue
                # Actually skip this iteration
                if docut:
                    noofskips[0] += 1
                    continue
            """
            index = np.ones(len(libitem["age"][:]), dtype=bool)

            for param in star.limits:
                index &= libitem[param][:] >= star.limits[param][0]
                index &= libitem[param][:] <= star.limits[param][1]

            if star.phase is not None:
                phaseindex = np.isin(libitem["phase"][:], iphases)
                index &= phaseindex

            # Check which models have l=0, lowest n within tolerance
            if inferencesettings.has_any_seismic_case:
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
                    if cl0.size == 0:
                        continue
                    elif cl0.size > 1:
                        cl0 = cl0[0]

                    cl0 = cl0.item()
                    anchordist = cl0 - obs[0, 0]
                    dnutype = "dnufit"
                    lower_threshold = min(
                        inferencesettings.boxpriors["dnufrac"].kwargs[dnutype]
                        / 2
                        * star.globalseismicparams.get_scaled(dnutype)[0],
                        3 * obs[1, 0],
                    )
                    upper_threshold = (
                        inferencesettings.boxpriors["dnufrac"].kwargs[dnutype]
                        * star.globalseismicparams.get_scaled(dnutype)[0]
                    )

                    # TODO(Amalie) This seems too restrictive as a default!
                    if lower_threshold < anchordist <= upper_threshold:
                        indexf[ind] = True
                index &= indexf

            # TODO(Amalie) rewrite this to a function
            # If any models are within tolerances, calculate statistics
            if np.any(index):
                chi2 = np.zeros(index.sum())
                # paramvalues = {}
                for param in star.classicalparams.params.keys():
                    if param not in ["parallax", "distance"]:
                        paramvals = libitem[param][index]
                        chi2 += (
                            (paramvals - star.classicalparams.params[param][0])
                            / star.classicalparams.params[param][1]
                        ) ** 2.0
                        # if param in set(inferencesettings.fitparams) | set(
                        #    plotconfig.cornerplots
                        # ):
                        #    paramvalues[param] = paramvals

                # Add parameters not in fitparams
                # for param in list(allparams):
                #    if param not in list(star.classicalparams.params.keys()):
                #        paramvalues[param] = libitem[param][index]

                # Frequency (and/or ratio and/or glitch) fitting
                if inferencesettings.has_any_seismic_case:
                    if inferencesettings.fit_surfacecorrected_dnu:
                        dnusurf = np.zeros(index.sum())
                    if inferencesettings.glitchfit:
                        glitchpar = np.zeros((index.sum(), 3))
                    for indd, ind in enumerate(np.where(index)[0]):
                        chi2_freq, warn, shapewarn, addpars = stats.chi2_astero(
                            libitem,
                            ind,
                            star,
                            inferencesettings,
                            # obskey,
                            # obs,
                            # obsfreqmeta,
                            # obsfreqdata,
                            # obsintervals,
                            # libitem,
                            # ind,
                            # fitfreqs,
                            warnings=warn,
                            shapewarn=shapewarn,
                            debug=outputoptions.debug,
                            verbose=outputoptions.verbose,
                        )
                        chi2[indd] += chi2_freq

                        if inferencesettings.fit_surfacecorrected_dnu:
                            dnusurf[indd] = addpars["dnusurf"]
                        if inferencesettings.glitchfit:
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
                if inferencesettings.has_distance_case:
                    for f in star.absolutemagnitudes["magnitudes"].keys():
                        mags = star.absolutemagnitudes["magnitudes"][f]["prior"]
                        absmags = libitem[f][index]
                        interp_mags = mags(absmags)

                        logPDF += util.inflog(interp_mags)
                        if outputoptions.debug:
                            magw += util.inflog(interp_mags)

                # Multiply priors into the weight
                if inferencesettings.imf is not None:
                    if inferencesettings.imf not in imfs.PRIOR_FUNCTIONS:
                        raise ValueError(f"Unknown IMF: {inferencesettings.imf}")
                    val = imfs.PRIOR_FUNCTIONS[inferencesettings.imf](libitem, index)
                    logPDF += util.inflog(val)
                    if outputoptions.debug:
                        IMFw += util.inflog(val)

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
                if inferencesettings.fit_surfacecorrected_dnu:
                    dnusurfmodels[group_name + name] = stats.Trackdnusurf(dnusurf)
                if inferencesettings.has_glitches:
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
    # TODO(Amalie) Rewrite
    """
    if gridcut:
        print(
            f"(Note: The use of 'gridcut' skipped {noofskips[0]} out of {noofskips[1]} {gridheader['gridtype']})\n"
        )
    """

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

    util._banner("Output and results from the fit")

    # Find and print highest likelihood model info
    maxPDF_path, maxPDF_ind = stats.get_highest_likelihood(
        Grid,
        selectedmodels,
        star=star,
        inferencesettings=inferencesettings,
        outputoptions=outputoptions,
    )
    stats.get_lowest_chi2(
        Grid,
        selectedmodels,
        star=star,
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
        absolutemagnitudes=star.absolutemagnitudes,
        star=star,
        filepaths=filepaths,
        runfiles=runfiles,
        inferencesettings=inferencesettings,
        outputoptions=outputoptions,
        plotconfig=plotconfig,
    )

    # Collect additional output for plotting and saving
    addstats: dict[str, Any] = {}
    if inferencesettings.fit_surfacecorrected_dnu:
        addstats["dnusurf"] = dnusurfmodels
    if inferencesettings.glitchfit:
        addstats["glitchparams"] = glitchmodels

    # Make frequency-related plots
    if plotconfig.freqplots and inferencesettings.has_any_seismic_case:
        plot_driver.plot_all_seismic(
            plotconfig,
            Grid=Grid,
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
