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
    process_output,
    remtor,
    stats,
    utils_priors,
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
    sys.stdout = remtor.Logger(filepaths.logfile)  # type: ignore
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
    remtor.print_bastaheader(
        t0=t0, seed=inferencesettings.seed, developermode=outputoptions.developermode
    )
    remtor.print_targetinformation(inputstar.starid)

    ## Load the grid
    Grid, gridheader, gridinfo = util.get_grid(inferencesettings)
    remtor.print_gridinfo(inferencesettings=inferencesettings, header=gridheader)
    bayweights, dweight = util.read_bayesianweights(
        Grid, gridinfo["entryname"], optional=not inferencesettings.usebayw
    )
    util.gridlimits(
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

    remtor.print_fitparams(star=star, inferencesettings=inferencesettings)
    remtor.print_seismic(inferencesettings, inputstar=inputstar)
    remtor.print_distances(star, outputoptions)
    remtor.print_additional(star)
    remtor.print_weights(bayweights, gridheader["gridtype"])
    remtor.print_priors(inferencesettings)

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
    iphases = util.list_phases(star)

    group_names = util.compute_group_names(
        gridinfo=gridinfo, metallicities=metallicities
    )

    # Prepare the main loop
    shapewarn = 0
    warn = True
    selectedmodels: dict[str, stats.priorlogPDF | stats.Trackstats] = {}
    noofind = 0
    noofposind = 0
    noofskips = [0, 0]

    # In some cases we need to store quantities computed at runtime
    quantities_at_runtime: dict[str, Any] = {}
    if inferencesettings.has_glitches:
        quantities_at_runtime["glitches"] = {}
    if inferencesettings.fit_surfacecorrected_dnu:
        quantities_at_runtime["surfacecorrected_dnu"] = {}

    # Before running the actual loop, all tracks/isochrones are counted to better
    # estimate the progress.
    trackcounter = sum(len(Grid[group_names[feh]].items()) for feh in metallicities)
    print(
        f"\n\nComputing likelihood of models in the grid ({trackcounter} {gridinfo['entryname']}) ..."
    )
    # Use a progress bar (with the package tqdm; will write to stderr)
    pbar = tqdm(total=trackcounter, desc="--> Progress", ascii=True)

    if inferencesettings.has_any_seismic_case:
        assert star.modes is not None
        obskey = np.asarray([star.modes.modes.l, star.modes.modes.n])
        obs = np.asarray([star.modes.modes.frequencies, star.modes.modes.errors])

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

            if util.should_skip_due_to_diffusion(libitem=libitem, star=star):
                continue

            index = np.ones(len(libitem["age"][:]), dtype=bool)

            for param in star.limits:
                if param in ["frequencies"]:
                    continue
                index &= star.limits[param][0] <= libitem[param][:]
                index &= libitem[param][:] <= star.limits[param][1]
            if np.sum(index) < 1:
                continue

            if star.phase is not None:
                phaseindex = np.isin(libitem["phase"][:], iphases)
                index &= phaseindex
            if np.sum(index) < 1:
                continue

            index &= util.apply_anchor_cut(
                index, star=star, libitem=libitem, inferencesettings=inferencesettings
            )

            # TODO(Amalie) rewrite this to a function
            # If any models are within tolerances, calculate statistics
            if np.any(index):
                chi2 = np.zeros(index.sum())
                for param in star.classicalparams.params.keys():
                    if param not in ["parallax", "distance"]:
                        paramvals = libitem[param][index]
                        chi2 += (
                            (paramvals - star.classicalparams.params[param][0])
                            / star.classicalparams.params[param][1]
                        ) ** 2.0

                # Frequency (and/or ratio and/or glitch) fitting
                if inferencesettings.has_any_seismic_case:
                    inds: list = []
                    ahe: list = []
                    dhe: list = []
                    tauhe: list = []
                    surfacecorrected_dnu: list = []
                    for indd, ind in enumerate(np.where(index)[0]):
                        chi2_freq, warn, shapewarn, addpars = stats.chi2_astero(
                            libitem,
                            ind,
                            star,
                            inferencesettings,
                            outputoptions,
                            warnings=warn,
                            shapewarn=shapewarn,
                            debug=outputoptions.debug,
                            verbose=outputoptions.verbose,
                        )
                        chi2[indd] += chi2_freq
                        if inferencesettings.has_glitches:
                            inds.append(indd)
                            ahe.append(addpars["glitchparams"][:, 0])
                            dhe.append(addpars["glitchparams"][:, 1])
                            tauhe.append(addpars["glitchparams"][:, 2])
                        if inferencesettings.fit_surfacecorrected_dnu:
                            surfacecorrected_dnu.append(addpars["surfacecorrected_dnu"])

                    if inferencesettings.has_glitches:
                        quantities_at_runtime["glitches"][group_name + name] = {
                            "idx": inds,
                            "aHe": addpars["glitchparams"][:, 0],
                            "dHe": addpars["glitchparams"][:, 1],
                            "tauHe": addpars["glitchparams"][:, 2],
                        }
                    if inferencesettings.fit_surfacecorrected_dnu:
                        quantities_at_runtime["surfacecorrected_dnu"][
                            group_name + name
                        ] = {
                            "idx": inds,
                            "surfacecorrected_dnu": addpars["surfacecorrected_dnu"],
                        }

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
                    assert star.absolutemagnitudes is not None
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
        # End loop over isochrones/tracks
        #######################################################################
    # End loop over metals
    ###########################################################################
    pbar.close()
    print(
        f"Done! Computed the likelihood of {noofind!s} models,",
        f"found {noofposind!s} models with non-zero likelihood!\n",
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

    remtor._banner("Output and results from the fit")

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
        star=star,
        filepaths=filepaths,
        runfiles=runfiles,
        inferencesettings=inferencesettings,
        outputoptions=outputoptions,
        plotconfig=plotconfig,
    )

    # Make frequency-related plots
    if plotconfig.freqplots and inferencesettings.has_any_seismic_case:
        plot_driver.plot_all_seismic(
            inputstar=inputstar,
            star=star,
            inferencesettings=inferencesettings,
            outputoptions=outputoptions,
            plotconfig=plotconfig,
            filepaths=filepaths,
            Grid=Grid,
            selectedmodels=selectedmodels,
            path=maxPDF_path,
            ind=maxPDF_ind,
            quantities_at_runtime=(
                quantities_at_runtime if inferencesettings.has_glitches else None
            ),
        )
    else:
        print(
            "Did not get any frequency file input, skipping ratios and echelle plots."
        )

    # TODO(Amalie) Write quantities_computed_at_runtime to json file

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
