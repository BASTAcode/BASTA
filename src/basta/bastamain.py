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
    selectedmodels: dict[str, stats.priorlogPDF | stats.Trackstats] = {}
    shapewarn = 0
    noofind = 0
    noofposind = 0

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

    for FeH in metallicities:
        group_name = group_names[FeH]
        group = Grid[group_name]

        for noingrid, (name, libitem) in enumerate(group.items()):
            # Update progress bar in the start of the loop to count skipped tracks
            pbar.update(1)

            index = stats.evaluate_track(
                libitem=libitem,
                gridheader=gridheader,
                star=star,
                inferencesettings=inferencesettings,
                iphases=iphases,
            )
            if np.sum(index) < 1:
                continue

            # Compute the log likelihood contributions of most stellar observables
            if np.any(index):
                log_likelihood, chi2, shapewarn = stats.compute_log_likelihood(
                    libitem,
                    index=index,
                    star=star,
                    inferencesettings=inferencesettings,
                    outputoptions=outputoptions,
                )

                # Bayesian weights (across tracks/isochrones)
                number_of_possible_models = np.sum(index)
                log_prior = np.zeros(number_of_possible_models)
                if outputoptions.debug:
                    bayw = np.zeros(number_of_possible_models)
                    IMFw = np.zeros(number_of_possible_models)

                if bayweights is not None:
                    for weight in bayweights:
                        log_prior += util.inflog(libitem[weight][()])
                        if outputoptions.debug:
                            bayw += util.inflog(libitem[weight][()])

                    # Within a given track/isochrone; these are called dweights
                    assert dweight is not None
                    log_prior += util.inflog(libitem[dweight][index])
                    if outputoptions.debug:
                        bayw += util.inflog(libitem[dweight][index])

                # IMF prior
                imf_prior = stats.evaluate_imf(
                    libitem, index=index, inferencesettings=inferencesettings
                )
                log_prior += imf_prior
                if outputoptions.debug:
                    IMFw += imf_prior

                # Calculate likelihood from weights, priors and chi2
                # PDF = weights * np.exp(-0.5 * chi2)
                posterior = log_prior + log_likelihood  # - 0.5 * chi2

                # Sum the number indexes and nonzero indexes
                noofind += len(posterior)
                noofposind += np.count_nonzero(~np.isinf(posterior))

                # Store statistical info
                if outputoptions.debug:
                    selectedmodels[group_name + name] = stats.priorlogPDF(
                        index, posterior, chi2, bayw, IMFw
                    )
                else:
                    selectedmodels[group_name + name] = stats.Trackstats(
                        index, posterior, chi2
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
    if shapewarn > 0:
        remtor.raise_shapewarning(
            shapewarn=shapewarn, inferencesettings=inferencesettings
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
    _, _ = stats.get_lowest_chi2(
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
            ind=int(maxPDF_ind),
            quantities_at_runtime=(
                quantities_at_runtime if inferencesettings.has_glitches else None
            ),
        )
    else:
        print(
            "Did not get any frequency file input, skipping ratios and echelle plots."
        )

    # TODO(Amalie) Write quantities_computed_at_runtime to json file
    if inferencesettings.fit_surfacecorrected_dnu or inferencesettings.has_glitches:
        pass

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
