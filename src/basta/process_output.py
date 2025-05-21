"""
Calculation and generation of output, and driver for producing plots
"""

import os
from copy import deepcopy

import matplotlib as mpl
import numpy as np

import basta.fileio as fio
from basta import core, plot_corner, plot_kiel, remtor, stats
from basta import utils_general as util
from basta.constants import parameters, statdata
from basta.distances import get_absorption, get_EBV_along_LOS
from basta.downloader import get_basta_dir
from basta.utils_distances import compute_distance_from_mag

# Change matplotlib backend before loading pyplot
mpl.use("Agg")
import matplotlib.pyplot as plt

# Set the style of all plots
plt.style.use(os.path.join(get_basta_dir(), "plots.mplstyle"))


def generate_sampled_indices(
    selectedmodels,
) -> tuple[np.ndarray, np.ndarray, int, np.ndarray, int, np.ndarray]:
    logy = np.concatenate([ts.logPDF for ts in selectedmodels.values()])
    noofind = len(logy)

    nonzeroprop = np.isfinite(logy)
    logy = logy[nonzeroprop]

    nsamples = min(statdata.nsamples, len(logy))

    # Normalize logy for numerical stability
    logy_max = np.amax(logy)
    lk = logy - logy_max
    exp_lk = np.exp(lk)
    p = exp_lk / np.sum(exp_lk)

    sampled_indices = np.random.choice(np.arange(len(p)), p=p, size=nsamples)

    return p, logy, noofind, nonzeroprop, nsamples, sampled_indices


def compute_posterior(
    starid,
    selectedmodels,
    Grid,
    gridheader: util.GridHeader,
    star: core.Star,
    filepaths: core.FilePaths,
    runfiles: core.RunFiles,
    inferencesettings: core.InferenceSettings,
    outputoptions: core.OutputOptions,
    plotconfig: core.PlotConfig,
    compareinputoutput=False,
) -> None:
    """
    This function computes the posterior distributions and produce plots.

    Parameters
    ----------
    starid : str
        Unique identifier of current target.
    selectedmodels : dict
        Contains information on all models with a non-zero likelihood.
    Grid : hdf5 object
        The already loaded grid, containing the tracks/isochrones.
    inputparams : dict
        Dict containing input from xml-file.
    gridtype : str
        Type of the grid (as read from the grid in bastamain) containing either 'tracks'
        or 'isochrones'.
    developermode : bool, optional
        If True, experimental features will be used in run.
    validationmode : bool, optional
        If True, assume a validation run with changed behaviour
    """
    # Extract relevant configuration and settings
    asciifile = runfiles.summarytable
    asciifile_dist = runfiles.distancesummarytable

    ckwargs = plotconfig.corner_style
    cornerplots = deepcopy(plotconfig.cornerplots)

    kielplots = plotconfig.kielplots

    params = util.unique_unsort(outputoptions.asciiparams + cornerplots)
    fitparams = (
        star.classicalparams.params
        | star.globalseismicparams.params
        | star.distanceparams.params
    )

    # Initialize necessary arrays for results
    hout = ["starid"]
    out = [star.starid]
    hout_dist = ["starid"]
    out_dist = [star.starid]

    # Generate log-likelihood values and sampled indices
    p, logy, noofind, nonzeroprop, nsamples, sampled_indices = generate_sampled_indices(
        selectedmodels
    )

    if outputoptions.debug:
        cs = np.concatenate([ts.chi2 for ts in selectedmodels.values()])
        ws = np.exp(logy + 0.5 * cs[nonzeroprop])
        ws /= np.sum(ws)
        expcs = np.exp(-0.5 * cs[nonzeroprop])
        expcs /= np.sum(expcs)
        lsampled_indices = np.random.choice(np.arange(len(p)), p=expcs, size=nsamples)
        wsampled_indices = np.random.choice(np.arange(len(ws)), p=ws, size=nsamples)

    # TODO(Amalie) This can be simplified
    # Compute distance posterior
    if inferencesettings.has_distance_case:
        distanceparams = star.distanceparams
        ms = distanceparams.magnitudes.keys()
        d_samples = np.zeros((nsamples, 2 * (len(ms) + 1)))
        LOS_EBV = get_EBV_along_LOS(distanceparams)
        if "distance" in cornerplots:
            plotout = np.zeros(3 * (2 * (len(ms) + 1)))
        j = 0
        dinterp, EBVinterp = [], []
        for idm, filt in enumerate(distanceparams.magnitudes.keys()):
            m_all = np.random.normal(
                distanceparams.magnitudes[filt][0],
                distanceparams.magnitudes[filt][1],
                noofind,
            )
            M_all = util.get_parameter_values(filt, Grid, selectedmodels, noofind)
            A_all = np.zeros(noofind)

            # Compute distances and extinction iteratively
            d_all = compute_distance_from_mag(m_all, M_all, A_all)
            for i in range(3):
                EBV_all = LOS_EBV(d_all)
                A_all = get_absorption(EBV_all, fitparams, filt)
                d_all = compute_distance_from_mag(m_all, M_all, A_all)

            # Create posteriors from weighted histograms
            dinterp.append(
                stats.posterior(
                    d_all, nonzeroprop, sampled_indices, nsigma=statdata.nsigma
                )
            )
            EBVinterp.append(
                stats.posterior(
                    EBV_all, nonzeroprop, sampled_indices, nsigma=statdata.nsigma
                )
            )

            # Compute centroid and uncertainties and print them
            xcen, xstdm, xstdp = stats.calc_key_stats(
                d_all[nonzeroprop][sampled_indices],
                outputoptions.centroid,
                outputoptions.uncert,
            )
            Acen, Astdm, Astdp = stats.calc_key_stats(
                A_all[nonzeroprop][sampled_indices],
                outputoptions.centroid,
                outputoptions.uncert,
            )
            Mcen, Mstdm, Mstdp = stats.calc_key_stats(
                M_all[nonzeroprop][sampled_indices],
                outputoptions.centroid,
                outputoptions.uncert,
            )

            if idm == 0:
                print("-----------------------------------------------------")
            remtor.print_param(
                "d(" + filt + ")",
                xcen,
                xstdm,
                xstdp,
                uncert=outputoptions.uncert,
                centroid=outputoptions.centroid,
            )
            remtor.print_param(
                "A(" + filt + ")",
                Acen,
                Astdm,
                Astdp,
                uncert=outputoptions.uncert,
                centroid=outputoptions.centroid,
            )
            if "distance" in cornerplots and outputoptions.uncert == "quantiles":
                plotout[6 * idm : 6 * idm + 3] = [xcen, xstdp - xcen, xcen - xstdm]
                plotout[6 * idm + 3 : 6 * idm + 6] = [Acen, Astdp - Acen, Acen - Astdm]
            elif "distance" in cornerplots:
                plotout[6 * idm : 6 * idm + 3] = [xcen, xstdm, xstdm]
                plotout[6 * idm + 3 : 6 * idm + 6] = [Acen, Astdm, Astdm]

            hout_dist, out_dist = util.add_out(
                hout_dist,
                out_dist,
                "distance_" + filt,
                xcen,
                xstdm,
                xstdp,
                outputoptions.uncert,
            )
            hout_dist, out_dist = util.add_out(
                hout_dist,
                out_dist,
                "A_" + filt,
                Acen,
                Astdm,
                Astdp,
                outputoptions.uncert,
            )
            hout_dist, out_dist = util.add_out(
                hout_dist,
                out_dist,
                "M_" + filt,
                Mcen,
                Mstdm,
                Mstdp,
                outputoptions.uncert,
            )

            d_samples[:, j] = d_all[nonzeroprop][sampled_indices]
            d_samples[:, j + 1] = A_all[nonzeroprop][sampled_indices]
            j += 2

        # Compute joint distance and extinction posteriors
        d_array = np.unique(
            [stats.quantile_1D(f.x, f.y, np.linspace(0, 1, 200)) for f in dinterp]
        )
        dposterior = np.prod([f(d_array) for f in dinterp], axis=0)
        EBV_array = np.unique(
            [stats.quantile_1D(f.x, f.y, np.linspace(0, 1, 200)) for f in EBVinterp]
        )
        EBVposterior = np.prod([f(EBV_array) for f in EBVinterp], axis=0)
        if np.nansum(dposterior) == 0 or np.nansum(EBVposterior) == 0:
            derrmessage = (
                "Joint distance posterior could not be computed as the "
                "distances derived for each magnitude are too different."
            )
            print(derrmessage)
            fio.write_star_to_errfile(starid, runfiles, derrmessage)
            if "distance" in outputoptions.asciiparams:
                hout_dist, out_dist = util.add_out(
                    hout_dist,
                    out_dist,
                    "distance_joint",
                    np.nan,
                    np.nan,
                    np.nan,
                    outputoptions.uncert,
                )
                hout_dist, out_dist = util.add_out(
                    hout_dist,
                    out_dist,
                    "EBV",
                    np.nan,
                    np.nan,
                    np.nan,
                    outputoptions.uncert,
                )
                hout, out = util.add_out(
                    hout, out, "distance", np.nan, np.nan, np.nan, outputoptions.uncert
                )
        else:
            xcen, xstdm, xstdp = stats.calc_key_stats(
                d_array,
                outputoptions.centroid,
                outputoptions.uncert,
                weights=dposterior,
            )
            EBVcen, EBVstdm, EBVstdp = stats.calc_key_stats(
                EBV_array,
                outputoptions.centroid,
                outputoptions.uncert,
                weights=EBVposterior,
            )

            remtor.print_param(
                "d(joint)",
                xcen,
                xstdm,
                xstdp,
                centroid=outputoptions.centroid,
                uncert=outputoptions.uncert,
            )
            remtor.print_param(
                "E(B-V)(joint)",
                EBVcen,
                EBVstdm,
                EBVstdp,
                centroid=outputoptions.centroid,
                uncert=outputoptions.uncert,
            )
            if "distance" in cornerplots and outputoptions.uncert == "quantiles":
                plotout[-6:-3] = [xcen, xstdp - xcen, xcen - xstdm]
                plotout[-3:] = [EBVcen, EBVstdp - EBVcen, EBVcen - EBVstdm]
            elif "distance" in cornerplots:
                plotout[-6:-3] = [xcen, xstdm, xstdm]
                plotout[-3:] = [EBVcen, EBVstdm, EBVstdm]

            d_samples[:, -2] = d_array[
                np.random.choice(
                    np.arange(len(dposterior)),
                    p=dposterior / np.sum(dposterior),
                    size=nsamples,
                )
            ]
            d_samples[:, -1] = EBV_array[
                np.random.choice(
                    np.arange(len(EBVposterior)),
                    p=EBVposterior / np.sum(EBVposterior),
                    size=nsamples,
                )
            ]

            # Create plots
            if "distance" in cornerplots:
                clabels = []
                for filt in distanceparams.magnitudes.keys():
                    clabels.append("d(" + filt + ")")
                    clabels.append("A(" + filt + ")")
                clabels = [*clabels, "d(joint)", "E(B-V)(joint)"]
                try:
                    cornerfig = plot_corner.corner(
                        d_samples,
                        labels=clabels,
                        plotout=plotout,
                        uncert=outputoptions.uncert,
                        **ckwargs,
                    )
                    filepaths.save_plot(cornerfig, kind="distance_corner")
                    plt.close()
                except Exception as error:
                    print(f"\nDistance corner plot failed with the error:{error}\n")

            # Add to output array
            if "distance" in outputoptions.asciiparams:
                hout_dist, out_dist = util.add_out(
                    hout_dist,
                    out_dist,
                    "distance_joint",
                    xcen,
                    xstdm,
                    xstdp,
                    outputoptions.uncert,
                )
                hout_dist, out_dist = util.add_out(
                    hout_dist,
                    out_dist,
                    "EBV",
                    EBVcen,
                    EBVstdm,
                    EBVstdp,
                    outputoptions.uncert,
                )
                hout, out = util.add_out(
                    hout, out, "distance", xcen, xstdm, xstdp, outputoptions.uncert
                )

    # Make sure that something is written to the ascii distance files! It will be
    # deleted later...
    else:
        hout_dist, out_dist = util.add_out(
            hout_dist,
            out_dist,
            "distance_joint",
            np.nan,
            np.nan,
            np.nan,
            outputoptions.uncert,
        )
        hout_dist, out_dist = util.add_out(
            hout_dist, out_dist, "EBV", np.nan, np.nan, np.nan, outputoptions.uncert
        )

    # Allocate arrays
    samples = np.zeros((nsamples, len(cornerplots)))
    if outputoptions.debug:
        lsamples = np.zeros((nsamples, len(cornerplots)))
        wsamples = np.zeros((nsamples, len(cornerplots)))

    # TODO(Amalie) make fillvalue in constants
    plotin = np.ones(2 * len(cornerplots)) * -9999
    plotout = np.zeros(3 * len(cornerplots))

    fitparams_scaled = {}

    for numpar, param in enumerate(params):
        # Generate list of x values
        x = util.get_parameter_values(param, Grid, selectedmodels, noofind)

        # Scale back to µHz before output/plot
        if param.startswith("dnu") or param.startswith("numax"):
            scale = star.globalseismicparams.get_scalefactor(param)
            x /= scale
            if param in fitparams:
                fitparams_scaled[param] = star.globalseismicparams.get_original(param)

        # Compute quantiles (using np.quantile is ~50 times faster than quantile_1D)
        xcen, xstdm, xstdp = stats.calc_key_stats(
            x[nonzeroprop][sampled_indices],
            outputoptions.centroid,
            outputoptions.uncert,
        )

        # Print info to log and console
        if numpar == 0:
            print("-----------------------------------------------------")
        remtor.print_param(
            param,
            xcen,
            xstdm,
            xstdp,
            uncert=outputoptions.uncert,
            centroid=outputoptions.centroid,
        )

        if param in cornerplots:
            if param in ["distance", "parallax"]:
                continue
            idx = cornerplots.index(param)
            if param in fitparams_scaled:
                xin, stdin = fitparams_scaled[param]
                plotin[2 * idx : 2 * idx + 2] = [xin, stdin]
            samples[:, idx] = x[nonzeroprop][sampled_indices]
            if outputoptions.debug:
                lsamples[:, idx] = x[nonzeroprop][lsampled_indices]
                wsamples[:, idx] = x[nonzeroprop][wsampled_indices]
            if outputoptions.uncert == "quantiles":
                plotout[3 * idx : 3 * idx + 3] = [xcen, xstdp - xcen, xcen - xstdm]
            else:
                plotout[3 * idx : 3 * idx + 3] = [xcen, xstdm, xstdm]

        if param in outputoptions.asciiparams:
            if param in ["distance", "parallax"]:
                continue
            hout, out = util.add_out(
                hout, out, param, xcen, xstdm, xstdp, outputoptions.uncert
            )

    # Create header for ascii file and save it
    if asciifile:
        asciifile.seek(0)
        if b"#" not in asciifile.readline():
            asciifile.write(f"# {' '.join(hout)} \n".encode())
        np.savetxt(
            asciifile, np.asarray(out).reshape(1, len(out)), fmt="%s", delimiter=" "
        )
        print(f"\nSaved results to {runfiles.summarytablepath}.")

    if asciifile_dist and "distance" in outputoptions.asciiparams:
        asciifile_dist.seek(0)
        if b"#" not in asciifile_dist.readline():
            asciifile_dist.write(f"# {' '.join(hout_dist)} \n".encode())
        np.savetxt(
            asciifile_dist,
            np.asarray(out_dist).reshape(1, len(out_dist)),
            fmt="%s",
            delimiter=" ",
        )
        print(
            f"Saved distance results for different filters to {runfiles.distancesummarytablepath}."
        )

    # Compare input to output and produce a comparison plot
    if compareinputoutput | outputoptions.developermode:
        comparewarn = util.compare_output_to_input(
            star=star,
            absolutemagnitudes=star.absolutemagnitudes,
            runfiles=runfiles,
            inferencesettings=inferencesettings,
            outputoptions=outputoptions,
            hout=hout,
            out=out,
            hout_dist=hout_dist,
            out_dist=out_dist,
        )
        if comparewarn:
            print(
                "DEBUG: The input values of the fitting parameters "
                "disagree with the outputted values."
            )
            if not len(kielplots):
                print("DEBUG: make Kiel diagram due to warning")
                library_param = (
                    "massini" if "tracks" in gridheader["gridtype"].lower() else "age"
                )
                x = util.get_parameter_values(
                    library_param, Grid, selectedmodels, noofind
                )
                lp_interval = np.quantile(
                    x[nonzeroprop][sampled_indices], statdata.quantiles[1:]
                )

                x = util.get_parameter_values("FeH", Grid, selectedmodels, noofind)
                feh_interval = np.quantile(
                    x[nonzeroprop][sampled_indices], statdata.quantiles[1:]
                )

                x = util.get_parameter_values("Teff", Grid, selectedmodels, noofind)
                Teffout = np.quantile(
                    x[nonzeroprop][sampled_indices], statdata.quantiles
                )

                x = util.get_parameter_values("logg", Grid, selectedmodels, noofind)
                loggout = np.quantile(
                    x[nonzeroprop][sampled_indices], statdata.quantiles
                )

                try:
                    fig = plot_kiel.kiel(
                        Grid=Grid,
                        selectedmodels=selectedmodels,
                        star=star,
                        inferencesettings=inferencesettings,
                        outputoptions=outputoptions,
                        plotconfig=plotconfig,
                        lp_interval=lp_interval,
                        feh_interval=feh_interval,
                        Teffout=Teffout,
                        loggout=loggout,
                        gridtype=gridheader["gridtype"],
                        nameinplot=starid if plotconfig.nameinplot else False,
                    )
                    filepaths.save_plot(fig, kind="warn_kiel")
                    plt.close()
                except Exception as error:
                    print("Warning Kiel diagram failed with the error:", error)

    # Create corner plot
    if len(cornerplots):
        try:
            cornerfig = plot_corner.corner(
                samples,
                labels=parameters.get_keys(cornerplots)[1],
                truth_color=parameters.get_keys(cornerplots)[3],
                plotin=plotin,
                plotout=plotout,
                nameinplot=starid if plotconfig.nameinplot else False,
                uncert=outputoptions.uncert,
                **ckwargs,
            )
            filepaths.save_plot(cornerfig, kind="corner")
            plt.close()
        except Exception as error:
            print("Corner plot failed with the error:", error)
        if outputoptions.debug:
            try:
                cornerfig = plot_corner.corner(
                    lsamples,
                    labels=parameters.get_keys(cornerplots)[1],
                    truth_color=parameters.get_keys(cornerplots)[3],
                    plotin=plotin,
                    plotout=plotout,
                    nameinplot=starid if plotconfig.nameinplot else False,
                    uncert=outputoptions.uncert,
                    **ckwargs,
                )
                filepaths.save_plot(cornerfig, kind="likelihood_corner")
                plt.close()
            except Exception as error:
                print("Likelihood corner plot failed with the error:", error)
            try:
                cornerfig = plot_corner.corner(
                    wsamples,
                    labels=parameters.get_keys(cornerplots)[1],
                    truth_color=parameters.get_keys(cornerplots)[3],
                    plotin=plotin,
                    plotout=plotout,
                    nameinplot=starid if plotconfig.nameinplot else False,
                    uncert=outputoptions.uncert,
                    **ckwargs,
                )
                filepaths.save_plot(fig, kind="prior_corner")
                plt.close()
            except Exception as error:
                print("Prior corner plot failed with the error:", error)

    # Create Kiel diagram
    if len(kielplots):
        # Find quantiles of massini/age and FeH to determine what tracks to plot
        library_param = (
            "massini" if "tracks" in gridheader["gridtype"].lower() else "age"
        )
        x = util.get_parameter_values(library_param, Grid, selectedmodels, noofind)
        lp_interval = np.quantile(
            x[nonzeroprop][sampled_indices], statdata.quantiles[1:]
        )

        # Use correct metallicity (only important for alpha enhancement)
        metalname = "MeH" if "MeH" in fitparams_scaled else "FeH"
        x = util.get_parameter_values(metalname, Grid, selectedmodels, noofind)
        feh_interval = np.quantile(
            x[nonzeroprop][sampled_indices], statdata.quantiles[1:]
        )

        x = util.get_parameter_values("Teff", Grid, selectedmodels, noofind)
        Teffout = np.quantile(x[nonzeroprop][sampled_indices], statdata.quantiles)

        x = util.get_parameter_values("logg", Grid, selectedmodels, noofind)
        loggout = np.quantile(x[nonzeroprop][sampled_indices], statdata.quantiles)

        try:
            fig = plot_kiel.kiel(
                Grid=Grid,
                selectedmodels=selectedmodels,
                star=star,
                inferencesettings=inferencesettings,
                outputoptions=outputoptions,
                plotconfig=plotconfig,
                lp_interval=lp_interval,
                feh_interval=feh_interval,
                Teffout=Teffout,
                loggout=loggout,
                gridtype=gridheader["gridtype"],
                nameinplot=starid if plotconfig.nameinplot else False,
                color_by_likelihood=False,
            )
            filepaths.save_plot(fig, kind="kiel")
            plt.close()
        except Exception as error:
            print("Kiel diagram failed with the error:", error)
            raise

    if outputoptions.debug and len(star.distanceparams.magnitudes.keys()) > 0:
        print("Make normalised distribution plot of terms in PDF computation")
        mins = []
        bayw = np.concatenate([ts.bayw for ts in selectedmodels.values()])
        bayw = bayw[nonzeroprop]
        bayw -= np.amax(bayw)
        mins.append(np.amin(bayw))

        magw = np.concatenate([ts.magw for ts in selectedmodels.values()])
        magw = magw[nonzeroprop]
        magw -= np.amax(magw)
        mins.append(np.amin(magw))

        IMFw = np.concatenate([ts.IMFw for ts in selectedmodels.values()])
        IMFw = IMFw[nonzeroprop]
        IMFw -= np.amax(IMFw)
        mins.append(np.amin(IMFw))

        csw = -0.5 * cs
        csw -= np.amax(csw)

        for param in ["massini"]:  # params:
            fig, axs = plt.subplots(5, sharex=True)
            x = util.get_parameter_values(param, Grid, selectedmodels, noofind)

            axs[0].plot(x[nonzeroprop], logy, "k.", label="Posterior", ms=3, alpha=0.1)
            axs[1].plot(
                x[nonzeroprop][sampled_indices],
                bayw[sampled_indices],
                "b.",
                label="Bayesian weights",
                ms=3,
                alpha=0.1,
            )
            axs[2].plot(
                x[nonzeroprop][sampled_indices],
                magw[sampled_indices],
                "g.",
                label="Absolute magnitude",
                ms=3,
                alpha=0.1,
            )
            axs[3].plot(
                x[nonzeroprop][sampled_indices],
                IMFw[sampled_indices],
                "c.",
                label="IMF",
                ms=3,
                alpha=0.1,
            )
            axs[4].plot(
                x[nonzeroprop][sampled_indices],
                csw[sampled_indices],
                "m.",
                label=r"$\chi^2$ part",
                ms=3,
                alpha=0.1,
            )

            plt.xlabel(param)
            shadowaxes = fig.add_subplot(111, frame_on=False)
            plt.tick_params(
                labelcolor="none", top=False, bottom=False, left=False, right=False
            )
            shadowaxes.set_ylabel("Scaled probability")
            for i in [1, 2, 3, 4]:
                axs[i].set_ylim([min(mins) - 0.5, 0.5])
            handles, labels = [
                (a + b + c + d + e)
                for a, b, c, d, e in zip(
                    axs[0].get_legend_handles_labels(),
                    axs[1].get_legend_handles_labels(),
                    axs[2].get_legend_handles_labels(),
                    axs[3].get_legend_handles_labels(),
                    axs[4].get_legend_handles_labels(),
                )
            ]
            fig.legend(handles, labels, loc="upper center", ncol=5)

            filepaths.save_plot(fig, kind="dist")
            plt.close()
