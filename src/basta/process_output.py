"""
Calculation and generation of output, and driver for producing plots
"""

import os
import copy
from io import IOBase
from copy import deepcopy

import numpy as np
import matplotlib

import basta.fileio as fio
from basta.constants import sydsun as sydc
from basta.constants import parameters, statdata
from basta.utils_distances import compute_distance_from_mag
from basta.distances import get_absorption, get_EBV_along_LOS
from basta import utils_general as util
from basta import stats, plot_corner, plot_kiel
from basta.downloader import get_basta_dir

# Change matplotlib backend before loading pyplot
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Set the style of all plots
plt.style.use(os.path.join(get_basta_dir(), "plots.mplstyle"))

# Define a color dictionary for easier change of color
colors = {"l0": "#D55E00", "l1": "#009E73", "l2": "#0072B2"}


def compute_posterior(
    starid,
    selectedmodels,
    Grid,
    inputparams,
    outfilename,
    gridtype,
    debug=False,
    developermode=False,
    validationmode=False,
    compareinputoutput=False,
):
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
    outfilename : str
        Name of directory of where to put plots outputted if debug is True.
    gridtype : str
        Type of the grid (as read from the grid in bastamain) containing either 'tracks'
        or 'isochrones'.
    debug : bool, optional
        Debug flag for developers.
    developermode : bool, optional
        If True, experimental features will be used in run.
    validationmode : bool, optional
        If True, assume a validation run with changed behaviour
    """
    # Load setings
    asciifile = inputparams.get("asciioutput")
    asciifile_dist = inputparams.get("asciioutput_dist")
    centroid = inputparams["centroid"]
    uncert = inputparams["uncert"]
    plottype = inputparams["plotfmt"]

    # Lists of params (copy to avoid problems when running multiple stars)
    outparams = deepcopy(inputparams["asciiparams"])
    cornerplots = deepcopy(inputparams["cornerplots"])
    params = util.unique_unsort(outparams + cornerplots)

    # List of params for plotting
    kielplots = inputparams["kielplots"]
    fitparams = inputparams["fitparams"]
    fitpar_kiel = copy.deepcopy(fitparams)

    # Initialise strings for printing
    hout = []
    out = []
    hout.append("starid")
    out.append(starid)
    hout_dist = []
    out_dist = []
    hout_dist.append("starid")
    out_dist.append(starid)

    # Generate PDF values
    logy = np.concatenate([ts.logPDF for ts in selectedmodels.values()])
    noofind = len(logy)
    nonzeroprop = np.isfinite(logy)
    logy = logy[nonzeroprop]
    nsamples = min(statdata.nsamples, noofind)

    # Likelihood is only defined up to a multiplicative constant of
    # proportionality, therefore we subtract max(logy) from logy to make sure
    # the greatest argument to np.exp is 1 and thus the sum is greater than 1
    # and we avoid dividing by zero when normalizing.
    lk = logy - np.amax(logy)
    p = np.exp(lk - np.log(np.sum(np.exp(lk))))
    sampled_indices = np.random.choice(np.arange(len(p)), p=p, size=nsamples)

    if debug:
        cs = np.concatenate([ts.chi2 for ts in selectedmodels.values()])
        ws = np.exp(logy + 0.5 * cs[nonzeroprop])
        ws /= np.sum(ws)
        expcs = np.exp(-0.5 * cs[nonzeroprop])
        expcs /= np.sum(expcs)
        lsampled_indices = np.random.choice(np.arange(len(p)), p=expcs, size=nsamples)
        wsampled_indices = np.random.choice(np.arange(len(ws)), p=ws, size=nsamples)

    # Corner plot kwargs
    ckwargs = {
        "show_titles": True,
        "quantiles": statdata.quantiles,
        "smooth": 1,
        "smooth1d": "kde",
        "title_kwargs": {"fontsize": 10},
        "plot_datapoints": False,
        "uncert": uncert,
    }

    # Compute distance posterior
    if "distance" in params:
        distanceparams = inputparams["distanceparams"]
        ms = list(distanceparams["filters"])
        d_samples = np.zeros((nsamples, 2 * (len(ms) + 1)))
        LOS_EBV = get_EBV_along_LOS(distanceparams)
        if "distance" in cornerplots:
            plotout = np.zeros(3 * (2 * (len(ms) + 1)))
        j = 0
        dinterp, EBVinterp = [], []
        for idm, m in enumerate(ms):
            m_all = np.random.normal(
                distanceparams["m"][m], distanceparams["m_err"][m], noofind
            )
            M_all = util.get_parameter_values(m, Grid, selectedmodels, noofind)
            A_all = np.zeros(noofind)

            # Compute distances and extinction iteratively
            d_all = compute_distance_from_mag(m_all, M_all, A_all)
            for i in range(3):
                EBV_all = LOS_EBV(d_all)
                A_all = get_absorption(EBV_all, fitparams, m)
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
                d_all[nonzeroprop][sampled_indices], centroid, uncert
            )
            Acen, Astdm, Astdp = stats.calc_key_stats(
                A_all[nonzeroprop][sampled_indices], centroid, uncert
            )
            Mcen, Mstdm, Mstdp = stats.calc_key_stats(
                M_all[nonzeroprop][sampled_indices], centroid, uncert
            )

            if idm == 0:
                print("-----------------------------------------------------")
            util.printparam(
                "d(" + m + ")", xcen, xstdm, xstdp, uncert=uncert, centroid=centroid
            )
            util.printparam(
                "A(" + m + ")", Acen, Astdm, Astdp, uncert=uncert, centroid=centroid
            )
            if "distance" in cornerplots and uncert == "quantiles":
                plotout[6 * idm : 6 * idm + 3] = [xcen, xstdp - xcen, xcen - xstdm]
                plotout[6 * idm + 3 : 6 * idm + 6] = [Acen, Astdp - Acen, Acen - Astdm]
            elif "distance" in cornerplots:
                plotout[6 * idm : 6 * idm + 3] = [xcen, xstdm, xstdm]
                plotout[6 * idm + 3 : 6 * idm + 6] = [Acen, Astdm, Astdm]

            hout_dist, out_dist = util.add_out(
                hout_dist, out_dist, "distance_" + m, xcen, xstdm, xstdp, uncert
            )
            hout_dist, out_dist = util.add_out(
                hout_dist, out_dist, "A_" + m, Acen, Astdm, Astdp, uncert
            )
            hout_dist, out_dist = util.add_out(
                hout_dist, out_dist, "M_" + m, Mcen, Mstdm, Mstdp, uncert
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
                + "distances derived for each magnitude are too different."
            )
            print(derrmessage)
            fio.write_star_to_errfile(starid, inputparams, derrmessage)
            if "distance" in outparams:
                hout_dist, out_dist = util.add_out(
                    hout_dist,
                    out_dist,
                    "distance_joint",
                    np.nan,
                    np.nan,
                    np.nan,
                    uncert,
                )
                hout_dist, out_dist = util.add_out(
                    hout_dist, out_dist, "EBV", np.nan, np.nan, np.nan, uncert
                )
                hout, out = util.add_out(
                    hout, out, "distance", np.nan, np.nan, np.nan, uncert
                )
            if "distance" in cornerplots:
                cornerplots.remove("distance")
        else:
            xcen, xstdm, xstdp = stats.calc_key_stats(
                d_array, centroid, uncert, weights=dposterior
            )
            EBVcen, EBVstdm, EBVstdp = stats.calc_key_stats(
                EBV_array, centroid, uncert, weights=EBVposterior
            )

            util.printparam(
                "d(joint)", xcen, xstdm, xstdp, centroid=centroid, uncert=uncert
            )
            util.printparam(
                "E(B-V)(joint)",
                EBVcen,
                EBVstdm,
                EBVstdp,
                centroid=centroid,
                uncert=uncert,
            )
            if "distance" in cornerplots and uncert == "quantiles":
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
                for m in ms:
                    clabels.append("d(" + m + ")")
                    clabels.append("A(" + m + ")")
                clabels = clabels + ["d(joint)", "E(B-V)(joint)"]
                try:
                    plot_corner.corner(
                        d_samples, labels=clabels, plotout=plotout, **ckwargs
                    )
                    cornerfile = outfilename + "_distance_corner." + plottype
                    plt.savefig(cornerfile)
                    plt.close()
                    print("\nSaved distance corner plot to {0}.\n".format(cornerfile))
                except Exception as error:
                    print(
                        "\nDistance corner plot failed with the error:{0}\n".format(
                            error
                        )
                    )

                # Plotting done: Remove keyword
                cornerplots.remove("distance")
            # Add to output array
            if "distance" in outparams:
                hout_dist, out_dist = util.add_out(
                    hout_dist, out_dist, "distance_joint", xcen, xstdm, xstdp, uncert
                )
                hout_dist, out_dist = util.add_out(
                    hout_dist, out_dist, "EBV", EBVcen, EBVstdm, EBVstdp, uncert
                )
                hout, out = util.add_out(
                    hout, out, "distance", xcen, xstdm, xstdp, uncert
                )

        # We have finished using distances: Remove keyword
        params.remove("distance")
        if "distance" in outparams:
            outparams.remove("distance")

    # Make sure that something is written to the ascii distance files! It will be
    # deleted later...
    else:
        hout_dist, out_dist = util.add_out(
            hout_dist, out_dist, "distance_joint", np.nan, np.nan, np.nan, uncert
        )
        hout_dist, out_dist = util.add_out(
            hout_dist, out_dist, "EBV", np.nan, np.nan, np.nan, uncert
        )

    # Allocate arrays
    samples = np.zeros((nsamples, len(cornerplots)))
    if debug:
        lsamples = np.zeros((nsamples, len(cornerplots)))
        wsamples = np.zeros((nsamples, len(cornerplots)))
    plotin = np.ones(2 * len(cornerplots)) * -9999
    plotout = np.zeros(3 * len(cornerplots))
    dnu_scales = inputparams.get("dnu_scales", {})
    for numpar, param in enumerate(params):
        # Generate list of x values
        x = util.get_parameter_values(param, Grid, selectedmodels, noofind)

        # Scale back to muHz before output/plot
        if param.startswith("dnu") and param not in ["dnufit", "dnufitMos12"]:
            dnu_rescal = dnu_scales.get(param, 1.00)
            x *= inputparams.get("dnusun", sydc.SUNdnu) / dnu_rescal
            if param in fitparams:
                fitparams[param] = (
                    np.asarray(fitparams[param])
                    * inputparams.get("dnusun", sydc.SUNdnu)
                    / dnu_rescal
                )

        elif param.startswith("numax"):
            x *= inputparams.get("numsun", sydc.SUNnumax)
            if param in fitparams:
                fitparams[param] = np.asarray(fitparams[param]) * inputparams.get(
                    "numsun", sydc.SUNnumax
                )
        elif param in ["dnufit", "dnufitMos12"]:
            dnu_rescal = dnu_scales.get(param, 1.00)
            x /= dnu_rescal
            if param in fitparams:
                fitparams[param] = np.asarray(fitparams[param]) / dnu_rescal

        # Compute quantiles (using np.quantile is ~50 times faster than quantile_1D)
        xcen, xstdm, xstdp = stats.calc_key_stats(
            x[nonzeroprop][sampled_indices], centroid, uncert
        )

        # Print info to log and console
        if numpar == 0:
            print("-----------------------------------------------------")
        util.printparam(param, xcen, xstdm, xstdp, uncert=uncert, centroid=centroid)

        if param in cornerplots:
            idx = cornerplots.index(param)
            if param in fitparams:
                xin, stdin = fitparams[param]
                plotin[2 * idx : 2 * idx + 2] = [xin, stdin]
            samples[:, idx] = x[nonzeroprop][sampled_indices]
            if debug:
                lsamples[:, idx] = x[nonzeroprop][lsampled_indices]
                wsamples[:, idx] = x[nonzeroprop][wsampled_indices]
            if uncert == "quantiles":
                plotout[3 * idx : 3 * idx + 3] = [xcen, xstdp - xcen, xcen - xstdm]
            else:
                plotout[3 * idx : 3 * idx + 3] = [xcen, xstdm, xstdm]

        if param in outparams:
            hout, out = util.add_out(hout, out, param, xcen, xstdm, xstdp, uncert)

    # Create header for ascii file and save it
    if asciifile is not False:
        hline = b"# "
        for i in range(len(hout)):
            hline += hout[i].encode() + " ".encode()
        if isinstance(asciifile, IOBase):
            asciifile.seek(0)
            if b"#" not in asciifile.readline():
                asciifile.write(hline + b"\n")
            np.savetxt(
                asciifile, np.asarray(out).reshape(1, len(out)), fmt="%s", delimiter=" "
            )
            print("\nSaved results to " + asciifile.name + ".")
        elif asciifile is False:
            pass
        else:
            np.savetxt(
                asciifile,
                np.asarray(out).reshape(1, len(out)),
                fmt="%s",
                header=hline,
                delimiter=" ",
            )
            print("Saved results to " + asciifile + ".")

    if asciifile_dist:
        if len(hout_dist) > 0:
            hline = b"# "
            for i in range(len(hout_dist)):
                hline += hout_dist[i].encode() + " ".encode()
            if isinstance(asciifile_dist, IOBase):
                asciifile_dist.seek(0)
                if b"#" not in asciifile_dist.readline():
                    asciifile_dist.write(hline + b"\n")
                np.savetxt(
                    asciifile_dist,
                    np.asarray(out_dist).reshape(1, len(out_dist)),
                    fmt="%s",
                    delimiter=" ",
                )
                if "distance" in outparams:
                    print(
                        "Saved distance results for different filters to %s."
                        % asciifile_dist.name
                    )
            elif asciifile_dist is False:
                pass
            else:
                np.savetxt(
                    asciifile_dist,
                    np.asarray(out_dist).reshape(1, len(out_dist)),
                    fmt="%s",
                    header=hline,
                    delimiter=" ",
                )
                if "distance" in outparams:
                    print(
                        "Saved distance results for different filters to %s."
                        % asciifile_dist
                    )

    # Compare input to output and produce a comparison plot
    if compareinputoutput | developermode:
        comparewarn = util.compare_output_to_input(
            starid, inputparams, hout, out, hout_dist, out_dist, uncert=uncert
        )
        if comparewarn:
            print(
                "DEBUG: The input values of the fitting parameters "
                + "disagree with the outputted values."
            )
            if not len(kielplots):
                print("DEBUG: make Kiel diagram due to warning")
                library_param = "massini" if "tracks" in gridtype.lower() else "age"
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
                        fitparams=fitpar_kiel,
                        inputparams=inputparams,
                        lp_interval=lp_interval,
                        feh_interval=feh_interval,
                        Teffout=Teffout,
                        loggout=loggout,
                        gridtype=gridtype,
                        nameinplot=starid if inputparams["nameinplot"] else False,
                        debug=debug,
                        developermode=developermode,
                        validationmode=validationmode,
                    )
                    kielfile = outfilename + "_warn_kiel." + plottype
                    fig.savefig(kielfile)
                    plt.close()
                    print("Saved warning Kiel diagram to " + kielfile + ".")
                except Exception as error:
                    print("Warning Kiel diagram failed with the error:", error)

    # Create corner plot
    if len(cornerplots):
        try:
            plot_corner.corner(
                samples,
                labels=parameters.get_keys(cornerplots)[1],
                truth_color=parameters.get_keys(cornerplots)[3],
                plotin=plotin,
                plotout=plotout,
                nameinplot=starid if inputparams["nameinplot"] else False,
                **ckwargs,
            )
            cornerfile = outfilename + "_corner." + plottype
            plt.savefig(cornerfile)
            plt.close()
            print("Saved corner plot to " + cornerfile + ".")
        except Exception as error:
            print("Corner plot failed with the error:", error)
        if debug:
            try:
                plot_corner.corner(
                    lsamples,
                    labels=parameters.get_keys(cornerplots)[1],
                    truth_color=parameters.get_keys(cornerplots)[3],
                    plotin=plotin,
                    plotout=plotout,
                    nameinplot=starid if inputparams["nameinplot"] else False,
                    **ckwargs,
                )
                cornerfile = outfilename + "_DEBUG_likelihood_corner." + plottype
                plt.savefig(cornerfile)
                plt.close()
                print("Saved likelihood corner plot to " + cornerfile + ".")
            except Exception as error:
                print("Likelihood corner plot failed with the error:", error)
            try:
                plot_corner.corner(
                    wsamples,
                    labels=parameters.get_keys(cornerplots)[1],
                    truth_color=parameters.get_keys(cornerplots)[3],
                    plotin=plotin,
                    plotout=plotout,
                    nameinplot=starid if inputparams["nameinplot"] else False,
                    **ckwargs,
                )
                cornerfile = outfilename + "_DEBUG_prior_corner." + plottype
                plt.savefig(cornerfile)
                plt.close()
                print("Saved prior corner plot to " + cornerfile + ".")
            except Exception as error:
                print("Prior corner plot failed with the error:", error)

    # Create Kiel diagram
    if len(kielplots):
        # Find quantiles of massini/age and FeH to determine what tracks to plot
        library_param = "massini" if "tracks" in gridtype.lower() else "age"
        x = util.get_parameter_values(library_param, Grid, selectedmodels, noofind)
        lp_interval = np.quantile(
            x[nonzeroprop][sampled_indices], statdata.quantiles[1:]
        )

        # Use correct metallicity (only important for alpha enhancement)
        metalname = "MeH" if "MeH" in fitparams else "FeH"
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
                fitparams=fitpar_kiel,
                inputparams=inputparams,
                lp_interval=lp_interval,
                feh_interval=feh_interval,
                Teffout=Teffout,
                loggout=loggout,
                gridtype=gridtype,
                nameinplot=starid if inputparams["nameinplot"] else False,
                debug=debug,
                developermode=developermode,
                validationmode=validationmode,
                color_by_likelihood=False,
            )
            kielfile = outfilename + "_kiel." + plottype
            fig.savefig(kielfile)
            plt.close()
            print("Saved Kiel diagram to " + kielfile + ".")
        except Exception as error:
            print("Kiel diagram failed with the error:", error)
            raise

    if debug and len(inputparams["magnitudes"]) > 0:
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

            distfile = outfilename + "_DEBUG_dist" + param + "." + plottype
            fig.savefig(distfile)
            plt.close()
            print("Saved distribution plot to " + distfile + ".")
