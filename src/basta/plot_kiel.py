"""
Production of Kiel diagrams
"""

import os
import numpy as np
import matplotlib
import matplotlib.collections

from basta import stats
from basta import fileio as fio
from basta import utils_seismic as su
from basta import utils_general as gu
from basta.constants import parameters
from basta.downloader import get_basta_dir

# Set the style of all plots
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.style.use(os.path.join(get_basta_dir(), "plots.mplstyle"))


def plot_param(Grid, ax, track, all_segments, label, color):
    """
    Function for plotting the parameter interval in the Kiel diagram

    Parameters
    ----------
    Grid : hdf5 object
        The already loaded grid, containing the tracks/isochrones
    ax : AxesSubplot object
        Axis in which to plot
    track : str
        Path for the current track/isochrone in the Gridfile
    all_segments : list
        The indeces in the track/isochrone where the parameter is within
        the limit in fitparams
    label : str
        The label desired for the legend, is '_nolegend_' if it has
        already been added to the plot
    color : string
        The designated plotting color for the parameter, see 'constants.py'
    """
    # Find out if there are multiple segments in track
    where_skip = np.where(np.diff(all_segments) != 1)[0]

    # If only one, plot the whole segment
    if len(where_skip) == 0:
        segments = [list(all_segments)]
    else:
        # If multiple segments, plot each individually
        where_skip = np.append(where_skip, len(all_segments) - 1)
        segments = []
        current = 0
        for skip in where_skip:
            segments.append(list(all_segments[current : skip + 1]))
            current = skip + 1

    # Dummy variable for making legend line
    dummy = False

    # Plot the segments
    for segment in segments:
        # Determine if the segment is a line or a single point
        if len(segment) < 2:
            markertype = "."
            lab = label
            if lab != "_nolegend_":
                dummy = True
                # Plot dummy, so legend entry becomes a line
                ax.plot([0, 0], [0, 0], "-", alpha=0.5, lw=3, color=color, label=lab)
                lab = "_nolegend_"
        else:
            markertype = "-"
            lab = label
        # The actual plotting
        ax.plot(
            Grid[track + "/Teff"][segment],
            Grid[track + "/logg"][segment],
            markertype,
            lw=3,
            markersize=6,
            color=color,
            zorder=3,
            alpha=0.5,
            label=lab,
        )
        # Label magic to limit the legend to having only a single line
        # entry per parameter
        if lab != "_nolegend_" or dummy:
            label = "_nolegend_"
    return label


def kiel(
    Grid,
    selectedmodels,
    fitparams,
    inputparams,
    lp_interval,
    feh_interval,
    Teffout,
    loggout,
    gridtype,
    nameinplot=False,
    debug=False,
    developermode=False,
    validationmode=False,
    color_by_likelihood=False,
):
    """
    Make a Kiel diagram of the relevant tracks/isochrones, where fitted
    parameters within their given uncertainties are marked on the tracks.

    The plotted tracks/isochrones are chosen as the tracks/isochrones with
    non-zero likelihood with mass/age within the 16th and 84th percenttile,
    and if [Fe/H] or [M/H] is fitted, within the given uncertainty.
    If they are not fitted, they are instead also chosen as the ones within
    the 16th and 84th percentile.

    They are also chosen to be within the fitted evolutionary constants,
    e.g. alphaMLT and overshooting.
    See list 'constants' for full list of fitting parameters.

    Parameters
    ----------
    Grid : hdf5 object
        The already loaded grid, containing the tracks/isochrones.
    selectedmodels : dict
        Contains information on all models with a non-zero likelihood.
    fitparams : dict
        A copy of the fitparams with grid-scaled frequency parameters.
    inputparams : dict
        All relevant input information about the fit.
    lp_interval : list
        16th and 84th percentile of the library parameter, mass for
        tracks and age for isochrones.
    feh_interval : list
        16th and 84th quantile of [Fe/H], for determination of plotted
        tracks/isochrones.
    Teffout : array
        Array with median, min, and max effective temperature.
    loggout : array
        Array with median, min, and max logg.
    gridtype : str
        Type of the grid (as read from the grid in bastamain) containing either 'tracks'
        or 'isochrones'.
    nameinplot : str or bool
        Star identifier if it is to be included in the figure
    debug : bool, optional
        Debug flag.
    developermode : bool, optional
        If True, experimental features will be used in run.
    validationmode : bool, optional
        If True, style the plots as required for validation runs

    Returns
    -------
    fig : figure canvas
        Kiel diagram
    """
    # Inflate parameter ranges if requested
    if developermode:
        print("\nACTIVATED EXPERIMENTAL FEATURE:")
        print(
            "Extending the selection ranges (from the default quantiles)",
            "in the Kiel diagram!\n",
        )
        scalefactor = 0.3
        lp_interval[0] *= 1 - scalefactor
        lp_interval[1] *= 1 + scalefactor
        if debug:
            print(
                "DEBUG: Interval after inflation by {0} pct. = {1}\n".format(
                    scalefactor * 100, lp_interval
                )
            )

    # Assign params
    kielplots = inputparams.get("kielplots")
    fitfreqs = inputparams.get("fitfreqs", False)
    toggle_freqs = True
    filters = [f for f in inputparams["magnitudes"]]

    # This is by design a "==" comparison to True, because it will otherwise
    # fail, as the type is numpy.bool_ !
    if not kielplots[0] == True:
        toggle_freqs = False
        new_filters = []
        new_fitpars = {}
        for par in kielplots:
            if par == "freqs":
                toggle_freqs = True
            elif par in filters:
                new_filters.append(par)
            else:
                new_fitpars[par] = fitparams[par]
        filters = new_filters
        fitparams = new_fitpars

    # Save the tracks in selectedmodels with appropriate massini and FeH
    tracks = []
    constants = ["alphaFe", "ove", "gcut", "eta", "alphaMLT"]
    metal = "MeH" if "MeH" in fitparams else "FeH"
    for modelpath in selectedmodels:
        if "tracks" in gridtype.lower():
            trackvalue = Grid[modelpath]["massini"][0]
        else:
            trackvalue = Grid[modelpath]["age"][0]
        if trackvalue >= lp_interval[0] and trackvalue <= lp_interval[1]:
            track_pass = True
            for param in constants:
                if param in fitparams:
                    err = fitparams[param][1]
                    param_interval = [
                        fitparams[param][0] - err,
                        fitparams[param][0] + err,
                    ]
                    trackvalue = Grid[modelpath + "/" + param][0]
                    if (
                        not trackvalue >= param_interval[0]
                        and trackvalue <= param_interval[1]
                    ):
                        track_pass = False
            if track_pass:
                metal_in_track = np.where(
                    np.logical_and(
                        Grid[modelpath + "/" + metal][:] >= feh_interval[0],
                        Grid[modelpath + "/" + metal][:] <= feh_interval[1],
                    )
                )[0]
                if list(metal_in_track):
                    tracks.append(modelpath)

    # Median teff and logg
    Teff, terrm, terrp = Teffout[0], Teffout[0] - Teffout[1], Teffout[2] - Teffout[0]
    logg, lerrm, lerrp = loggout[0], loggout[0] - loggout[1], loggout[2] - loggout[0]

    # The highest likelihood is used to control the plot below. If desired, the
    # model with lowest chi^2 can be extracted and added to the plot
    if validationmode:
        minchi2_path, minchi2_ind = stats.lowest_chi2(selectedmodels)
        hlm_chi2, lcm_chi2 = stats.chi_for_plot(selectedmodels)

    # Define the max-likelihood model, and define teff/logg
    # intervals for limit control
    maxPDF_path, maxPDF_ind = stats.most_likely(selectedmodels)
    if maxPDF_path not in tracks:
        tracks.append(maxPDF_path)
    teffrange = [min(Grid[maxPDF_path + "/Teff"]), max(Grid[maxPDF_path + "/Teff"])]
    loggrange = [min(Grid[maxPDF_path + "/logg"]), max(Grid[maxPDF_path + "/logg"])]
    nsigma = 2
    if Teff < teffrange[0]:
        teffrange[0] = Teff - nsigma * terrm
    if Teff > teffrange[1]:
        teffrange[1] = Teff + nsigma * terrp
    if logg < loggrange[0]:
        loggrange[0] = logg - nsigma * lerrm
    if logg > loggrange[1]:
        loggrange[1] = logg + nsigma * lerrp
    dteff = teffrange[1] - teffrange[0]
    dlogg = loggrange[1] - loggrange[0]

    # Limits for full track, adjusted to even values
    tefflim = [
        100 * np.floor(teffrange[0] / 100) - 200,
        100 * np.ceil(teffrange[1] / 100) + 200,
    ]
    logglim = [0.1 * np.floor(loggrange[0] / 0.1), 0.1 * np.ceil(loggrange[1] / 0.1)]

    # "Standard" value for span in axis
    if "tracks" in gridtype.lower():
        teff_std = 350
        logg_std = 0.5
    else:
        teff_std = 150
        logg_std = 0.30
    make_subplot = [dteff > teff_std * 2, dlogg > logg_std * 2]

    # If the range is too large, make a zoomed subplot, keep ratio of original
    if True in make_subplot:
        tefflim_sub = [
            100 * np.floor((Teff - teff_std) / 100),
            100 * np.ceil((Teff + teff_std) / 100),
        ]
        ratio = (logglim[1] - logglim[0]) / (tefflim[1] - tefflim[0])
        logglim_sub = [
            logg - 0.5 * (tefflim_sub[1] - tefflim_sub[0]) * ratio,
            logg + 0.5 * (tefflim_sub[1] - tefflim_sub[0]) * ratio,
        ]

    # Make list with both limits
    tefflim = [tefflim_sub, tefflim] if True in make_subplot else [tefflim]
    logglim = [logglim_sub, logglim] if True in make_subplot else [logglim]

    # Get labels and colors for sorted params
    keys = list(fitparams.keys()) + filters
    sorted_parameters = np.array(keys)[np.argsort(keys)]
    _, labels, _, colors = parameters.get_keys(sorted_parameters)

    ################
    # Figure starts
    ################

    # Set up the figure
    if True in make_subplot:
        fig, axis = plt.subplots(2, 1, figsize=(12.8, 17.6))
        axis[1].plot(
            [tefflim[0][0], tefflim[0][1], tefflim[0][1], tefflim[0][0], tefflim[0][0]],
            [logglim[0][0], logglim[0][0], logglim[0][1], logglim[0][1], logglim[0][0]],
            "-",
            color="darkgrey",
            alpha=0.5,
            zorder=5,
            label="_nolegend_",
        )
        iteration = zip(axis, tefflim, logglim)
    else:
        fig, axis = plt.subplots(1, 1, figsize=(8.47, 6))
        iteration = zip([axis], tefflim, logglim)

    # Iteration over the subplots, the same is done with different limits
    for ax, tlim, glim in iteration:
        max_logPDF = selectedmodels[maxPDF_path].logPDF.max()
        for track in tracks:
            # Make a copy to allow manipulation
            xs = gu.h5py_to_array(Grid[track + "/Teff"])
            ys = gu.h5py_to_array(Grid[track + "/logg"])

            # Special treatment to plot points color-coded by likelihood
            if color_by_likelihood:
                # Extract pdf information
                logpdf = np.zeros_like(xs) + 0.1
                m = selectedmodels[track]
                logpdf[m.index] = 0.2 + 0.5 * np.exp(m.logPDF - max_logPDF)

                # Make segments to colorcode
                points = np.transpose([xs, ys]).reshape(-1, 1, 2)
                segments = np.concatenate([points[:-1], points[1:]], axis=1)
                lc = matplotlib.collections.LineCollection(segments, cmap="gray_r")
                lc.set_array(logpdf)
                lc.set_linewidth(1)
                ax.add_collection(lc)
                continue

            # Plot as points for validation mode
            if validationmode:
                ax.plot(
                    xs,
                    ys,
                    ".",
                    zorder=1,
                    color="darkgrey",
                    alpha=0.8,
                    label="_nolegend_",
                )
            else:
                ax.plot(
                    xs,
                    ys,
                    zorder=1,
                    color="darkgrey",
                    alpha=0.8,
                    label="_nolegend_",
                )

        # Plot the max likelihood and median model
        if validationmode:
            bfmmodlab = "Highest likelihood model (chi2 = {0:1.4e})".format(hlm_chi2)
        else:
            bfmmodlab = "Best fit model"
        ax.plot(
            Grid[maxPDF_path + "/Teff"][maxPDF_ind],
            Grid[maxPDF_path + "/logg"][maxPDF_ind],
            "*",
            color="#000000",
            markersize=20,
            zorder=5,
            label=bfmmodlab,
        )
        ax.plot(
            Teff, logg, "o", color="k", markersize=15, zorder=np.inf, label="Median"
        )

        # Add chi^2 model?
        if validationmode:
            ax.plot(
                Grid[minchi2_path + "/Teff"][minchi2_ind],
                Grid[minchi2_path + "/logg"][minchi2_ind],
                "p",
                color="k",
                markersize=15,
                label="Lowest chi^2 model (chi2 = {0:1.4e})".format(lcm_chi2),
            )

        # Plot parameter intervals of fitparams
        ncol = 2
        for i, param in enumerate(sorted_parameters):
            label = labels[i]
            # Set background marking of Teff
            if param == "Teff":
                ncol += 1
                err = fitparams[param][1]
                Tmin = np.ones(2) * fitparams[param][0] - err
                Tmax = np.ones(2) * fitparams[param][0] + err
                ax.fill_betweenx(
                    glim,
                    Tmin,
                    Tmax,
                    facecolor=colors[i],
                    zorder=2,
                    alpha=0.3,
                    label=label,
                )

            # Set background marking of logg
            elif param == "logg":
                ncol += 1
                err = fitparams[param][1]
                gmin = np.ones(2) * fitparams[param][0] - err
                gmax = np.ones(2) * fitparams[param][0] + err
                # xlim = ax.get_xlim()
                ax.fill_between(
                    tlim,
                    gmin,
                    gmax,
                    facecolor=colors[i],
                    zorder=2,
                    alpha=0.3,
                    label=label,
                )

            # All parameters with no special cases
            elif (
                (param != metal) and ("mass" not in param) and (param not in constants)
            ):
                ncol += 1
                # Set up the parameter-limit
                if param in fitparams:
                    err = fitparams[param][1]
                    parmin = fitparams[param][0] - err
                    parmax = fitparams[param][0] + err
                # If not regular fitparam, check if it is in filters
                elif param in filters:
                    errm = inputparams["magnitudes"][param]["errm"]
                    errp = inputparams["magnitudes"][param]["errp"]
                    parmin = inputparams["magnitudes"][param]["median"] - errm
                    parmax = inputparams["magnitudes"][param]["median"] + errp
                for track in tracks:
                    # For each track, check what indices is within
                    # the paramlimits
                    all_segments = np.where(
                        np.logical_and(
                            Grid[track + "/" + param][:] > parmin,
                            Grid[track + "/" + param][:] < parmax,
                        )
                    )[0]
                    # If none are, skip the track
                    if len(all_segments) == 0:
                        continue
                    # Call the plot function
                    label = plot_param(Grid, ax, track, all_segments, label, colors[i])

        # Highlight where frequencies are limited to
        # Calculation follows that of bastamain
        if fitfreqs["active"] and toggle_freqs:
            ncol += 1
            label = "Freq. constrain"
            dnufrac = fitfreqs.get("dnufrac", 0.15)
            obskey, obs, _ = fio.read_freq(fitfreqs["freqfile"])

            for track in tracks:
                libitem = Grid[track]
                index = np.ones(len(libitem["age"][:]), dtype=bool)

                # Locate where the lowest l=0 is within set limit
                for ind in np.where(index)[0]:
                    rawmod = libitem["osc"][ind]
                    rawmodkey = libitem["osckey"][ind]
                    mod = su.transform_obj_array(rawmod)
                    modkey = su.transform_obj_array(rawmodkey)
                    modkeyl0, modl0 = su.get_givenl(l=0, osc=mod, osckey=modkey)
                    # As mod is ordered, [0, 0] is the lowest l=0 mode
                    same_n = modkeyl0[1, :] == obskey[1, 0]
                    cl0 = modl0[0, same_n]
                    cl0 = cl0[0] if len(cl0 > 1) else cl0
                    if not (
                        (
                            cl0
                            >= (
                                obs[0, 0]
                                - max(
                                    (dnufrac / 2 * fitfreqs["dnufit"]),
                                    (3 * obs[1, 0]),
                                )
                            )
                        )
                        and ((cl0 - obs[0, 0]) <= (dnufrac * fitfreqs["dnufit"]))
                    ):
                        index[ind] = False

                # Plot the region
                if True in index:
                    all_segments = np.where(index)[0]
                    label = plot_param(Grid, ax, track, all_segments, label, "#AA3377")

        # General settings of plot
        ax.legend(
            bbox_to_anchor=(0.0, 1.02, 1.0, 0.102),
            loc=8,
            ncol=ncol,
            mode="expand",
            borderaxespad=0.0,
            title=nameinplot if nameinplot else "",
        )
        _, axlabels, _, _ = parameters.get_keys(["Teff", "logg"])
        ax.set_xlabel(axlabels[0])
        ax.set_ylabel(axlabels[1])
        ax.set_xlim(tlim)
        ax.set_ylim(glim)
        ax.invert_xaxis()
        ax.invert_yaxis()

    # Make list of metallicities in isochrones for annotation
    if "isochrones" in gridtype.lower():
        metal_list = np.asarray([Grid[track + "/" + metal][0] for track in tracks])

        # Assumes the lowest metallicity is at the highest Teff
        metal_list = np.sort(np.unique(metal_list))
        if len(metal_list) <= 5:
            metal_str = ", ".join([str(x) for x in list(metal_list)])
        else:
            metal_str = "{:.3f},...,{:.3f}".format(min(metal_list), max(metal_list))

        _, mlabel, _, _ = parameters.get_keys([metal])
        text = mlabel[0] + ": " + metal_str

        # The cases for single or divided plot
        if True in make_subplot:
            pos = [
                tefflim[1][1] - 0.03 * (tefflim[1][1] - tefflim[1][0]),
                logglim[1][0] + 0.06 * (logglim[1][1] - logglim[1][0]),
            ]
            axis[1].text(pos[0], pos[1], text, fontsize=12)
        else:
            pos = [
                tefflim[0][1] - 0.03 * (tefflim[0][1] - tefflim[0][0]),
                logglim[0][0] + 0.06 * (logglim[0][1] - logglim[0][0]),
            ]
            axis.text(pos[0], pos[1], text, fontsize=12)

    fig.tight_layout()

    return fig
