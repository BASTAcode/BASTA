"""
Production of Kiel diagrams
"""

import os

import matplotlib as mpl
import numpy as np

from basta import core, constants, stats
from basta import fileio as fio
from basta import utils_general as util
from basta import utils_seismic as su
from basta.downloader import get_basta_dir

# Set the style of all plots
mpl.use("Agg")
import matplotlib.pyplot as plt

plt.style.use(os.path.join(get_basta_dir(), "plots.mplstyle"))


def plot_param(
    grid,
    ax: mpl.axes.Axes,
    track: str,
    all_segments: np.ndarray,
    label: str,
    color: str,
) -> str:
    """
    Function for plotting the parameter interval in the Kiel diagram

    Parameters
    ----------
    grid : hdf5 object
        The already loaded grid, containing the tracks/isochrones
    ax : AxesSubplot object
        Axis in which to plot
    track : str
        Path for the current track/isochrone in the grid
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
    segment_breaks = np.where(np.diff(all_segments) != 1)[0]

    if len(segment_breaks) == 0:
        segments = [list(all_segments)]
    else:
        segment_breaks = np.append(segment_breaks, len(all_segments) - 1)
        segments = [
            list(all_segments[start : end + 1])
            for start, end in zip(np.append(0, segment_breaks[:-1] + 1), segment_breaks)
        ]

    # Plot dummy, so legend entry becomes a line
    dummy_line_plotted = False

    for segment in segments:
        # Determine if the segment is a line or a single point
        is_single_point = len(segment) < 2
        plot_type = "." if is_single_point else "-"
        current_label = label

        if is_single_point and label != "_nolegend_":
            dummy_line_plotted = True
            ax.plot([0, 0], [0, 0], "-", alpha=0.5, lw=3, color=color, label=label)
            current_label = "_nolegend_"

        ax.plot(
            grid[track + "/Teff"][segment],
            grid[track + "/logg"][segment],
            plot_type,
            lw=3,
            markersize=6,
            color=color,
            zorder=3,
            alpha=0.5,
            label=current_label,
        )

        # Label magic to limit the legend to having only a single line
        # entry per parameter
        if current_label != "_nolegend_" or dummy_line_plotted:
            label = "_nolegend_"

    return label


def calculate_limits(
    value: float, err_m: float, err_p: float, nsigma: int = 2
) -> tuple[float, float]:
    return value - nsigma * err_m, value + nsigma * err_p


def kiel(
    grid,
    selectedmodels,
    star: core.Star,
    inferencesettings: core.InferenceSettings,
    plotconfig: core.PlotConfig,
    outputoptions: core.OutputOptions,
    lp_interval: list[float],
    feh_interval: list[float],
    Teffout: list[float],
    loggout: list[float],
    gridtype: str,
    nameinplot: bool = False,
    color_by_likelihood: bool = False,
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
    grid : hdf5 object
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
    if outputoptions.developermode:
        print("\nACTIVATED EXPERIMENTAL FEATURE:")
        print(
            "Extending the selection ranges (from the default quantiles)",
            "in the Kiel diagram!\n",
        )
        scalefactor = 0.3
        lp_interval[0] *= 1 - scalefactor
        lp_interval[1] *= 1 + scalefactor

    # Assign params
    fitparams = inferencesettings.fitparams
    filters = (
        list(star.absolutemagnitudes["magnitudes"].keys())
        if inferencesettings.has_distance_case and star.absolutemagnitudes is not None
        else []
    )
    constant_parameters = ["alphaFe", "ove", "gcut", "eta", "alphaMLT"]
    metal = "MeH" if "MeH" in fitparams else "FeH"

    # Save the tracks in selectedmodels with appropriate massini and FeH
    tracks = []
    for modelpath in selectedmodels:
        trackvalue = (
            grid[modelpath]["massini"][0]
            if "tracks" in gridtype.lower()
            else grid[modelpath]["age"][0]
        )

        if lp_interval[0] <= trackvalue <= lp_interval[1]:
            track_pass = all(
                lp_interval[0] <= grid[modelpath + f"/{param}"][0] <= lp_interval[1]
                for param in constant_parameters
                if param in fitparams
            )
            if track_pass:
                metal_in_track = np.where(
                    (grid[modelpath + f"/{metal}"][:] >= feh_interval[0])
                    & (grid[modelpath + f"/{metal}"][:] <= feh_interval[1])
                )[0]
                if metal_in_track.size > 0:
                    tracks.append(modelpath)

    # Median teff and logg
    teff, terrm, terrp = Teffout[0], Teffout[0] - Teffout[1], Teffout[2] - Teffout[0]
    logg, lerrm, lerrp = loggout[0], loggout[0] - loggout[1], loggout[2] - loggout[0]

    # The highest likelihood is used to control the plot below. If desired, the
    # model with lowest chi^2 can be extracted and added to the plot
    if outputoptions.validationmode:
        minchi2_path, minchi2_ind = stats.lowest_chi2(selectedmodels)
        hlm_chi2, lcm_chi2 = stats.chi_for_plot(selectedmodels)

    # Define the max-likelihood model, and define teff/logg
    # intervals for limit control
    maxPDF_path, maxPDF_ind = stats.most_likely(selectedmodels)
    if maxPDF_path not in tracks:
        tracks.append(maxPDF_path)

    teffrange = [min(grid[maxPDF_path + "/Teff"]), max(grid[maxPDF_path + "/Teff"])]
    loggrange = [min(grid[maxPDF_path + "/logg"]), max(grid[maxPDF_path + "/logg"])]

    teffrange = list(calculate_limits(teff, terrm, terrp))
    loggrange = list(calculate_limits(logg, lerrm, lerrp))

    # Limits for full track, adjusted to even values
    tefflim = [
        100 * np.floor(teffrange[0] / 100) - 200,
        100 * np.ceil(teffrange[1] / 100) + 200,
    ]
    logglim = [0.1 * np.floor(loggrange[0] / 0.1), 0.1 * np.ceil(loggrange[1] / 0.1)]

    # "Standard" value for span in axis
    teff_std, logg_std = (350, 0.5) if "tracks" in gridtype.lower() else (150, 0.3)
    make_subplot = [
        (teffrange[1] - teffrange[0]) > teff_std * 2,
        (loggrange[1] - loggrange[0]) > logg_std * 2,
    ]

    # If the range is too large, make a zoomed subplot, keep ratio of original
    if any(make_subplot):
        tefflim_sub = [
            100 * np.floor((teff - teff_std) / 100),
            100 * np.ceil((teff + teff_std) / 100),
        ]
        ratio = (logglim[1] - logglim[0]) / (tefflim[1] - tefflim[0])
        logglim_sub = [
            logg - 0.5 * (tefflim_sub[1] - tefflim_sub[0]) * ratio,
            logg + 0.5 * (tefflim_sub[1] - tefflim_sub[0]) * ratio,
        ]
        tefflim = [tefflim_sub, tefflim]
        logglim = [logglim_sub, logglim]
    else:
        tefflim = [tefflim]
        logglim = [logglim]

    # Make list with both limits
    tefflim = [tefflim_sub, tefflim] if True in make_subplot else [tefflim]
    logglim = [logglim_sub, logglim] if True in make_subplot else [logglim]

    # Get labels and colors for sorted params
    keys = [
        k
        for k in fitparams + filters
        if k not in constants.freqtypes.alltypes and k != "parallax"
    ]
    sorted_parameters = np.array(keys)[np.argsort(keys)]
    _, labels, _, colors = constants.parameters.get_keys(sorted_parameters)

    fig, axes = (
        plt.subplots(2, 1, figsize=(12.8, 17.6))
        if any(make_subplot)
        else plt.subplots(1, 1, figsize=(8.47, 6))
    )
    axes = axes if isinstance(axes, np.ndarray) else [axes]

    for ax, tlim, glim in zip(axes, tefflim, logglim):
        max_logPDF = selectedmodels[maxPDF_path].logPDF.max()
        for track in tracks:
            # Make a copy to allow manipulation
            xs = util.h5py_to_array(grid[track + "/Teff"])
            ys = util.h5py_to_array(grid[track + "/logg"])

            # Special treatment to plot points color-coded by likelihood
            if color_by_likelihood:
                # Extract pdf information
                logpdf = np.zeros_like(xs) + 0.1
                m = selectedmodels[track]
                logpdf[m.index] = 0.2 + 0.5 * np.exp(m.logPDF - max_logPDF)

                # Make segments to colorcode
                points = np.transpose([xs, ys]).reshape(-1, 1, 2)
                segments = np.concatenate([points[:-1], points[1:]], axis=1)
                lc = mpl.collections.LineCollection(segments, cmap="gray_r")
                lc.set_array(logpdf)
                lc.set_linewidth(1)
                ax.add_collection(lc)
                continue

            # Plot as points for validation mode
            if outputoptions.validationmode:
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
        if outputoptions.validationmode:
            bfmmodlab = f"Highest likelihood model (chi2 = {hlm_chi2:1.4e})"
        else:
            bfmmodlab = "Best fit model"
        ax.plot(
            grid[maxPDF_path + "/Teff"][maxPDF_ind],
            grid[maxPDF_path + "/logg"][maxPDF_ind],
            "*",
            color="#000000",
            markersize=20,
            zorder=5,
            label=bfmmodlab,
        )
        ax.plot(
            teff, logg, "o", color="k", markersize=15, zorder=np.inf, label="Median"
        )

        # Add chi^2 model?
        if outputoptions.validationmode:
            ax.plot(
                grid[minchi2_path + "/Teff"][minchi2_ind],
                grid[minchi2_path + "/logg"][minchi2_ind],
                "p",
                color="k",
                markersize=15,
                label=f"Lowest chi^2 model (chi2 = {lcm_chi2:1.4e})",
            )

        # Plot parameter intervals of fitparams
        ncol = 2
        for i, param in enumerate(sorted_parameters):
            label = labels[i]
            if param == "Teff":
                ncol += 1
                val, err = star.classicalparams.params[param]
                Tmin = np.ones(2) * val - err
                Tmax = np.ones(2) * val + err
                ax.fill_betweenx(
                    glim[0],
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
                val, err = star.classicalparams.params[param]
                gmin = np.ones(2) * val - err
                gmax = np.ones(2) * val + err
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
                (param != metal)
                and ("mass" not in param)
                and (param not in constant_parameters)
            ):
                ncol += 1
                # Set up the parameter-limit
                if param in star.globalseismicparams.params.keys():
                    val, err = star.globalseismicparams.get_scaled(param)
                    parmin = val - err
                    parmax = val + err
                # If not regular fitparam, check if it is in filters
                elif param in filters:
                    assert star.absolutemagnitudes is not None
                    errm = star.absolutemagnitudes["magnitudes"][param]["errm"]
                    errp = star.absolutemagnitudes["magnitudes"][param]["errp"]
                    med = star.absolutemagnitudes["magnitudes"][param]["median"]
                    parmin = med - errm
                    parmax = med + errp
                else:
                    val, err = star.classicalparams.params[param]
                    parmin = val - err
                    parmax = val + err
                for track in tracks:
                    # For each track, check what indices is within
                    # the paramlimits
                    all_segments = np.where(
                        np.logical_and(
                            grid[track + "/" + param][:] > parmin,
                            grid[track + "/" + param][:] < parmax,
                        )
                    )[0]
                    # If none are, skip the track
                    if len(all_segments) == 0:
                        continue
                    # Call the plot function
                    label = plot_param(grid, ax, track, all_segments, label, colors[i])

        # Highlight where frequencies are limited to
        # Calculation follows that of bastamain
        # TODO(Amalie) This can be simplified
        # TODO(Amalie) Why is this repeated here?
        if inferencesettings.has_frequencies:
            ncol += 1
            label = "Freq. constrain"
            assert star.modes is not None
            obskey = np.asarray([star.modes.modes.l, star.modes.modes.n])
            obs = np.asarray([star.modes.modes.frequencies, star.modes.modes.errors])

            for track in tracks:
                libitem = grid[track]
                index = np.ones(len(libitem["age"][:]), dtype=bool)

                # Locate where the lowest l=0 is within set limit
                index = util.apply_anchor_cut(
                    index=index,
                    star=star,
                    libitem=libitem,
                    inferencesettings=inferencesettings,
                )
                """
                # TODO(Amalie) Why is this code repeated in here?
                for ind in range(len(libitem["age"][:])):
                    rawmod = libitem["osc"][ind]
                    rawmodkey = libitem["osckey"][ind]
                    mod = su.transform_obj_array(rawmod)
                    modkey = su.transform_obj_array(rawmodkey)
                    modkeyl0, modl0 = su.get_givenl(l=0, osc=mod, osckey=modkey)
                    # As mod is ordered, [0, 0] is the lowest l=0 mode
                    same_n = modkeyl0[1, :] == obskey[1, 0]
                    cl0 = modl0[0, same_n]
                    if cl0.size == 0:
                        continue
                    elif cl0.size > 1:
                        cl0 = cl0[0]

                    cl0 = cl0.item()
                    anchordist = cl0 - obs[0, 0]
                    dnutype = "dnufit"
                    dnufrac = inferencesettings.boxpriors["dnufrac"].kwargs[dnutype]
                    dnu = star.globalseismicparams.get_scaled(dnutype)[0]
                    lower_threshold = -max(dnufrac / 2 * dnu, 3 * obs[1, 0])
                    upper_threshold = dnufrac * dnu
                    index.append(lower_threshold < anchordist <= upper_threshold)
                """

                # Plot the region
                if True in index:
                    all_segments = np.where(index)[0]
                    label = plot_param(grid, ax, track, all_segments, label, "#AA3377")

        # General settings of plot
        ax.legend(
            bbox_to_anchor=(0.0, 1.02, 1.0, 0.102),
            loc=8,
            ncol=ncol,
            mode="expand",
            borderaxespad=0.0,
            title=nameinplot if nameinplot else "",
        )
        _, axlabels, _, _ = constants.parameters.get_keys(["Teff", "logg"])
        ax.set_xlabel(axlabels[0])
        ax.set_ylabel(axlabels[1])
        ax.set_xlim(tlim[0])
        ax.set_ylim(glim[0])
        ax.invert_xaxis()
        ax.invert_yaxis()

    # Make list of metallicities in isochrones for annotation
    if "isochrones" in gridtype.lower():
        metal_list = np.asarray([grid[track + "/" + metal][0] for track in tracks])

        # Assumes the lowest metallicity is at the highest Teff
        metal_list = np.sort(np.unique(metal_list))
        if len(metal_list) <= 5:
            metal_str = ", ".join([str(x) for x in list(metal_list)])
        else:
            metal_str = f"{min(metal_list):.3f},...,{max(metal_list):.3f}"

        _, mlabel, _, _ = constants.parameters.get_keys([metal])
        text = mlabel[0] + ": " + metal_str

        # The cases for single or divided plot
        if True in make_subplot:
            pos = [
                tefflim[1][1] - 0.03 * (tefflim[1][1] - tefflim[1][0]),
                logglim[1][0] + 0.06 * (logglim[1][1] - logglim[1][0]),
            ]
            axes[1].text(pos[0], pos[1], text, fontsize=12)
        else:
            pos = [
                tefflim[0][1] - 0.03 * (tefflim[0][1] - tefflim[0][0]),
                logglim[0][0] + 0.06 * (logglim[0][1] - logglim[0][0]),
            ]
            axes.text(pos[0], pos[1], text, fontsize=12)

    fig.tight_layout()

    return fig
