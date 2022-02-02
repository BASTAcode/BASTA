"""
Production of asteroseismic plots
"""
import os
import numpy as np
import matplotlib

import basta.fileio as fio
from basta import utils_seismic as su
from basta import stats, freq_fit

# Set the style of all plots
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

plt.style.use(os.path.join(os.environ["BASTADIR"], "basta/plots.mplstyle"))

# Define a color dictionary for easier change of color
colors = {"l0": "#D55E00", "l1": "#009E73", "l2": "#0072B2"}


def duplicateechelle(
    selectedmodels,
    Grid,
    freqfile,
    mod=None,
    modkey=None,
    dnu=None,
    join=None,
    joinkeys=False,
    coeffs=None,
    scalnu=None,
    freqcor="BG14",
    output=None,
):
    """
    Echelle diagram that is place twice so patterns across the moduli-limit
    is easier seen.

    Parameters
    ----------
    selectedmodels : dict
        Contains information on all models with a non-zero likelihood.
    Grid : hdf5 object
        The already loaded grid, containing the tracks/isochrones.
    freqfile : str
        Name of file containing frequencies
    mod : array or None
        Array of modes in a given model.
        If None, `mod` will be found from the most likely model in
        `selectedmodels`.
    modkey : array or None
        Array of mode identification of modes from the model.
        If None, `modkey` will be found from the most likely model in
        `selectedmodels`.
    dnu : float or None
        The large frequency separation. If None, `dnufit` will be used.
    join : array or None
        Array containing the matched observed and modelled modes.
        If None, this information is not added to the plot.
    joinkeys : array or None
        Array containing the mode identification of the matched observed
        and modelled modes.
        If None, this information is not added to the plot.
    coeffs : array or None
        Coefficients for the near-surface frequency correction specified
        in `freqcor`. If None, the frequencies on the plot will not
        be corrected.
    scalnu : float or None
        Value used to scale frequencies in frequencies correction.
        `numax` is often used.
    freqcor : str {'None', 'HK08', 'BG14', 'cubicBG14'}
        Flag determining the frequency correction.
    output : str or None
        Filename for saving the figure.
    """
    if dnu is None:
        print("Note: No deltanu specified, using dnufit for echelle diagram.")
        maxPDF_path, maxPDF_ind = stats.most_likely(selectedmodels)
        dnu = Grid[maxPDF_path + "/dnufit"][maxPDF_ind]
    if (mod is None) and (modkey is None):
        maxPDF_path, maxPDF_ind = stats.most_likely(selectedmodels)
        rawmod = Grid[maxPDF_path + "/osc"][maxPDF_ind]
        rawmodkey = Grid[maxPDF_path + "/osckey"][maxPDF_ind]
        mod = su.transform_obj_array(rawmod)
        modkey = su.transform_obj_array(rawmodkey)
        mod = mod[:, modkey[0, :] < 2.5]
        modkey = modkey[:, modkey[0, :] < 2.5]
    cormod = np.copy(mod)
    if coeffs is not None:
        if freqcor == "HK08":
            corosc = freq_fit.apply_HK08(
                modkey=modkey, mod=mod, coeffs=coeffs, scalnu=scalnu
            )
            cormod[0, :] = corosc
        elif freqcor == "BG14":
            corosc = freq_fit.apply_BG14(
                modkey=modkey, mod=mod, coeffs=coeffs, scalnu=scalnu
            )
            cormod[0, :] = corosc
        elif freqcor == "cubicBG14":
            corosc = freq_fit.apply_cubicBG14(
                modkey=modkey, mod=mod, coeffs=coeffs, scalnu=scalnu
            )
            cormod[0, :] = corosc

    obskey, obs, _ = fio.read_freq(freqfile, nottrustedfile=None)
    _, modl0 = su.get_givenl(l=0, osc=cormod, osckey=modkey)
    _, modl1 = su.get_givenl(l=1, osc=cormod, osckey=modkey)
    _, modl2 = su.get_givenl(l=2, osc=cormod, osckey=modkey)
    fmodl0_all = modl0[0, :] / dnu
    fmodl1_all = modl1[0, :] / dnu
    fmodl2_all = modl2[0, :] / dnu
    s = su.scale_by_inertia(modkey, cormod)

    # Get all observation including the not-trusted ones
    obskey, obs, _ = fio.read_freq(freqfile, nottrustedfile=None)
    _, obsl0 = su.get_givenl(l=0, osc=obs, osckey=obskey)
    _, obsl1 = su.get_givenl(l=1, osc=obs, osckey=obskey)
    _, obsl2 = su.get_givenl(l=2, osc=obs, osckey=obskey)
    fobsl0_all = obsl0[0, :] / dnu
    fobsl1_all = obsl1[0, :] / dnu
    fobsl2_all = obsl2[0, :] / dnu
    eobsl0_all = obsl0[1, :] / dnu
    eobsl1_all = obsl1[1, :] / dnu
    eobsl2_all = obsl2[1, :] / dnu

    if join is not None:
        _, joinl0 = su.get_givenl(l=0, osc=join, osckey=joinkeys)
        _, joinl1 = su.get_givenl(l=1, osc=join, osckey=joinkeys)
        _, joinl2 = su.get_givenl(l=2, osc=join, osckey=joinkeys)

        fmodl0 = joinl0[0, :] / dnu
        fmodl1 = joinl1[0, :] / dnu
        fmodl2 = joinl2[0, :] / dnu
        fobsl0 = joinl0[2, :] / dnu
        fobsl1 = joinl1[2, :] / dnu
        fobsl2 = joinl2[2, :] / dnu
        eobsl0 = joinl0[3, :] / dnu
        eobsl1 = joinl1[3, :] / dnu
        eobsl2 = joinl2[3, :] / dnu
        sjoin = su.scale_by_inertia(joinkeys[0:2], join[0:2])

    # Create plot
    fig, ax1 = plt.subplots()
    ax2 = ax1.twinx()
    # Plot something to set the scale on one y-axis
    ax1.errorbar(
        fobsl0_all % 1,
        fobsl0_all * dnu,
        xerr=eobsl0_all,
        fmt="o",
        mfc=colors["l0"],
        ecolor=colors["l0"],
        alpha=0.5,
        zorder=0,
    )

    ax2.axvline(x=0, linestyle="--", color="0.8", zorder=0)

    # Plot all observed modes
    if len(fobsl0_all) != 0:
        ax2.errorbar(
            fobsl0_all % 1,
            fobsl0_all,
            xerr=eobsl0_all,
            fmt="o",
            mfc=colors["l0"],
            ecolor=colors["l0"],
            zorder=1,
            alpha=0.5,
        )
        ax2.errorbar(
            fobsl0_all % 1 - 1,
            fobsl0_all,
            xerr=eobsl0_all,
            fmt="o",
            mfc=colors["l0"],
            ecolor=colors["l0"],
            zorder=1,
            alpha=0.5,
        )
    if len(fobsl1_all) != 0:
        ax2.errorbar(
            fobsl1_all % 1,
            fobsl1_all,
            xerr=eobsl1_all,
            fmt="o",
            mfc=colors["l1"],
            ecolor=colors["l1"],
            zorder=1,
            alpha=0.5,
        )
        ax2.errorbar(
            fobsl1_all % 1 - 1,
            fobsl1_all,
            xerr=eobsl1_all,
            fmt="o",
            mfc=colors["l1"],
            ecolor=colors["l1"],
            zorder=1,
            alpha=0.5,
        )
    if len(fobsl2_all) != 0:
        ax2.errorbar(
            fobsl2_all % 1,
            fobsl2_all,
            xerr=eobsl2_all,
            fmt="o",
            mfc=colors["l2"],
            ecolor=colors["l2"],
            zorder=1,
            alpha=0.5,
        )
        ax2.errorbar(
            fobsl2_all % 1 - 1,
            fobsl2_all,
            xerr=eobsl2_all,
            fmt="o",
            mfc=colors["l2"],
            ecolor=colors["l2"],
            zorder=1,
            alpha=0.5,
        )

    # Plot all modes in the model
    ax2.scatter(
        fmodl0_all % 1,
        fmodl0_all,
        s=s[0],
        c=colors["l0"],
        marker="D",
        alpha=0.5,
        zorder=2,
    )
    ax2.scatter(
        fmodl1_all % 1,
        fmodl1_all,
        s=s[1],
        c=colors["l1"],
        marker="^",
        alpha=0.5,
        zorder=2,
    )
    ax2.scatter(
        fmodl2_all % 1,
        fmodl2_all,
        s=s[2],
        c=colors["l2"],
        marker="v",
        alpha=0.5,
        zorder=2,
    )
    ax2.scatter(
        fmodl0_all % 1 - 1,
        fmodl0_all,
        s=s[0],
        c=colors["l0"],
        marker="D",
        alpha=0.5,
        zorder=2,
    )
    ax2.scatter(
        fmodl1_all % 1 - 1,
        fmodl1_all,
        s=s[1],
        c=colors["l1"],
        marker="^",
        alpha=0.5,
        zorder=2,
    )
    ax2.scatter(
        fmodl2_all % 1 - 1,
        fmodl2_all,
        s=s[2],
        c=colors["l2"],
        marker="v",
        alpha=0.5,
        zorder=2,
    )

    # Plot the matched modes in negative and positive side
    if join is not None:
        if len(fmodl0) != 0:
            ax2.scatter(
                fmodl0 % 1 - 1,
                fmodl0,
                s=sjoin[0],
                c=colors["l0"],
                marker="D",
                linewidths=1,
                edgecolors="k",
                zorder=3,
            )
            ax2.scatter(
                fmodl0 % 1,
                fmodl0,
                s=sjoin[0],
                c=colors["l0"],
                marker="D",
                linewidths=1,
                edgecolors="k",
                zorder=3,
                label=r"Best fit $l=0$",
            )
            ax2.errorbar(
                fobsl0 % 1,
                fobsl0,
                xerr=eobsl0,
                fmt="o",
                mfc=colors["l0"],
                ecolor=colors["l0"],
                zorder=1,
                label=r"Measured $l=0$",
            )
            ax2.errorbar(
                fobsl0 % 1 - 1,
                fobsl0,
                xerr=eobsl0,
                fmt="o",
                mfc=colors["l0"],
                ecolor=colors["l0"],
                zorder=1,
            )
        if len(fmodl1) != 0:
            ax2.scatter(
                fmodl1 % 1 - 1,
                fmodl1,
                s=sjoin[1],
                c=colors["l1"],
                marker="^",
                linewidths=1,
                edgecolors="k",
                zorder=3,
            )
            ax2.scatter(
                fmodl1 % 1,
                fmodl1,
                s=sjoin[1],
                c=colors["l1"],
                marker="^",
                linewidths=1,
                edgecolors="k",
                zorder=3,
                label=r"Best fit $l=1$",
            )
            ax2.errorbar(
                fobsl1 % 1,
                fobsl1,
                xerr=eobsl1,
                fmt="o",
                mfc=colors["l1"],
                ecolor=colors["l1"],
                zorder=1,
                label=r"Measured $l=1$",
            )
            ax2.errorbar(
                fobsl1 % 1 - 1,
                fobsl1,
                xerr=eobsl1,
                fmt="o",
                mfc=colors["l1"],
                ecolor=colors["l1"],
                zorder=1,
            )
        if len(fmodl2) != 0:
            ax2.scatter(
                fmodl2 % 1 - 1,
                fmodl2,
                s=sjoin[2],
                c=colors["l2"],
                marker="v",
                linewidths=1,
                edgecolors="k",
                zorder=3,
            )
            ax2.scatter(
                fmodl2 % 1,
                fmodl2,
                s=sjoin[2],
                c=colors["l2"],
                marker="v",
                linewidths=1,
                edgecolors="k",
                zorder=3,
                label=r"Best fit $l=2$",
            )
            ax2.errorbar(
                fobsl2 % 1,
                fobsl2,
                xerr=eobsl2,
                fmt="o",
                mfc=colors["l2"],
                ecolor=colors["l2"],
                zorder=1,
                label=r"Measured $l=2$",
            )
            ax2.errorbar(
                fobsl2 % 1 - 1,
                fobsl2,
                xerr=eobsl2,
                fmt="o",
                mfc=colors["l2"],
                ecolor=colors["l2"],
                zorder=1,
            )

        # Make line segments connecting the observed and associated mode
        linelimit = 0.75
        for l, (fmod, fobs) in enumerate(
            zip([fmodl0, fmodl1, fmodl2], [fobsl0, fobsl1, fobsl2])
        ):
            for i in range(len(fmod)):
                if ((fmod[i] % 1) > linelimit) & ((fobs[i] % 1) < (1 - linelimit)):
                    ax2.plot(
                        (fmod[i] % 1 - 1, fobs[i] % 1),
                        (fmod[i], fobs[i]),
                        c=colors["l" + str(l)],
                        alpha=0.7,
                        zorder=1,
                    )
                elif ((fobs[i] % 1) > linelimit) & ((fmod[i] % 1) < (1 - linelimit)):
                    ax2.plot(
                        (fobs[i] % 1 - 1, fmod[i] % 1),
                        (fobs[i], fmod[i]),
                        c=colors["l" + str(l)],
                        alpha=0.7,
                        zorder=1,
                    )
                else:
                    ax2.plot(
                        (fmod[i] % 1, fobs[i] % 1),
                        (fmod[i], fobs[i]),
                        c=colors["l" + str(l)],
                        alpha=0.7,
                        zorder=1,
                    )
                    ax2.plot(
                        (fmod[i] % 1 - 1, fobs[i] % 1 - 1),
                        (fmod[i], fobs[i]),
                        c=colors["l" + str(l)],
                        alpha=0.7,
                        zorder=1,
                    )

    ax1.set_xlim([-1, 1])
    ax1.set_ylim(ax2.set_ylim()[0] * dnu, ax2.set_ylim()[1] * dnu)
    lgnd = ax2.legend(
        bbox_to_anchor=(0.0, 1.02, 1.0, 0.102),
        loc=8,
        ncol=6,
        mode="expand",
        borderaxespad=0.0,
    )
    for i in range(len(lgnd.legendHandles)):
        lgnd.legendHandles[i]._sizes = [50]
    ax2.set_ylabel(r"Frequency normalised by $\Delta \nu$")
    ax1.set_xlabel(
        r"Frequency normalised by $\Delta \nu$ modulo 1 ($\Delta \nu =$%s $\mu$Hz)"
        % dnu
    )
    ax1.set_ylabel(r"Frequency ($\mu$Hz)")

    if output is not None:
        plt.savefig(output, bbox_inches="tight")
        print("Saved figure to " + output)


def ratioplot(freqfile, datos, joinkeys, join, output=None, nonewfig=False):
    """
    Plot frequency ratios.

    Parameters
    ----------
    freqfile : str
        Name of file containing frequencies and ratio types
    datos : array
        Individual frequencies, uncertainties, and combinations read
        directly from the observational input files
    joinkeys : array
        Array containing the mode identification of the matched observed
        and modelled modes.
    join : array
        Array containing the matched observed and modelled modes.
    output : str or None, optional
        Filename for saving the plot. MUST BE PDF!
    nonewfig : bool, optional
        If True, this creates a new canvas. Otherwise, the plot is added
        to the existing canvas.
    """
    # Load input ratios
    orders, ratio, ratio_types, errors, errors_m, errors_p = fio.read_ratios_xml(
        freqfile
    )
    f, f_err, f_n, f_l = fio.read_freqs_xml(freqfile)

    # Observed ratios (prefer ratios from xml file)
    # r02
    if b"r02" in ratio_types:
        nr = orders[ratio_types == b"r02"]
        datos02 = np.zeros((3, len(nr)))
        datos02[0, :] = ratio[ratio_types == b"r02"]
        datos02[2, :] = errors[ratio_types == b"r02"]
        for i, n in enumerate(nr):
            ind = (f_n == n) & (f_l == 0)
            datos02[1, i] = f[ind]
    else:
        datos02 = datos[1]

    # r01
    if b"r01" in ratio_types:
        nr = orders[ratio_types == b"r01"]
        datos01 = np.zeros((3, len(nr)))
        datos01[0, :] = ratio[ratio_types == b"r01"]
        datos01[2, :] = errors[ratio_types == b"r01"]
        for i, n in enumerate(nr):
            ind = (f_n == n) & (f_l == 0)
            datos01[1, i] = f[ind]
    else:
        datos01 = datos[3]

    # r10
    if b"r10" in ratio_types:
        nr = orders[ratio_types == b"r10"]
        datos10 = np.zeros((3, len(nr)))
        datos10[0, :] = ratio[ratio_types == b"r10"]
        datos10[2, :] = errors[ratio_types == b"r10"]
        for i, n in enumerate(nr):
            ind = (f_n == n) & (f_l == 1)
            datos10[1, i] = f[ind]
    else:
        datos10 = datos[4]

    # Best fit model ratios
    frq = np.zeros((joinkeys[:, joinkeys[0, :] < 3].shape[1], 4))
    frq[:, 0] = joinkeys[0, joinkeys[0, :] < 3]
    frq[:, 1] = joinkeys[1, joinkeys[0, :] < 3]
    frq[:, 2] = join[0, joinkeys[0, :] < 3]
    r02, r01, r10 = freq_fit.ratios(frq)

    if r02 is None:
        print("WARNING: missing radial orders! Skipping ratios plot.")
    else:
        # Plotting...
        if output is not None:
            pp = PdfPages(output)

        for ratio_type in ["r02", "r01", "r10"]:
            if ratio_type == "r02":
                obsratio = datos02
                modratio = r02
            elif ratio_type == "r01":
                obsratio = datos01
                modratio = r01
            elif ratio_type == "r10":
                obsratio = datos10
                modratio = r10

            # Open figure and set style
            if nonewfig is False:
                plt.figure()

            ax1 = plt.subplot(2, 1, 1)
            ax1.plot(
                modratio[:, 3], modratio[:, 1], "*", markersize=20, label="Best fit"
            )
            ax1.errorbar(
                obsratio[1, :],
                obsratio[0, :],
                yerr=obsratio[2, :],
                marker="o",
                linestyle="None",
                label="Measured",
            )
            ax1.legend(frameon=False)
            ax1.set_ylabel('Ratio type "{0}"'.format(ratio_type[1:]))
            ax1.tick_params(axis="x", labelbottom=False)

            ax2 = plt.subplot(2, 1, 2, sharex=ax1)
            ax2.axhline(y=0.0, color="black", linestyle=":")
            ax2.plot(
                obsratio[1, :],
                (obsratio[0, :] - modratio[:, 1]) / obsratio[2, :],
                "d",
                color="black",
                markersize=10,
            )
            ax2.set_xlabel(r"Frequency ($\mu$Hz)")
            ax2.set_ylabel(r"Standardized residuals")

            if output is not None:
                pp.savefig(bbox_inches="tight")

        if output is not None:
            print("Saved figure to " + output)
            pp.close()


def glitchplot(datos, rt, selectedmodels, output=None, nonewfig=False):
    """
    Plot glitch parameters.

    Parameters
    ----------
    datos : array
        Individual frequencies, uncertainties, and combinations read
        directly from the observational input files
    rt : list
        Type of fits available for individual frequencies
    selectedmodels : dict
        Contains information on all models with a non-zero likelihood.
    output : str or None, optional
        Filename for saving the plot. MUST BE PDF!
    nonewfig : bool, optional
        If True, this creates a new canvas. Otherwise, the plot is added
        to the existing canvas.
    """
    if "glitches" in rt:
        obs_amp, obs_width, obs_depth = (
            datos[7][0, -3],
            datos[7][0, -2],
            datos[7][0, -1],
        )
        err_amp, err_width, err_depth = (
            datos[7][2, -3],
            datos[7][2, -2],
            datos[7][2, -1],
        )
    if "gr010" in rt:
        obs_amp, obs_width, obs_depth = (
            datos[8][0, -3],
            datos[8][0, -2],
            datos[8][0, -1],
        )
        err_amp, err_width, err_depth = (
            datos[8][2, -3],
            datos[8][2, -2],
            datos[8][2, -1],
        )
    if "gr02" in rt:
        obs_amp, obs_width, obs_depth = (
            datos[9][0, -3],
            datos[9][0, -2],
            datos[9][0, -1],
        )
        err_amp, err_width, err_depth = (
            datos[9][2, -3],
            datos[9][2, -2],
            datos[9][2, -1],
        )
    if "gr01" in rt:
        obs_amp, obs_width, obs_depth = (
            datos[10][0, -3],
            datos[10][0, -2],
            datos[10][0, -1],
        )
        err_amp, err_width, err_depth = (
            datos[10][2, -3],
            datos[10][2, -2],
            datos[10][2, -1],
        )
    if "gr10" in rt:
        obs_amp, obs_width, obs_depth = (
            datos[11][0, -3],
            datos[11][0, -2],
            datos[11][0, -1],
        )
        err_amp, err_width, err_depth = (
            datos[11][2, -3],
            datos[11][2, -2],
            datos[11][2, -1],
        )
    if "gr012" in rt:
        obs_amp, obs_width, obs_depth = (
            datos[12][0, -3],
            datos[12][0, -2],
            datos[12][0, -1],
        )
        err_amp, err_width, err_depth = (
            datos[12][2, -3],
            datos[12][2, -2],
            datos[12][2, -1],
        )
    if "gr102" in rt:
        obs_amp, obs_width, obs_depth = (
            datos[13][0, -3],
            datos[13][0, -2],
            datos[13][0, -1],
        )
        err_amp, err_width, err_depth = (
            datos[13][2, -3],
            datos[13][2, -2],
            datos[13][2, -1],
        )

    # Highest probability (or best-fit) model parameters
    # maxPDF_path, maxPDF_ind = stats.most_likely(selectedmodels)
    maxPDF = -np.inf
    for path, trackstats in selectedmodels.items():
        i = np.argmax(trackstats.logPDF)
        if trackstats.logPDF[i] > maxPDF:
            maxPDF = trackstats.logPDF[i]
            best_amp, best_width, best_depth = trackstats.glhparams[i, :]
    if maxPDF == -np.inf:
        print("Warning: The logPDF are all -np.inf! Skipping glitches plot.")
    else:
        # Plotting...
        if output is not None:
            pp = PdfPages(output)

        for i in range(3):

            # Open figure and set style
            if nonewfig is False:
                plt.figure()

            if i == 0:
                for path, trackstats in selectedmodels.items():
                    glhparams = trackstats.glhparams
                    glhparams = glhparams[glhparams[:, 0] > 1e-14, :]
                    plt.plot(
                        glhparams[:, 1],
                        glhparams[:, 0],
                        ".",
                        ms=5,
                        color="grey",
                        zorder=0,
                    )
                plt.errorbar(
                    obs_width,
                    obs_amp,
                    xerr=err_width,
                    yerr=err_amp,
                    marker="o",
                    linestyle="None",
                    color="#D55E00",
                    zorder=1,
                    label="Measured",
                )
                plt.plot(
                    best_width,
                    best_amp,
                    "*",
                    ms=20,
                    color="#56B4E9",
                    zorder=2,
                    label="Best fit",
                )
                plt.legend(frameon=False)
                plt.xlabel(r"Acoustic width (s)")
                plt.ylabel(r"Average amplitude ($\mu$Hz)")
            elif i == 1:
                for path, trackstats in selectedmodels.items():
                    glhparams = trackstats.glhparams
                    glhparams = glhparams[glhparams[:, 0] > 1e-14, :]
                    plt.plot(
                        glhparams[:, 2],
                        glhparams[:, 0],
                        ".",
                        ms=5,
                        color="grey",
                        zorder=0,
                    )
                plt.errorbar(
                    obs_depth,
                    obs_amp,
                    xerr=err_depth,
                    yerr=err_amp,
                    marker="o",
                    linestyle="None",
                    color="#D55E00",
                    zorder=1,
                    label="Measured",
                )
                plt.plot(
                    best_depth,
                    best_amp,
                    "*",
                    ms=20,
                    color="#56B4E9",
                    zorder=2,
                    label="Best fit",
                )
                plt.legend(frameon=False)
                plt.xlabel(r"Acoustic depth (s)")
                plt.ylabel(r"Average amplitude ($\mu$Hz)")
            elif i == 2:
                for path, trackstats in selectedmodels.items():
                    glhparams = trackstats.glhparams
                    glhparams = glhparams[glhparams[:, 0] > 1e-14, :]
                    plt.plot(
                        glhparams[:, 2],
                        glhparams[:, 1],
                        ".",
                        ms=5,
                        color="grey",
                        zorder=0,
                    )
                plt.errorbar(
                    obs_depth,
                    obs_width,
                    xerr=err_depth,
                    yerr=err_width,
                    marker="o",
                    linestyle="None",
                    color="#D55E00",
                    zorder=1,
                    label="Measured",
                )
                plt.plot(
                    best_depth,
                    best_width,
                    "*",
                    ms=20,
                    color="#56B4E9",
                    zorder=2,
                    label="Best fit",
                )
                plt.legend(frameon=False)
                plt.xlabel(r"Acoustic depth (s)")
                plt.ylabel(r"Acoustic width (s)")

            if output is not None:
                pp.savefig(bbox_inches="tight")

        if output is not None:
            print("Saved figure to " + output)
            pp.close()


def pairechelle(
    selectedmodels,
    Grid,
    freqfile,
    mod=None,
    modkey=None,
    dnu=None,
    join=False,
    joinkeys=False,
    coeffs=None,
    freqcor="BG14",
    scalnu=None,
    output=None,
    plotlines=False,
):
    """
    Echelle diagram that is either a normal Echelle diagram or where
    matched/paired model and observed modes are connected with a line for
    easier visual inspection.

    Parameters
    ----------
    selectedmodels : dict
        Contains information on all models with a non-zero likelihood.
    Grid : hdf5 object
        The already loaded grid, containing the tracks/isochrones.
    freqfile : str
        Name of file containing frequencies
    mod : array or None
        Array of modes in a given model.
        If None, `mod` will be found from the most likely model in
        `selectedmodels`.
    modkey : array or None
        Array of mode identification of modes from the model.
        If None, `modkey` will be found from the most likely model in
        `selectedmodels`.
    dnu : float or None
        The large frequency separation. If None, `dnufit` will be used.
    join : array or None
        Array containing the matched observed and modelled modes.
        If None, this information is not added to the plot.
    joinkeys : array or None
        Array containing the mode identification of the matched observed
        and modelled modes.
        If None, this information is not added to the plot.
    coeffs : array or None
        Coefficients for the near-surface frequency correction specified
        in `freqcor`. If None, the frequencies on the plot will not
        be corrected.
    scalnu : float or None
        Value used to scale frequencies in frequencies correction.
        `numax` is often used.
    freqcor : str {'None', 'HK08', 'BG14', 'cubicBG14'}
        Flag determining the frequency correction.
    output : str or None
        Filename for saving the figure.
    plotlines : bool
        Flag determinning whether the matched modes should be connected with
        a line segment.
    """
    if dnu is None:
        print("Note: No deltanu specified, using dnufit for echelle diagram.")
        maxPDF_path, maxPDF_ind = stats.most_likely(selectedmodels)
        dnu = Grid[maxPDF_path + "/dnufit"][maxPDF_ind]
    if (mod is None) and (modkey is None):
        maxPDF_path, maxPDF_ind = stats.most_likely(selectedmodels)
        # Find most likely model and compute join
        rawmod = Grid[maxPDF_path + "/osc"][maxPDF_ind]
        rawmodkey = Grid[maxPDF_path + "/osckey"][maxPDF_ind]
        mod = su.transform_obj_array(rawmod)
        modkey = su.transform_obj_array(rawmodkey)
        mod = mod[:, modkey[0, :] < 2.5]
        modkey = modkey[:, modkey[0, :] < 2.5]
    cormod = np.copy(mod)
    if coeffs is not None:
        if freqcor == "HK08":
            corosc = freq_fit.apply_HK08(
                modkey=modkey, mod=mod, coeffs=coeffs, scalnu=scalnu
            )
            cormod[0, :] = corosc
        elif freqcor == "BG14":
            corosc = freq_fit.apply_BG14(
                modkey=modkey, mod=mod, coeffs=coeffs, scalnu=scalnu
            )
            cormod[0, :] = corosc
        elif freqcor == "cubicBG14":
            corosc = freq_fit.apply_cubicBG14(
                modkey=modkey, mod=mod, coeffs=coeffs, scalnu=scalnu
            )
            cormod[0, :] = corosc

    obskey, obs, _ = fio.read_freq(freqfile, nottrustedfile=None)
    _, modl0 = su.get_givenl(l=0, osc=cormod, osckey=modkey)
    _, modl1 = su.get_givenl(l=1, osc=cormod, osckey=modkey)
    _, modl2 = su.get_givenl(l=2, osc=cormod, osckey=modkey)
    fmodl0_all = modl0[0, :]
    fmodl1_all = modl1[0, :]
    fmodl2_all = modl2[0, :]
    s = su.scale_by_inertia(modkey, cormod)

    # Get all observation including the not-trusted ones
    _, obsl0 = su.get_givenl(l=0, osc=obs, osckey=obskey)
    _, obsl1 = su.get_givenl(l=1, osc=obs, osckey=obskey)
    _, obsl2 = su.get_givenl(l=2, osc=obs, osckey=obskey)
    fobsl0_all = obsl0[0, :]
    fobsl1_all = obsl1[0, :]
    fobsl2_all = obsl2[0, :]
    eobsl0_all = obsl0[1, :]
    eobsl1_all = obsl1[1, :]
    eobsl2_all = obsl2[1, :]

    if join is False:
        joins = freq_fit.calc_join(mod, modkey, obs, obskey)
        if joins is None:
            print("No associated_model_index found!")
            return
        else:
            joinkeys, join = joins
    if join is not None:
        _, joinl0 = su.get_givenl(l=0, osc=join, osckey=joinkeys)
        _, joinl1 = su.get_givenl(l=1, osc=join, osckey=joinkeys)
        _, joinl2 = su.get_givenl(l=2, osc=join, osckey=joinkeys)

        fmodl0 = joinl0[0, :]
        fmodl1 = joinl1[0, :]
        fmodl2 = joinl2[0, :]
        fobsl0 = joinl0[2, :]
        fobsl1 = joinl1[2, :]
        fobsl2 = joinl2[2, :]
        eobsl0 = joinl0[3, :]
        eobsl1 = joinl1[3, :]
        eobsl2 = joinl2[3, :]
        sjoin = su.scale_by_inertia(joinkeys[0:2], join[0:2])

    # Set style and open figure
    plt.figure()

    # Plot all observed modes
    if len(fobsl0_all) != 0:
        plt.errorbar(
            fobsl0_all % dnu,
            fobsl0_all,
            xerr=eobsl0_all,
            fmt="o",
            mfc=colors["l0"],
            ecolor=colors["l0"],
            zorder=1,
            alpha=0.5,
        )
    if len(fobsl1_all) != 0:
        plt.errorbar(
            fobsl1_all % dnu,
            fobsl1_all,
            xerr=eobsl1_all,
            fmt="o",
            mfc=colors["l1"],
            ecolor=colors["l1"],
            zorder=1,
            alpha=0.5,
        )
    if len(fobsl2_all) != 0:
        plt.errorbar(
            fobsl2_all % dnu,
            fobsl2_all,
            xerr=eobsl2_all,
            fmt="o",
            mfc=colors["l2"],
            ecolor=colors["l2"],
            zorder=1,
            alpha=0.5,
        )

    # Plot all modes in the model
    plt.scatter(
        fmodl0_all % dnu,
        fmodl0_all,
        s=s[0],
        c=colors["l0"],
        marker="D",
        alpha=0.5,
        zorder=2,
    )
    plt.scatter(
        fmodl1_all % dnu,
        fmodl1_all,
        s=s[1],
        c=colors["l1"],
        marker="^",
        alpha=0.5,
        zorder=2,
    )
    plt.scatter(
        fmodl2_all % dnu,
        fmodl2_all,
        s=s[2],
        c=colors["l2"],
        marker="v",
        alpha=0.5,
        zorder=2,
    )

    # Plot the matched modes
    if join is not None:
        if len(fmodl0) != 0:
            plt.scatter(
                fmodl0 % dnu,
                fmodl0,
                s=sjoin[0],
                c=colors["l0"],
                marker="D",
                linewidths=1,
                edgecolors="k",
                zorder=3,
                label=r"Best fit $l=0$",
            )
            plt.errorbar(
                fobsl0 % dnu,
                fobsl0,
                xerr=eobsl0,
                fmt="o",
                mfc=colors["l0"],
                ecolor=colors["l0"],
                zorder=1,
                label=r"Measured $l=0$",
            )
        if len(fmodl1) != 0:
            plt.scatter(
                fmodl1 % dnu,
                fmodl1,
                s=sjoin[1],
                c=colors["l1"],
                marker="^",
                linewidths=1,
                edgecolors="k",
                zorder=3,
                label=r"Best fit $l=1$",
            )
            plt.errorbar(
                fobsl1 % dnu,
                fobsl1,
                xerr=eobsl1,
                fmt="o",
                mfc=colors["l1"],
                ecolor=colors["l1"],
                zorder=1,
                label=r"Measured $l=1$",
            )
        if len(fmodl2) != 0:
            plt.scatter(
                fmodl2 % dnu,
                fmodl2,
                s=sjoin[2],
                c=colors["l2"],
                marker="v",
                linewidths=1,
                edgecolors="k",
                zorder=3,
                label=r"Best fit $l=2$",
            )
            plt.errorbar(
                fobsl2 % dnu,
                fobsl2,
                xerr=eobsl2,
                fmt="o",
                mfc=colors["l2"],
                ecolor=colors["l2"],
                zorder=1,
                label=r"Measured $l=2$",
            )

        # Make line segments connecting the observed and associated mode
        if plotlines:
            for l, (fmod, fobs) in enumerate(
                zip([fmodl0, fmodl1, fmodl2], [fobsl0, fobsl1, fobsl2])
            ):
                for i in range(len(fmod)):
                    plt.plot(
                        (fmod[i] % dnu, fobs[i] % dnu),
                        (fmod[i], fobs[i]),
                        c=colors["l" + str(l)],
                        alpha=0.7,
                        zorder=1,
                    )

    # Decoration
    lgnd = plt.legend(
        bbox_to_anchor=(0.0, 1.02, 1.0, 0.102),
        loc=8,
        ncol=6,
        mode="expand",
        borderaxespad=0.0,
    )
    for i in range(len(lgnd.legendHandles)):
        lgnd.legendHandles[i]._sizes = [50]
    plt.xlabel(r"Frequency modulo $\Delta \nu=$" + str(dnu) + r" $\mu$Hz")
    plt.ylabel(r"Frequency ($\mu$Hz)")
    plt.xlim(0, dnu)

    if output is not None:
        plt.savefig(output, bbox_inches="tight")
        print("Saved figure to " + output)
