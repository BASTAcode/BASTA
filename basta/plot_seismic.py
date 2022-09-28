"""
Production of asteroseismic plots
"""
import os
import numpy as np
import matplotlib

from scipy.interpolate import CubicSpline

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


def ratioplot(
    freqfile, datos, joinkeys, join, output=None, nonewfig=False, threepoint=False
):
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
    threepoint : bool
        If True, use three point definition of r01 and r10 ratios instead
        of default five point definition.
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
    names = ["l", "n", "freq", "err"]  # 'err' is redundant here!
    fmts = [int, int, float, float]
    nmodes = joinkeys[:, joinkeys[0, :] < 3].shape[1]
    freq = np.zeros(nmodes, dtype={"names": names, "formats": fmts})
    freq[:]["l"] = joinkeys[0, joinkeys[0, :] < 3]
    freq[:]["n"] = joinkeys[1, joinkeys[0, :] < 3]
    freq[:]["freq"] = join[0, joinkeys[0, :] < 3]
    r02, r01, r10 = freq_fit.ratios(freq, threepoint=threepoint)

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

            plt.plot(
                modratio[:, 3], modratio[:, 1], "*", markersize=20, label="Best fit"
            )
            plt.errorbar(
                obsratio[1, :],
                obsratio[0, :],
                yerr=obsratio[2, :],
                marker="o",
                linestyle="None",
                label="Measured",
            )
            plt.legend(frameon=False)
            plt.xlabel(r"Frequency ($\mu$Hz)")
            plt.ylabel('Ratio type "{0}"'.format(ratio_type[1:]))

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


def epsilon_difference_diagram(
    mod,
    modkey,
    moddnu,
    obsepsdiff,
    covinv,
    output,
):
    """
    Full comparison figure of observed and best-fit model epsilon
    differences, with individual epsilons and correlation map.

    Parameters
    ----------
    mod : array
        Array of frequency modes in best-fit model.
    modkey : array
        Array of mode identification of modes in the best-fit model.
    moddnu : float
        Average large frequency separation (dnufit) of best-fit model.
    obsepsdiff : array
        Array of computed observed epsilon differences, with mode
        identification and related frequency.
    covinv : array
        Inverse covariance matrix of the observed epsilon differences.
    output : str
        Name and path of output plotfile.
    """

    # Labels and markers
    delab = r"$\delta\epsilon^{%s}_{0%d}$"
    fmt = [".", "^", "s"]
    fmtev = [".", "1", "2"]

    # Extract l degrees available and apply restriction to model
    l_available = [int(ll) for ll in set(obsepsdiff[2])]
    index = np.zeros(mod.shape[1], dtype=bool)
    for ll in [0, *l_available]:
        index |= modkey[0] == ll
    mod = mod[:, index]
    modkey = modkey[:, index]

    # Get model epsilon differences
    modepsdiff = freq_fit.compute_epsilon_diff(modkey, mod, moddnu)

    # Uncertainty from inverse covariance
    uncert = np.sqrt(np.diag(np.linalg.pinv(covinv, rcond=1e-12)))

    # Start of figure
    fig, ax = plt.subplots(1, 1)
    handles, legends = [], []
    for ll in l_available:
        indobs = obsepsdiff[2] == ll
        indmod = modepsdiff[2] == ll
        indmod &= modepsdiff[1] > min(obsepsdiff[1]) - 3 * moddnu
        indmod &= modepsdiff[1] < max(obsepsdiff[1]) + 3 * moddnu
        spline = CubicSpline(modepsdiff[1][indmod], modepsdiff[0][indmod])
        fnew = np.linspace(min(modepsdiff[1][indmod]), max(modepsdiff[1][indmod]), 100)

        # Model with spline
        (moddot,) = ax.plot(
            modepsdiff[1][indmod],
            modepsdiff[0][indmod],
            fmt[0],
            color=colors["l%d" % ll],
        )
        ax.plot(fnew, spline(fnew), "-", color=colors["l%d" % ll])

        # Model at observed
        (modobs,) = ax.plot(
            obsepsdiff[1][indobs],
            spline(obsepsdiff[1][indobs]),
            fmtev[ll],
            color="k",
            markeredgewidth=2,
            alpha=0.7,
        )

        # Observed with uncertainties
        obsdot = ax.errorbar(
            obsepsdiff[1][indobs],
            obsepsdiff[0][indobs],
            yerr=uncert[indobs],
            fmt=fmt[ll],
            color=colors["l%d" % ll],
            markeredgewidth=0.5,
            markeredgecolor="k",
        )

        handles.extend([moddot, obsdot, modobs])
        legends.extend(
            [
                delab % ("mod", ll),
                delab % ("obs", ll),
                delab % ("mod", ll) + r"$(\nu^{obs})$",
            ]
        )

    # To get the right order of entries in the legend
    h, l = [], []
    for i in range(3):
        h.extend(handles[i::3])
        l.extend(legends[i::3])
    ax.legend(h, l, fontsize=16, loc=2, bbox_to_anchor=(1.02, 1))
    # Labels
    ax.set_xlabel(r"$\nu\,(\mu {\rm Hz})$")
    ax.set_ylabel(r"$\delta\epsilon_{0\ell}$")

    fig.tight_layout()
    fig.savefig(output)
    print("Saved figure to " + output)


def epsilon_difference_all_diagram(
    mod,
    modkey,
    moddnu,
    obsepsdiff,
    covinv,
    obs,
    obskey,
    obsdnu,
    output,
):
    """
    Full comparison figure of observed and best-fit model epsilon
    differences, with individual epsilons and correlation map.

    Parameters
    ----------
    mod : array
        Array of frequency modes in best-fit model.
    modkey : array
        Array of mode identification of modes in the best-fit model.
    moddnu : float
        Average large frequency separation (dnufit) of best-fit model.
    obsepsdiff : array
        Array of computed observed epsilon differences, with mode
        identification and related frequency.
    covinv : array
        Inverse covariance matrix of the observed epsilon differences.
    obs : array
        Array of observed frequency modes.
    obskey : array
        Array of mode identification of observed frequency modes.
    obsdnu : float
        Inputted average large frequency separation (dnu) of observations.
    output : str
        Name and path of output plotfile.
    """

    # Prepared labels and markers
    delab = r"$\delta\epsilon^{%s}_{0%d}$"
    elab = r"$\epsilon_{%d}$"
    colab = r"$\delta\epsilon_{0%d}(%d)$"
    fmt = [".", "^", "s"]
    fmtev = [".", "1", "2"]

    # Extract l degrees available, and only work with these
    l_available = [int(ll) for ll in set(obsepsdiff[2])]
    index = np.zeros(mod.shape[1], dtype=bool)
    for ll in [0, *l_available]:
        index |= modkey[0] == ll
    mod = mod[:, index]
    modkey = modkey[:, index]

    # Determine model epsilon differences
    modepsdiff = freq_fit.compute_epsilon_diff(modkey, mod, moddnu)

    # Recompute to determine if possible but extrapolated modes
    edextrapol = freq_fit.compute_epsilon_diff(obskey, obs, obsdnu)
    nu12 = edextrapol[1][edextrapol[2] > 0]
    nu0 = obs[0][obskey[0] == 0]
    expol = np.where(np.logical_or(nu12 < min(nu0), nu12 > max(nu0)))[0]

    # All parameters needed from inverse covariance
    cov = np.linalg.pinv(covinv, rcond=1e-12)
    uncert = np.sqrt(np.diag(cov))
    Dinv = np.diag(1 / np.sqrt(np.diag(cov)))
    cor = Dinv @ cov @ Dinv

    # Definition of figure
    figsize = np.array([11.69, 11.69]) * 1.5
    fig, ax = plt.subplots(
        3,
        3,
        figsize=figsize,
        gridspec_kw={"width_ratios": [1, 1, 0.05], "height_ratios": [2, 1, 1]},
    )
    fig.delaxes(ax[1, 2])
    fig.delaxes(ax[2, 2])

    # Epsilon differences
    handles, legends = [], []
    for ll in l_available:
        legends.extend(
            [
                delab % ("mod", ll),
                delab % ("obs", ll),
                delab % ("mod", ll) + r"$(\nu^{obs})$",
            ]
        )

        indobs = obsepsdiff[2] == ll
        indmod = modepsdiff[2] == ll
        spline = CubicSpline(modepsdiff[1][indmod], modepsdiff[0][indmod])
        fnew = np.linspace(min(modepsdiff[1][indmod]), max(modepsdiff[1][indmod]), 1000)

        # Raw model with spline
        ax[1, 1].errorbar(
            modepsdiff[1][indmod],
            modepsdiff[0][indmod],
            yerr=np.zeros(sum(indmod)),
            fmt=fmt[ll],
            color=colors["l%d" % ll],
            markeredgewidth=0.5,
            markeredgecolor="k",
            label=delab % ("", ll),
        )
        ax[1, 1].plot(fnew, spline(fnew), "-", color=colors["l%d" % ll])

        # Constrained model range
        indmod &= modepsdiff[1] > min(obsepsdiff[1]) - 3 * moddnu
        indmod &= modepsdiff[1] < max(obsepsdiff[1]) + 3 * moddnu
        spline = CubicSpline(modepsdiff[1][indmod], modepsdiff[0][indmod])
        fnew = np.linspace(min(modepsdiff[1][indmod]), max(modepsdiff[1][indmod]), 100)

        # Model with spline
        (moddot,) = ax[0, 0].plot(
            modepsdiff[1][indmod],
            modepsdiff[0][indmod],
            fmt[0],
            color=colors["l%d" % ll],
        )
        ax[0, 0].plot(fnew, spline(fnew), "-", color=colors["l%d" % ll])

        # Model at observed
        (modobs,) = ax[0, 0].plot(
            obsepsdiff[1][indobs],
            spline(obsepsdiff[1][indobs]),
            fmtev[ll],
            color="k",
            markeredgewidth=2,
            alpha=0.7,
        )

        # Observed with uncertainties
        obsdot = ax[0, 0].errorbar(
            obsepsdiff[1][indobs],
            obsepsdiff[0][indobs],
            yerr=uncert[indobs],
            fmt=fmt[ll],
            color=colors["l%d" % ll],
            markeredgewidth=0.5,
            markeredgecolor="k",
        )

        # Spline observed for separate plot
        spline = CubicSpline(obsepsdiff[1][indobs], obsepsdiff[0][indobs])
        fnew = np.linspace(min(obsepsdiff[1][indobs]), max(obsepsdiff[1][indobs]), 100)

        # Observed with uncertainties and spline
        ax[1, 0].errorbar(
            obsepsdiff[1][indobs],
            obsepsdiff[0][indobs],
            yerr=uncert[indobs],
            fmt=fmt[ll],
            color=colors["l%d" % ll],
            markeredgewidth=0.5,
            markeredgecolor="k",
        )
        ax[1, 0].plot(fnew, spline(fnew), "--k")

        handles.extend([moddot, obsdot, modobs])

    # Correlation map and colorbar
    im = ax[0, 1].imshow(cor, cmap="RdBu_r", vmin=-1, vmax=1)
    labs = [
        colab % (obsepsdiff[2][j], obsepsdiff[3][j]) for j in range(obsepsdiff.shape[1])
    ]
    plt.colorbar(im, cax=ax[0, 2], shrink=0.5, pad=0.05)

    # Potential extrapolated points
    if len(expol):
        for ll in set(edextrapol[2][expol].astype(int)):
            ax[1, 0].plot(
                edextrapol[1][expol],
                edextrapol[0][expol],
                fmt[ll],
                color="k",
                label=r"$\nu(\ell={0})\,\notin\,\nu(\ell=0)$".format(ll),
            )
        ax[1, 0].legend()

    # Individual epsilons
    for ll in [0, *l_available]:
        # Extract observed quantities
        indobs = obskey[0] == ll
        fre = obs[0][indobs]
        eps = fre / obsdnu - obskey[1][indobs] - ll / 2
        err = obs[1][indobs] / obsdnu
        intpol = CubicSpline(fre, eps)
        fnew = np.linspace(min(fre), max(fre), 100)

        # Plot observed w. spline
        ax[2, 0].errorbar(
            fre, eps, yerr=err, fmt=fmt[ll], color=colors["l%d" % ll], label=elab % ll
        )
        ax[2, 0].plot(fnew, intpol(fnew), "-", color=colors["l%d" % ll])

        # Extract model quantities
        indmod = modkey[0] == ll
        fre = mod[0][indmod]
        eps = fre / moddnu - modkey[1][indmod] - ll / 2
        err = mod[1][indmod] / moddnu
        intpol = CubicSpline(fre, eps)
        fnew = np.linspace(min(fre), max(fre), 1000)

        # Plot model w. spline
        ax[2, 1].errorbar(
            fre, eps, yerr=err, fmt=fmt[ll], color=colors["l%d" % ll], label=elab % ll
        )
        ax[2, 1].plot(fnew, intpol(fnew), "-", color=colors["l%d" % ll])

    # Limits in the bottom plots
    for i in [0, 1]:
        xl1 = list(ax[1, i].get_xlim())
        xl2 = list(ax[2, i].get_xlim())
        xlim = [min(xl1[0], xl2[0]), max(xl1[1], xl2[1])]
        yl1 = list(ax[i + 1, 0].get_ylim())
        yl2 = list(ax[i + 1, 1].get_ylim())
        ylim = [min(yl1[0], yl2[0]), max(yl1[1], yl2[1])]
        for j in [0, 1]:
            ax[j + 1, i].set_xlim(xlim)
            ax[i + 1, j].set_ylim(ylim)
        ax[1, i].set_xticklabels([])
        ax[i + 1, 1].set_yticklabels([])

    # To get the right order of entries in the legend
    h, l = [], []
    for i in range(3):
        h.extend(handles[i::3])
        l.extend(legends[i::3])
    # Legends
    ax[0, 0].legend(
        h,
        l,
        fontsize=16,
        bbox_to_anchor=(0, 1.02, 1, 0.102),
        ncol=3,
        loc=8,
        mode="expand",
        borderpad=0,
        borderaxespad=0.0,
    )
    ax[1, 1].legend(fontsize=16, bbox_to_anchor=(1.02, 1), loc=2, borderaxespad=0.0)
    ax[2, 1].legend(fontsize=16, bbox_to_anchor=(1.02, 1), loc=2)

    # Axes labels
    ax[0, 0].set_xlabel(r"$\nu\,(\mu {\rm Hz})$")
    ax[0, 0].set_ylabel(r"$\delta\epsilon_{0\ell}$")
    ax[0, 1].set_xticklabels([])
    ax[0, 1].set_yticks(range(obsepsdiff.shape[1]))
    ax[0, 1].set_yticklabels(labs)
    ax[1, 0].set_ylabel(r"$\delta\epsilon_{0\ell}$")
    ax[2, 0].set_xlabel(r"$\nu\, (\mu {\rm Hz})$")
    ax[2, 0].set_ylabel(r"$\epsilon_{\ell}$")
    ax[2, 1].set_xlabel(r"$\nu\, (\mu {\rm Hz})$")

    # Titles
    ax[0, 1].set_title(r"Correlation map", fontsize=18)
    ax[1, 0].set_title(r"Observed", fontsize=18)
    ax[1, 1].set_title(r"Model", fontsize=18)

    fig.tight_layout()
    fig.subplots_adjust(wspace=0.2, hspace=0.4)

    # Dear god oh why does this have to be so difficult...
    p10 = ax[1, 0].get_position().get_points().flatten()
    p11 = ax[1, 1].get_position().get_points().flatten()
    p20 = ax[2, 0].get_position().get_points().flatten()
    dx = p10[2] - p10[0] + (p11[0] - p10[2]) / 2
    xs = [p10[0], p10[0] + dx]
    dy = p20[3] - p20[1] + (p10[1] - p20[3]) / 2
    ys = [p20[0], p20[0] + dy]

    ax[1, 0].set_position([xs[0], ys[1], dx, dy])
    ax[1, 1].set_position([xs[1], ys[1], dx, dy])
    ax[2, 0].set_position([xs[0], ys[0], dx, dy])
    ax[2, 1].set_position([xs[1], ys[0], dx, dy])

    fig.savefig(output)
    print("Saved figure to " + output)


def epsilon_diff_and_correlation(depsN, depsL, covN, covL, osc, osckey, avgdnu):

    titles = [r"Correlation $n$-sorted", r"Correlation $\ell$-sorted"]
    l_avail = [int(ll) for ll in set(depsN[2])]
    fig, ax = plt.subplots(
        2, 3, figsize=(17, 15), gridspec_kw={"width_ratios": [1, 1, 0.05]}
    )
    fig.delaxes(ax[1, 2])

    ####################
    # Correlation maps #
    ####################
    for i, (deps, cov) in enumerate(zip([depsN, depsL], [covN, covL])):
        Dinv = np.diag(1 / np.sqrt(np.diag(cov)))
        cor = Dinv @ cov @ Dinv
        im = ax[0, i].imshow(cor, cmap="RdBu_r", vmin=-1, vmax=1)

        labs = [
            r"$\delta\epsilon_{0%d}(%d)$" % (deps[2][j], deps[3][j])
            for j in range(deps.shape[1])
        ]

        ax[0, i].set_xticks(range(deps.shape[1]))
        ax[0, i].set_xticklabels(labs, rotation=90)
        ax[0, i].set_yticks(range(deps.shape[1]))
        ax[0, i].set_yticklabels(labs)
        if sum(abs(np.diff(deps[2]))) < 2:
            ax[0, i].set_title(titles[1], fontsize=18)
        else:
            ax[0, i].set_title(titles[0], fontsize=18)

    ax[0, 1].yaxis.tick_right()
    plt.colorbar(
        im,
        cax=ax[0, 2],
        shrink=0.5,
        drawedges=False,
        anchor=(0.5, 0.5),
        use_gridspec=False,
        pad=20,
        fraction=0.8,
        aspect=10,
    )

    ################
    # Pure epsilon #
    ################
    for ll in [0, *l_avail]:
        nn = osckey[1, :][osckey[0] == ll]
        eps = osc[0, :][osckey[0] == ll] / avgdnu - nn - ll / 2
        err = osc[1, :][osckey[0] == ll] / avgdnu
        fre = osc[0, :][osckey[0] == ll]
        ax[1, 0].errorbar(
            fre,
            eps,
            yerr=err,
            fmt=".",
            color=colors["l%d" % ll],
            label=r"$\epsilon_{%d}$" % (ll),
        )
        intpol = CubicSpline(fre, eps)
        if ll == 0:
            fnew = np.linspace(min(osc[0]) - avgdnu, max(osc[0]) + avgdnu, 100)

            ax[1, 0].plot(fnew, intpol(fnew), "-", color=colors["l%d" % ll])
        else:
            fnew = np.linspace(fre[0], fre[-1], 100)
            ax[1, 0].plot(fnew, intpol(fnew), "--k", alpha=0.7)

    ax[1, 0].legend(fontsize=16)
    ax[1, 0].set_xlabel(r"$\nu\,(\mu {\rm Hz})$")
    ax[1, 0].set_ylabel(r"$\epsilon_\ell$")

    #######################
    # Epsilon differences #
    #######################
    for ll in l_avail:
        deps = depsN[0][depsN[2] == ll]
        fre = depsN[1][depsN[2] == ll]
        err = np.sqrt(np.diag(covN))[depsN[2] == ll]
        intpol = CubicSpline(fre, deps)
        fnew = np.linspace(min(fre), max(fre), 100)
        ax[1, 1].errorbar(
            fre,
            deps,
            yerr=err,
            fmt=".",
            color=colors["l%d" % ll],
            label=r"$\delta\epsilon_{0%d}$" % (ll),
        )
        ax[1, 1].plot(fnew, intpol(fnew), "--", color="k", alpha=0.7)

    nu12 = depsN[1][depsN[2] > 0]
    nu0 = osc[0][osckey[0] == 0]
    expol = np.where(np.logical_or(nu12 < min(nu0), nu12 > max(nu0)))[0]
    if len(expol):
        ax[1, 1].plot(
            depsN[1][expol], depsN[0][expol], "o", color="k", label=r"Extrapolation"
        )

    ax[1, 1].legend(fontsize=16)
    ax[1, 1].set_xlabel(r"$\nu\,(\mu {\rm Hz})$")
    ax[1, 1].set_ylabel(r"$\delta\epsilon_{0\ell}$")
    ax[1, 1].yaxis.tick_right()
    ax[1, 1].yaxis.set_label_position("right")

    fig.tight_layout()
    fig.savefig("covariance_map.pdf")
    plt.close(fig)
