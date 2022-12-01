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
colors = {
    "l0": "#D55E00",
    "l1": "#009E73",
    "l2": "#0072B2",
    "r01": "#D36E70",
    "r10": "#CCBB44",
    "r02": "#228833",
    "r012": "#549EB3",
    "r010": "#60AB9E",
    "r102": "#A778B4",
}
modmarkers = {
    "l0": "D",
    "l1": "^",
    "l2": "v",
    "ratio": "d",
}
obsmarker = "o"
splinemarkers = [".", "2", "1"]
splinecolor = "0.7"


def echelle(
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
    pair=False,
    duplicate=False,
    output=None,
):
    """
    Echelle diagram. It is possible to either make a single Echelle diagram
    or plot it twice making patterns across the moduli-limit easier to see.

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
    pair : bool
        Flag determining whether to link matched observed and modelled
        frequencies.
    duplicate : bool
        Flag determining whether to plot two echelle diagrams next to one
        another.
    output : str or None
        Filename for saving the figure.
    """
    if pair:
        lw = 1
    else:
        lw = 0

    if dnu is None:
        print("Note: No deltanu specified, using dnufit for echelle diagram.")
        maxPDF_path, maxPDF_ind = stats.most_likely(selectedmodels)
        dnu = Grid[maxPDF_path + "/dnufit"][maxPDF_ind]

    if duplicate:
        modx = 1
        scalex = dnu
    else:
        modx = dnu
        scalex = 1

    obskey, obs, _ = fio.read_freq(freqfile, nottrustedfile=None)
    obsls = np.unique(obskey[0, :]).astype(str)

    if (mod is None) and (modkey is None):
        maxPDF_path, maxPDF_ind = stats.most_likely(selectedmodels)
        rawmod = Grid[maxPDF_path + "/osc"][maxPDF_ind]
        rawmodkey = Grid[maxPDF_path + "/osckey"][maxPDF_ind]
        mod = su.transform_obj_array(rawmod)
        modkey = su.transform_obj_array(rawmodkey)
        mod = mod[:, modkey[0, :] <= np.amax(obsls.astype(int))]
        modkey = modkey[:, modkey[0, :] <= np.amax(obsls)]

    cormod = np.copy(mod)

    if coeffs is not None:
        if freqcor == "HK08":
            corosc = freq_fit.apply_HK08(
                modkey=modkey, mod=mod, coeffs=coeffs, scalnu=scalnu
            )
        elif freqcor == "BG14":
            corosc = freq_fit.apply_BG14(
                modkey=modkey, mod=mod, coeffs=coeffs, scalnu=scalnu
            )
        elif freqcor == "cubicBG14":
            corosc = freq_fit.apply_cubicBG14(
                modkey=modkey, mod=mod, coeffs=coeffs, scalnu=scalnu
            )
        cormod[0, :] = corosc

    s = su.scale_by_inertia(modkey, cormod)
    if join is not None:
        sjoin = su.scale_by_inertia(joinkeys[0:2], join[0:2])

    fmod = {}
    fmod_all = {}
    fobs = {}
    fobs_all = {}
    eobs = {}
    eobs_all = {}
    for l in np.arange(np.amax(obsls.astype(int)) + 1):
        _, mod = su.get_givenl(l=l, osc=cormod, osckey=modkey)
        _, lobs = su.get_givenl(l=l, osc=obs, osckey=obskey)
        fmod_all[str(l)] = mod[0, :] / scalex
        fobs_all[str(l)] = lobs[0, :] / scalex
        eobs_all[str(l)] = lobs[1, :] / scalex
        if join is not None:
            _, ljoin = su.get_givenl(l=l, osc=join, osckey=joinkeys)
            fmod[str(l)] = ljoin[0, :] / scalex
            fobs[str(l)] = ljoin[2, :] / scalex
            eobs[str(l)] = ljoin[3, :] / scalex

    # Create plot
    fig, ax1 = plt.subplots()
    ax2 = ax1.twinx()
    if duplicate:
        # Plot something to set the scale on one y-axis
        ax1.errorbar(
            fobs_all[obsls[0]] % modx,
            fobs_all[obsls[0]] * dnu,
            xerr=eobs_all[obsls[0]],
            fmt=obsmarker,
            mfc=colors["l" + obsls[0]],
            ecolor=colors["l" + obsls[0]],
            alpha=0.5,
            zorder=0,
        )
        ax1.axvline(x=0, linestyle="--", color="0.8", zorder=0)
        ax = ax2
        aax = ax1
    else:
        ax2.errorbar(
            fobs_all[obsls[0]],
            fobs_all[obsls[0]] / dnu,
            xerr=eobs_all[obsls[0]],
            fmt=obsmarker,
            mfc=colors["l" + obsls[0]],
            ecolor=colors["l" + obsls[0]],
            alpha=0.5,
            zorder=0,
        )
        ax = ax1
        aax = ax2

    # Plot all observed modes
    for l in obsls:
        ax.errorbar(
            fobs_all[l] % modx,
            fobs_all[l],
            xerr=eobs_all[l],
            fmt=obsmarker,
            mfc=colors["l" + l],
            ecolor=colors["l" + l],
            zorder=1,
            alpha=0.5,
        )
        if duplicate:
            ax.errorbar(
                fobs_all[l] % modx - modx,
                fobs_all[l],
                xerr=eobs_all[l],
                fmt=obsmarker,
                mfc=colors["l" + l],
                ecolor=colors["l" + l],
                zorder=1,
                alpha=0.5,
            )

    for l in np.arange(np.amax(obsls.astype(int)) + 1):
        l = str(l)
        ax.scatter(
            fmod_all[l] % modx,
            fmod_all[l],
            s=s[int(l)],
            c=colors["l" + l],
            marker=modmarkers["l" + l],
            alpha=0.5,
            zorder=2,
        )
        if duplicate:
            ax.scatter(
                fmod_all[l] % modx - modx,
                fmod_all[l],
                s=s[int(l)],
                c=colors["l" + l],
                marker=modmarkers["l" + l],
                alpha=0.5,
                zorder=2,
            )

    # Plot the matched modes in negative and positive side
    linelimit = 0.75 * modx
    if join is not None:
        for l in obsls:
            if len(fmod[l]) > 0:
                ax.scatter(
                    fmod[l] % modx,
                    fmod[l],
                    s=sjoin[int(l)],
                    c=colors["l" + l],
                    marker=modmarkers["l" + l],
                    linewidths=1,
                    edgecolors="k",
                    zorder=3,
                    label=f"Best fit $\ell={l}$",
                )
                if duplicate:
                    ax.scatter(
                        fmod[l] % modx - modx,
                        fmod[l],
                        s=sjoin[int(l)],
                        c=colors["l" + l],
                        marker=modmarkers["l" + l],
                        linewidths=1,
                        edgecolors="k",
                        zorder=3,
                    )
                ax.errorbar(
                    fobs[l] % modx,
                    fobs[l],
                    xerr=eobs[l],
                    fmt=obsmarker,
                    mfc=colors["l" + l],
                    ecolor=colors["l" + l],
                    zorder=1,
                    label=f"Measured $\ell={l}$",
                )
                if duplicate:
                    ax.errorbar(
                        fobs[l] % modx - modx,
                        fobs[l],
                        xerr=eobs[l],
                        fmt=obsmarker,
                        mfc=colors["l" + l],
                        ecolor=colors["l" + l],
                        zorder=1,
                    )

                if pair:
                    fm = fmod[l]
                    fo = fobs[l]
                    for i in range(len(fm)):
                        if ((fm[i] % modx) > linelimit) & (
                            (fo[i] % modx) < (modx - linelimit)
                        ):
                            a = (fm[i] - fo[i]) / ((fm[i] - fo[i]) % modx)
                            if duplicate:
                                x0 = -1
                            else:
                                x0 = 0
                            ax.plot(
                                (fm[i] % modx - modx, fo[i] % modx),
                                (fm[i], fo[i]),
                                c=colors["l" + str(l)],
                                alpha=0.7,
                                zorder=10,
                            )
                            ax.plot(
                                (fm[i] % modx, modx),
                                (fm[i], a * np.ceil(fm[i] / modx) * modx),
                                c=colors["l" + str(l)],
                                alpha=0.7,
                                zorder=10,
                            )
                            ax.plot(
                                (x0, fo[i] % modx - modx),
                                (a * np.ceil(fm[i] / modx) * modx, fo[i]),
                                c=colors["l" + str(l)],
                                alpha=0.7,
                                zorder=10,
                            )
                        elif ((fo[i] % modx) > linelimit) & (
                            (fm[i] % modx) < (modx - linelimit)
                        ):
                            if duplicate:
                                x0 = -1
                            else:
                                x0 = 0
                            a = (fm[i] - fo[i]) / ((fm[i] - fo[i]) % modx)
                            ax.plot(
                                (fo[i] % modx - modx, fm[i] % modx),
                                (fo[i], fm[i]),
                                c=colors["l" + str(l)],
                                alpha=0.7,
                                zorder=10,
                            )
                            ax.plot(
                                (fo[i] % modx, modx),
                                (fo[i], a * np.ceil(fo[i] / modx) * modx),
                                c=colors["l" + str(l)],
                                alpha=0.7,
                                zorder=10,
                            )
                            ax.plot(
                                (x0, fm[i] % modx - modx),
                                (a * np.ceil(fo[i] / modx) * modx, fm[i]),
                                c=colors["l" + str(l)],
                                alpha=0.7,
                                zorder=10,
                            )
                        else:
                            ax.plot(
                                (fm[i] % modx, fo[i] % modx),
                                (fm[i], fo[i]),
                                c=colors["l" + str(l)],
                                alpha=0.7,
                                zorder=10,
                                lw=lw,
                            )
                            if duplicate:
                                ax.plot(
                                    (fm[i] % modx - modx, fo[i] % modx - modx),
                                    (fm[i], fo[i]),
                                    c=colors["l" + str(l)],
                                    alpha=0.7,
                                    zorder=10,
                                    lw=lw,
                                )

                    """
                    elif (
                        duplicate
                        & ((fo[i] % modx) > linelimit)
                        & ((fm[i] % modx) < (modx - linelimit))
                    ):
                        ax.plot(
                            (fo[i] % modx - modx, fm[i] % modx),
                            (fo[i], fm[i]),
                            c=colors["l" + str(l)],
                            alpha=0.7,
                            zorder=10,
                            lw=lw,
                        )
                    """
    lgnd = ax.legend(
        bbox_to_anchor=(0.0, 1.02, 1.0, 0.102),
        loc=8,
        ncol=6,
        mode="expand",
        borderaxespad=0.0,
    )
    for i in range(len(lgnd.legendHandles)):
        lgnd.legendHandles[i]._sizes = [50]

    if duplicate:
        ax.set_xlim([-1, 1])
        aax.set_ylim(ax.set_ylim()[0] * dnu, ax.set_ylim()[1] * dnu)
        aax.set_xlabel(
            r"Frequency normalised by $\Delta \nu$ modulo 1 ($\Delta \nu =$%s $\mu$Hz)"
            % dnu
        )
        aax.set_ylabel(r"Frequency ($\mu$Hz)")
        ax.set_ylabel(r"Frequency normalised by $\Delta \nu$")
    else:
        ax.set_xlim([0, modx])
        aax.set_ylim(ax.set_ylim()[0] / dnu, ax.set_ylim()[1] / dnu)
        ax.set_xlabel(
            r"Frequency normalised by $\Delta \nu$ modulo 1 ($\Delta \nu =$%s $\mu$Hz)"
            % dnu
        )
        ax.set_ylabel(r"Frequency ($\mu$Hz)")
        aax.set_ylabel(r"Frequency normalised by $\Delta \nu$")

    if output is not None:
        plt.savefig(output, bbox_inches="tight")
        print("Saved figure to " + output)


def ratioplot(
    freqfile,
    obsfreqdata,
    obsfreqmeta,
    joinkeys,
    join,
    output=None,
    nonewfig=False,
    threepoint=False,
):
    """
    Plot frequency ratios.

    Parameters
    ----------
    freqfile : str
        Name of file containing frequencies and ratio types
    obsfreqdata : dict
        Requested frequency-dependent data such as glitches, ratios, and
        epsilon difference. It also contains the covariance matrix and its
        inverse of the individual frequency modes.
        The keys correspond to the science case, e.g. `r01a, `glitch`, or
        `e012`.
        Inside each case, you find the data (`data`), the covariance matrix
        (`cov`), and its inverse (`covinv`).
    obsfreqmeta : dict
        The requested information about which frequency products to fit or
        plot, unpacked for easier access later.
    joinkeys : array
        Array containing the mode identification of the matched observed
        and modelled modes.
    join : array
        Array containing the matched observed and modelled modes.
    output : str or None, optional
        Filename for saving the plot.
    nonewfig : bool, optional
        If True, this creates a new canvas. Otherwise, the plot is added
        to the existing canvas.
    threepoint : bool
        If True, use three point definition of r01 and r10 ratios instead
        of default five point definition.
    """
    if output is not None:
        pp = PdfPages(output)

    allfig, allax = plt.subplots(1, 1)
    for ratiotype in obsfreqmeta["ratios"]["plot"]:
        fig, ax = plt.subplots(1, 1)

        obsratio = obsfreqdata[ratiotype]["data"]
        obsratio_cov = obsfreqdata[ratiotype]["cov"]
        obsratio_err = np.sqrt(np.diag(obsratio_cov))
        modratio = freq_fit.compute_ratioseqs(
            joinkeys, join[0:2, :], ratiotype, threepoint=threepoint
        )

        allax.scatter(
            modratio[1, :],
            modratio[0, :],
            marker=modmarkers["ratio"],
            color=colors[ratiotype],
            edgecolors="k",
            zorder=3,
            label=f"Best fit ({ratiotype[1:]})",
        )
        allax.plot(
            modratio[1, :],
            modratio[0, :],
            "-",
            color=colors[ratiotype],
            zorder=-1,
        )

        allax.errorbar(
            obsratio[1, :],
            obsratio[0, :],
            yerr=obsratio_err,
            marker=obsmarker,
            color=colors[ratiotype],
            mec="k",
            mew=0.5,
            linestyle="None",
            zorder=3,
            label=f"Measured ({ratiotype[1:]})",
        )
        allax.plot(
            obsratio[1, :],
            obsratio[0, :],
            "-",
            color=colors[ratiotype],
            zorder=-1,
        )

        ax.scatter(
            modratio[1, :],
            modratio[0, :],
            marker=modmarkers["ratio"],
            color=colors[ratiotype],
            edgecolors="k",
            zorder=3,
            label=f"Best fit ({ratiotype[1:]})",
        )
        ax.plot(
            modratio[1, :],
            modratio[0, :],
            "-",
            color=colors[ratiotype],
            zorder=-1,
        )

        ax.errorbar(
            obsratio[1, :],
            obsratio[0, :],
            yerr=obsratio_err,
            marker=obsmarker,
            color=colors[ratiotype],
            mec="k",
            mew=0.5,
            linestyle="None",
            zorder=3,
            label=f"Measured ({ratiotype[1:]})",
        )
        ax.plot(
            obsratio[1, :],
            obsratio[0, :],
            "-",
            color=colors[ratiotype],
            zorder=-1,
        )

        lgnd = ax.legend(
            bbox_to_anchor=(0.0, 1.02, 1.0, 0.102),
            loc=8,
            ncol=2,
            mode="expand",
            borderaxespad=0.0,
        )
        for i in range(len(lgnd.legendHandles)):
            lgnd.legendHandles[i]._sizes = [50]

        ax.set_xlabel(r"Frequency ($\mu$Hz)")
        ax.set_ylabel(f"Frequency ratio ({ratiotype})")

        if output is not None:
            pp.savefig(fig, bbox_inches="tight")

    ncols = np.amin([len(obsfreqmeta["ratios"]["plot"]) * 2, 6])
    lgnd = allax.legend(
        bbox_to_anchor=(0.0, 1.02, 1.0, 0.102),
        loc=8,
        ncol=ncols,
        mode="expand",
        borderaxespad=0.0,
    )
    for i in range(len(lgnd.legendHandles)):
        lgnd.legendHandles[i]._sizes = [50]

    allax.set_xlabel(r"Frequency ($\mu$Hz)")
    allax.set_ylabel(r"Frequency ratio")

    pp.savefig(allfig, bbox_inches="tight")

    if output is not None:
        print("Saved figure to " + output)
        pp.close()


def epsilon_difference_diagram(
    mod,
    modkey,
    moddnu,
    obsfreqdata,
    obsfreqmeta,
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
    obsfreqdata : dict
        Requested frequency-dependent data such as glitches, ratios, and
        epsilon difference. It also contains the covariance matrix and its
        inverse of the individual frequency modes.
        The keys correspond to the science case, e.g. `r01a, `glitch`, or
        `e012`.
        Inside each case, you find the data (`data`), the covariance matrix
        (`cov`), and its inverse (`covinv`).
    obsfreqmeta : dict
        The requested information about which frequency products to fit or
        plot, unpacked for easier access later.
    output : str
        Name and path of output plotfile.
    """

    delab = r"$\delta\epsilon^{%s}_{0%d}$"

    epsdifftype = obsfreqmeta["epsdiff"]["plot"][0]

    obsepsdiff = obsfreqdata[epsdifftype]["data"]
    obsepsdiff_cov = obsfreqdata[epsdifftype]["cov"]
    obsepsdiff_err = np.sqrt(np.diag(obsepsdiff_cov))

    l_available = [int(ll) for ll in set(obsepsdiff[2])]
    lindex = np.zeros(mod.shape[1], dtype=bool)

    for ll in [0, *l_available]:
        lindex |= modkey[0] == ll
    mod = mod[:, lindex]
    modkey = modkey[:, lindex]

    modepsdiff = freq_fit.compute_epsilondiffseqs(
        modkey,
        mod,
        moddnu,
        epsdifftype,
    )

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
            marker=modmarkers["l" + str(ll)],
            color=colors["l" + str(ll)],
            lw=0,
        )
        ax.plot(fnew, spline(fnew), "-", color=splinecolor, zorder=-1)

        # Model at observed
        (modobs,) = ax.plot(
            obsepsdiff[1][indobs],
            spline(obsepsdiff[1][indobs]),
            marker=splinemarkers[ll],
            color="k",
            markeredgewidth=2,
            alpha=0.7,
            lw=0,
        )

        # Observed with uncertainties
        obsdot = ax.errorbar(
            obsepsdiff[1][indobs],
            obsepsdiff[0][indobs],
            yerr=obsepsdiff_err[indobs],
            marker=obsmarker,
            color=colors["l" + str(ll)],
            markeredgewidth=0.5,
            markeredgecolor="k",
            zorder=3,
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

    lgnd = ax.legend(
        h,
        l,
        bbox_to_anchor=(0.0, 1.02, 1.0, 0.102),
        loc=8,
        ncol=9,
        mode="expand",
        borderaxespad=0.0,
    )
    for i in range(len(lgnd.legendHandles)):
        lgnd.legendHandles[i]._sizes = [50]

    ax.set_xlabel(r"Frequency ($\mu$Hz)")
    ax.set_ylabel(r"Epsilon difference $\delta\epsilon_{0\ell}$")

    fig.tight_layout()
    if output is not None:
        print("Saved figure to " + output)
        fig.savefig(output)
    else:
        return fig


def epsilon_difference_all_diagram(
    mod,
    modkey,
    moddnu,
    obs,
    obskey,
    dnudata,
    obsfreqdata,
    obsfreqmeta,
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
    obs : array
        Array of observed frequency modes.
    obskey : array
        Array of mode identification of observed frequency modes.
    dnudata : float
        Inputted average large frequency separation (dnu) of observations.
    obsfreqdata : dict
        Requested frequency-dependent data such as glitches, ratios, and
        epsilon difference. It also contains the covariance matrix and its
        inverse of the individual frequency modes.
        The keys correspond to the science case, e.g. `r01a, `glitch`, or
        `e012`.
        Inside each case, you find the data (`data`), the covariance matrix
        (`cov`), and its inverse (`covinv`).
    obsfreqmeta : dict
        The requested information about which frequency products to fit or
        plot, unpacked for easier access later.
    output : str
        Name and path of output plotfile.
    """

    # Prepared labels and markers
    delab = r"$\delta\epsilon^{%s}_{0%d}$"
    elab = r"$\epsilon_{%d}$"
    colab = r"$\delta\epsilon_{0%d}(%d)$"

    epsdifftype = obsfreqmeta["epsdiff"]["plot"][0]

    obsepsdiff = obsfreqdata[epsdifftype]["data"]
    obsepsdiff_cov = obsfreqdata[epsdifftype]["cov"]
    obsepsdiff_err = np.sqrt(np.diag(obsepsdiff_cov))

    l_available = [int(ll) for ll in set(obsepsdiff[2])]
    lindex = np.zeros(mod.shape[1], dtype=bool)

    for ll in [0, *l_available]:
        lindex |= modkey[0] == ll
    mod = mod[:, lindex]
    modkey = modkey[:, lindex]

    modepsdiff = freq_fit.compute_epsilondiffseqs(
        modkey,
        mod,
        moddnu,
        epsdifftype,
    )

    # Recompute to determine if possible but extrapolated modes
    edextrapol = freq_fit.compute_epsilondiffseqs(
        obskey,
        obs,
        dnudata,
        epsdifftype,
    )
    nu12 = edextrapol[1][edextrapol[2] > 0]
    nu0 = obs[0][obskey[0] == 0]
    expol = np.where(np.logical_or(nu12 < min(nu0), nu12 > max(nu0)))[0]

    # All parameters needed from inverse covariance
    Dinv = np.diag(1 / np.sqrt(np.diag(obsepsdiff_cov)))
    cor = Dinv @ obsepsdiff_cov @ Dinv

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
            marker=modmarkers["l" + str(ll)],
            color=colors["l" + str(ll)],
            markeredgewidth=0.5,
            markeredgecolor="k",
            zorder=3,
            linestyle="",
            label=delab % ("", ll),
        )
        ax[1, 1].plot(fnew, spline(fnew), "-", color=splinecolor, zorder=-1)

        # Constrained model range
        indmod &= modepsdiff[1] > min(obsepsdiff[1]) - 3 * moddnu
        indmod &= modepsdiff[1] < max(obsepsdiff[1]) + 3 * moddnu
        spline = CubicSpline(modepsdiff[1][indmod], modepsdiff[0][indmod])
        fnew = np.linspace(min(modepsdiff[1][indmod]), max(modepsdiff[1][indmod]), 100)

        # Model
        (moddot,) = ax[0, 0].plot(
            modepsdiff[1][indmod],
            modepsdiff[0][indmod],
            marker=modmarkers["l" + str(ll)],
            color=colors["l" + str(ll)],
            lw=0,
        )
        ax[0, 0].plot(fnew, spline(fnew), "-", color=splinecolor, zorder=-1)

        # Model at observed
        (modobs,) = ax[0, 0].plot(
            obsepsdiff[1][indobs],
            spline(obsepsdiff[1][indobs]),
            marker=splinemarkers[ll],
            color="k",
            markeredgewidth=2,
            alpha=0.7,
            lw=0,
        )

        # Observed with uncertainties
        obsdot = ax[0, 0].errorbar(
            obsepsdiff[1][indobs],
            obsepsdiff[0][indobs],
            yerr=obsepsdiff_err[indobs],
            marker=obsmarker,
            color=colors["l" + str(ll)],
            markeredgewidth=0.5,
            markeredgecolor="k",
            zorder=3,
        )

        # Spline observed for separate plot
        spline = CubicSpline(obsepsdiff[1][indobs], obsepsdiff[0][indobs])
        fnew = np.linspace(min(obsepsdiff[1][indobs]), max(obsepsdiff[1][indobs]), 100)

        # Observed with uncertainties and spline
        ax[1, 0].errorbar(
            obsepsdiff[1][indobs],
            obsepsdiff[0][indobs],
            yerr=obsepsdiff_err[indobs],
            marker=obsmarker,
            color=colors["l" + str(ll)],
            markeredgewidth=0.5,
            markeredgecolor="k",
            linestyle="",
            zorder=3,
        )
        ax[1, 0].plot(fnew, spline(fnew), "-", color="0.7", zorder=-1)

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
                edextrapol[1][expol][edextrapol[2][expol] == ll],
                edextrapol[0][expol][edextrapol[2][expol] == ll],
                marker=obsmarker,
                lw=0,
                alpha=0.5,
                color=colors["l" + str(ll)],
                label=r"$\nu(\ell={0})\,\notin\,\nu(\ell=0)$".format(ll),
            )
        ax[1, 0].legend()

    # Individual epsilons
    for ll in [0, *l_available]:
        # Extract observed quantities
        indobs = obskey[0] == ll
        fre = obs[0][indobs]
        eps = fre / dnudata - obskey[1][indobs] - ll / 2
        err = obs[1][indobs] / dnudata
        intpol = CubicSpline(fre, eps)
        fnew = np.linspace(min(fre), max(fre), 100)

        # Plot observed w. spline
        ax[2, 0].errorbar(
            fre,
            eps,
            yerr=err,
            linestyle=None,
            fmt=obsmarker,
            color=colors["l" + str(ll)],
            zorder=3,
            label=elab % ll,
        )
        ax[2, 0].plot(
            fnew,
            intpol(fnew),
            "-",
            color=splinecolor,
        )

        # Extract model quantities
        indmod = modkey[0] == ll
        fre = mod[0][indmod]
        eps = fre / moddnu - modkey[1][indmod] - ll / 2
        err = mod[1][indmod] / moddnu
        intpol = CubicSpline(fre, eps)
        fnew = np.linspace(min(fre), max(fre), 1000)

        # Plot model w. spline
        ax[2, 1].errorbar(
            fre,
            eps,
            yerr=err,
            linestyle=None,
            fmt=modmarkers["l" + str(ll)],
            color=colors["l" + str(ll)],
            zorder=3,
            label=elab % ll,
        )
        ax[2, 1].plot(fnew, intpol(fnew), "-", color=splinecolor)

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

    ax[0, 0].set_xlabel(r"Frequency ($\mu$Hz)")
    ax[0, 0].set_ylabel(r"Epsilon difference $\delta\epsilon_{0\ell}$")
    ax[0, 1].set_xticklabels([])
    ax[0, 1].set_yticks(range(obsepsdiff.shape[1]))
    ax[0, 1].set_yticklabels(labs)
    ax[1, 0].set_ylabel(r"Epsilon difference $\delta\epsilon_{0\ell}$")
    ax[2, 0].set_xlabel(r"Frequency ($\mu$Hz)")
    ax[2, 0].set_ylabel(r"Epsilon $\epsilon_{\ell}$")
    ax[2, 1].set_xlabel(r"Frequency ($\mu$Hz)")

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


def epsilon_diff_and_correlation(
    depsN, depsL, covN, covL, osc, osckey, avgdnu, filename
):
    """
    Subroutine only run when --debug as been invoked. It produces correlation maps and
    diagrams of the observed epsilon differences. The correlation maps are produced
    using the two different sorting methods, to detail the differences between the
    "01" and "02" sequence, if both are available.

    Parameters
    ----------
    depsN : array
        The epsilon differences and their identifying information, sorted by n before l.
    depsL : array
        Same as depsN, but sorted as l before n.
    covN : array
        Covariance matrix of the n before l sorted epsilon differences.
    covL : array
        Same as covN, but sorted as l before n.
    osc : array
        Observed oscillation frequencies.
    osckey : array
        l and n of the observed oscillation frequencies.
    avgdnu : float
        Average large frequency separation for computation of epsilons.
    filename : str
        Name to save the figure to.
    """
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
            color=colors["l" + str(ll)],
            zorder=3,
            label=r"$\epsilon_{%d}$" % (ll),
        )
        intpol = CubicSpline(fre, eps)
        if ll == 0:
            fnew = np.linspace(min(osc[0]) - avgdnu, max(osc[0]) + avgdnu, 100)

            ax[1, 0].plot(fnew, intpol(fnew), "-", color=colors["l" + str(ll)])
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
            color=colors["l" + str(ll)],
            zorder=3,
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
    fig.savefig(filename)
    plt.close(fig)


def ratio_cormap(obsfreqdata, obsfreqmeta, output):
    """
    Routine for plotting a correlation map of the plotted ratios

    Parameters
    ----------
    obsfreqdata : dict
        All necessary frequency related data from observations
    obsfreqmeta : dict
        Metadata defining the content of `obsfreqdata`
    output : str
        Name and path to output figure
    """

    # Extract data
    sequence = obsfreqmeta["ratios"]["plot"][0]
    data = obsfreqdata[sequence]["data"]
    cov = obsfreqdata[sequence]["cov"]

    # Compute correlations
    Dinv = np.diag(1 / np.sqrt(np.diag(cov)))
    cor = Dinv @ cov @ Dinv

    # Make labels
    labs = [
        r"$r_{%02d}(%d)$" % (int(l), int(n)) for l, n in zip(data[2, :], data[3, :])
    ]

    # Produce figure
    fig, ax = plt.subplots(1, 1, figsize=(7.3, 6))
    im = ax.imshow(cor, cmap="RdBu_r", vmin=-1, vmax=1)
    plt.colorbar(im)

    # Beautify
    ax.set_xticks(range(data.shape[1]))
    ax.set_xticklabels(labs, rotation=90)
    ax.set_yticks(range(data.shape[1]))
    ax.set_yticklabels(labs)
    fig.tight_layout()
    fig.savefig(output)
    plt.close(fig)
