"""
Production of asteroseismic plots
"""

import os
import h5py
import numpy as np
import matplotlib
import typing

from scipy.interpolate import interp1d, CubicSpline

from basta import utils_seismic as su
from basta import stats, freq_fit
from basta.constants import freqtypes
from basta.downloader import get_basta_dir

# Set the style of all plots
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.transforms as transforms

plt.style.use(os.path.join(get_basta_dir(), "plots.mplstyle"))

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
    selectedmodels: dict,
    Grid: h5py.File,
    obs: np.ndarray,
    obskey: np.ndarray,
    mod: typing.Optional[np.ndarray] = None,
    modkey: typing.Optional[np.ndarray] = None,
    dnu: float = None,
    join: typing.Optional[np.ndarray] = None,
    joinkeys: typing.Optional[np.ndarray] | bool = False,
    coeffs: typing.Optional[np.ndarray] | None = None,
    scalnu: float | None = None,
    freqcor: str = "BG14",
    pairmode: bool = False,
    duplicatemode: bool = False,
    outputfilename: str | None = None,
) -> None:
    """
    Echelle diagram. It is possible to either make a single Echelle diagram
    or plot it twice making patterns across the moduli-limit easier to see.

    Parameters
    ----------
    selectedmodels : dict
        Contains information on all models with a non-zero likelihood.
    Grid : hdf5 object
        The already loaded grid, containing the tracks/isochrones.
    obs : array or None
        Array of observed modes.
    obskey : array or None
        Array of mode identification observed modes.
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
    pairmode : bool
        Flag determining whether to link matched observed and modelled
        frequencies.
    duplicatemode : bool
        Flag determining whether to plot two echelle diagrams next to one
        another.
    outputfilename : str or None
        Filename for saving the figure.
    """
    if pairmode:
        lw = 1
    else:
        lw = 0

    if dnu is None:
        print("Note: No deltanu specified, using dnufit for echelle diagram.")
        maxPDF_path, maxPDF_ind = stats.most_likely(selectedmodels)
        dnu = Grid[maxPDF_path + "/dnufit"][maxPDF_ind]

    if duplicatemode:
        modx = 1
        scalex = dnu
    else:
        modx = dnu
        scalex = 1

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
    if duplicatemode:
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
        if duplicatemode:
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
        if duplicatemode:
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
                    label=f"Best fit $\\ell={l}$",
                )
                if duplicatemode:
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
                    label=f"Measured $\\ell={l}$",
                )
                if duplicatemode:
                    ax.errorbar(
                        fobs[l] % modx - modx,
                        fobs[l],
                        xerr=eobs[l],
                        fmt=obsmarker,
                        mfc=colors["l" + l],
                        ecolor=colors["l" + l],
                        zorder=1,
                    )

                if pairmode:
                    fm = fmod[l]
                    fo = fobs[l]
                    for i in range(len(fm)):
                        if ((fm[i] % modx) > linelimit) & (
                            (fo[i] % modx) < (modx - linelimit)
                        ):
                            if duplicatemode:
                                x0 = -1
                            else:
                                x0 = 0
                            a = (fo[i] - fm[i]) / abs(
                                fo[i] % modx - (fm[i] % modx - modx)
                            )
                            ax.plot(
                                (fm[i] % modx - modx, fo[i] % modx),
                                (fm[i], fo[i]),
                                c=colors["l" + str(l)],
                                alpha=0.7,
                                zorder=10,
                            )
                            ax.plot(
                                (fm[i] % modx, modx),
                                (fm[i], fm[i] + a * (modx - fm[i] % modx)),
                                c=colors["l" + str(l)],
                                alpha=0.7,
                                zorder=10,
                            )
                            ax.plot(
                                (x0, fo[i] % modx - modx),
                                (fm[i] + a * (modx - fm[i] % modx), fo[i]),
                                c=colors["l" + str(l)],
                                alpha=0.7,
                                zorder=10,
                            )
                        elif ((fo[i] % modx) > linelimit) & (
                            (fm[i] % modx) < (modx - linelimit)
                        ):
                            if duplicatemode:
                                x0 = -1
                            else:
                                x0 = 0
                            a = (fm[i] - fo[i]) / abs(
                                fm[i] % modx - (fo[i] % modx - modx)
                            )
                            ax.plot(
                                (fo[i] % modx - modx, fm[i] % modx),
                                (fo[i], fm[i]),
                                c=colors["l" + str(l)],
                                alpha=0.7,
                                zorder=10,
                            )
                            ax.plot(
                                (fo[i] % modx, modx),
                                (fo[i], fo[i] + a * (modx - fo[i] % modx)),
                                c=colors["l" + str(l)],
                                alpha=0.7,
                                zorder=10,
                            )
                            ax.plot(
                                (x0, fm[i] % modx - modx),
                                (fo[i] + a * (modx - fo[i] % modx), fm[i]),
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
                            if duplicatemode:
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
                        duplicatemode
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
    for i in range(len(lgnd.legend_handles)):
        lgnd.legend_handles[i]._sizes = [50]

    if duplicatemode:
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

    if outputfilename is not None:
        plt.savefig(outputfilename, bbox_inches="tight")
        print("Saved figure to " + outputfilename)
        plt.close(fig)


def ratioplot(
    obsfreqdata,
    joinkeys,
    join,
    modkey,
    mod,
    ratiotype,
    outputfilename=None,
    threepoint=False,
    interp_ratios=True,
):
    """
    Plot frequency ratios.

    Parameters
    ----------
    obsfreqdata : dict
        Requested frequency-dependent data such as glitches, ratios, and
        epsilon difference. It also contains the covariance matrix and its
        inverse of the individual frequency modes.
        The keys correspond to the science case, e.g. `r01a, `glitch`, or
        `e012`.
        Inside each case, you find the data (`data`), the covariance matrix
        (`cov`), and its inverse (`covinv`).
    joinkeys : array
        Array containing the mode identification of the matched observed
        and modelled modes.
    join : array
        Array containing the matched observed and modelled modes.
    modkey : array
        Array containing the mode identification of the modelled modes.
    mod : array
        Array containing the modelled modes.
    ratiotype : str
        Key for the ratio sequence to be plotted (e.g. `r01`, `r02`, `r012`)
    outputfilename : str or None, optional
        Filename for saving the plot.
    nonewfig : bool, optional
        If True, this creates a new canvas. Otherwise, the plot is added
        to the existing canvas.
    threepoint : bool
        If True, use three point definition of r01 and r10 ratios instead
        of default five point definition.
    interp_ratios : bool
        If True (default), plot how the model ratios are linearly interpolated
        to the frequencies of the observed ratios, in order to compare the
        sequences at the same frequencies.
    """
    fig, ax = plt.subplots(1, 1)

    obsratio = obsfreqdata[ratiotype]["data"]
    # Exit if there are no ratios to plot
    if obsratio is None:
        plt.close(fig)
        return

    obsratio_cov = obsfreqdata[ratiotype]["cov"]
    obsratio_err = np.sqrt(np.diag(obsratio_cov))

    if interp_ratios:
        modratio = freq_fit.compute_ratioseqs(
            modkey, mod, ratiotype, threepoint=threepoint
        )
    else:
        modratio = freq_fit.compute_ratioseqs(
            joinkeys, join[0:2, :], ratiotype, threepoint=threepoint
        )

    handles = []
    for rtype in set(obsratio[2, :]):
        obsmask = obsratio[2, :] == rtype
        modmask = modratio[2, :] == rtype
        rtname = "r{:02d}".format(int(rtype))
        modp = ax.scatter(
            modratio[1, modmask],
            modratio[0, modmask],
            marker=modmarkers["ratio"],
            color=colors[rtname],
            edgecolors="k",
            zorder=3,
            label=r"Best fit ($r_{{{:02d}}}$)".format(int(rtype)),
        )
        ax.plot(
            modratio[1, modmask],
            modratio[0, modmask],
            "-",
            color="darkgrey",
            alpha=0.9,
            zorder=-1,
        )

        obsp = ax.errorbar(
            obsratio[1, obsmask],
            obsratio[0, obsmask],
            yerr=obsratio_err[obsmask],
            marker=obsmarker,
            color=colors[rtname],
            mec="k",
            mew=0.5,
            linestyle="None",
            zorder=3,
            label=r"Measured ($r_{{{:02d}}}$)".format(int(rtype)),
        )
        ax.plot(
            obsratio[1, obsmask],
            obsratio[0, obsmask],
            "-",
            color=colors[rtname],
            zorder=-1,
        )

        if interp_ratios:
            intfunc = interp1d(
                modratio[1, modmask], modratio[0, modmask], kind="linear"
            )
            # When only plotting, not fitting, model freqs can be outside observed range
            rangemask = np.ones(sum(obsmask), dtype=bool)
            rangemask &= obsratio[1, obsmask] > min(modratio[1, modmask])
            rangemask &= obsratio[1, obsmask] < max(modratio[1, modmask])
            newmod = intfunc(obsratio[1, obsmask][rangemask])
            marker = splinemarkers[1] if "1" in str(rtype) else splinemarkers[2]
            (intp,) = ax.plot(
                obsratio[1, obsmask][rangemask],
                newmod,
                marker=marker,
                color="k",
                markeredgewidth=2,
                alpha=0.7,
                lw=0,
                zorder=5,
                label=r"$r_{{{:02d}}}(\nu^{{\mathrm{{obs}}}})$".format(int(rtype)),
            )
            handles.extend([modp, intp, obsp])
        else:
            handles.extend([modp, obsp])

    nbase = 3 if interp_ratios else 2
    lgnd = ax.legend(
        handles,
        [h.get_label() for h in handles],
        bbox_to_anchor=(0.0, 1.02, 1.0, 0.102),
        loc=8,
        ncol=nbase * len(set(obsratio[2, :])),
        mode="expand",
        borderaxespad=0.0,
    )
    for i in range(len(lgnd.legend_handles)):
        lgnd.legend_handles[i]._sizes = [50]

    ax.set_xlabel(r"Frequency ($\mu$Hz)")
    ax.set_ylabel(f"Frequency ratio ({ratiotype})")
    ylim = ax.get_ylim()
    ax.set_ylim(max(ylim[0], 0), ylim[1])

    if outputfilename is not None:
        fig.savefig(outputfilename, bbox_inches="tight")
        print("Saved figure to " + outputfilename)
        plt.close(fig)


def confidence_ellipse(
    mean_x, std_x, mean_y, std_y, cov, ax, facecolor="none", **kwargs
):
    """
    Create a plot of the covariance confidence ellipse of *x* and *y*.

    Parameters
    ----------
    mean_x : float
        Mean of input x
    std_x : float
        Standard deviation of input x
    mean_y : float
        Mean of input y
    std_y : float
        Standard deviation of input y
    cov : float
        Covariance of x and y
    ax : matplotlib.axes.Axes
        Axes object to draw the ellipse into.
    **kwargs
        Forwarded to `~matplotlib.patches.Ellipse`
    Returns
    -------
    matplotlib.patches.Ellipse
    """

    pearson_correlation = cov / (std_x * std_y)

    ell_radius_x = np.sqrt(1 + pearson_correlation)
    ell_radius_y = np.sqrt(1 - pearson_correlation)

    ellipse = patches.Ellipse(
        (0, 0),
        width=ell_radius_x * 2,
        height=ell_radius_y * 2,
        facecolor=facecolor,
        **kwargs,
    )

    transf = (
        transforms.Affine2D()
        .rotate_deg(45)
        .scale(std_x, std_y)
        .translate(mean_x, mean_y)
    )

    ellipse.set_transform(transf + ax.transData)

    return ax.add_patch(ellipse)


def glitchplot(
    obsfreqdata,
    sequence,
    modelvalues,
    maxPath,
    maxInd,
    outputfilename,
):
    labels = {
        7: r"$\langle A_{\mathrm{He}}\rangle$ ($\mu$Hz)",
        8: r"$\Delta_{\mathrm{He}}$ (s)",
        9: r"$\tau_{\mathrm{He}}$ (s)",
    }

    # Read in data
    obsparams = obsfreqdata[sequence]["data"]
    obs_cov = obsfreqdata[sequence]["cov"]
    obs_err = np.sqrt(np.diag(obs_cov))

    # Start figure
    fig, ax = plt.subplots(2, 2, figsize=(8, 8))
    fig.delaxes(ax[0, 1])

    # Loop over each track to plot
    for path, trackparams in modelvalues.items():
        AHe = trackparams.AHe
        dHe = trackparams.dHe[AHe > 1e-14]
        tauHe = trackparams.tauHe[AHe > 1e-14]
        AHe = AHe[AHe > 1e-14]

        ax[1, 0].plot(AHe, dHe, ".", color="grey", ms=5, zorder=1)
        ax[0, 0].plot(AHe, tauHe, ".", color="grey", ms=5, zorder=1)
        ax[1, 1].plot(tauHe, dHe, ".", color="grey", ms=5, zorder=1)

    # AHe vs dHe
    ax[1, 0].errorbar(
        obsparams[0, obsparams[2, :] == 7.0],
        obsparams[0, obsparams[2, :] == 8.0],
        xerr=obs_err[obsparams[2, :] == 7.0],
        yerr=obs_err[obsparams[2, :] == 8.0],
        marker=".",
        linestyle="None",
        color="#D55E00",
        zorder=1,
        label="Measured",
    )
    ax[1, 0].plot(
        modelvalues[maxPath].AHe[maxInd],
        modelvalues[maxPath].dHe[maxInd],
        "*",
        ms=20,
        color="#0072B2",
        zorder=2,
        label="Best fit",
    )
    confidence_ellipse(
        obsparams[0, obsparams[2, :] == 7.0],
        obs_err[obsparams[2, :] == 7.0],
        obsparams[0, obsparams[2, :] == 8.0],
        obs_err[obsparams[2, :] == 8.0],
        obs_cov[obsparams[2, :] == 7.0, obsparams[2, :] == 8.0],
        ax[1, 0],
        edgecolor="#D55E00",
        lw=1.5,
        alpha=0.5,
    )
    ax[1, 0].set_xlabel(labels[7])
    ax[1, 0].set_ylabel(labels[8])

    # AHe vs tauHe
    ax[0, 0].errorbar(
        obsparams[0, obsparams[2, :] == 7.0],
        obsparams[0, obsparams[2, :] == 9.0],
        xerr=obs_err[obsparams[2, :] == 7.0],
        yerr=obs_err[obsparams[2, :] == 9.0],
        marker=".",
        linestyle="None",
        color="#D55E00",
        zorder=1,
        label="Measured",
    )
    ax[0, 0].plot(
        modelvalues[maxPath].AHe[maxInd],
        modelvalues[maxPath].tauHe[maxInd],
        "*",
        ms=20,
        color="#0072B2",
        zorder=2,
        label="Best fit",
    )
    confidence_ellipse(
        obsparams[0, obsparams[2, :] == 7.0],
        obs_err[obsparams[2, :] == 7.0],
        obsparams[0, obsparams[2, :] == 9.0],
        obs_err[obsparams[2, :] == 9.0],
        obs_cov[obsparams[2, :] == 7.0, obsparams[2, :] == 9.0],
        ax[0, 0],
        edgecolor="#D55E00",
        lw=1.5,
        alpha=0.5,
    )
    ax[0, 0].set_ylabel(labels[9])
    ax[0, 0].legend(bbox_to_anchor=(1.01, 1), loc="upper left", ncol=1)

    # tauHe vs dHe
    ax[1, 1].errorbar(
        obsparams[0, obsparams[2, :] == 9.0],
        obsparams[0, obsparams[2, :] == 8.0],
        xerr=obs_err[obsparams[2, :] == 9.0],
        yerr=obs_err[obsparams[2, :] == 8.0],
        marker=".",
        linestyle="None",
        color="#D55E00",
        zorder=1,
        label="Measured",
    )
    ax[1, 1].plot(
        modelvalues[maxPath].tauHe[maxInd],
        modelvalues[maxPath].dHe[maxInd],
        "*",
        ms=20,
        color="#0072B2",
        zorder=2,
        label="Best fit",
    )
    confidence_ellipse(
        obsparams[0, obsparams[2, :] == 9.0],
        obs_err[obsparams[2, :] == 9.0],
        obsparams[0, obsparams[2, :] == 8.0],
        obs_err[obsparams[2, :] == 8.0],
        obs_cov[obsparams[2, :] == 9.0, obsparams[2, :] == 8.0],
        ax[1, 1],
        edgecolor="#D55E00",
        lw=1.5,
        alpha=0.5,
    )
    ax[1, 1].set_xlabel(labels[9])

    if outputfilename is not None:
        fig.savefig(outputfilename, bbox_inches="tight")
        print("Saved figure to " + outputfilename)
        plt.close(fig)


def epsilon_difference_diagram(
    mod,
    modkey,
    moddnu,
    sequence,
    obsfreqdata,
    outputfilename,
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
    sequence : str
        The sequence to be plotted
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
    outputfilename : str
        Name and path of outputfilename plotfile.
    """

    delab = r"$\delta\epsilon^{%s}_{0%d}$"

    obsepsdiff = obsfreqdata[sequence]["data"]
    obsepsdiff_cov = obsfreqdata[sequence]["cov"]
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
        sequence,
    )

    # Mixed modes results in negative differences. Flag using nans, not displayed
    mask = np.where(modepsdiff[0, :] < 0)[0]
    modepsdiff[0, mask] = np.nan

    fig, ax = plt.subplots(1, 1)
    handles, legends = [], []
    for ll in l_available:
        indobs = obsepsdiff[2] == ll
        indmod = modepsdiff[2] == ll
        indmod &= modepsdiff[1] > min(obsepsdiff[1]) - 3 * moddnu
        indmod &= modepsdiff[1] < max(obsepsdiff[1]) + 3 * moddnu
        indmod &= ~np.isnan(modepsdiff[0])

        # Model with spline
        (moddot,) = ax.plot(
            modepsdiff[1][indmod],
            modepsdiff[0][indmod],
            marker=modmarkers["l" + str(ll)],
            color=colors["l" + str(ll)],
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

        if sum(indmod) > 1:
            spline = CubicSpline(
                modepsdiff[1][indmod], modepsdiff[0][indmod], extrapolate=False
            )
            fnew = np.linspace(
                min(modepsdiff[1][indmod]), max(modepsdiff[1][indmod]), 100
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

            handles.extend([moddot, obsdot, modobs])
            legends.extend(
                [
                    delab % ("mod", ll),
                    delab % ("obs", ll),
                    delab % ("mod", ll) + r"$(\nu^{obs})$",
                ]
            )
        else:
            handles.extend([moddot, obsdot])
            legends.extend(
                [
                    delab % ("mod", ll),
                    delab % ("obs", ll),
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
    for i in range(len(lgnd.legend_handles)):
        lgnd.legend_handles[i]._sizes = [50]

    ax.set_xlabel(r"Frequency ($\mu$Hz)")
    ax.set_ylabel(r"Epsilon difference $\delta\epsilon_{0\ell}$")
    ylim = ax.get_ylim()
    ax.set_ylim(max(ylim[0], 0), ylim[1])

    fig.tight_layout()
    if outputfilename is not None:
        print("Saved figure to " + outputfilename)
        fig.savefig(outputfilename)
        plt.close(fig)
    else:
        return fig


def correlation_map(fittype, obsfreqdata, outputfilename, obskey=None):
    """
    Routine for plotting a correlation map of the plotted ratios

    Parameters
    ----------
    fittype : str
        The type of frequency product (individual, ratios, epsilon
        differences) for which to to the correlation map of.
    obsfreqdata : dict
        All necessary frequency related data from observations.
    outputfilename : str
        Name and path to outputfilename figure.
    obskey : array, optional
        Contains radial order and degree of frequencies, used if plotting
        for individual frequencies.
    """

    # Determine information for constructing labels
    if fittype in freqtypes.freqs:
        fmtstr = r"$\nu({:d}, {:d})$"
        ln_zip = zip(obskey[0, :], obskey[1, :])

    elif fittype in freqtypes.rtypes:
        data = obsfreqdata[fittype]["data"]
        if data is None:
            return
        fmtstr = r"$r_{{{:02d}}}({{{:d}}})$"
        ln_zip = zip(data[2, :], data[3, :])

    elif fittype in freqtypes.epsdiff:
        data = obsfreqdata[fittype]["data"]
        if data is None:
            return
        fmtstr = r"$\delta\epsilon_{{{:02d}}}({{{:d}}})$"
        ln_zip = zip(data[2, :], data[3, :])

    elif fittype in freqtypes.glitches:
        data = obsfreqdata[fittype]["data"]
        if data is None:
            return
        fmtstr = r"$r_{{{:02d}}}({{{:d}}})$"
        if fittype != "glitches":
            ln_zip = zip(data[2, :-3], data[3, :-3])
        else:
            ln_zip = []

    # Construct labels
    labs = []
    for l, n in ln_zip:
        labs.append(fmtstr.format(int(l), int(n)))

    # Append special glitches labels
    if fittype in freqtypes.glitches:
        glitchlabels = {
            7: r"$\langle A_{\mathrm{He}}\rangle$ ($\mu$Hz)",
            8: r"$\Delta_{\mathrm{He}}$ (s)",
            9: r"$\tau_{\mathrm{He}}$ (s)",
        }

        for glitchtype in data[2, -3:]:
            labs.append(glitchlabels[int(glitchtype)])

    # Compute correlations
    cov = obsfreqdata[fittype]["cov"]
    Dinv = np.diag(1 / np.sqrt(np.diag(cov)))
    cor = Dinv @ cov @ Dinv

    # Produce figure
    fig, ax = plt.subplots(1, 1, figsize=(7.3, 6))
    im = ax.imshow(cor, cmap="RdBu_r", vmin=-1, vmax=1)
    plt.colorbar(im)

    # Beautify
    ax.set_xticks(range(cov.shape[1]))
    ax.set_xticklabels(labs, rotation=90)
    ax.set_yticks(range(cov.shape[1]))
    ax.set_yticklabels(labs)
    fig.tight_layout()

    if outputfilename is not None:
        fig.savefig(outputfilename, bbox_inches="tight")
        print("Saved figure to " + outputfilename)
        plt.close(fig)


###############
# DEBUG PLOTS #
###############


def epsilon_difference_components_diagram(
    mod,
    modkey,
    moddnu,
    obs,
    obskey,
    dnudata,
    obsfreqdata,
    obsfreqmeta,
    outputfilename,
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
    outputfilename : str
        Name and path of outputfilename plotfile.
    """

    # Prepared labels and markers
    delab = r"$\delta\epsilon^{%s}_{0%d}$"
    elab = r"$\epsilon_{%d}$"

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

    # Definition of figure
    fig, ax = plt.subplots(2, 2, figsize=(8.47, 6), sharex="col", sharey="row")

    # Epsilon differences
    for ll in l_available:
        indobs = obsepsdiff[2] == ll
        indmod = modepsdiff[2] == ll

        # Constrained model range
        indmod &= modepsdiff[1] > min(obsepsdiff[1]) - 3 * moddnu
        indmod &= modepsdiff[1] < max(obsepsdiff[1]) + 3 * moddnu
        spline = CubicSpline(modepsdiff[1][indmod], modepsdiff[0][indmod])
        fnew = np.linspace(min(modepsdiff[1][indmod]), max(modepsdiff[1][indmod]), 1000)

        # Raw model with spline
        ax[0, 1].errorbar(
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
        ax[0, 1].plot(fnew, spline(fnew), "-", color=splinecolor, zorder=-1)

        # Spline observed for separate plot
        spline = CubicSpline(obsepsdiff[1][indobs], obsepsdiff[0][indobs])
        fnew = np.linspace(min(obsepsdiff[1][indobs]), max(obsepsdiff[1][indobs]), 100)

        # Observed with uncertainties and spline
        ax[0, 0].errorbar(
            obsepsdiff[1][indobs],
            obsepsdiff[0][indobs],
            yerr=obsepsdiff_err[indobs],
            marker=modmarkers["l" + str(ll)],
            color=colors["l" + str(ll)],
            markeredgewidth=0.5,
            markeredgecolor="k",
            linestyle="",
            zorder=3,
        )
        ax[0, 0].plot(fnew, spline(fnew), "-", color="0.7", zorder=-1)

    # Potential extrapolated points
    if len(expol):
        for ll in set(edextrapol[2][expol].astype(int)):
            ax[0, 0].plot(
                edextrapol[1][expol][edextrapol[2][expol] == ll],
                edextrapol[0][expol][edextrapol[2][expol] == ll],
                marker=obsmarker,
                lw=0,
                alpha=0.5,
                color=colors["l" + str(ll)],
                label=r"$\nu(\ell={0})\,\notin\,\nu(\ell=0)$".format(ll),
            )
        ax[0, 0].legend()

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
        ax[1, 0].errorbar(
            fre,
            eps,
            yerr=err,
            linestyle=None,
            fmt=modmarkers["l" + str(ll)],
            color=colors["l" + str(ll)],
            zorder=3,
            label=elab % ll,
        )
        ax[1, 0].plot(
            fnew,
            intpol(fnew),
            "-",
            color=splinecolor,
        )

        # Extract model quantities
        indmod = modkey[0] == ll
        indmod &= mod[0] > min(obsepsdiff[1]) - 3 * moddnu
        indmod &= mod[0] < max(obsepsdiff[1]) + 3 * moddnu
        fre = mod[0][indmod]
        eps = fre / moddnu - modkey[1][indmod] - ll / 2
        err = mod[1][indmod] / moddnu
        intpol = CubicSpline(fre, eps)
        fnew = np.linspace(min(fre), max(fre), 1000)

        # Plot model w. spline
        ax[1, 1].errorbar(
            fre,
            eps,
            yerr=err,
            linestyle=None,
            fmt=modmarkers["l" + str(ll)],
            color=colors["l" + str(ll)],
            zorder=3,
            label=elab % ll,
        )
        ax[1, 1].plot(fnew, intpol(fnew), "-", color=splinecolor)

    ax[0, 1].legend(fontsize=16, bbox_to_anchor=(1.02, 1), loc=2, borderaxespad=0.0)
    ax[1, 1].legend(fontsize=16, bbox_to_anchor=(1.02, 1), loc=2)

    # Axes labels
    ax[0, 0].set_ylabel(r"Epsilon difference $\delta\epsilon_{0\ell}$")
    ax[1, 0].set_xlabel(r"Frequency ($\mu$Hz)")
    ax[1, 0].set_ylabel(r"Epsilon $\epsilon_{\ell}$")
    ax[1, 1].set_xlabel(r"Frequency ($\mu$Hz)")

    # Titles
    ax[0, 0].set_title(r"Observed", fontsize=18)
    ax[0, 1].set_title(r"Model", fontsize=18)

    fig.tight_layout()
    # fig.subplots_adjust(wspace=0.2, hspace=0.4)

    fig.savefig(outputfilename)
    print("Saved figure to " + outputfilename)
    plt.close(fig)
