"""
Production of asteroseismic plots
"""

import os
import typing
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import h5py  # type: ignore[import]
import matplotlib as mpl
import numpy as np
from scipy.interpolate import CubicSpline, interp1d  # type: ignore[import]

from basta import core, freq_fit, stats, surfacecorrections
from basta import utils_seismic as su
from basta.constants import freqtypes
from basta.downloader import get_basta_dir
from basta.utils_general import compute_matrix_inverse

# Set the style of all plots
mpl.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import patches, transforms

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


@dataclass(kw_only=True, frozen=True)
class EchellePlotBase:
    selectedmodels: dict
    Grid: h5py.File
    model_modes: core.ModelFrequencies
    joinedmodes: core.JoinedModes | None = None
    coeffs: np.ndarray | None = None
    star: core.Star
    inferencesettings: core.InferenceSettings
    plotconfig: core.PlotConfig
    outputoptions: core.OutputOptions


def _connect_model_observed(
    ax: plt.Axes,
    fmod: np.ndarray,
    fobs: np.ndarray,
    modx: float,
    duplicatemode: bool,
    color: str,
    line_kwargs_base: dict,
) -> None:
    """
    Draw lines connecting model and observed frequencies.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        The axis to draw on.
    fmod : np.ndarray
        Model frequencies.
    fobs : np.ndarray
        Observed frequencies.
    modx : float
        Modulo frequency.
    duplicatemode : bool
        Whether to plot duplicated modx range.
    color : str
        Color of the connecting line.
    line_kwargs_base : dict
        Base kwargs passed to ax.plot.
    """
    linelimit = 0.75 * modx
    for i in range(len(fmod)):
        mod_phase = fmod[i] % modx
        obs_phase = fobs[i] % modx

        if mod_phase > linelimit and obs_phase < (modx - linelimit):
            x0 = -1 if duplicatemode else 0
            slope = (fobs[i] - fmod[i]) / abs(obs_phase - (mod_phase - modx))
            ax.plot(
                [mod_phase - modx, obs_phase],
                [fmod[i], fobs[i]],
                color=color,
                **line_kwargs_base,
            )
            ax.plot(
                [mod_phase, modx],
                [fmod[i], fmod[i] + slope * (modx - mod_phase)],
                color=color,
                **line_kwargs_base,
            )
            ax.plot(
                [x0, obs_phase - modx],
                [fmod[i] + slope * (modx - mod_phase), fobs[i]],
                color=color,
                **line_kwargs_base,
            )

        elif obs_phase > linelimit and mod_phase < (modx - linelimit):
            x0 = -1 if duplicatemode else 0
            slope = (fmod[i] - fobs[i]) / abs(mod_phase - (obs_phase - modx))
            ax.plot(
                [obs_phase - modx, mod_phase],
                [fobs[i], fmod[i]],
                color=color,
                **line_kwargs_base,
            )
            ax.plot(
                [obs_phase, modx],
                [fobs[i], fobs[i] + slope * (modx - obs_phase)],
                color=color,
                **line_kwargs_base,
            )
            ax.plot(
                [x0, mod_phase - modx],
                [fobs[i] + slope * (modx - obs_phase), fmod[i]],
                color=color,
                **line_kwargs_base,
            )
        else:
            ax.plot(
                [mod_phase, obs_phase],
                [fmod[i], fobs[i]],
                color=color,
                **line_kwargs_base,
            )
            if duplicatemode:
                ax.plot(
                    [mod_phase - modx, obs_phase - modx],
                    [fmod[i], fobs[i]],
                    color=color,
                    **line_kwargs_base,
                )


def echelle(
    x: EchellePlotBase,
    pairmode: bool = False,
    duplicatemode: bool = False,
    outputfilename: Path | None = None,
) -> None:
    """
    Echelle diagram. It is possible to either make a single Echelle diagram
    or plot it twice making patterns across the moduli-limit easier to see.

    Parameters
    ----------
    pairmode : bool
        Flag determining whether to link matched observed and modelled
        frequencies.
    duplicatemode : bool
        Flag determining whether to plot two echelle diagrams next to one
        another.
    outputfilename : str or None
        Filename for saving the figure.
    """
    star = x.star
    if star.modes is None:
        return
    dnu = star.globalseismicparams.get_original("dnufit")[0]
    selectedmodels = x.selectedmodels
    joinedmodes = x.joinedmodes
    plotconfig = x.plotconfig

    mpl.rcParams.update(plotconfig.mpl_style)

    lw = 1 if pairmode else 0
    modx = 1.0 if duplicatemode else dnu
    scalex = dnu if duplicatemode else 1

    if x.model_modes is None:
        maxPDF_path, maxPDF_ind = stats.most_likely(selectedmodels)
        model_modes = core.make_model_modes_from_ln_freqinertia(
            x.Grid[maxPDF_path + "/osckey"][maxPDF_ind],
            x.Grid[maxPDF_path + "/osc"][maxPDF_ind],
        )
    else:
        model_modes = x.model_modes

    corrected_model_modes = surfacecorrections.apply_surfacecorrection_coefficients(
        coeffs=x.coeffs, star=star, modes=model_modes
    )
    corrected_joinedmodes = surfacecorrections.apply_surfacecorrection_coefficients(
        coeffs=x.coeffs, star=star, modes=joinedmodes
    )

    s = su.scale_by_inertia(modes=corrected_model_modes)
    sjoin = (
        su.scale_by_inertia(modes=corrected_joinedmodes)
        if corrected_joinedmodes
        else None
    )

    obsls = star.modes.modes.possible_angular_degrees
    fmod, fmod_all, fobs, fobs_all, eobs, eobs_all = {}, {}, {}, {}, {}, {}

    for l in obsls:
        l = int(l)
        mod_givenl = corrected_model_modes.of_angular_degree(l)
        obs_givenl = star.modes.modes.of_angular_degree(l)
        fmod_all[l] = mod_givenl["frequency"] / scalex
        fobs_all[l] = obs_givenl["frequency"] / scalex
        eobs_all[l] = obs_givenl["error"] / scalex
        if corrected_joinedmodes is not None:
            ljoin = corrected_joinedmodes.of_angular_degree(l)
            fmod[l] = ljoin["model_frequency"] / scalex
            fobs[l] = ljoin["observed_frequency"] / scalex
            eobs[l] = ljoin["error"] / scalex

    # Create plot
    fig, ax1 = plt.subplots()
    ax2 = ax1.twinx()
    ax = ax2 if duplicatemode else ax1
    aax = ax1 if duplicatemode else ax2

    errorbar_kwargs_base = {
        "alpha": 0.5,
    }
    scatter_kwargs_case = {
        "alpha": 0.5,
    }
    line_kwargs_base = {"alpha": 0.7, "lw": lw}

    if duplicatemode:
        # Plot something to set the scale on one y-axis
        ax1.errorbar(
            fobs_all[obsls[0]] % modx,
            fobs_all[obsls[0]] * dnu,
            xerr=eobs_all[obsls[0]],
            fmt=obsmarker,
            mfc=colors[f"l{obsls[0]}"],
            ecolor=colors[f"l{obsls[0]}"],
            zorder=0,
            **errorbar_kwargs_base,  # type: ignore
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
            mfc=colors[f"l{obsls[0]}"],
            ecolor=colors[f"l{obsls[0]}"],
            zorder=0,
            **errorbar_kwargs_base,  # type: ignore
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
            mfc=colors[f"l{l}"],
            ecolor=colors[f"l{l}"],
            zorder=1,
            **errorbar_kwargs_base,  # type: ignore
        )
        if duplicatemode:
            ax.errorbar(
                fobs_all[l] % modx - modx,
                fobs_all[l],
                xerr=eobs_all[l],
                fmt=obsmarker,
                mfc=colors[f"l{l}"],
                ecolor=colors[f"l{l}"],
                zorder=1,
                **errorbar_kwargs_base,  # type: ignore
            )

    for l in obsls:
        ax.scatter(
            fmod_all[l] % modx,
            fmod_all[l],
            s=s[int(l)],
            c=colors[f"l{l}"],
            marker=modmarkers[f"l{l}"],
            zorder=2,
            **scatter_kwargs_case,  # type: ignore
        )
        if duplicatemode:
            ax.scatter(
                fmod_all[l] % modx - modx,
                fmod_all[l],
                s=s[int(l)],
                c=colors[f"l{l}"],
                marker=modmarkers[f"l{l}"],
                zorder=2,
                **scatter_kwargs_case,  # type: ignore
            )

    # Plot the matched modes in negative and positive side
    linelimit = 0.75 * modx
    if corrected_joinedmodes is not None:
        assert sjoin is not None
        for l in obsls:
            if len(fmod[l]) > 0:
                ax.scatter(
                    fmod[l] % modx,
                    fmod[l],
                    s=sjoin[int(l)],
                    c=colors[f"l{l}"],
                    marker=modmarkers[f"l{l}"],
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
                        c=colors[f"l{l}"],
                        marker=modmarkers[f"l{l}"],
                        linewidths=1,
                        edgecolors="k",
                        zorder=3,
                    )
                ax.errorbar(
                    fobs[l] % modx,
                    fobs[l],
                    xerr=eobs[l],
                    fmt=obsmarker,
                    mfc=colors[f"l{l}"],
                    ecolor=colors[f"l{l}"],
                    zorder=1,
                    label=f"Measured $\\ell={l}$",
                )
                if duplicatemode:
                    ax.errorbar(
                        fobs[l] % modx - modx,
                        fobs[l],
                        xerr=eobs[l],
                        fmt=obsmarker,
                        mfc=colors[f"l{l}"],
                        ecolor=colors[f"l{l}"],
                        zorder=1,
                    )

                if pairmode:
                    _connect_model_observed(
                        ax=ax,
                        fmod=fmod[l],
                        fobs=fobs[l],
                        modx=modx,
                        duplicatemode=duplicatemode,
                        color=colors[f"l{l}"],
                        line_kwargs_base=line_kwargs_base,
                    )

    if plotconfig.seismic_legend_on_top:
        lgnd = ax.legend(
            bbox_to_anchor=(0.0, 1.02, 1.0, 0.102),
            loc=8,
            ncol=2 * len(obsls),
            mode="expand",
            borderaxespad=0.0,
        )
    else:
        lgnd = ax.legend(
            bbox_to_anchor=(1.02, 0.68, 0.2, 0.202),
            loc="upper left",
            ncol=1,
            mode="expand",
            borderaxespad=0.0,
        )

    for i in range(len(lgnd.legend_handles)):
        typing.cast(Any, lgnd.legend_handles[i])._sizes = [50]

    if duplicatemode:
        if plotconfig.seismic_twinax:
            ax.set_xlim((-1, 1))
            ax.set_ylabel(r"Frequency normalised by $\Delta \nu$")
        else:
            ax.set_yticks([])
        aax.set_ylim(ax.set_ylim()[0] * dnu, ax.set_ylim()[1] * dnu)
        aax.set_xlabel(
            rf"Frequency normalised by $\Delta \nu$ modulo 1 ($\Delta \nu =${dnu} $\mu$Hz)"
        )
        aax.set_ylabel(r"Frequency ($\mu$Hz)")
    else:
        if plotconfig.seismic_twinax:
            ax.set_xlim((0, modx))
            ax.set_xlabel(
                rf"Frequency normalised by $\Delta \nu$ modulo 1 ($\Delta \nu =${dnu} $\mu$Hz)"
            )
            ax.set_ylabel(r"Frequency ($\mu$Hz)")
        else:
            ax.set_yticks([])
        aax.set_ylim(ax.set_ylim()[0] / dnu, ax.set_ylim()[1] / dnu)
        aax.set_ylabel(r"Frequency normalised by $\Delta \nu$")

    if outputfilename is not None:
        plt.savefig(outputfilename, bbox_inches="tight")
        print(f"Saved figure to {outputfilename}")
        plt.close(fig)


def ratioplot(
    star: core.Star,
    joinedmodes: core.JoinedModes,
    model_modes: core.ModelFrequencies,
    sequence: str,
    obs_ratios: np.ndarray | None = None,
    obs_ratios_covinv: np.ndarray | None = None,
    outputfilename: Path | None = None,
    kwargs_ratios: dict[str, Any] = {},
    interp_ratios: bool | int = True,
) -> None:
    """
    Plot frequency ratios.

    Parameters
    ----------
    threepoint : bool
        If True, use three point definition of r01 and r10 ratios instead
        of default five point definition.
    interp_ratios : bool
        If True (default), plot how the model ratios are linearly interpolated
        to the frequencies of the observed ratios, in order to compare the
        sequences at the same frequencies.
    """
    if obs_ratios is None or obs_ratios_covinv is None:
        if star.ratios is None:
            return
        obs_ratios = star.ratios[sequence].values
        obs_ratios_covinv = star.ratios[sequence].inverse_covariance

    if len(obs_ratios) < 1:
        return
    obs_ratios_err = np.sqrt(1 / np.diag(obs_ratios_covinv))

    if interp_ratios:
        model_ratios = freq_fit.compute_ratio_sequences(
            modes=model_modes,
            sequence=sequence,
            threepoint=kwargs_ratios.get("threepoint", False),
        )
    else:
        model_ratios = freq_fit.compute_ratio_sequences(
            modes=joinedmodes,
            sequence=sequence,
            threepoint=kwargs_ratios.get("threepoint", False),
        )

    assert model_ratios is not None

    fig, ax = plt.subplots(1, 1)
    handles: list[Any] = []
    xlabel = "frequency"
    ylabel = "ratio"
    for sequence in np.unique(obs_ratios["id"]):
        obsmask = obs_ratios["id"] == sequence
        modmask = model_ratios["id"] == sequence
        rtname = f"r{int(sequence):02d}"
        modp = ax.scatter(
            model_ratios[modmask][xlabel],
            model_ratios[modmask][ylabel],
            marker=modmarkers["ratio"],
            color=colors[rtname],
            edgecolors="k",
            zorder=3,
            label=f"Best fit ($r_{{{int(sequence):02d}}}$)",
        )
        ax.plot(
            model_ratios[modmask][xlabel],
            model_ratios[modmask][ylabel],
            "-",
            color="darkgrey",
            alpha=0.9,
            zorder=-1,
        )

        obsp = ax.errorbar(
            obs_ratios[obsmask][xlabel],
            obs_ratios[obsmask][ylabel],
            yerr=obs_ratios_err[obsmask],
            marker=obsmarker,
            color=colors[rtname],
            mec="k",
            mew=0.5,
            linestyle="None",
            zorder=3,
            label=f"Measured ($r_{{{int(sequence):02d}}}$)",
        )
        ax.plot(
            obs_ratios[obsmask][xlabel],
            obs_ratios[obsmask][ylabel],
            "-",
            color=colors[rtname],
            zorder=-1,
        )

        if interp_ratios:
            intfunc = interp1d(
                model_ratios[modmask][xlabel],
                model_ratios[modmask][ylabel],
                kind="linear",
            )
            # When only plotting, not fitting, model freqs can be outside observed range
            rangemask = np.ones(sum(obsmask), dtype=bool)
            rangemask &= obs_ratios[obsmask][xlabel] > min(
                model_ratios[modmask][xlabel]
            )
            rangemask &= obs_ratios[obsmask][xlabel] < max(
                model_ratios[modmask][xlabel]
            )
            newmod = intfunc(obs_ratios[obsmask][rangemask][xlabel])
            marker = splinemarkers[1] if "1" in str(sequence) else splinemarkers[2]
            (intp,) = ax.plot(
                obs_ratios[obsmask][rangemask][xlabel],
                newmod,
                marker=marker,
                color="k",
                markeredgewidth=2,
                alpha=0.7,
                lw=0,
                zorder=5,
                label=rf"$r_{{{int(sequence):02d}}}(\nu^{{\mathrm{{obs}}}})$",
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
        ncol=nbase * len(set(obs_ratios["id"])),
        mode="expand",
        borderaxespad=0.0,
    )
    for i in range(len(lgnd.legend_handles)):
        typing.cast(Any, lgnd.legend_handles[i])._sizes = [50]

    ax.set_xlabel(r"Frequency ($\mu$Hz)")
    ax.set_ylabel(f"Frequency ratio ({sequence})")
    ylim = ax.get_ylim()
    ax.set_ylim(max(ylim[0], 0), ylim[1])

    if outputfilename is not None:
        fig.savefig(outputfilename, bbox_inches="tight")
        print(f"Saved figure to {outputfilename}")
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
    star: core.Star,
    sequence: str,
    quantities_at_runtime: dict[str, np.ndarray],
    max_path: str,
    max_index: np.ndarray,
    outputfilename: Path | None,
) -> None:

    if star.glitches is None or sequence not in star.glitches:
        return
    labels = {
        7: r"$\langle A_{\mathrm{He}}\rangle$ ($\mu$Hz)",
        8: r"$\Delta_{\mathrm{He}}$ (s)",
        9: r"$\tau_{\mathrm{He}}$ (s)",
    }

    glitches = star.glitches[sequence]
    values = glitches.values
    inv_cov = glitches.inverse_covariance
    errors = np.sqrt(1 / np.diag(inv_cov))
    cov = compute_matrix_inverse(inv_cov)

    def extract_obs_data(param_id):
        mask = values["id"] == param_id
        return values["value"][mask], errors[mask], mask

    obs_aHe, error_obs_aHe, mask_obs_aHe = extract_obs_data(7)
    obs_dHe, error_obs_dHe, mask_obs_dHe = extract_obs_data(8)
    obs_tauHe, error_obs_tauHe, mask_obs_tauHe = extract_obs_data(9)

    model_aHe = (quantities_at_runtime[max_path]["glitchparameters"]["aHe"][max_index],)
    model_dHe = (quantities_at_runtime[max_path]["glitchparameters"]["dHe"][max_index],)
    model_tauHe = (
        quantities_at_runtime[max_path]["glitchparameters"]["tauHe"][max_index],
    )

    # Start figure
    fig, ax = plt.subplots(2, 2, figsize=(8, 8))
    fig.delaxes(ax[0, 1])

    # Loop over each track to plot
    for trackparams in quantities_at_runtime.values():
        aHe = trackparams["glitchparameters"]["aHe"]
        dHe = trackparams["glitchparameters"]["dHe"][aHe > 1e-14]
        tauHe = trackparams["glitchparameters"]["tauHe"][aHe > 1e-14]
        aHe = aHe[aHe > 1e-14]

        ax[1, 0].plot(aHe, dHe, ".", color="grey", ms=5, zorder=1)
        ax[0, 0].plot(aHe, tauHe, ".", color="grey", ms=5, zorder=1)
        ax[1, 1].plot(tauHe, dHe, ".", color="grey", ms=5, zorder=1)

    # AHe vs dHe
    ax[1, 0].errorbar(
        obs_aHe,
        obs_dHe,
        xerr=error_obs_aHe,
        yerr=error_obs_dHe,
        marker=".",
        linestyle="None",
        color="#D55E00",
        zorder=1,
        label="Measured",
    )
    ax[1, 0].plot(
        model_aHe,
        model_dHe,
        "*",
        ms=20,
        color="#0072B2",
        zorder=2,
        label="Best fit",
    )
    confidence_ellipse(
        obs_aHe,
        error_obs_aHe,
        obs_dHe,
        error_obs_dHe,
        cov[mask_obs_aHe, mask_obs_dHe],
        ax[1, 0],
        edgecolor="#D55E00",
        lw=1.5,
        alpha=0.5,
    )
    ax[1, 0].set_xlabel(labels[7])
    ax[1, 0].set_ylabel(labels[8])

    # AHe vs tauHe
    ax[0, 0].errorbar(
        obs_aHe,
        obs_tauHe,
        xerr=error_obs_aHe,
        yerr=error_obs_tauHe,
        marker=".",
        linestyle="None",
        color="#D55E00",
        zorder=1,
        label="Measured",
    )
    ax[0, 0].plot(
        model_aHe,
        model_tauHe,
        "*",
        ms=20,
        color="#0072B2",
        zorder=2,
        label="Best fit",
    )
    confidence_ellipse(
        obs_aHe,
        error_obs_aHe,
        obs_tauHe,
        error_obs_tauHe,
        cov[mask_obs_aHe, mask_obs_tauHe],
        ax[0, 0],
        edgecolor="#D55E00",
        lw=1.5,
        alpha=0.5,
    )
    ax[0, 0].set_ylabel(labels[9])
    ax[0, 0].legend(bbox_to_anchor=(1.01, 1), loc="upper left", ncol=1)

    # tauHe vs dHe
    ax[1, 1].errorbar(
        obs_tauHe,
        obs_dHe,
        xerr=error_obs_tauHe,
        yerr=error_obs_dHe,
        marker=".",
        linestyle="None",
        color="#D55E00",
        zorder=1,
        label="Measured",
    )
    ax[1, 1].plot(
        model_tauHe,
        model_dHe,
        "*",
        ms=20,
        color="#0072B2",
        zorder=2,
        label="Best fit",
    )
    confidence_ellipse(
        obs_tauHe,
        error_obs_tauHe,
        obs_dHe,
        error_obs_dHe,
        cov[mask_obs_tauHe, mask_obs_dHe],
        ax[1, 1],
        edgecolor="#D55E00",
        lw=1.5,
        alpha=0.5,
    )
    ax[1, 1].set_xlabel(labels[9])

    if outputfilename is not None:
        fig.savefig(outputfilename, bbox_inches="tight")
        print(f"Saved figure to {outputfilename}")
        plt.close(fig)


def epsilon_difference_diagram(
    *,
    sequence: str,
    model_modes: core.ModelFrequencies,
    model_dnu: float,
    star: core.Star,
    outputfilename: Path | None,
):
    """
    Full comparison figure of observed and best-fit model epsilon
    differences, with individual epsilons and correlation map.

    Parameters
    ----------
    sequence : str
        The sequence to be plotted
    model_dnu : float
        Average large frequency separation (dnufit) of best-fit model.
    outputfilename : str
        Name and path of outputfilename plotfile.
    """

    if star.epsilondifferences is None or sequence not in star.epsilondifferences:
        return
    delab = r"$\delta\epsilon^{%s}_{0%d}$"

    obsepsdiff = star.epsilondifferences[sequence].values
    obsepsdiff_covinv = star.epsilondifferences[sequence].inverse_covariance
    diag = np.diag(obsepsdiff_covinv)
    safe_diag = np.where(diag == 0, np.nan, diag)
    obsepsdiff_err = np.sqrt(1 / safe_diag)

    modepsdiff = freq_fit.compute_sequence_of_epsilondifferences(
        modes=model_modes,
        average_dnu=model_dnu,
        sequence=sequence,
    )

    # Mixed modes results in negative differences. Flag using nans, not displayed
    mask = np.where(modepsdiff["epsilondifference"] < 0)[0]
    modepsdiff["epsilondifference"][mask] = np.nan

    fig, ax = plt.subplots(1, 1)
    handles, legends = [], []
    for ll in np.unique(obsepsdiff["l"]):
        indobs = obsepsdiff["l"] == ll
        indmod = modepsdiff["l"] == ll
        indmod &= modepsdiff["frequency"] > min(obsepsdiff["frequency"]) - 3 * model_dnu
        indmod &= modepsdiff["frequency"] < max(obsepsdiff["frequency"]) + 3 * model_dnu
        indmod &= ~np.isnan(modepsdiff["epsilondifference"])

        # Model with spline
        (moddot,) = ax.plot(
            modepsdiff["frequency"][indmod],
            modepsdiff["epsilondifference"][indmod],
            marker=modmarkers["l" + str(ll)],
            color=colors["l" + str(ll)],
            lw=0,
        )

        # Observed with uncertainties
        obsdot = ax.errorbar(
            obsepsdiff["frequency"][indobs],
            obsepsdiff["epsilondifference"][indobs],
            yerr=obsepsdiff_err[:-1][indobs],
            marker=obsmarker,
            color=colors["l" + str(ll)],
            markeredgewidth=0.5,
            markeredgecolor="k",
            zorder=3,
        )

        if sum(indmod) > 1:
            spline = CubicSpline(
                modepsdiff["frequency"][indmod],
                modepsdiff["epsilondifference"][indmod],
                extrapolate=False,
            )
            fnew = np.linspace(
                min(modepsdiff["frequency"][indmod]),
                max(modepsdiff["epsilondifference"][indmod]),
                100,
            )
            ax.plot(fnew, spline(fnew), "-", color=splinecolor, zorder=-1)

            # Model at observed
            (modobs,) = ax.plot(
                obsepsdiff["frequency"][indobs],
                spline(obsepsdiff["frequency"][indobs]),
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
    h: list[Any] = []
    l: list[Any] = []
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
        legend_handle: Any = lgnd.legend_handles[i]
        legend_handle._sizes = [50]

    ax.set_xlabel(r"Frequency ($\mu$Hz)")
    ax.set_ylabel(r"Epsilon difference $\delta\epsilon_{0\ell}$")
    ylim = ax.get_ylim()
    ax.set_ylim(max(ylim[0], 0), ylim[1])

    fig.tight_layout()
    if outputfilename is not None:
        print(f"Saved figure to {outputfilename}")
        fig.savefig(outputfilename)
        plt.close(fig)
        return None
    return fig


def correlation_map(fittype, star, outputfilename: Path | None) -> None:
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
        obskey = np.asarray([star.modes.modes.l, star.modes.modes.n])
        ln_zip: Iterable[tuple[Any, Any]] = zip(obskey[0, :], obskey[1, :])

    elif fittype in freqtypes.rtypes:
        if fittype not in star.ratios:
            return
        data = star.ratios[fittype].values
        covinv = star.ratios[fittype].inverse_covariance
        fmtstr = r"$r_{{{:02d}}}({{{:d}}})$"
        ln_zip = zip(data[2, :], data[3, :])

    elif fittype in freqtypes.epsdiff:
        if fittype not in star.epsilondifferences:
            return
        data = star.epsilondifferences[fittype].values
        covinv = star.epsilondifferences[fittype].inverse_covariance
        fmtstr = r"$\delta\epsilon_{{{:02d}}}({{{:d}}})$"
        ln_zip = zip(data[2, :], data[3, :])

    elif fittype in freqtypes.glitches:
        if fittype not in star.glitches:
            return
        data = star.glitches[fittype].values
        covinv = star.glitches[fittype].inverse_covariance
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

        for sequence in data[2, -3:]:
            labs.append(glitchlabels[int(sequence)])

    # Compute correlations
    Dinv = np.diag(np.sqrt(np.diag(covinv)))
    cor = Dinv @ (1 / covinv) @ Dinv

    # Produce figure
    fig, ax = plt.subplots(1, 1, figsize=(7.3, 6))
    im = ax.imshow(cor, cmap="RdBu_r", vmin=-1, vmax=1)
    plt.colorbar(im)

    # Beautify
    ax.set_xticks(range(covinv.shape[1]))
    ax.set_xticklabels(labs, rotation=90)
    ax.set_yticks(range(covinv.shape[1]))
    ax.set_yticklabels(labs)
    fig.tight_layout()

    if outputfilename is not None:
        fig.savefig(outputfilename, bbox_inches="tight")
        print(f"Saved figure to {outputfilename}")
        plt.close(fig)
