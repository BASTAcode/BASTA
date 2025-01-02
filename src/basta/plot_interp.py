"""
Production of interpolation plots
"""

import os
import numpy as np
import matplotlib.pyplot as plt

import basta.utils_general as gu
import basta.constants as bc


def base_corner(baseparams, base, newbase, tri, sobol=1.0, outbasename=""):
    """
    Plots the new vs. old base of across interpolation, as long as dim(base) > 1,
    and produces a corner plot for dim(base) > 2.

    Parameters
    ----------
    baseparams : dict
        Parameters forming the base of the grid, and their required resolution.

    base : array
        Old base, in list form with each column corresponding to a different base
        parameter, and each row corresponding to a given point (track/isochrone).

    newbase : array
        Same as base, but with the newly determined values for across interpolation.

    tri : object
        Triangulation of the old base.

    sobol : float
        Number of points increase factor.

    outbasename : str, optional
        If set, it is the name and location of where to put the figure. If not given,
        it won't produce the figure.

    Returns
    -------
    success : bool
        True of False of whether the figure has been produced
    """
    if outbasename == "":
        return False

    _, parlab, _, _ = bc.parameters.get_keys([par for par in baseparams])
    if sobol >= 10.0:
        alpha = 0.2
    elif sobol <= 2.0:
        alpha = 0.7
    else:
        alpha = 0.7 - 0.5 * (sobol - 2) / 10

    # For adding statistics to the plot
    numstr = "Old tracks: {:d}\nNew tracks: {:d}"
    numstr = numstr.format(base.shape[0], newbase.shape[0])
    if sobol:
        numstr = "Scale: {:.1f}\n".format(sobol) + numstr
    # Size of figure, stolen from basta/corner.py
    K = len(baseparams) - 1
    factor = 2.0 if K > 1 else 3.0
    whspace = 0.05
    lbdim = 0.5 * factor
    trdim = 0.2 * factor
    plotdim = factor * K + factor * (K - 1.0) * whspace
    dim = lbdim + plotdim + trdim

    fig, axes = plt.subplots(K, K, figsize=(dim, dim))
    # Format figure
    lb = lbdim / dim
    tr = (lbdim + plotdim) / dim
    fig.subplots_adjust(
        left=lb, bottom=lb, right=tr, top=tr, wspace=whspace, hspace=whspace
    )
    # Some index magic in the following, as we are not intereset in the 'diagonal'
    for j in range(K):
        for i in range(j, K + 1):
            if j == i:
                continue
            elif K == 1:
                ax = axes
                ax.triplot(base[:, j], base[:, i], tri.simplices, zorder=1)
            else:
                ax = axes[i - 1, j]
            # Old subgrid
            ax.plot(
                base[:, j],
                base[:, i],
                "X",
                color="black",
                markersize=6,
                label=r"Base",
                zorder=10,
            )
            # New subgrid
            ax.plot(
                newbase[:, j],
                newbase[:, i],
                ".",
                alpha=alpha,
                color="#DC050C",
                markersize=7,
                label=r"New points",
                zorder=3,
            )
            if i == K:
                # Set xlabel and rotate ticklabels for no overlap
                ax.set_xlabel(parlab[j])
                [l.set_rotation(45) for l in ax.get_xticklabels()]
                if not K == 1:
                    ax.xaxis.set_label_coords(0.5, -0.35)
            else:
                # Remove xticks from non-bottom subplots
                ax.set_xticks([])
            if j == 0:
                # Set ylabel
                ax.set_ylabel(parlab[i])
                if not K == 1:
                    ax.yaxis.set_label_coords(-0.35, 0.5)
            else:
                # Remove yticks from non-leftmost subplots
                ax.set_yticks([])

            # Legend magic depending on case
            if (i == 1 and j == 0) and K > 1:
                ax.legend(
                    bbox_to_anchor=(-0.34, 1.0),
                    loc="lower left",
                    ncol=2,
                    frameon=False,
                )
            elif (i == 1 and j == 0) and K == 1:
                ax.legend(
                    bbox_to_anchor=(0.0, 1.02),
                    loc="lower left",
                    ncol=2,
                    frameon=False,
                )
            if i == 1 and j == 0:
                # Number info
                ax.text(
                    1.0 + whspace,
                    0.0,
                    numstr,
                    transform=ax.transAxes,
                    verticalalignment="bottom",
                )

        for i in range(K):
            # Remove the empty subplots
            if i < j:
                ax = axes[i, j]
                ax.set_frame_on(False)
                ax.set_xticks([])
                ax.set_yticks([])

    # Easiest pretty layout for single-subplot
    if K == 1:
        fig.tight_layout()
    # Save and close
    fig.savefig(outbasename)
    plt.close()
    return True


def across_debug(
    grid, outfile, basepath, basevar, inttrack, envtracks, selmods, outname
):
    """
    If run with the --debug option, this produces a plot for each interpolated
    track, comparing the interpolated track to the enveloping tracks.

    Parameters
    ----------
    grid : h5py file
        Handle of original grid

    outfile : h5py file
        Handle of output grid

    basepath : str
        Base path to access tracks/isochrones in grid

    basevar : str
        Base parameter used for interpolation to map tracks/isochrones along

    inttrack : str
        Name (with path) of interpolated track/isochrone in outfile

    envtracks : str
        Name of enveloping tracks in grid used for interpolated track/isochrone

    selmods : dict
        Selectedmodels of enveloping tracks, to show what models were used for
        interpolation

    outname : str
        Name and path of outputted plots

    Returns
    -------
    """

    # Set of colors for the enveloping tracks
    cols = [
        "#882E72",
        "#1965B0",
        "#5289C7",
        "#7BAFDE",
        "#4EB265",
        "#CAE0AB",
        "#F7F056",
        "#F4A736",
        "#E8601C",
        "#DC050C",
        "#72190E",
    ]
    # Pretty labels
    _, labels, _, _ = bc.parameters.get_keys(["Teff", "logg", basevar])

    # We use these mulitple times, so better to abbriviate them now
    Teff = gu.h5py_to_array(outfile[inttrack]["Teff"])
    logg = gu.h5py_to_array(outfile[inttrack]["logg"])
    base = gu.h5py_to_array(outfile[inttrack][basevar])

    # Define figure and plot interpolated track
    fig, ax = plt.subplots(2, 1, figsize=(12.8, 17.6))
    ax[0].plot(Teff, logg, ".k", label=r"Interpolated", zorder=20, markersize=8)
    ax[1].plot(Teff, base, ".k", zorder=20, markersize=8)

    # Plot enveloping tracks
    for i, track in enumerate(envtracks):
        name = os.path.join(basepath, track)
        if "track" in track:
            lab = track.split("/")[-1]
        else:
            lab = track
        pltTeff = gu.h5py_to_array(grid[name]["Teff"])
        pltlogg = gu.h5py_to_array(grid[name]["logg"])
        pltbase = gu.h5py_to_array(grid[name][basevar])
        ax[0].plot(
            pltTeff,
            pltlogg,
            ",",
            color=cols[i],
            zorder=5,
            alpha=0.4,
        )
        ax[0].plot(
            pltTeff[selmods[i]],
            pltlogg[selmods[i]],
            ".",
            color=cols[i],
            label=lab,
            zorder=5,
            markersize=8,
        )
        ax[1].plot(
            pltTeff,
            pltbase,
            ",",
            color=cols[i],
            zorder=5,
            alpha=0.4,
        )
        ax[1].plot(
            pltTeff[selmods[i]],
            pltbase[selmods[i]],
            ".",
            color=cols[i],
            zorder=5,
            markersize=8,
        )

    # Set labels
    ax[0].set_xlabel(labels[0])
    ax[0].set_ylabel(labels[1])
    ax[1].set_xlabel(labels[0])
    ax[1].set_ylabel(r"Base parameter: " + labels[2])

    # Invert axis for Kiel-diagram
    ax[0].invert_xaxis()
    ax[1].invert_xaxis()
    ax[0].invert_yaxis()

    # Check and produce inset if needed
    xlim = ax[0].get_xlim()
    ylim = ax[0].get_ylim()
    blim = ax[1].get_ylim()
    dx = np.amax(Teff) - np.amin(Teff)
    dy = np.amax(logg) - np.amin(logg)
    db = np.amax(base) - np.amin(base)
    if abs(np.diff(xlim)) > 3 * dx or abs(np.diff(ylim)) > 3 * dy:
        # Locate most empty quadrant
        halfx = min(xlim) + 0.5 * abs(np.diff(xlim))
        halfy = min(ylim) + 0.5 * abs(np.diff(ylim))
        halfb = min(blim) + 0.5 * abs(np.diff(blim))

        # Index juggling due to inverted axis
        indexes = [[[0, 0], [0, 1], [1, 0], [1, 1]], [[0, 1], [1, 1], [0, 0], [1, 0]]]

        # Take the different variables on the y-axis into account
        yvars = ["logg", basevar]
        halfs = [halfy, halfb]
        indx = []
        for j, yvar in enumerate(yvars):
            sums = [0, 0, 0, 0]
            for i, track in enumerate(envtracks):
                name = os.path.join(basepath, track)

                index = np.ones(len(grid[name]["Teff"][:]), dtype=bool)
                index &= grid[name]["Teff"][:] > halfx
                index &= grid[name][yvar][:] < halfs[j]
                sums[0] += sum(index)

                index = np.ones(len(grid[name]["Teff"][:]), dtype=bool)
                index &= grid[name]["Teff"][:] < halfx
                index &= grid[name][yvar][:] < halfs[j]
                sums[1] += sum(index)

                index = np.ones(len(grid[name]["Teff"][:]), dtype=bool)
                index &= grid[name]["Teff"][:] > halfx
                index &= grid[name][yvar][:] > halfs[j]
                sums[2] += sum(index)

                index = np.ones(len(grid[name]["Teff"][:]), dtype=bool)
                index &= grid[name]["Teff"][:] < halfx
                index &= grid[name][yvar][:] > halfs[j]
                sums[3] += sum(index)

            indx.append(indexes[j][np.argmin(sums)])

        inset_place_and_size = [
            [[0.03, 0.6, 0.37, 0.37], [0.6, 0.6, 0.37, 0.37]],
            [[0.03, 0.05, 0.37, 0.37], [0.6, 0.05, 0.37, 0.37]],
        ]

        # Finally create the insets in the emptiest quadrant
        axins0 = ax[0].inset_axes(inset_place_and_size[indx[0][1]][indx[0][0]])
        axins1 = ax[1].inset_axes(inset_place_and_size[indx[1][1]][indx[1][0]])

        # Move ytick labels correspondingly
        if not indx[0][0]:
            axins0.yaxis.set_ticks_position("right")
        if not indx[1][0]:
            axins1.yaxis.set_ticks_position("right")

        # Don't set legend on top of inset
        if not indx[0][1] and not indx[0][0]:
            ax[0].legend(loc=4)
        else:
            ax[0].legend()

        # Plot things again in inset
        axins0.plot(Teff, logg, ".k", zorder=20, markersize=8)
        axins1.plot(Teff, base, ".k", zorder=20, markersize=8)
        for i, track in enumerate(envtracks):
            name = os.path.join(basepath, track)
            pltTeff = gu.h5py_to_array(grid[name]["Teff"])
            pltlogg = gu.h5py_to_array(grid[name]["logg"])
            pltbase = gu.h5py_to_array(grid[name][basevar])
            axins0.plot(
                pltTeff,
                pltlogg,
                ",",
                color=cols[i],
                zorder=5,
                alpha=0.4,
            )
            axins0.plot(
                pltTeff[selmods[i]],
                pltlogg[selmods[i]],
                ".",
                color=cols[i],
                zorder=5,
                markersize=8,
            )
            axins1.plot(
                pltTeff,
                pltbase,
                ",",
                color=cols[i],
                zorder=5,
                alpha=0.4,
            )
            axins1.plot(
                pltTeff[selmods[i]],
                pltbase[selmods[i]],
                ".",
                color=cols[i],
                zorder=5,
                markersize=8,
            )

            axins0.set_xlim([np.amin(Teff) - 0.3 * dx, np.amax(Teff) + 0.3 * dx])
            axins0.set_ylim([np.amin(logg) - 0.3 * dy, np.amax(logg) + 0.3 * dy])
            axins1.set_xlim([np.amin(Teff) - 0.3 * dx, np.amax(Teff) + 0.3 * dx])
            axins1.set_ylim([np.amin(base) - 0.3 * db, np.amax(base) + 0.3 * db])

            axins0.invert_xaxis()
            axins1.invert_xaxis()
            axins0.invert_yaxis()

    else:
        ax[0].legend()
    fig.tight_layout()
    fig.savefig(outname)
    plt.close()
