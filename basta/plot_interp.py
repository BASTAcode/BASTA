"""
Production of interpolation plots
"""
import matplotlib.pyplot as plt

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
