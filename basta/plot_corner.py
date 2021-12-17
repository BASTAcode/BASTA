"""
Production of corner plots.

Modified from a fork of https://github.com/dfm/corner.py .
Original code: Copyright (c) 2013-2020 Daniel Foreman-Mackey
Full license: https://github.com/dfm/corner.py/blob/main/LICENSE

This modified version add the observed quantities to the corner plots.
"""
import logging
import colorsys

import numpy as np
import matplotlib
from scipy.stats import gaussian_kde
from matplotlib.ticker import MaxNLocator, NullLocator
from matplotlib.colors import LinearSegmentedColormap, colorConverter
from matplotlib.ticker import ScalarFormatter
import matplotlib.colors as mc

matplotlib.use("Agg")
import matplotlib.pyplot as pl

fontdic = {"size": 12}

try:
    from scipy.ndimage import gaussian_filter
except ImportError:
    gaussian_filter = None

__all__ = ["corner", "hist2d", "quantile"]


def corner(
    xs,
    bins=20,
    range=None,
    weights=None,
    color="k",
    smooth=None,
    smooth1d="kde",
    labels=None,
    label_kwargs=fontdic,
    show_titles=False,
    title_fmt=".3f",
    title_kwargs=fontdic,
    truths=None,
    truth_color="#4682b4",
    scale_hist=False,
    quantiles=None,
    verbose=False,
    fig=None,
    max_n_ticks=5,
    top_ticks=False,
    use_math_text=False,
    reverse=False,
    plotin=None,
    plotout=None,
    hist_kwargs=None,
    autobins=True,
    binrule_fallback="scott",
    uncert="quantiles",
    **hist2d_kwargs
):
    """
    Make a *sick* corner plot showing the projections of a data set in a
    multi-dimensional space. kwargs are passed to hist2d() or used for
    `matplotlib` styling.

    Parameters
    ----------
    xs : array_like[nsamples, ndim]
        The samples. This should be a 1- or 2-dimensional array. For a 1-D
        array this results in a simple histogram. For a 2-D array, the zeroth
        axis is the list of samples and the next axis are the dimensions of
        the space.

    bins : int or array_like[ndim,]
        The number of bins to use in histograms, either as a fixed value for
        all dimensions, or as a list of integers for each dimension.

    weights : array_like[nsamples,]
        The weight of each sample. If `None` (default), samples are given
        equal weight.

    color : str
        A ``matplotlib`` style color for all histograms.

    smooth, smooth1d : float
       The standard deviation for Gaussian kernel passed to
       `scipy.ndimage.gaussian_filter` to smooth the 2-D and 1-D histograms
       respectively. If `None` (default), no smoothing is applied.

    labels : iterable (ndim,)
        A list of names for the dimensions. If a ``xs`` is a
        ``pandas.DataFrame``, labels will default to column names.

    label_kwargs : dict
        Any extra keyword arguments to send to the `set_xlabel` and
        `set_ylabel` methods.

    show_titles : bool
        Displays a title above each 1-D histogram showing the 0.5 quantile
        with the upper and lower errors supplied by the quantiles argument.

    title_fmt : string
        The format string for the quantiles given in titles. If you explicitly
        set ``show_titles=True`` and ``title_fmt=None``, the labels will be
        shown as the titles. (default: ``.2f``)

    title_kwargs : dict
        Any extra keyword arguments to send to the `set_title` command.

    range : iterable (ndim,)
        A list where each element is either a length 2 tuple containing
        lower and upper bounds or a float in range (0., 1.)
        giving the fraction of samples to include in bounds, e.g.,
        [(0.,10.), (1.,5), 0.999, etc.].
        If a fraction, the bounds are chosen to be equal-tailed.

    truths : iterable (ndim,)
        A list of reference values to indicate on the plots.  Individual
        values can be omitted by using ``None``.

    truth_color : str or dict
        A ``matplotlib`` style color for the ``truths`` makers.
        or a dict with the colors with keys being the labels.

    scale_hist : bool
        Should the 1-D histograms be scaled in such a way that the zero line
        is visible?

    quantiles : iterable
        A list of fractional quantiles to show on the 1-D histograms as
        vertical dashed lines.

    verbose : bool
        If true, print the values of the computed quantiles.

    plot_contours : bool
        Draw contours for dense regions of the plot.

    use_math_text : bool
        If true, then axis tick labels for very large or small exponents will
        be displayed as powers of 10 rather than using `e`.

    reverse : bool
        If true, plot the corner plot starting in the upper-right corner instead
        of the usual bottom-left corner

    max_n_ticks: int
        Maximum number of ticks to try to use

    top_ticks : bool
        If true, label the top ticks of each axis

    fig : matplotlib.Figure
        Overplot onto the provided figure object.

    hist_kwargs : dict
        Any extra keyword arguments to send to the 1-D histogram plots.

    autobins : bool
        Switch between automatically determining bin edges, or use bin number decided from
        data size. True for automic rule.

    binrule_fallback : str
        In case auto-binning fails for the posterior distribution (usually due to too many
        zeros, which causes a memory leak), use this rule for posterior binning instead.

    uncert : str
        If uncertainties are given in terms of 'quantiles' or 'std' (standard deviation),
        included here to change formatting when reporting inferred quantities in titles.

    **hist2d_kwargs
        Any remaining keyword arguments are sent to `corner.hist2d` to generate
        the 2-D histogram plots.

    """
    if quantiles is None:
        quantiles = []
    if title_kwargs is None:
        title_kwargs = dict()
    if label_kwargs is None:
        label_kwargs = dict()

    # Try filling in labels from pandas.DataFrame columns.
    if labels is None:
        try:
            labels = xs.columns
        except AttributeError:
            pass

    # Deal with 1D sample lists.
    xs = np.atleast_1d(xs)
    if len(xs.shape) == 1:
        xs = np.atleast_2d(xs)
    else:
        assert len(xs.shape) == 2, "The input sample array must be 1- or 2-D."
        xs = xs.T
    assert xs.shape[0] <= xs.shape[1], (
        "I don't believe that you want more " "dimensions than samples!"
    )

    # Parse the weight array.
    if weights is not None:
        weights = np.asarray(weights)
        if weights.ndim != 1:
            raise ValueError("Weights must be 1-D")
        if xs.shape[1] != weights.shape[0]:
            raise ValueError("Lengths of weights must match number of samples")

    # Parse the parameter ranges.
    if range is None:
        if "extents" in hist2d_kwargs:
            logging.warn(
                "Deprecated keyword argument 'extents'. " "Use 'range' instead."
            )
            range = hist2d_kwargs.pop("extents")
        else:
            mins = np.array([x.min() for x in xs])
            maxs = np.array([x.max() for x in xs])

            # Set dummy ranges [v-1, v+1] for parameters that never change..
            m = mins == maxs
            mins[m] -= 1
            maxs[m] += 1

            range = np.transpose((mins, maxs)).tolist()

    else:
        # If any of the extents are percentiles, convert them to ranges.
        # Also make sure it's a normal list.
        range = list(range)
        for i, _ in enumerate(range):
            try:
                emin, emax = range[i]
            except TypeError:
                q = [0.5 - 0.5 * range[i], 0.5 + 0.5 * range[i]]
                range[i] = quantile(xs[i], q, weights=weights)

    if len(range) != xs.shape[0]:
        raise ValueError("Dimension mismatch between samples and range")

    # Parse the bin specifications.
    if autobins:
        bins = []
        for i, x in enumerate(xs):
            try:
                xbin = np.histogram_bin_edges(x, bins="auto", range=np.sort(range[i]))
            except MemoryError:
                print(
                    "WARNING! Using 'auto' as bin-rule causes a memory crash!"
                    "Switching to '{0}'".format(binrule_fallback),
                    "for the parameter '{0}'!".format(labels[i]),
                )
                xbin = np.histogram_bin_edges(
                    x, bins=binrule_fallback, range=np.sort(range[i])
                )
            bins.append(xbin)
    else:
        try:
            bins = [int(bins) for _ in range]
        except TypeError:
            if len(bins) != len(range):
                raise ValueError("Dimension mismatch between bins and range")

    # Some magic numbers for pretty axis layout.
    K = len(xs)
    factor = 2.0  # size of one side of one panel
    if reverse:
        lbdim = 0.2 * factor  # size of left/bottom margin
        trdim = 0.5 * factor  # size of top/right margin
    else:
        lbdim = 0.5 * factor  # size of left/bottom margin
        trdim = 0.2 * factor  # size of top/right margin
    whspace = 0.05  # w/hspace size
    plotdim = factor * K + factor * (K - 1.0) * whspace
    dim = lbdim + plotdim + trdim

    # Create a new figure if one wasn't provided.
    if fig is None:
        fig, axes = pl.subplots(K, K, figsize=(dim, dim))
    else:
        try:
            axes = np.array(fig.axes).reshape((K, K))
        except Exception:
            raise ValueError(
                "Provided figure has {0} axes, but data has "
                "dimensions K={1}".format(len(fig.axes), K)
            )

    # Format the figure.
    lb = lbdim / dim
    tr = (lbdim + plotdim) / dim
    fig.subplots_adjust(
        left=lb, bottom=lb, right=tr, top=tr, wspace=whspace, hspace=whspace
    )

    # Set up the default histogram keywords.
    if hist_kwargs is None:
        hist_kwargs = dict()
    hist_kwargs["color"] = hist_kwargs.get("color", color)
    if smooth1d is None:
        hist_kwargs["histtype"] = hist_kwargs.get("histtype", "step")

    for i, x in enumerate(xs):
        # Deal with masked arrays.
        if hasattr(x, "compressed"):
            x = x.compressed()

        if np.shape(xs)[0] == 1:
            ax = axes
        else:
            if reverse:
                ax = axes[K - i - 1, K - i - 1]
            else:
                ax = axes[i, i]

        if isinstance(truth_color, str):
            tcolor = truth_color
        else:
            tcolor = lighten_color(truth_color[i], 0.5)

        # Plot the histograms.
        if smooth1d is None:
            n, _, _ = ax.hist(
                x, bins=bins[i], weights=weights, range=np.sort(range[i]), **hist_kwargs
            )
        else:
            if gaussian_filter is None:
                raise ImportError("Please install scipy for smoothing")
            n, b = np.histogram(
                x, bins=bins[i], weights=weights, range=np.sort(range[i])
            )
            if smooth1d != "kde":
                n = gaussian_filter(n, smooth1d)
                x0 = np.array(list(zip(b[:-1], b[1:]))).flatten()
                y0 = np.array(list(zip(n, n))).flatten()
            else:
                try:
                    kernel = gaussian_kde(x, bw_method="silverman")
                except np.linalg.LinAlgError:
                    print("WARNING! Unable to create KDE. Skipping plot...")
                    raise
                x0 = np.linspace(np.amin(x), np.amax(x), num=250)
                y0 = kernel(x0)
                y0 /= np.amax(y0)
                n = gaussian_filter(n, 1)
                x0_hist = np.array(list(zip(b[:-1], b[1:]))).flatten()
                y0_hist = np.array(list(zip(n, n))).flatten() / np.amax(n)
                ax.fill_between(
                    x0_hist, y0_hist, y2=-1, interpolate=True, color=tcolor, alpha=0.15
                )
            ax.plot(x0, y0, **hist_kwargs)
            ax.fill_between(x0, y0, y2=-1, interpolate=True, color=tcolor, alpha=0.15)

        if truths is not None and truths[i] is not None:
            ax.axvline(truths[i], color=tcolor)

        # Plot quantiles
        q = plotout[3 * i]
        p = plotout[3 * i + 1]
        m = plotout[3 * i + 2]

        ax.axvline(q, ls="solid", color=color)
        ax.axvline(q + p, ls="dashed", color=color)
        ax.axvline(q - m, ls="dashed", color=color)

        # Plot input parameters when they are given
        if plotin is not None:
            if plotin[2 * i] != -9999:
                inx = plotin[2 * i]
                instd = plotin[2 * i + 1]
                ax.axvline(inx, ls="dashdot", color="0.4")
                ax.axvline(inx - instd, ls="dotted", color="0.4")
                ax.axvline(inx + instd, ls="dotted", color="0.4")

        if show_titles:
            title = None
            if title_fmt is not None:
                # Format the quantile display.
                fmt = "{{0:{0}}}".format(title_fmt).format
                if uncert == "quantiles":
                    title = r"${{{0}}}_{{-{1}}}^{{+{2}}}$"
                    title = title.format(fmt(q), fmt(m), fmt(p))
                else:
                    title = r"${{{0}}}\pm{{{1}}}$"
                    title = title.format(fmt(q), fmt(p))

                # Add in the column name if it's given.
                if labels is not None:
                    title = "{0} = {1}".format(labels[i], title)

            elif labels is not None:
                title = "{0}".format(labels[i])

            if title is not None:
                if reverse:
                    ax.set_xlabel(title, **title_kwargs)
                else:
                    ax.set_title(title, **title_kwargs)

        # Set up the axes.
        ax.set_xlim(range[i])
        if scale_hist:
            maxn = np.max(n)
            ax.set_ylim(-0.1 * maxn, 1.05 * maxn)
        elif smooth1d == "kde":
            maxn = np.amax(y0)
            ax.set_ylim(-0.1 * maxn, 1.05 * maxn)
        else:
            ax.set_ylim(0, 1.05 * np.max(n))
        ax.set_yticklabels([])
        if max_n_ticks == 0:
            ax.xaxis.set_major_locator(NullLocator())
            ax.yaxis.set_major_locator(NullLocator())
        else:
            ax.xaxis.set_major_locator(MaxNLocator(max_n_ticks, prune="lower"))
            ax.yaxis.set_major_locator(NullLocator())

        if i < K - 1:
            if top_ticks:
                ax.xaxis.set_ticks_position("top")
                [l.set_rotation(45) for l in ax.get_xticklabels()]
            else:
                ax.set_xticklabels([])
        else:
            if reverse:
                ax.xaxis.tick_top()
            [l.set_rotation(45) for l in ax.get_xticklabels()]
            if labels is not None:
                if reverse:
                    ax.set_title(labels[i], y=1.25, **label_kwargs)
                else:
                    ax.set_xlabel(labels[i], **label_kwargs)

            # use MathText for axes ticks
            ax.xaxis.set_major_formatter(ScalarFormatter(useMathText=use_math_text))

        for j, y in enumerate(xs):
            if np.shape(xs)[0] == 1:
                ax = axes
            else:
                if reverse:
                    ax = axes[K - i - 1, K - j - 1]
                else:
                    ax = axes[i, j]
            if j > i:
                ax.set_frame_on(False)
                ax.set_xticks([])
                ax.set_yticks([])
                continue
            elif j == i:
                continue

            if isinstance(truth_color, str):
                tcolor = truth_color
            else:
                tcolor = lighten_color(truth_color[j], 0.5)

            # Deal with masked arrays.
            if hasattr(y, "compressed"):
                y = y.compressed()

            hist2d(
                y,
                x,
                ax=ax,
                range=[range[j], range[i]],
                weights=weights,
                color=tcolor,
                smooth=smooth,
                bins=[bins[j], bins[i]],
                **hist2d_kwargs
            )

            if truths is not None:
                if truths[i] is not None and truths[j] is not None:
                    ax.plot(truths[j], truths[i], "s", color=tcolor)
                if truths[j] is not None:
                    ax.axhline(truths[j], color=tcolor)
                if truths[i] is not None:
                    ax.axvline(truths[i], color=tcolor)

            if max_n_ticks == 0:
                ax.xaxis.set_major_locator(NullLocator())
                ax.yaxis.set_major_locator(NullLocator())
            else:
                ax.xaxis.set_major_locator(MaxNLocator(max_n_ticks, prune="lower"))
                ax.yaxis.set_major_locator(MaxNLocator(max_n_ticks, prune="lower"))

            if i < K - 1:
                ax.set_xticklabels([])
            else:
                if reverse:
                    ax.xaxis.tick_top()
                [l.set_rotation(45) for l in ax.get_xticklabels()]
                if labels is not None:
                    ax.set_xlabel(labels[j], **label_kwargs)
                    if reverse:
                        ax.xaxis.set_label_coords(0.5, 1.4)
                    else:
                        ax.xaxis.set_label_coords(0.5, -0.35)

                # use MathText for axes ticks
                ax.xaxis.set_major_formatter(ScalarFormatter(useMathText=use_math_text))

            if j > 0:
                ax.set_yticklabels([])
            else:
                if reverse:
                    ax.yaxis.tick_right()
                [l.set_rotation(45) for l in ax.get_yticklabels()]
                if labels is not None:
                    if reverse:
                        ax.set_ylabel(labels[i], rotation=-90, **label_kwargs)
                        ax.yaxis.set_label_coords(1.3, 0.5)
                    else:
                        ax.set_ylabel(labels[i], **label_kwargs)
                        ax.yaxis.set_label_coords(-0.35, 0.5)

                # use MathText for axes ticks
                ax.yaxis.set_major_formatter(ScalarFormatter(useMathText=use_math_text))

    return fig


def quantile(x, q, weights=None):
    """
    Compute sample quantiles with support for weighted samples.

    Note
    ----
    When ``weights`` is ``None``, this method simply calls numpy's percentile
    function with the values of ``q`` multiplied by 100.

    Parameters
    ----------
    x : array_like[nsamples,]
       The samples.

    q : array_like[nquantiles,]
       The list of quantiles to compute. These should all be in the range
       ``[0, 1]``.

    weights : Optional[array_like[nsamples,]]
        An optional weight corresponding to each sample. These

    Returns
    -------
    quantiles : array_like[nquantiles,]
        The sample quantiles computed at ``q``.

    Raises
    ------
    ValueError
        For invalid quantiles; ``q`` not in ``[0, 1]`` or dimension mismatch
        between ``x`` and ``weights``.

    """
    x = np.atleast_1d(x)
    q = np.atleast_1d(q)

    if np.any(q < 0.0) or np.any(q > 1.0):
        raise ValueError("Quantiles must be between 0 and 1")

    if weights is None:
        return np.percentile(x, list(100.0 * q))
    else:
        weights = np.atleast_1d(weights)
        if len(x) != len(weights):
            raise ValueError("Dimension mismatch: len(weights) != len(x)")
        idx = np.argsort(x)
        sw = weights[idx]
        cdf = np.cumsum(sw)[:-1]
        cdf /= cdf[-1]
        cdf = np.append(0, cdf)
        return np.interp(q, cdf, x[idx]).tolist()


def hist2d(
    x,
    y,
    bins=20,
    range=None,
    weights=None,
    levels=None,
    smooth=None,
    ax=None,
    color=None,
    plot_datapoints=True,
    plot_density=True,
    plot_contours=True,
    no_fill_contours=True,
    fill_contours=True,
    contour_kwargs=None,
    contourf_kwargs=None,
    data_kwargs=None,
    **kwargs
):
    """
    Plot a 2-D histogram of samples.

    Parameters
    ----------
    x : array_like[nsamples,]
       The samples.

    y : array_like[nsamples,]
       The samples.

    levels : array_like
        The contour levels to draw.

    ax : matplotlib.Axes
        A axes instance on which to add the 2-D histogram.

    plot_datapoints : bool
        Draw the individual data points.

    plot_density : bool
        Draw the density colormap.

    plot_contours : bool
        Draw the contours.

    no_fill_contours : bool
        Add no filling at all to the contours (unlike setting
        ``fill_contours=False``, which still adds a white fill at the densest
        points).

    fill_contours : bool
        Fill the contours.

    contour_kwargs : dict
        Any additional keyword arguments to pass to the `contour` method.

    contourf_kwargs : dict
        Any additional keyword arguments to pass to the `contourf` method.

    data_kwargs : dict
        Any additional keyword arguments to pass to the `plot` method when
        adding the individual data points.

    """
    if ax is None:
        ax = pl.gca()

    # Set the default range based on the data range if not provided.
    if range is None:
        if "extent" in kwargs:
            logging.warn(
                "Deprecated keyword argument 'extent'. " "Use 'range' instead."
            )
            range = kwargs["extent"]
        else:
            range = [[x.min(), x.max()], [y.min(), y.max()]]

    # Set up the default plotting arguments.
    if color is None:
        color = "k"

    # Choose the default "sigma" contour levels,
    # https://corner.readthedocs.io/en/latest/pages/sigmas.html
    if levels is None:
        levels = 1.0 - np.exp(-0.5 * np.arange(0.5, 2.1, 0.5) ** 2)

    # This is the color map for the density plot, over-plotted to indicate the
    # density of the points near the center.
    density_cmap = LinearSegmentedColormap.from_list(
        "density_cmap", [color, (1, 1, 1, 0)]
    )

    # This color map is used to hide the points at the high density areas.
    white_cmap = LinearSegmentedColormap.from_list(
        "white_cmap", [(1, 1, 1), (1, 1, 1)], N=2
    )

    # This "color map" is the list of colors for the contour levels if the
    # contours are filled.
    rgba_color = colorConverter.to_rgba(color)
    contour_cmap = [list(rgba_color) for l in levels] + [rgba_color]
    for i, l in enumerate(levels):
        contour_cmap[i][-1] *= float(i) / (len(levels) + 1)

    # We'll make the 2D histogram to directly estimate the density.
    H, X, Y = np.histogram2d(
        x.flatten(),
        y.flatten(),
        bins=bins,
        range=list(map(np.sort, range)),
        weights=weights,
    )

    if smooth is not None:
        if gaussian_filter is None:
            raise ImportError("Please install scipy for smoothing")
        H = gaussian_filter(H, smooth)

    # Compute the density levels.
    if not (np.all(x == x[0]) or np.all(y == y[0])):
        if plot_contours or plot_density:
            Hflat = H.flatten()
            inds = np.argsort(Hflat)[::-1]
            Hflat = Hflat[inds]
            sm = np.cumsum(Hflat)
            sm /= sm[-1]
            V = np.empty(len(levels))
            for i, v0 in enumerate(levels):
                try:
                    V[i] = Hflat[sm <= v0][-1]
                except Exception:
                    V[i] = Hflat[0]
            V.sort()
            m = np.diff(V) == 0
            if np.any(m):
                logging.warning("Too few points to create valid contours")
            while np.any(m):
                V[np.where(m)[0][0]] *= 1.0 - 1e-4
                m = np.diff(V) == 0
            V.sort()

            # Compute the bin centers.
            X1, Y1 = 0.5 * (X[1:] + X[:-1]), 0.5 * (Y[1:] + Y[:-1])

            # Extend the array for the sake of the contours at the plot edges.
            H2 = H.min() + np.zeros((H.shape[0] + 4, H.shape[1] + 4))
            H2[2:-2, 2:-2] = H
            H2[2:-2, 1] = H[:, 0]
            H2[2:-2, -2] = H[:, -1]
            H2[1, 2:-2] = H[0]
            H2[-2, 2:-2] = H[-1]
            H2[1, 1] = H[0, 0]
            H2[1, -2] = H[0, -1]
            H2[-2, 1] = H[-1, 0]
            H2[-2, -2] = H[-1, -1]
            X2 = np.concatenate(
                [
                    X1[0] + np.array([-2, -1]) * np.diff(X1[:2]),
                    X1,
                    X1[-1] + np.array([1, 2]) * np.diff(X1[-2:]),
                ]
            )
            Y2 = np.concatenate(
                [
                    Y1[0] + np.array([-2, -1]) * np.diff(Y1[:2]),
                    Y1,
                    Y1[-1] + np.array([1, 2]) * np.diff(Y1[-2:]),
                ]
            )

    if plot_datapoints:
        if data_kwargs is None:
            data_kwargs = dict()
        data_kwargs["color"] = data_kwargs.get("color", color)
        data_kwargs["ms"] = data_kwargs.get("ms", 2.0)
        data_kwargs["mec"] = data_kwargs.get("mec", "none")
        data_kwargs["alpha"] = data_kwargs.get("alpha", 0.1)
        ax.plot(x, y, "o", zorder=-1, rasterized=True, **data_kwargs)

    # Plot the base fill to hide the densest data points.
    if not (np.all(x == x[0]) or np.all(y == y[0])):
        if (plot_contours or plot_density) and not no_fill_contours:
            ax.contourf(
                X2, Y2, H2.T, [V.min(), H.max()], cmap=white_cmap, antialiased=False
            )

        if plot_contours and fill_contours:
            if contourf_kwargs is None:
                contourf_kwargs = dict()
            contourf_kwargs["colors"] = contourf_kwargs.get("colors", contour_cmap)
            contourf_kwargs["antialiased"] = contourf_kwargs.get("antialiased", False)
            ax.contourf(
                X2,
                Y2,
                H2.T,
                np.concatenate([[0], V, [H.max() * (1 + 1e-4)]]),
                **contourf_kwargs
            )

        # Plot the density map. This can't be plotted at the same time as the
        # contour fills.
        elif plot_density:
            ax.pcolor(X, Y, H.max() - H.T, cmap=density_cmap)

        # Plot the contour edge colors.
        if plot_contours:
            if contour_kwargs is None:
                contour_kwargs = dict()
            contour_kwargs["colors"] = contour_kwargs.get("colors", color)
            ax.contour(X2, Y2, H2.T, V, **contour_kwargs)

    ax.set_xlim(range[0])
    ax.set_ylim(range[1])


def lighten_color(color, amount=0.5):
    """
    Lightens the given color by multiplying (1-luminosity) by the given amount.
    Input can be matplotlib color string, hex string, or RGB tuple.

    Examples:
        >> lighten_color('g', 0.3)
        >> lighten_color('#F034A3', 0.6)
        >> lighten_color((.3,.55,.1), 0.5)
    """
    try:
        c = mc.cnames[color]
    except:
        c = color

    c = colorsys.rgb_to_hls(*mc.to_rgb(c))
    return colorsys.hls_to_rgb(c[0], max(0, min(1, amount * c[1])), c[2])
