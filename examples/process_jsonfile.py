"""
An example of how to extract information from a BASTA .json file
"""
import os
import sys

import h5py
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde
from scipy.ndimage import gaussian_filter

import basta.fileio as fio


def extract_from_json(jsonfile, gridfile, parameter):
    """
    Extract information from a BASTA json file.

    During a fit, if "optionaloutputs" is True, a dump of the statistics for each star
    will be stored in a .json file. The grid used to perform the fit is required to
    obtain any useful information from the json-dump.

    Parameters
    ----------
    jsonfile : str
        Path to the .json file from the BASTA fit

    gridfile : str
        Path to the grid used to compute the fit

    parameter : str
        The parameter values to extract (must be a valid name!)

    Returns
    -------
    parametervals : array_like
        Values of the given parameter for all models (with non-zero likelihood) in the
        grid

    loglikes : array_like
        The log(likelihood) of all models in the grid, excluding zero-likelihood models

    chi2 : array_like
        The chi**2 values of all models (with non-zero likelihood) in the grid
    """
    # The json file can be read into selectedmodels (in BASTA lingo), which contains
    # chi2 and log-likelihood (called logPDF internally) of all models in the grid. It
    # is stored by track (see below).
    selectedmodels = fio.load_selectedmodels(jsonfile)

    # Typically, we want to extract the likelihoods and the values of a given parameter
    # for all models in the grid/fit
    with h5py.File(gridfile, "r") as grid:
        parametervals = []
        loglikes = []
        chi2s = []

        # In selectedmodels, the information is accessed by looping over the tracks
        for trackpath, trackstats in sorted(selectedmodels.items()):
            # The log-likelihood (called logPDF) and chi2 can be stored directly
            loglikes.append(trackstats.logPDF)
            chi2s.append(trackstats.chi2)

            # The corresponding parameter values can be extracted from the grid
            gridval = grid["{0}/{1}".format(trackpath, parameter)]
            parametervals.append(gridval[trackstats.index])

        # After extraction it is useful to collapse the lists into numpy arrays
        parametervals = np.concatenate(parametervals)
        loglikes = np.concatenate(loglikes)
        chi2s = np.concatenate(chi2s)

    return parametervals, loglikes, chi2s


def sample_posterior(vals, logys):
    """
    Computation of the posterior of a parameter by numerical sampling. Duplication of
    how it is performed in basta/process_output.py.

    Parameters
    ----------
    vals : array_like
        Parameter values

    logys : array_like
        Corresponding log(likelihood)'s

    Returns
    ------
    samples : array_like
        Samples of the posterior. Can be used to make the posterior distribution.
    """
    lk = logys - np.amax(logys)
    post = np.exp(lk - np.log(np.sum(np.exp(lk))))
    nsamples = min(100000, len(logys))
    sampled_indices = np.random.choice(a=np.arange(len(post)), p=post, size=nsamples)
    return vals[sampled_indices]


if __name__ == "__main__":
    # Init
    plt.close("all")
    outdir = os.path.join("output", "json-analysis")
    if not os.path.exists(outdir):
        os.makedirs(outdir)

    # Extract a given parameter and corresponding statistics
    param = "massini"
    infile = os.path.join("output", "json", "16CygA.json")
    gridf = os.path.join(os.environ["BASTADIR"], "grids", "Garstec_16CygA.hdf5")
    try:
        paramvals, loglike, _ = extract_from_json(
            jsonfile=infile, gridfile=gridf, parameter=param
        )
    except FileNotFoundError:
        print(
            "Cannot read '{0}'! Did you run 'xmlinput/input_json.xml'?".format(infile)
        )
        sys.exit(1)

    # The extracted information can then be used to, e.g., sample the posterior
    samples = sample_posterior(vals=paramvals, logys=paramvals)

    # ... which can then be plotted in a simple, smoothed histogram
    _, ax = plt.subplots()
    counts, bins = np.histogram(a=samples, bins=100)  # Fixed bins
    counts = gaussian_filter(input=counts, sigma=2)  # Slightly smoothed
    x0 = np.array(list(zip(bins[:-1], bins[1:]))).flatten()
    y0 = np.array(list(zip(counts, counts))).flatten()
    ax.plot(x0, y0, label="Smoothed posterior histogram (100 bins)")
    ax.set_xlabel(param)
    ax.set_ylabel("Counts")
    ax.legend(loc="best", facecolor="none", edgecolor="none", fontsize="small")
    plt.savefig(
        os.path.join(outdir, f"posterior_{param}_simple.pdf"), bbox_inches="tight"
    )

    # ... or in a KDE smoothed version, like the BASTA cornerplots
    histargs = {"color": "tab:cyan", "alpha": 0.17}
    histargs_line = {"color": "tab:orange", "alpha": 0.7}
    histargs_fill = {"color": "tab:orange", "alpha": 0.1}
    _, ax = plt.subplots()
    counts, bins = np.histogram(a=samples, bins="auto")
    counts = gaussian_filter(input=counts, sigma=1)
    kernel = gaussian_kde(samples, bw_method="silverman")
    x0 = np.linspace(np.amin(samples), np.amax(samples), num=500)
    y0 = kernel(x0)
    y0 /= np.amax(y0)
    x0_hist = np.array(list(zip(bins[:-1], bins[1:]))).flatten()
    y0_hist = np.array(list(zip(counts, counts))).flatten() / np.amax(counts)
    ax.fill_between(
        x0_hist, y0_hist, interpolate=True, label="Posterior histogram", **histargs
    )
    ax.plot(x0, y0, label="Posterior KDE", **histargs_line)
    ax.fill_between(x0, y0, interpolate=True, **histargs_fill)
    ax.set_xlabel(param)
    ax.set_ylabel("Density")
    ax.legend(loc="best", facecolor="none", edgecolor="none", fontsize="small")
    ax.get_yaxis().set_ticks([])
    plt.savefig(os.path.join(outdir, f"posterior_{param}.pdf"), bbox_inches="tight")

    # Done!
    print(f"Plots saved to '{outdir}'!")
