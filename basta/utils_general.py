"""
General mix of utility functions
"""
import sys
from io import IOBase

import numpy as np


def prt_center(text, llen):
    """
    Prints a centered line

    Parameters
    ----------
    text : str
        The text string to print

    llen : int
        Length of the line

    Returns
    -------
    None
    """
    print("{0}{1}{0}".format(int((llen - len(text)) / 2) * " ", text))


class Logger(object):
    """
    Class used to redefine stdout to terminal and an output file.

    Parameters
    ----------
    outfilename : str
        Absolute path to an output file
    """

    # Credit: http://stackoverflow.com/a/14906787
    def __init__(self, outfilename):
        self.terminal = sys.stdout
        self.log = open(outfilename + ".log", "a")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        # this flush method is needed for python 3 compatibility.
        # this handles the flush command by doing nothing.
        # you might want to specify some extra behavior here.
        pass


def list_metallicities(Grid, defaultpath, inputparams, limits):
    """
    Get a list of metallicities in the grid that we loop over

    Parameters
    ----------
    Grid : hdf5 object
        The already loaded grid, containing the tracks/isochrones.
    defaultpath : str
        Path in Grid
    inputparams : dict
        Dictionary of all controls and input.
    limits : dict
        Dict of flat priors used in run.

    Returns
    -------
    metal : list
        List of possible metalliticies that should be looped over in
        `bastamain`.
    """
    if "grid" in defaultpath:
        metal = range(1)
    else:
        metal = [x for x in Grid[defaultpath].items() if "=" in x[0]]
        for i in range(len(metal)):
            metal[i] = float(metal[i][0][4:])
        metal = np.asarray(metal)

        metal_name = "MeH" if "MeH" in limits else "FeH"
        if metal_name in limits:
            metal = metal[
                (metal >= limits[metal_name][0]) & (metal <= limits[metal_name][1])
            ]
    return metal


def unique_unsort(params):
    """
    As we want to check for unique elements to not copy elements, but retain the
    order they were given in, we have to do this, until numpy implements an 'unsort'
    key to numpy.unique...

    Parameters
    ----------
    params : list
        List of parameters

    Returns
    -------
    params : list
        List of unique params, retaining order
    """
    indexes = np.unique(params, return_index=True)[1]
    return [params[index] for index in sorted(indexes)]


def compare_output_to_input(
    starid, inputparams, hout, out, hout_dist, out_dist, uncert="qunatiles", sigmacut=1
):
    """
    This function compares the outputted value of all fitting parameters
    to the input that was fitted.

    If one or more fitting parameters deviates more than 'sigmacut' number
    of the effective symmetric uncertainty away from their input parameter,
    a warning is printed and 'starid' is appended to the .warn-file.

    Parameters
    ----------
    starid : str
        Unique identifier of current target.
    inputparms : dict
        Dict containing input from xml-file.
    hout : list
        List of column headers for output
    out : list
        List of output values for the columns given in `hout`.
    uncert : str
        Type of reported uncertainty to use for comparison.
    sigmacut : float, optional
        Number of standard deviation used for determining when to issue
        a warning.

    Returns
    -------
    comparewarn : bool
        Flag to determine whether or not a warning was raised.
    """
    if inputparams["warnoutput"] is False:
        return False
    fitparams = inputparams["fitparams"]
    warnfile = inputparams["warnoutput"]
    comparewarn = False
    ps = []
    sigmas = []

    for p in fitparams:
        if p in hout:
            idx = np.nonzero([p == xout for xout in hout])[0][0]
            xin, xinerr = fitparams[p]
            if uncert == "quantiles":
                outerr = (out[idx + 1] + out[idx + 2]) / 2
            else:
                outerr = out[idx + 1]
            serr = np.sqrt(outerr ** 2 + xinerr ** 2)
            sigma = np.abs(out[idx] - xin) / serr
            bigdiff = sigma >= sigmacut
            if bigdiff:
                comparewarn = True
                ps.append(p)
                sigmas.append(sigma)

    if len(inputparams["magnitudes"]) > 0:
        for m in list(inputparams["distanceparams"]["filters"]):
            mdist = "M_" + m
            if mdist in hout_dist:
                idx = np.nonzero([x == mdist for x in hout_dist])[0][0]
                priorM = inputparams["magnitudes"][m]["median"]
                priorerrp = inputparams["magnitudes"][m]["errp"]
                priorerrm = inputparams["magnitudes"][m]["errm"]
                if uncert == "quantiles":
                    outerr = (out_dist[idx + 1] + out_dist[idx + 2]) / 2
                else:
                    outerr = out_dist[idx + 1]
                serr = np.sqrt(((priorerrp + priorerrm) / 2) ** 2 + outerr ** 2)
                sigma = np.abs(out_dist[idx] - priorM) / serr
                bigdiff = sigma >= sigmacut
                if bigdiff:
                    comparewarn = True
                    ps.append(mdist)
                    sigmas.append(sigma)

        if "distance" in hout_dist:
            idx = np.nonzero([x == "distance_joint" for x in hout_dist])[0][0]
            priordistqs = inputparams["distanceparams"]["priordistance"]
            priorerrm = priordistqs[1] - priordistqs[0]
            priorerrp = priordistqs[2] - priordistqs[1]
            if uncert == "quantiles":
                outerr = (out_dist[idx + 1] + out_dist[idx + 2]) / 2
            else:
                outerr = out_dist[idx + 1]
            serr = np.sqrt(((priorerrp + priorerrm) / 2) ** 2 + outerr ** 2)
            sigma = np.abs(out_dist[idx] - priordistqs[1]) / serr
            bigdiff = sigma >= sigmacut
            if bigdiff:
                comparewarn = True
                ps.append("distance")
                sigmas.append(sigma)

    if comparewarn:
        print("A >%s sigma difference was found between input and output of" % sigmacut)
        print(ps)
        print("with sigma differences of")
        print(sigmas)
        if isinstance(warnfile, IOBase):
            warnfile.write("{}\t{}\t{}\n".format(starid, ps, sigmas))
        else:
            with open(warnfile, "a") as wf:
                wf.write("{}\t{}\t{}\n".format(starid, ps, sigmas))

    return comparewarn


def inflog(x):
    "np.log(x), but where x=0 returns -inf without a warning"
    with np.errstate(divide="ignore"):
        return np.log(x)


def add_out(hout, out, par, x, xm, xp, uncert):
    """
    Add entries in out list, according to the wanted uncertainty.

    Parameters
    ----------
    hout : list
        Names in header
    out : list
        Parameter values
    par : str
        Parameter name
    x : float
        Centroid value
    xm : float
        Lower bound uncertainty, or symmetric unceartainty
    xp : float, None
        Upper bound uncertainty, if not symmetric
    uncert : str
        Type of reported uncertainty, "quantiles" or "std"

    Returns
    -------
    hout : list
        Header list with added names
    out : list
        Parameter list with added entries
    """
    if uncert == "quantiles":
        hout += [par, par + "_errm", par + "_errp"]
        out += [x, xp - x, x - xm]
    else:
        hout += [par, par + "_err"]
        out += [x, xm]
    return hout, out


def normfactor(alphas, ms):
    # Algorithm from App. A in Pflamm-Altenburg & Kroupa (2006)
    # https://ui.adsabs.harvard.edu/abs/2006MNRAS.373..295P/abstract
    ks = np.zeros(len(alphas))
    ks[0] = (1 / ms[1]) ** alphas[0]
    ks[1] = (1 / ms[1]) ** alphas[1]
    if len(ks) == 2:
        return ks
    ks[2] = (ms[2] / ms[1]) ** alphas[1] * (1 / ms[2]) ** alphas[2]
    if len(ks) == 3:
        return ks
    if len(ks) == 4:
        ks[3] = (
            (ms[2] / ms[1]) ** alphas[1]
            * (ms[3] / ms[2]) ** alphas[2]
            * (1 / ms[3]) ** alphas[3]
        )
        return ks
    else:
        print("Mistake in normfactor")


def get_parameter_values(parameter, Grid, selectedmodels, noofind):
    """
    Get parameter values from grid

    Parameters
    ----------
    parameter : str
        Grid, hdf5 object
    selectedmodels :
        models to return
    noofind :
        number of parameter values

    Returns
    -------
    x_all : array
        parameter values
    """
    x_all = np.zeros(noofind)
    i = 0
    for modelpath in selectedmodels:
        N = len(selectedmodels[modelpath].logPDF)
        try:
            x_all[i : i + N] = selectedmodels[modelpath].paramvalues[parameter]
        except Exception:
            x_all[i : i + N] = Grid[modelpath + "/" + parameter][
                selectedmodels[modelpath].index
            ]
        i += N
    return x_all


def printparam(param, xmed, xstdm, xstdp, uncert="quantiles", centroid="median"):
    if uncert == "quantiles":
        print(centroid + " " + param + ":", xmed)
        print("stdm " + param + "  :", xmed - xstdm)
        print("stdp " + param + "  :", xstdp - xmed)
        print("-----------------------------------------------------")
    else:
        print(centroid + " " + param + ":", xmed)
        print("std " + param + "  :", xstdm)
        print("-----------------------------------------------------")
