"""
Auxiliary functions for file operations
"""

import os
import json
import h5py
import warnings
import numpy as np
from io import IOBase
from copy import deepcopy
from xml.etree import ElementTree
from xml.etree.ElementTree import Element, SubElement, tostring
from xml.dom import minidom

from basta import stats, freq_fit, glitch_fit
from basta import utils_seismic as su
from basta import utils_general as util
from basta.constants import freqtypes


def _export_selectedmodels(selectedmodels: dict) -> dict:
    res = {}
    for trackno, ts in selectedmodels.items():
        index = ts.index.nonzero()[0].tolist()
        res[trackno] = {
            "chi2": ts.chi2.tolist(),
            "index": index,
            "n": len(ts.index),
            "logPDF": ts.logPDF.tolist(),
        }

    return res


def _import_selectedmodels(data: dict) -> dict:
    res = {}
    for trackno, ts in data.items():
        index = np.zeros(ts["n"], dtype=bool)
        index[ts["index"]] = True
        res[trackno] = stats.Trackstats(
            chi2=np.asarray(ts["chi2"]), index=index, logPDF=np.asarray(ts["logPDF"])
        )

    return res


def save_selectedmodels(fname: str, selectedmodels):
    s = json.dumps(_export_selectedmodels(selectedmodels))
    with open(fname, "w") as fp:
        fp.write(s)


def load_selectedmodels(fname: str):
    with open(fname) as fp:
        data = json.load(fp)

    return _import_selectedmodels(data)


def write_star_to_errfile(starid: str, inputparams: dict, errormessage: str):
    """
    Write starid and error message to .err-file

    Parameters
    ----------
    starid : str
        Unique identifier for this target.
    inputparams : dict
        Dictionary of all controls and input.
    errormessage : str
        String explaining error which will be written to the .err-file
    """
    errfile = inputparams.get("erroutput")

    if isinstance(errfile, IOBase):
        errfile.write("{}\t{}\n".format(starid, errormessage))
    else:
        with open(errfile, "a") as ef:
            ef.write("{}\t{}\n".format(starid, errormessage))


def no_models(starid: str, inputparams: dict, errormessage: str):
    """
    If no models are found in the grid, create an outputfile with nans (for consistency reasons).

    The approach mirrors process_output.compute_posterior()

    Parameters
    ----------
    starid : str
        Unique identifier for this target.
    inputparams : dict
        Dictionary of all controls and input.
    errormessage : str
        String explaining error which will be written to the .err-file
    """

    # Extract the output parameters
    asciifile = inputparams.get("asciioutput")
    asciifile_dist = inputparams.get("asciioutput_dist")
    params = deepcopy(inputparams["asciiparams"])

    # Init vectors and add the Star ID
    hout = []
    out = []
    hout_dist = []
    out_dist = []
    hout.append("starid")
    out.append(starid)
    hout_dist.append("starid")
    out_dist.append(starid)
    uncert = inputparams.get("uncert")

    # The distance parameters
    if "distance" in params:
        distanceparams = inputparams["distanceparams"]
        ms = list(distanceparams["filters"])

        for m in ms:
            hout_dist, out_dist = util.add_out(
                hout_dist, out_dist, "distance_" + m, np.nan, np.nan, np.nan, uncert
            )
            hout_dist, out_dist = util.add_out(
                hout_dist, out_dist, "A_" + m, np.nan, np.nan, np.nan, uncert
            )
            hout_dist, out_dist = util.add_out(
                hout_dist, out_dist, "M_" + m, np.nan, np.nan, np.nan, uncert
            )

        hout_dist, out_dist = util.add_out(
            hout_dist, out_dist, "distance", np.nan, np.nan, np.nan, uncert
        )
        hout_dist, out_dist = util.add_out(
            hout_dist, out_dist, "EBV_", np.nan, np.nan, np.nan, uncert
        )
        hout, out = util.add_out(hout, out, "distance", np.nan, np.nan, np.nan, uncert)
        params.remove("distance")

    # The normal parameters
    for param in params:
        hout, out = util.add_out(hout, out, param, np.nan, np.nan, np.nan, uncert)

    # Write to file
    if asciifile is not False:
        hline = b"# "
        for i in range(len(hout)):
            hline += hout[i].encode() + " ".encode()

        if isinstance(asciifile, IOBase):
            asciifile.seek(0)
            if b"#" not in asciifile.readline():
                asciifile.write(hline + b"\n")
            np.savetxt(
                asciifile, np.asarray(out).reshape(1, len(out)), fmt="%s", delimiter=" "
            )
            print("Saved results to " + asciifile.name + ".")
        elif asciifile is False:
            pass
        else:
            np.savetxt(
                asciifile,
                np.asarray(out).reshape(1, len(out)),
                fmt="%s",
                header=hline,
                delimiter=" ",
            )
            print("Saved results to " + asciifile + ".")

    # Write to file
    if asciifile_dist:
        hline = b"# "
        for i in range(len(hout_dist)):
            hline += hout_dist[i].encode() + " ".encode()

        if isinstance(asciifile_dist, IOBase):
            asciifile_dist.seek(0)
            if b"#" not in asciifile_dist.readline():
                asciifile_dist.write(hline + b"\n")
            np.savetxt(
                asciifile_dist,
                np.asarray(out_dist).reshape(1, len(out_dist)),
                fmt="%s",
                delimiter=" ",
            )
            print(
                "Saved distance results for different filters to "
                + asciifile_dist.name
                + "."
            )
        else:
            np.savetxt(
                asciifile_dist,
                np.asarray(out_dist).reshape(1, len(out_dist)),
                fmt="%s",
                header=hline,
                delimiter=" ",
            )
            print(
                "Saved  distance results for different filters to "
                + asciifile_dist
                + "."
            )

    write_star_to_errfile(starid, inputparams, errormessage)


def read_freq_xml(filename: str) -> tuple[np.array, np.array, np.array, np.array]:
    """
    Read frequencies from an xml file

    Parameters
    ----------
    filename : str
        Name of file to read

    Returns
    -------
    frequencies : array
        Individual frequencies
    errors : array
        Uncertainties in the frequencies
    orders : array
        Radial orders
    degrees : array
        Angular degree
    """

    # Parse the XML file:
    tree = ElementTree.parse(filename)
    root = tree.getroot()

    # Find a list of all the frequency ratios:
    freq_modes = root.findall("mode")

    # Simply get the orders and frequency ratios as numpy arrays:
    orders = np.array(
        [mode.find("order").get("value") for mode in freq_modes], dtype=int
    )
    degrees = np.array(
        [mode.find("degree").get("value") for mode in freq_modes], dtype=int
    )
    errors = np.array(
        [mode.find("frequency").get("error") for mode in freq_modes], dtype="float64"
    )
    frequencies = np.array(
        [mode.find("frequency").get("value") for mode in freq_modes], dtype="float64"
    )
    return frequencies, errors, orders, degrees


def _read_freq_cov_xml(filename: str, obskey: np.array) -> tuple[np.array, np.array]:
    """
    Read frequency covariances from xml-file

    Parameters
    ----------
    filename : str
        Name of xml-file containing the frequencies
    obskey : array
        Observed modes after filtering

    Returns
    -------
    corr : array
        Correlation matrix of frequencies
    cov : array
        Covariance matrix of frequencies
    """
    # Parse the XML file:
    tree = ElementTree.parse(filename)
    root = tree.getroot()

    # Find a list of all the frequency ratios:
    freqs = root.findall("mode")

    # Set up the matrices for collection
    corr = np.identity(len(obskey[0, :]))
    cov = np.zeros((len(obskey[0, :]), len(obskey[0, :])))

    # Loop over all modes to collect input
    for i, mode1 in enumerate(freqs):
        id1 = mode1.get("id")
        column = root.findall("frequency_corr[@id1='%s']" % id1)
        if not len(column):
            errmsg = "Correlations in frequency fit requested, but not pr"
            errmsg += "ovided! If not available, set 'correlations=False'."
            raise KeyError(errmsg)
        # Loop over possible all modes again
        for j, mode2 in enumerate(freqs):
            id2 = mode2.get("id")
            # Loop to find matching entries in matrix
            for row in column:
                if row.get("id2") == id2:
                    n1 = int(mode1.find("order").get("value"))
                    l1 = int(mode1.find("degree").get("value"))
                    n2 = int(mode2.find("order").get("value"))
                    l2 = int(mode2.find("degree").get("value"))

                    i = np.where(np.logical_and(obskey[0, :] == l1, obskey[1, :] == n1))
                    j = np.where(np.logical_and(obskey[0, :] == l2, obskey[1, :] == n2))

                    corr[i, j] = corr[j, i] = row.find("correlation").get("value")
                    cov[i, j] = cov[j, i] = row.find("covariance").get("value")
                    break
    return corr, cov


def read_freq(
    filename: str,
    excludemodes: str | None = None,
    onlyradial: bool = False,
    flag_onlyls: list[int] = [],
    covarfre: bool = False,
) -> tuple[np.array, np.array, np.array]:
    """
    Routine to extract the frequencies in the desired n-range, and the
    corresponding covariance matrix

    Parameters
    ----------
    filename : str
        Name of file to read
    excludemodes : str or None, optional
        Name of file containing the (l, n) values of frequencies to be
        omitted in the fit. If None, no modes will be excluded.
    onlyradial : bool
        Flag to determine to only fit the l=0 modes
    covarfre : bool, optional
        Read also the covariances in the individual frequencies

    Returns
    -------
    obskey : array
        Array containing the angular degrees and radial orders of obs
    obs : array
        Individual frequencies
    covarfreq : array
        Array including covariances in the frequencies. If ``covarfre`` is
        ``False`` then a diagonal matrix is produced
    """
    # Read frequencies from file
    frecu, errors, norder, ldegree = read_freq_xml(filename)

    # Build osc and osckey in a sorted manner
    f = np.asarray([])
    n = np.asarray([])
    e = np.asarray([])
    l = np.asarray([])
    for li in [0, 1, 2]:
        given_l = ldegree == li
        incrn = np.argsort(norder[given_l], kind="mergesort")
        l = np.concatenate([l, ldegree[given_l][incrn]])
        n = np.concatenate([n, norder[given_l][incrn]])
        f = np.concatenate([f, frecu[given_l][incrn]])
        e = np.concatenate([e, errors[given_l][incrn]])
    assert len(f) == len(n) == len(e) == len(l)
    obskey = np.asarray([l, n], dtype=int)
    obs = np.array([f, e])

    # Remove untrusted frequencies
    if excludemodes not in [None, "", "None", "none", "False", "false"]:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            nottrustedobskey = np.genfromtxt(excludemodes, comments="#", encoding=None)
        # If there is only one not-trusted mode, it misses a dimension
        if nottrustedobskey.size == 0:
            print("File for not-trusted frequencies was empty")
        else:
            if nottrustedobskey.shape == (2,):
                nottrustedobskey = [nottrustedobskey]
            for l, n in nottrustedobskey:
                nottrustedmask = (obskey[0] == l) & (obskey[1] == n)
                print(
                    *(f"Removed mode at {x} µHz" for x in obs[:, nottrustedmask][0]),
                    sep="\n",
                )
                obskey = obskey[:, ~nottrustedmask]
                obs = obs[:, ~nottrustedmask]
    if flag_onlyls or onlyradial:
        if onlyradial:
            flag_onlyls = [
                0,
            ]
        assert all(isinstance(x, int) for x in flag_onlyls)
        onlylsmask = np.isin(obskey[0], flag_onlyls)
        print("\n".join(f"Removed mode at {x} µHz" for x in obs[:, ~onlylsmask][0]))
        obskey = obskey[:, onlylsmask]
        obs = obs[:, onlylsmask]

    if covarfre:
        corrfre, covarfreq = _read_freq_cov_xml(filename, obskey)
    else:
        covarfreq = np.diag(obs[1, :]) ** 2

    return obskey, obs, covarfreq


def _read_precomputed_glitches(
    filename: str, type: str = "glitches"
) -> tuple[np.array, np.array]:
    """
    Read glitch parameters. If fitted together with ratios, these must be
    provided in this file as well, for covariance between them.

    Parameters
    ----------
    filename : str
        Name of file to read
    grtype : str
        Parameter combination to be read from: glitches, gr02, gr01, gr10
        gr010, gr012, gr102.

    Returns
    -------
    gdata : array
        Array of median glitch parameters (and ratios)
    gcov : array
        Covariance matrix, for glitch parameters or glitch parameters
        and ratios
    """
    # Read datafile
    try:
        datfile = h5py.File(filename, "r")
    except:
        return NameError("Could not find f{filename}")

    # Read ratio type in file, and check that it matches requested fittype
    if type != "glitches":
        rtype = datfile["rto/rtype"][()].decode("utf-8")
        if rtype != type[1:]:
            raise KeyError(f"Requested ratio type {type[1:]} not found in {filename}")

    # Read data and covariance matrix
    gdata = datfile["cov/params"][()]
    gcov = datfile["cov/cov"][()]

    return gdata, gcov


def _read_precomputed_ratios_xml(
    filename: str,
    ratiotype: str,
    obskey: np.array,
    obs: np.array,
    excludemodes: str | None = None,
    correlations: bool = True,
) -> tuple[np.array, np.array]:
    """
    Read the precomputed ratios and covariance matrix from xml-file.

    Parameters
    ----------
    filename : str
        Path to xml-file to be read
    ratiotype : str
        Ratio sequence to be read, see constants.freqtypes.rtypes for
        possible sequences.
    obskey : array
        Array containing the angular degrees and radial orders of obs
    obs : array
        Individual frequencies and uncertainties.
    excludemodes : str or None, optional
        Name of file containing the (l, n) values of frequencies to be
        omitted in the fit. Does however only trigger a warning for
        precomputed ratios.
    correlations : bool
        True for reading covariance matrix from xml, False to assume no
        correlations, and simply use individual errors on ratios.

    Returns
    -------
    ratios : array
        Contains frequency ratios, their identifying integers of sequence and
        radial order, and their frequency location (matching l=0 frequency).
    cov : array
        Covariance matrix matching the ratios.
    """
    # Print warning to user
    if excludemodes != None:
        wstr = "Warning: Removing precomputed ratios based on "
        wstr += "not-trusted-file is not yet supported!"
        print(wstr)

    # Read in xml tree/root
    tree = ElementTree.parse(filename)
    root = tree.getroot()

    # Get all ratios available in xml
    all_ratios = root.findall("frequency_ratio")

    # Make numpy arrays of all ratios
    # Developers note: Standard examples also contain unused "error_minus" and "error_plus"
    orders = np.array([ratio.get("order") for ratio in all_ratios], dtype=int)
    ratval = np.array([ratio.get("value") for ratio in all_ratios], dtype=float)
    types = np.array([ratio.get("type") for ratio in all_ratios], dtype="U3")
    errors = np.array([ratio.get("error") for ratio in all_ratios], dtype=float)

    # Make sorting mask for the desired ratio sequence
    if ratiotype == "r012":
        mask = np.where(np.logical_or(types == "r01", types == "r02"))[0]
    elif ratiotype == "r102":
        mask = np.where(np.logical_or(types == "r10", types == "r02"))[0]
    elif ratiotype == "r010":
        mask = np.where(np.logical_or(types == "r01", types == "r10"))[0]
    else:
        mask = np.where(types == ratiotype)[0]

    # Pack into data structure
    ratios = np.zeros((4, len(mask)))
    ratios[0, :] = ratval[mask]
    ratios[3, :] = orders[mask]
    ratios[2, :] = [int(r[1:]) for r in types[mask]]

    # Get frequency location from obs and obskey
    try:
        for i, nn in enumerate(ratios[3, :]):
            l0mask = obskey[0, :] == 0
            ratios[1, i] = obs[0, l0mask][obskey[1, l0mask] == nn]
    except ValueError as e:
        wstr = "Could not find l=0, n={0:d} frequency to match {1}(n={0:d})!"
        raise KeyError(wstr.format(int(nn), types[mask][i])) from e

    # Sort n-before-l
    sorting = np.argsort(ratios[3, :] + 0.01 * ratios[2, :])
    ratios = ratios[:, sorting]

    # Either read covariance matrix or assume uncorrelated
    if correlations:
        cov = _read_ratios_cov_xml(root, types[mask][sorting], orders[mask][sorting])
    else:
        cov = np.diag(errors[mask][sorting])
    return ratios, cov


def _read_ratios_cov_xml(xmlroot, types: np.array, order: np.array) -> np.array:
    """
    Read the precomputed covariance matrix of the read precomputed ratios.

    Parameters
    ----------
    xmlroot : Element
        Element tree of xml-file.
    types : array
        Ratios types of all ratios to be read
    order : array
        Radial order of all ratios to be read

    Returns
    -------
    cov : array
        Covariance matrix matching the ratios.
    """
    # Empty matrix to fill out and format string to look for
    cov = np.zeros((len(types), len(types)))
    fstr = (
        "frequency_ratio_corr[@type1='{0}'][@order1='{1}'][@type2='{2}'][@order2='{3}']"
    )

    # Loop for each index in matrix and find it in xml
    for ind1, (type1, order1) in enumerate(zip(types, order)):
        for ind2, (type2, order2) in enumerate(zip(types, order)):
            try:
                element = xmlroot.findall(fstr.format(type1, order1, type2, order2))[0]
            except IndexError as e:
                wstr = (
                    "Could not find covariance between {0}(n={1}) and {2}(n={3}) in xml"
                )
                raise KeyError(wstr.format(type1, order1, type2, order2)) from e
            cov[ind1, ind2] = float(element.find("covariance").get("value"))

    return cov


def _make_obsfreqs(
    obskey: np.array,
    obs: np.array,
    obscov: np.array,
    allfits: list[str, ...],
    freqplots: list[str, ...],
    numax: float,
    debug: bool = False,
) -> tuple[dict, dict]:
    """
    Make a dictionary of frequency-dependent data

    Parameters
    ----------
    obskey : array
        Array containing the angular degrees and radial orders of obs
    obs : array
        Individual frequencies and uncertainties.
    obscov : array
        Covariance matrix of the individual frequencies
    allfits : list
        Type of fits available for individual frequencies
    freqplots : list
        List of frequency-dependent fits
    numax : float
        The input numax of the target
    debug : bool, optional
        Activate additional output for debugging (for developers)

    Returns
    -------
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
    """
    obsfreqdata = {}
    obsfreqmeta = {}

    allfits = np.asarray(list(allfits))
    allplots = np.asarray(list(freqplots))
    fitratiotypes = []
    plotratiotypes = []
    fitglitchtypes = []
    plotglitchtypes = []
    fitepsdifftypes = []
    plotepsdifftypes = []

    getratios = False
    getglitch = False
    getepsdiff = False

    obscovinv = np.linalg.pinv(obscov, rcond=1e-8)
    obsls = np.unique(obskey[0, :])

    # Large frequency separation from individual frequencies
    dnudata, dnudata_err = freq_fit.compute_dnu_wfit(obskey, obs, numax)

    for fit in allfits:
        obsfreqdata[fit] = {}
        # Look for ratios
        if fit in freqtypes.rtypes:
            getratios = True
            fitratiotypes.append(fit)
            if not "ratios" in obsfreqmeta.keys():
                obsfreqmeta["ratios"] = {}
        # Look for glitches
        elif fit in freqtypes.glitches:
            getglitch = True
            fitglitchtypes.append(fit)
            if not "glitch" in obsfreqmeta.keys():
                obsfreqmeta["glitch"] = {}
        # Look for epsdiff
        elif fit in freqtypes.epsdiff:
            getepsdiff = True
            fitepsdifftypes.append(fit)
            if not "epsdiff" in obsfreqmeta.keys():
                obsfreqmeta["epsdiff"] = {}
        elif fit not in freqtypes.freqs:
            print(f"Fittype {fit} not recognised")
            raise ValueError

    obsfreqdata["freqs"] = {
        "cov": obscov,
        "covinv": obscovinv,
        "dnudata": dnudata,
        "dnudata_err": dnudata_err,
    }

    # If all frequency plots enabled, turn on defaults
    if len(freqplots) and freqplots[0] == True:
        getratios = True
        getepsdiff = True

        plotratiotypes = list(set(freqtypes.defaultrtypes) | set(fitratiotypes))
        plotepsdifftypes = list(set(freqtypes.defaultepstypes) | set(fitepsdifftypes))

        # Only turn on glitches if they are fitted (expensive)
        if getglitch:
            plotglitchtypes = list(set(fitglitchtypes))

    elif len(freqplots):
        for plot in allplots:
            # Look for ratios
            if plot in ["ratios", *freqtypes.rtypes]:
                getratios = True
                if plot in [
                    "ratios",
                ]:
                    for rtype in freqtypes.defaultrtypes:
                        if rtype not in plotratiotypes:
                            plotratiotypes.append(rtype)
                else:
                    if plot not in plotratiotypes:
                        plotratiotypes.append(plot)
            # Look for glitches
            if plot in freqtypes.glitches:
                getglitch = True
                if plot not in plotglitchtypes:
                    plotglitchtypes.append(plot)
            # Look for epsdiff
            if plot in ["epsdiff", *freqtypes.epsdiff]:
                getepsdiff = True
                if plot in ["epsdiff"]:
                    for etype in freqtypes.defaultepstypes:
                        if etype not in plotepsdifftypes:
                            plotepsdifftypes.append(etype)
                else:
                    if plot not in plotepsdifftypes:
                        plotepsdifftypes.append(plot)

    # Check that there is observational data available for fits and plots
    if getratios or getepsdiff:
        for fittype in set(fitratiotypes) | set(fitepsdifftypes) | set(fitglitchtypes):
            if not all(x in obsls.astype(str) for x in fittype if x.isdigit()):
                for l in fittype[1:]:
                    if not l in obsls.astype(str):
                        print(f"* No l={l} modes were found in the observations")
                        print(f"* It is not possible to fit {fittype}")
                        raise ValueError
        for fittype in (
            set(plotratiotypes) | set(plotepsdifftypes) | set(plotglitchtypes)
        ):
            if not all(x in obsls.astype(str) for x in fittype if x.isdigit()):
                if debug:
                    print(f"*BASTA {fittype} cannot be plotted")
                if fittype in plotratiotypes:
                    plotratiotypes.remove(fittype)
                if fittype in plotepsdifftypes:
                    plotepsdifftypes.remove(fittype)
                if fittype in plotglitchtypes:
                    plotglitchtypes.remove(fittype)
        if getratios and ((len(fitratiotypes) == 0) & (len(plotratiotypes) == 0)):
            getratios = False
        if getepsdiff and ((len(fitepsdifftypes) == 0) & (len(plotepsdifftypes) == 0)):
            getepsdiff = False

    if getratios:
        obsfreqmeta["ratios"] = {}
        obsfreqmeta["ratios"]["fit"] = fitratiotypes
        obsfreqmeta["ratios"]["plot"] = plotratiotypes

    if getglitch:
        obsfreqmeta["glitch"] = {}
        obsfreqmeta["glitch"]["fit"] = fitglitchtypes
        obsfreqmeta["glitch"]["plot"] = plotglitchtypes

    if getepsdiff:
        obsfreqmeta["epsdiff"] = {}
        obsfreqmeta["epsdiff"]["fit"] = fitepsdifftypes
        obsfreqmeta["epsdiff"]["plot"] = plotepsdifftypes

    obsfreqmeta["getratios"] = getratios
    obsfreqmeta["getglitch"] = getglitch
    obsfreqmeta["getepsdiff"] = getepsdiff

    return obsfreqdata, obsfreqmeta


def read_allseismic(
    fitfreqs: dict,
    freqplots: list,
    verbose: bool = False,
    debug: bool = False,
) -> tuple[np.array, np.array, dict, dict]:
    """
    Routine to all necesary data from individual frequencies for the
    desired fit

    Parameters
    ----------
    fitfreqs : dict
        Contains all frequency related input needed for reading.
    freqplots : list
        List of frequency-dependent fits
    verbose : bool, optional
        If True, extra text will be printed to log (for developers).
    debug : bool, optional
        Activate additional output for debugging (for developers)

    Returns
    -------
    obskey : array
        Array containing the angular degrees and radial orders of obs
    obs : array
        Individual frequencies and uncertainties.
    obsfreqdata : dict
        Requested frequency-dependent data such as glitches, ratios, and
        epsilon difference. It also contains the covariance matrix and its
        inverse of the individual frequency modes.
        The keys correspond to the science case, e.g. `r01`, `glitch`, or
        `e012`.
        Inside each case, you find the data (`data`), the covariance matrix
        (`cov`), and its inverse (`covinv`).
    obsfreqmeta : dict
        The requested information about which frequency products to fit or
        plot, unpacked for easier access later.
    """

    if "freqs" in fitfreqs["fittypes"] and fitfreqs["correlations"]:
        obskey, obs, obscov = read_freq(
            fitfreqs["freqfile"],
            excludemodes=fitfreqs["excludemodes"],
            onlyradial=fitfreqs["onlyradial"],
            flag_onlyls=fitfreqs["onlyls"],
            covarfre=True,
        )
    else:
        obskey, obs, obscov = read_freq(
            fitfreqs["freqfile"],
            excludemodes=fitfreqs["excludemodes"],
            onlyradial=fitfreqs["onlyradial"],
            flag_onlyls=fitfreqs["onlyls"],
            covarfre=False,
        )

    # Construct data and metadata dictionaries
    obsfreqdata, obsfreqmeta = _make_obsfreqs(
        obskey,
        obs,
        obscov,
        fitfreqs["fittypes"],
        freqplots,
        numax=fitfreqs["numax"],
        debug=debug,
    )

    # Add large frequency separation bias (default is 0)
    if fitfreqs["dnubias"]:
        print(
            f"Added {fitfreqs['dnubias']}muHz bias/systematic to dnu error, from {obsfreqdata['freqs']['dnudata_err']:.3f}",
            end=" ",
        )
        obsfreqdata["freqs"]["dnudata_err"] = np.sqrt(
            obsfreqdata["freqs"]["dnudata_err"] ** 2.0 + fitfreqs["dnubias"] ** 2.0
        )
        print(f"to {obsfreqdata['freqs']['dnudata_err']:.3f}")

    # Compute or dataread in required ratios
    if obsfreqmeta["getratios"]:
        if fitfreqs["readratios"]:
            # Read all requested ratio sequences
            for ratiotype in set(obsfreqmeta["ratios"]["fit"]) | set(
                obsfreqmeta["ratios"]["plot"]
            ):
                datos = _read_precomputed_ratios_xml(
                    fitfreqs["freqfile"],
                    ratiotype,
                    obskey,
                    obs,
                    ratiotype,
                    fitfreqs["excludemodes"],
                    threepoint=fitfreqs["threepoint"],
                    verbose=verbose,
                )
                obsfreqdata[ratiotype] = {}
                obsfreqdata[ratiotype]["data"] = datos[0]
                obsfreqdata[ratiotype]["cov"] = datos[1]
        else:
            for ratiotype in set(obsfreqmeta["ratios"]["fit"]) | set(
                obsfreqmeta["ratios"]["plot"]
            ):
                obsfreqdata[ratiotype] = {}
                datos = freq_fit.compute_ratios(
                    obskey, obs, ratiotype, threepoint=fitfreqs["threepoint"]
                )
                if datos is not None:
                    obsfreqdata[ratiotype]["data"] = datos[0]
                    obsfreqdata[ratiotype]["cov"] = datos[1]
                elif ratiotype in obsfreqmeta["ratios"]["fit"]:
                    # Fail
                    raise ValueError(
                        f"Fitting parameter {ratiotype} could not be computed."
                    )
                else:
                    # Do not fail as much
                    print(f"Ratio {ratiotype} could not be computed.")
                    obsfreqdata[ratiotype]["data"] = None
                    obsfreqdata[ratiotype]["cov"] = None
                    obsfreqdata[ratiotype]["covinv"] = None

    # Get glitches
    if obsfreqmeta["getglitch"]:
        for glitchtype in set(obsfreqmeta["glitch"]["fit"]) | set(
            obsfreqmeta["glitch"]["plot"]
        ):
            obsfreqdata[glitchtype] = {}
            if fitfreqs["readglitchfile"]:
                datos = _read_precomputed_glitches(fitfreqs["glitchfile"], glitchtype)
                # Precomputed from glitchpy lacks the data structure, so sample once to obtain that
                obsseq = glitch_fit.compute_glitchseqs(
                    obskey, obs, glitchtype, obsfreqdata["freqs"]["dnudata"], fitfreqs
                )
                # Store data in new structure, overwrite old
                obsseq[0] = datos[0]
                datos = (obsseq, datos[1])
            else:
                datos = glitch_fit.compute_observed_glitches(
                    obskey,
                    obs,
                    glitchtype,
                    obsfreqdata["freqs"]["dnudata"],
                    fitfreqs,
                    debug=debug,
                )
            if datos is not None:
                obsfreqdata[glitchtype]["data"] = datos[0]
                obsfreqdata[glitchtype]["cov"] = datos[1]
            elif glitchtype in obsfreqmeta["glitches"]["fit"]:
                # Fail
                raise ValueError(
                    f"Fitting parameter {glitchtype} could not be computed."
                )
            else:
                # Do not fail as much
                print(f"Glitch type {glitchtype} could not be computed.")
                obsfreqdata[glitchtype]["data"] = None
                obsfreqdata[glitchtype]["cov"] = None
                obsfreqdata[glitchtype]["covinv"] = None

    # Get epsilon differences
    if obsfreqmeta["getepsdiff"]:
        for epsdifffit in set(obsfreqmeta["epsdiff"]["fit"]) | set(
            obsfreqmeta["epsdiff"]["plot"]
        ):
            obsfreqdata[epsdifffit] = {}
            if epsdifffit in obsfreqmeta["epsdiff"]["fit"]:
                datos = freq_fit.compute_epsilondiff(
                    obskey,
                    obs,
                    obsfreqdata["freqs"]["dnudata"],
                    sequence=epsdifffit,
                    nsorting=fitfreqs["nsorting"],
                    debug=debug,
                )
                obsfreqdata[epsdifffit]["data"] = datos[0]
                obsfreqdata[epsdifffit]["cov"] = datos[1]
            elif epsdifffit in obsfreqmeta["epsdiff"]["plot"]:
                datos = freq_fit.compute_epsilondiff(
                    obskey,
                    obs,
                    obsfreqdata["freqs"]["dnudata"],
                    sequence=epsdifffit,
                    nsorting=fitfreqs["nsorting"],
                    nrealisations=2000,
                    debug=debug,
                )
                obsfreqdata[epsdifffit]["data"] = datos[0]
                obsfreqdata[epsdifffit]["cov"] = datos[1]

    # As the inverse covariance matrix is actually what is used, compute it once
    # Diagonalise covariance matrices if correlations is set to False
    for key in obsfreqdata.keys():
        if not fitfreqs["correlations"]:
            obsfreqdata[key]["cov"] = (
                np.identity(obsfreqdata[key]["cov"].shape[0]) * obsfreqdata[key]["cov"]
            )
        if "covinv" not in obsfreqdata[key].keys():
            obsfreqdata[key]["covinv"] = np.linalg.pinv(
                obsfreqdata[key]["cov"], rcond=1e-8
            )

    return obskey, obs, obsfreqdata, obsfreqmeta


##############################################################
# Routines related to reading ascii-files and convert to xml #
##############################################################
def freqs_ascii_to_xml(
    directory: str,
    starid: str,
    freqsfile: str | None = None,
    covfile: str | None = None,
    ratiosfile: str | None = None,
    cov010file: str | None = None,
    cov02file: str | None = None,
    symmetric_errors: bool = True,
    check_radial_orders: bool = False,
    verbose: bool = True,
):
    """
    Creates frequency xml-file based on ascii-files.

    Parameters
    ----------
    directory : str
        Absolute path of the location of the ascii-files.
        This is also the location where the xml-file will be created.
        The directory must contain the ascii-file with the extension
        '.fre' (containing frequencies), and may contain the
        ascii-files with the extensions '.cov' (frequency covariances),
        '.ratios', '.cov010', and '.cov02' (ratios and their covariances).
    starid : str
        id of the star. The ascii files must be named 'starid.xxx' and
        the generated xml-file will be named 'starid.xml'
    freqsfile : str | None, optional
        File containing individual frequency modes, ends in `.fre`.
    covfile : str | None, optional
        File containing covariances, ends in `.cov`.
    ratiosfile : str | None, optional
        File containing precomputed ratios, end in `.ratios`.
    cov010file : str | None, optional
        File containing precomputed ratio 010 covariances, end in `.cov010`.
    cov02file : str | None, optional
        File containing precomputed ratio 02 covariances, end in `.cov02`.
    symmetric_errors : bool, optional
        If True, the ascii files are assumed to only include symmetric
        errors. Otherwise, asymmetric errors are assumed. Default is
        True.
    check_radial_orders : bool or float, optional
        If True, the routine will correct the radial order printed in the
        xml, based on the calculated epsilon value, with its own dnufit.
        If float, does the same, but uses the inputted float as dnufit.
    quiet : bool, optional
        Toggle to silence the output (useful for running batches)
    """

    # (Potential) filepaths
    if freqsfile is None:
        freqsfile = os.path.join(directory, starid + ".fre")
    if covfile is None:
        covfile = os.path.join(directory, starid + ".cov")
    if ratiosfile is None:
        ratiosfile = os.path.join(directory, starid + ".ratios")
    if cov010file is None:
        cov010file = os.path.join(directory, starid + ".cov010")
    if cov02file is None:
        cov02file = os.path.join(directory, starid + ".cov02")

    # Flags for existence of ratios and covariances
    cov_flag, ratios_flag, cov010_flag, cov02_flag = 1, 1, 1, 1

    #####################################
    # Flags which are redundant for now #
    #####################################
    cov01_flag, cov10_flag, cov012_flag, cov102_flag = 0, 0, 0, 0

    # Make sure that the frequency file exists, and read the frequencies
    if os.path.exists(freqsfile):
        freqs = _read_freq_ascii(freqsfile, symmetric_errors)
        # If covariances are available, read them
        if os.path.exists(covfile):
            cov = _read_freq_cov_ascii(covfile)
        else:
            cov_flag = 0
    else:
        raise RuntimeError("Frequency file not found")

    # Check the value of epsilon, to estimate if the radial orders
    # are correctly identified. Correct them, if user allows to.
    ncorrection = su.check_epsilon_of_freqs(
        freqs,
        starid,
        check_radial_orders,
        quiet=~verbose,
    )
    if check_radial_orders:
        freqs["order"] += ncorrection
        if verbose:
            print("The proposed correction has been implemented.\n")
    else:
        if verbose:
            print("No correction made.\n")

    # Look for ratios and their covariances, read if available
    if os.path.exists(ratiosfile):
        ratios = _read_ratios_ascii(ratiosfile, symmetric_errors)
        if os.path.exists(cov010file):
            cov010 = _read_ratios_cov_ascii(cov010file)
        else:
            cov010_flag = 0
        if os.path.exists(cov02file):
            cov02 = _read_ratios_cov_ascii(cov02file)
        else:
            cov02_flag = 0
    else:
        ratios_flag, cov010_flag, cov02_flag = 0, 0, 0

    # Main xml element
    main = Element("frequencies", {"kic": starid})

    flags = SubElement(
        main,
        "flags",
        {
            "cov": str(cov_flag),
            "ratios": str(ratios_flag),
            "cov010": str(cov010_flag),
            "cov02": str(cov02_flag),
            "cov01": str(cov01_flag),
            "cov10": str(cov10_flag),
            "cov012": str(cov012_flag),
            "cov102": str(cov102_flag),
        },
    )

    # If needed for covariance, store mode id
    if cov_flag:
        mode_ids = np.zeros((len(freqs)), dtype=[("f_id", "int"), ("modeid", "8U")])

    # Frequency elements
    for i in range(len(freqs)):
        modeid = "mode" + str(i + 1)
        order = freqs["order"][i]
        degree = freqs["degree"][i]
        frequency = freqs["frequency"][i]
        error = freqs["error"][i]

        mode_element = SubElement(main, "mode", {"id": modeid})
        SubElement(mode_element, "order", {"value": str(order)})
        SubElement(mode_element, "degree", {"value": str(degree)})
        SubElement(
            mode_element,
            "frequency",
            {"error": str(error), "unit": "uHz", "value": str(frequency)},
        )

        if cov_flag:
            mode_ids["f_id"][i] = freqs["order"][i] * 10 + freqs["degree"][i]
            mode_ids["modeid"][i] = modeid

    # Frequency correlation elements
    if cov_flag:
        # Prepare sorting arrays, order before degree
        fsort = [freqs["order"][i] * 10 + freqs["degree"][i] for i in range(len(freqs))]
        csort = np.array(
            [
                [cov["n1"][i] * 10 + cov["l1"][i] for i in range(len(cov))],
                [cov["n2"][i] * 10 + cov["l2"][i] for i in range(len(cov))],
            ]
        )

        # Double loop over covariance matrix row and columns
        for f1_id in fsort:
            # Get modeid and create subsection of full covariance list
            f1_modeid = mode_ids["modeid"][np.where(mode_ids["f_id"] == f1_id)[0]][0]
            f1_mask = np.where(csort[0] == f1_id)[0]
            covf1 = cov[f1_mask]
            covs1 = csort[1][f1_mask]

            for f2_id in fsort:
                f2_modeid = mode_ids["modeid"][np.where(mode_ids["f_id"] == f2_id)[0]][
                    0
                ]
                f2_index = np.where(covs1 == f2_id)[0][0]

                order1 = covf1["n1"][f2_index]
                order2 = covf1["n2"][f2_index]
                deg1 = covf1["l1"][f2_index]
                deg2 = covf1["l2"][f2_index]
                covariance = covf1["covariance"][f2_index]
                correlation = covf1["correlation"][f2_index]

                freq_corr_element = SubElement(
                    main,
                    "frequency_corr",
                    {
                        "id1": f1_modeid,
                        "id2": f2_modeid,
                        "order1": str(order1),
                        "order2": str(order2),
                        "degree1": str(deg1),
                        "degree2": str(deg2),
                    },
                )
                SubElement(freq_corr_element, "covariance", {"value": str(covariance)})
                SubElement(
                    freq_corr_element, "correlation", {"value": str(correlation)}
                )

    # Ratio elements
    if ratios_flag:
        n02 = 0
        for i in range(len(ratios)):
            ratioid = "ratio" + str(i + 1)
            order = ratios["n"][i]
            rtype = ratios["ratio_type"][i].decode("utf-8")
            ratio = ratios["value"][i]
            error = ratios["error"][i]

            # Count the number of r02 ratios
            if rtype == "r02":
                n02 += 1

            # Note: setting error_minus = error_plus = error
            SubElement(
                main,
                "frequency_ratio",
                {
                    "error": str(error),
                    "error_minus": str(error),
                    "error_plus": str(error),
                    "id": ratioid,
                    "order": str(order),
                    "type": rtype,
                    "value": str(ratio),
                },
            )

    if cov010_flag:
        # Number of r010 ratios
        n010 = len(ratios) - n02
        # Ratio 010 correlation elements
        ratio1_id = n02
        for i in range(len(cov010)):
            id2_index = np.mod(i, n010)
            if id2_index == 0:
                ratio1_id += 1
            ratio2_id = n02 + id2_index + 1

            order1 = cov010["n1"][i]
            order2 = cov010["n2"][i]
            rtype1 = cov010["ratio_type1"][i].decode("utf-8")
            rtype2 = cov010["ratio_type2"][i].decode("utf-8")
            covariance = cov010["covariance"][i]
            correlation = cov010["correlation"][i]

            for rtype in ("01", "10"):
                if rtype in rtype1:
                    rtype1 = "r" + rtype
                if rtype in rtype2:
                    rtype2 = "r" + rtype

            ratio_corr_element = SubElement(
                main,
                "frequency_ratio_corr",
                {
                    "id1": "ratio" + str(ratio1_id),
                    "id2": "ratio" + str(ratio2_id),
                    "order1": str(order1),
                    "order2": str(order2),
                    "type1": rtype1,
                    "type2": rtype2,
                },
            )
            SubElement(ratio_corr_element, "covariance", {"value": str(covariance)})
            SubElement(ratio_corr_element, "correlation", {"value": str(correlation)})

    if cov02_flag:
        # Ratio 02 correlation elements
        ratio1_id = 0
        for i in range(len(cov02)):
            id2_index = np.mod(i, n02)
            if id2_index == 0:
                ratio1_id += 1
            ratio2_id = id2_index + 1

            order1 = cov02["n1"][i]
            order2 = cov02["n2"][i]
            rtype1 = "r02"
            rtype2 = "r02"
            covariance = cov02["covariance"][i]
            correlation = cov02["correlation"][i]

            ratio_corr_element = SubElement(
                main,
                "frequency_ratio_corr",
                {
                    "id1": "ratio" + str(ratio1_id),
                    "id2": "ratio" + str(ratio2_id),
                    "order1": str(order1),
                    "order2": str(order2),
                    "type1": rtype1,
                    "type2": rtype2,
                },
            )
            SubElement(ratio_corr_element, "covariance", {"value": str(covariance)})
            SubElement(ratio_corr_element, "correlation", {"value": str(correlation)})

    # Create xml string and make it pretty
    xml = tostring(main)
    reparsed = minidom.parseString(xml)
    pretty_xml = reparsed.toprettyxml()

    # Write output to file starid.xml
    with open(os.path.join(directory, starid + ".xml"), "w") as xmlfile:
        print(pretty_xml, file=xmlfile)


def _read_freq_ascii(
    filename: str,
    symmetric_errors: bool = True,
) -> np.array:
    """
    Read individual frequencies from an ascii file

    Parameters
    ----------
    filename : str
        Name of file to read
    symmetric_errors : bool, optional
        If True, assumes the ascii file to contain only one column with
        frequency errors. Otherwise, the file must include asymmetric
        errors (error_plus and error_minus). Default is True.

    Returns
    -------
    freqs : array
        Contains the radial order, angular degree, frequency, uncertainty,
        and flag
    """
    cols = []
    data = np.genfromtxt(filename, dtype=None, encoding=None, names=True)

    rdict = {
        "frequency": {
            "dtype": "float",
            "recognisednames": [
                "f",
                "freq",
                "freqs",
                "frequency",
                "modefrequency",
            ],
        },
        "order": {
            "dtype": "int",
            "recognisednames": [
                "n",
                "ns",
                "order",
                "radialorder",
            ],
        },
        "degree": {
            "dtype": "int",
            "recognisednames": [
                "l",
                "ls",
                "ell",
                "ells",
                "degree",
                "angulardegree",
            ],
        },
        "error": {
            "dtype": "float",
            "recognisednames": [
                "error",
                "err",
                "e",
                "uncertainty",
                "uncert",
            ],
        },
        "error_plus": {
            "dtype": "float",
            "recognisednames": [
                "error_plus",
                "err_plus",
                "error_upper",
                "upper_error",
            ],
        },
        "error_minus": {
            "dtype": "float",
            "recognisednames": [
                "error_minus",
                "err_minus",
                "error_lower",
                "lower_error",
            ],
        },
        "flag": {
            "dtype": "int",
            "recognisednames": [
                "flag",
            ],
        },
        "frequency_mode": {
            "dtype": "float",
            "recognisednames": [
                "frequency_mode",
            ],
        },
    }

    for colname in data.dtype.names:
        param = [p for p in rdict if colname.lower() in rdict[p]["recognisednames"]]
        if not param:
            raise ValueError(f"BASTA cannot recognize {colname}")
        assert len(param) == 1, (colname, param)
        cols.append((param[0], rdict[param[0]]["dtype"]))

    sym = np.any([col[0] == "error" for col in cols])
    asym = all(
        [
            item in col[0]
            for col in cols
            for item in rdict["error_plus"]["recognisednames"]
        ]
    )
    if not sym ^ asym:
        if sym | asym:
            raise ValueError(
                "BASTA found too many uncertainties, please specify which to use."
            )
        else:
            raise ValueError("BASTA is missing frequency uncertainties.")
    if not symmetric_errors and not asym:
        raise ValueError(
            "BASTA is looking for asymmetric frequency uncertainties, but did not find them"
        )

    freqs = np.genfromtxt(
        filename,
        dtype=cols,
        encoding=None,
    )

    if not np.any(["order" in col[0] for col in cols]):
        print("BASTA did not find column with radial orders, they will be generated")

        freqcol = [
            col
            for col in data.dtype.names
            if col in rdict["frequency"]["recognisednames"]
        ]
        lcol = [
            col for col in data.dtype.names if col in rdict["degree"]["recognisednames"]
        ]
        assert len(freqcol) == 1, freqcol
        assert len(lcol) == 1, lcol
        mask = data[lcol[0]] == 0
        l0s = data[freqcol[0]][mask]
        assert len(l0s) > 1, "More than 1 l=0 mode is required for estimation of dnu"
        dnu = np.median(np.diff(l0s))
        ns = np.asarray(data[freqcol[0]]) // dnu - 1
        from numpy.lib.recfunctions import append_fields

        freqs = append_fields(freqs, "order", data=ns.astype(int))

    return freqs


def _read_freq_cov_ascii(filename: str) -> np.array:
    """
    Read covariance and correlations for individual frequencies from an
    ascii file

    Parameters
    ----------
    filename : str
        Name of file to read

    Returns
    -------
    cov : array
        Contains the angular degree, radial order, covariances and
        correlations between the individual frequencies
    """
    cov = np.genfromtxt(
        filename,
        dtype=[
            ("l1", "int"),
            ("n1", "int"),
            ("l2", "int"),
            ("n2", "int"),
            ("covariance", "float"),
            ("correlation", "float"),
        ],
        encoding=None,
    )
    return cov


def _read_ratios_ascii(filename: str, symmetric_errors: bool = True) -> np.array:
    """
    Read frequency ratios from an ascii file

    Parameters
    ----------
    filename : str
        Name of file to read
    symmetric_errors : bool, optional
        If True, assumes the ascii file to contain only one column with
        frequency errors. Otherwise, the file must include asymmetric
        errors (error_plus and error_minus). Default is True.

    Returns
    -------
    ratios : array
        Contains the frequency ratios type, radial order, value,
        uncertainty added in quadrature, upper uncertainty, and lower
        uncertainty
    """
    if symmetric_errors:
        ratios = np.genfromtxt(
            filename,
            dtype=[
                ("ratio_type", "S3"),
                ("n", "int"),
                ("value", "float"),
                ("error", "float"),
            ],
            encoding=None,
        )
    else:
        ratios = np.genfromtxt(
            filename,
            dtype=[
                ("ratio_type", "S3"),
                ("n", "int"),
                ("value", "float"),
                ("error", "float"),
                ("error_plus", "float"),
                ("error_minus", "float"),
            ],
            encoding=None,
        )
    return ratios


def _read_ratios_cov_ascii(filename: str) -> np.array:
    """
    Read covariance and correlations for frequency ratios from an
    ascii file

    Parameters
    ----------
    filename : str
        Name of file to read

    Returns
    -------
    cov : array
        Contains the ratio type, radial order, covariances and
        correlations between the frequency ratios
    """
    cov = np.genfromtxt(
        filename,
        dtype=[
            ("ratio_type1", "S4"),
            ("n1", "int"),
            ("ratio_type2", "S4"),
            ("n2", "int"),
            ("covariance", "float"),
            ("correlation", "float"),
        ],
        encoding=None,
    )
    return cov
