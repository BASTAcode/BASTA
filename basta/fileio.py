"""
Auxiliary functions for file operations
"""
import os
import json
import warnings
from io import IOBase
from copy import deepcopy
from xml.etree import ElementTree
from xml.etree.ElementTree import Element, SubElement, tostring
from xml.dom import minidom

import numpy as np
import h5py

from basta import stats, freq_fit
from basta import utils_seismic as su
from basta import utils_general as util
from basta.constants import freqtypes
import basta.supportGlitch as sg


def _export_selectedmodels(selectedmodels):
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


def _import_selectedmodels(data):
    res = {}
    for trackno, ts in data.items():
        index = np.zeros(ts["n"], dtype=np.bool)
        index[ts["index"]] = True
        res[trackno] = stats.Trackstats(
            chi2=np.asarray(ts["chi2"]), index=index, logPDF=np.asarray(ts["logPDF"])
        )

    return res


def save_selectedmodels(fname, selectedmodels):
    s = json.dumps(_export_selectedmodels(selectedmodels))
    with open(fname, "w") as fp:
        fp.write(s)


def load_selectedmodels(fname):
    with open(fname) as fp:
        data = json.load(fp)

    return _import_selectedmodels(data)


def write_star_to_errfile(starid, inputparams, errormessage):
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


def no_models(starid, inputparams, errormessage):
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


def read_ratios_xml(filename):
    """
    Read frequency ratios from an xml file

    Parameters
    ----------
    filename : str
        Name of file to read

    Returns
    -------
    orders : array
        Radial order of the ratios
    ratios : array
        Value of the ratios
    ratio_types : array
        Type of ratio
    errors_m : array
        Lower uncertainty in the ratio
    errors_p : array
        Upper uncertaity in the ratio
    """

    # Parse the XML file:
    tree = ElementTree.parse(filename)
    root = tree.getroot()

    # Find a list of all the frequency ratios:
    freq_ratios = root.findall("frequency_ratio")

    # Simply get the orders and frequency ratios as numpy arrays:
    orders = np.array([ratio.get("order") for ratio in freq_ratios], dtype=int)
    ratios = np.array([ratio.get("value") for ratio in freq_ratios], dtype="float64")
    ratio_types = np.array([ratio.get("type") for ratio in freq_ratios], dtype="|S3")
    errors = np.array([ratio.get("error") for ratio in freq_ratios], dtype="float64")
    errors_m = np.array(
        [ratio.get("error_minus") for ratio in freq_ratios], dtype="float64"
    )
    errors_p = np.array(
        [ratio.get("error_plus") for ratio in freq_ratios], dtype="float64"
    )

    return orders, ratios, ratio_types, errors, errors_m, errors_p


def read_freqs_xml(filename):
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


def read_corr_xml(filename):
    """
    Read correlation matrix for frequency ratios from an xml file

    Parameters
    ----------
    filename : str
        Name of file to read

    Returns
    -------
    corr : array
        Contains the correlations between the frequency ratios
    """

    # Parse the XML file:
    tree = ElementTree.parse(filename)
    root = tree.getroot()

    # Find a list of all the frequency ratios:
    freq_ratios = root.findall("frequency_ratio")

    # Loop over the frequency ratios to load them one-by-one and construct the
    # correlation and covariance matrices:
    corr = np.identity(len(freq_ratios))
    cov = np.zeros((len(freq_ratios), len(freq_ratios)))
    for k, ratio in enumerate(freq_ratios):
        # Loop over the ratios again to construct correlation matrix:
        # The reason for the slightly un-elegant loops is a limitation in the
        # ElementTree XML parser to only allow to search for one attribute at a
        # time
        # Search for all "frequency_ratio_corr" elements with id1-attribute
        # equal
        elmlist = root.findall("frequency_ratio_corr[@id1='%s']" % ratio.get("id"))
        for j, ratio2 in enumerate(freq_ratios):
            id2 = ratio2.get("id")
            for elm in elmlist:
                if elm.get("id2") == id2:
                    corr[k, j] = corr[j, k] = elm.find("correlation").get("value")
                    cov[k, j] = cov[j, k] = elm.find("covariance").get("value")
                    break

    # Return the correlation and covariance matrices:
    return corr


def read_cov_xml(filename, allparams=False):
    """
    Read covariance matrix for frequency ratios from an xml file

    Parameters
    ----------
    filename : str
        Name of file to read

    Returns
    -------
    cov : array
        Contains the covariances between the frequency ratios
    """

    # Parse the XML file:
    tree = ElementTree.parse(filename)
    root = tree.getroot()

    # Find a list of all the frequency ratios:
    freq_ratios = root.findall("frequency_ratio")

    # Loop over the frequency ratios to load them one-by-one and construct the
    # correlation and covariance matrices:
    corr = np.identity(len(freq_ratios))
    cov = np.zeros((len(freq_ratios), len(freq_ratios)))
    if allparams:
        order1 = np.zeros(len(freq_ratios) ** 2, dtype="int")
        order2 = np.zeros(len(freq_ratios) ** 2, dtype="int")
        type1 = np.zeros(len(freq_ratios) ** 2, dtype="|S3")
        type2 = np.zeros(len(freq_ratios) ** 2, dtype="|S3")
        corrvec = np.zeros(len(freq_ratios) ** 2, dtype="float64")
        covvec = np.zeros(len(freq_ratios) ** 2, dtype="float64")

    p = 0
    for k, ratio in enumerate(freq_ratios):
        # Loop over the ratios again to construct correlation matrix:
        # The reason for the slightly un-elegant loops is a limitation in the
        # ElementTree XML parser to only allow to search for one attribute at a
        # time
        # Search for all "frequency_ratio_corr" elements with id1-attribute
        # equal
        elmlist = root.findall("frequency_ratio_corr[@id1='%s']" % ratio.get("id"))
        for j, ratio2 in enumerate(freq_ratios):
            id2 = ratio2.get("id")
            for elm in elmlist:
                if elm.get("id2") == id2:
                    corr[k, j] = corr[j, k] = elm.find("correlation").get("value")
                    cov[k, j] = cov[j, k] = elm.find("covariance").get("value")
                    if allparams:
                        order1[p] = elm.get("order1")
                        order2[p] = elm.get("order2")
                        type1[p] = elm.get("type1")
                        type2[p] = elm.get("type2")
                        corrvec[p] = corr[k, j]
                        covvec[p] = cov[k, j]
                        p += 1
                    break

    # Return the correlation and covariance matrices:
    if allparams:
        lid = len(type1[type1 != b""])
        return (
            cov,
            corr,
            covvec[:lid],
            corrvec[:lid],
            order1[:lid],
            order2[:lid],
            type1[:lid],
            type2[:lid],
        )
    else:
        return cov


def read_freq_cov_xml(filename, obskey):
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


def read_ratios(filename, symmetric_errors=True):
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


def read_fre(filename, symmetric_errors=True, nbeforel=True):
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
    nbeforel : bool, optional
        If True (default), the column containing the orders n is [0], and
        the column containing the degrees l is [1].
        If False, it is the other way around.

    Returns
    -------
    freqs : array
        Contains the radial order, angular degree, frequency, uncertainty,
        and flag
    """
    if nbeforel:
        if symmetric_errors:
            freqs = np.genfromtxt(
                filename,
                dtype=[
                    ("order", "int"),
                    ("degree", "int"),
                    ("frequency", "float"),
                    ("error", "float"),
                    ("flag", "int"),
                ],
                encoding=None,
            )
        else:
            freqs = np.genfromtxt(
                filename,
                dtype=[
                    ("order", "int"),
                    ("degree", "int"),
                    ("frequency", "float"),
                    ("error", "float"),
                    ("error_plus", "float"),
                    ("error_minus", "float"),
                    ("frequency_mode", "float"),
                    ("flag", "int"),
                ],
                encoding=None,
            )
    else:
        if symmetric_errors:
            freqs = np.genfromtxt(
                filename,
                dtype=[
                    ("degree", "int"),
                    ("order", "int"),
                    ("frequency", "float"),
                    ("error", "float"),
                    ("flag", "int"),
                ],
                encoding=None,
            )
        else:
            freqs = np.genfromtxt(
                filename,
                dtype=[
                    ("degree", "int"),
                    ("order", "int"),
                    ("frequency", "float"),
                    ("error", "float"),
                    ("error_plus", "float"),
                    ("error_minus", "float"),
                    ("frequency_mode", "float"),
                    ("flag", "int"),
                ],
                encoding=None,
            )
    return freqs


def read_cov_freqs(filename):
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


def read_cov_ratios(filename):
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


def read_freq(filename, nottrustedfile=None, covarfre=False):
    """
    Routine to extract the frequencies in the desired n-range, and the
    corresponding covariance matrix

    Parameters
    ----------
    filename : str
        Name of file to read
    nottrustedfile : str or None, optional
        Name of file containing the (l, n) values of frequencies to be
        omitted in the fit. If None, no modes will be excluded.
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
    frecu, errors, norder, ldegree = read_freqs_xml(filename)

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
    obskey = np.asarray([l, n], dtype=np.int)
    obs = np.array([f, e])

    # Remove untrusted frequencies
    if nottrustedfile not in [None, "", "None", "none", "False", "false"]:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            nottrustedobskey = np.genfromtxt(
                nottrustedfile, comments="#", encoding=None
            )
        # If there is only one not-trusted mode, it misses a dimension
        if nottrustedobskey.size == 0:
            print("File for not-trusted frequencies was empty")
        else:
            print("Not-trusted frequencies found")
            if nottrustedobskey.shape == (2,):
                nottrustedobskey = [nottrustedobskey]
            for (l, n) in nottrustedobskey:
                nottrustedmask = (obskey[0] == l) & (obskey[1] == n)
                print("Removed mode at", obs[:, nottrustedmask][0][0], "µHz")
                obskey = obskey[:, ~nottrustedmask]
                obs = obs[:, ~nottrustedmask]

    if covarfre:
        corrfre, covarfreq = read_freq_cov_xml(filename, obskey)
        """
        # Read covariances of individual frequency pairs from .cov file
        # These covariances are not (yet) implemented in the xml file
        covfile = os.path.splitext(filename)[0] + ".cov"
        if os.path.isfile(covfile):
            cov = read_cov_freqs(covfile)
            covarfre = np.zeros((len(obs[0, :]), len(obs[0, :])))

            nldata = list(zip(obskey[0], obskey[1]))
            for i, (l1, n1) in enumerate(nldata):
                for j, (n2, l2) in enumerate(nldata[i:], start=i):
                    cov_index = np.where(
                        (cov["n1"] == n1)
                        & (cov["l1"] == l1)
                        & (cov["n2"] == n2)
                        & (cov["l2"] == l2)
                    )[0][0]

                    covarfre[i, j] = covarfre[j, i] = cov["covariance"][cov_index]
        else:
            print(covfile + " not found!")
            return
        """
    else:
        covarfreq = np.diag(obs[1, :]) ** 2

    return obskey, obs, covarfreq


def read_r010(filename, rrange, nottrustedfile, verbose=False):
    """
    Routine to extract the frequency ratios r01 and r10 in the desired
    n-range, and the corresponding covariance matrix

    Parameters
    ----------
    filename : str
        Name of file to read.
    rrange : list
        Radial range in ratios to be used for different angular degree.
    nottrustedfile : str or None
        Name of file containing the (l, n) values of frequencies to be
        omitted in the fit. If None, no modes will be excluded.
    verbose : bool, optional
        If True, extra text will be printed (for developers).

    Returns
    -------
    r010var : array
        Value of the frequency ratios, the associated central frequency,
        and the uncertainty in the ratios
    covar010 : array
        Covariances between frequency ratios
    """
    # Read ratios from xml file
    orders, ratios, ratio_types, errors, errors_m, errors_p = read_ratios_xml(filename)
    if (b"r01" not in ratio_types) or (b"r10" not in ratio_types):
        r010var, covar010 = None, None
        return r010var, covar010

    # Compute frequency ratios array
    if verbose:
        print("Computing frequency ratios array...")

    obskey, obs, _ = read_freq(filename, nottrustedfile=nottrustedfile)
    obskey_l0, obs_l0 = su.get_givenl(l=0, osc=obs, osckey=obskey)
    obskey_l1, obs_l1 = su.get_givenl(l=1, osc=obs, osckey=obskey)

    # Unpack rrange
    nirdl0, nfrdl0, nirdl1, nfrdl1 = rrange[:4]

    ind = (ratio_types == b"r01") | (ratio_types == b"r10")
    nr010 = orders[ind]
    r010dat = np.zeros((2, len(nr010)))
    r010dat[0, :] = ratios[ind]
    r010dat[1, :] = errors[ind]

    i, j, k = 0, 0, 0
    if verbose:
        print("Building r010var...")
    # start together, finish together
    nl0mask = (obskey_l0[1, :] >= nirdl0) & (obskey_l0[1, :] <= nfrdl0)
    nl1mask = (obskey_l1[1, :] >= nirdl1) & (obskey_l1[1, :] <= nfrdl1)
    obs_l0n = obs_l0[0, nl0mask]
    obs_l1n = obs_l1[0, nl1mask]
    if nirdl0 == nirdl1 and nfrdl0 == nfrdl1:
        if verbose:
            print("start together, finish together")
        id1 = np.where(nr010 == nirdl0)[0][0]
        id2 = 1 + np.where(nr010 == nfrdl0)[0][1]
        r010var = np.zeros((3, len(r010dat[0, id1:id2])))
        r010var[0, :] = r010dat[0, id1:id2]
        r010var[2, :] = r010dat[1, id1:id2]
        while j <= len(r010var[0, :]) - 2:
            r010var[1, j] = obs_l0n[i]
            r010var[1, j + 1] = obs_l1n[i]
            j += 2
            i += 1

    # start together, finish in l=0
    elif nirdl0 == nirdl1 and nfrdl0 > nfrdl1:
        if verbose:
            print("Start together, finish in l=0")
        id1 = np.where(nr010 == nirdl0)[0][0]
        id2 = np.where(nr010 == nfrdl0)[0][0] + 1
        r010var = np.zeros((3, len(r010dat[0, id1:id2])))
        r010var[0, :] = r010dat[0, id1:id2]
        r010var[2, :] = r010dat[1, id1:id2]
        nlo = nirdl0
        ndif = nfrdl1 - nlo
        while j <= len(r010var[0, :]) - 1:
            if i <= ndif:  # fill both
                r010var[1, j] = obs_l0n[i]
                r010var[1, j + 1] = obs_l1n[i]
                j += 2
                i += 1
            else:  # fill only l=0
                r010var[1, j] = obs_l0n[i]
                j += 2
                i += 1
    # start l=1, finish together
    elif nirdl0 > nirdl1 and nfrdl0 == nfrdl1:
        if verbose:
            print("Start l=1, finish together")
        # l=0 exist in the file, take the second nirdl1
        if any((orders == nirdl1) & (ratio_types == b"r01")):
            if verbose:
                print("l=0 exist in the file, take the second nirdl1")
            id1 = np.where(nr010 == nirdl1)[0][1]
            id2 = 1 + np.where(nr010 == nfrdl0)[0][1]
        else:
            id1 = np.where(nr010 == nirdl1)[0][0]
            id2 = 1 + np.where(nr010 == nfrdl0)[0][1]
        r010var = np.zeros((3, len(r010dat[0, id1:id2])))
        r010var[0, :] = r010dat[0, id1:id2]
        r010var[2, :] = r010dat[1, id1:id2]

        ndif = nirdl0 - nirdl1
        while j <= len(r010var[0, :]) - 2:
            if k < ndif:  # fill only l=1
                if verbose:
                    print("fill only l=1")
                r010var[1, j] = obs_l1n[k]
                j += 1
                k += 1
            else:  # fill l=0 and 1
                if verbose:
                    print("fill l=0 and 1")
                r010var[1, j] = obs_l0n[i]
                r010var[1, j + 1] = obs_l1n[k]
                j += 2
                i += 1
                k += 1
    # start l=1, finish l=0
    elif nirdl0 > nirdl1 and nfrdl0 > nfrdl1:
        if verbose:
            print("Start l=1, finish l=0")
        # l=0 exist in the file, take the second nirdl1
        if any((orders == nirdl1) & (ratio_types == b"r01")):
            if verbose:
                print("l=0 exist in the file, take the second nirdl1")
            id1 = np.where(nr010 == nirdl1)[0][1]
            id2 = 1 + np.where(nr010 == nfrdl0)[0][0]
        else:
            id1 = np.where(nr010 == nirdl1)[0][0]
            id2 = 1 + np.where(nr010 == nfrdl0)[0][0]
        r010var = np.zeros((3, len(r010dat[0, id1:id2])))
        r010var[0, :] = r010dat[0, id1:id2]
        r010var[2, :] = r010dat[1, id1:id2]
        ndif = nirdl0 - nirdl1
        while j <= len(r010var[0, :]) - 1:
            if k < ndif:  # fill only l=1
                if verbose:
                    print("fill only l=1")
                r010var[1, j + 1] = obs_l1n[k]
                j += 1
                k += 1
            elif nirdl0 + i <= nfrdl1:  # fill l=0 and 1
                if verbose:
                    print("fill l=0 and 1")
                r010var[1, j] = obs_l0n[i]
                r010var[1, j + 1] = obs_l1n[k]
                j += 2
                i += 1
                k += 1
            else:
                r010var[1, j] = obs_l0n[i]
                j += 1
                i += 1
    else:
        print("PROBLEM WITH RATIOS CASE!")
        return

    # Build covar of ratio types 01/10
    cov, corr, CovG, corG, n1, n2, rty1, rty2 = read_cov_xml(
        filename, allparams=True
    )  # loading as expected

    r01ind = (rty1 == b"r01") | (rty1 == b"r10")
    CovG = CovG[r01ind]
    corG = corG[r01ind]
    n1 = n1[r01ind]
    n2 = n2[r01ind]
    rty1 = rty1[r01ind]
    rty2 = rty2[r01ind]
    lr010 = len(r010var[0, :])
    covar010 = np.zeros((lr010, lr010))

    if verbose:
        print("Building Covext...")
    # start together, finish together
    if nirdl0 == nirdl1 and nfrdl0 == nfrdl1:
        if verbose:
            print("Start together, finish together")
        id1 = np.where((n1 == nirdl0) & (n2 == nirdl0))[0][0]
        Covext = CovG[id1 : id1 + lr010]
        id1 = np.where((n1 == nirdl0) & (n2 == nirdl0))[0][2]
        Covext = np.append(Covext, CovG[id1 : id1 + lr010])
        for icov in range(nirdl0 + 1, nfrdl0 + 1):
            id1 = np.where((n1 == icov) & (n2 == nirdl0))[0][0]
            Covext = np.append(Covext, CovG[id1 : id1 + lr010])
            id1 = np.where((n1 == icov) & (n2 == nirdl0))[0][2]
            Covext = np.append(Covext, CovG[id1 : id1 + lr010])
    # start together, finish in l=0
    elif nirdl0 == nirdl1 and nfrdl0 > nfrdl1:
        if verbose:
            print("Start together, finish in l=0")
        id1 = np.where((n1 == nirdl0) & (n2 == nirdl0))[0][0]
        Covext = CovG[id1 : id1 + lr010]
        id1 = np.where((n1 == nirdl0) & (n2 == nirdl0))[0][2]
        Covext = np.append(Covext, CovG[id1 : id1 + lr010])
        for icov in range(nirdl0 + 1, nfrdl0):
            id1 = np.where((n1 == icov) & (n2 == nirdl0))[0][0]
            Covext = np.append(Covext, CovG[id1 : id1 + lr010])
            id1 = np.where((n1 == icov) & (n2 == nirdl0))[0][2]
            Covext = np.append(Covext, CovG[id1 : id1 + lr010])
        id1 = np.where((n1 == nfrdl0) & (n2 == nirdl0))[0][0]
        Covext = np.append(Covext, CovG[id1 : id1 + lr010])
    # start l=1, finish together
    elif nirdl0 > nirdl1 and nfrdl0 == nfrdl1:
        if verbose:
            print("Start l=1, finish together")
        # l=0 exist in the file, take the second pair of nirDl1
        if any((n1 == nirdl1) & (rty1 == b"r01")):
            if verbose:
                print("l=0 exist in the file, take the second pair of nirDl1")
            id1 = np.where((n1 == nirdl1) & (n2 == nirdl1))[0][3]
            Covext = CovG[id1 : id1 + lr010]
            for icov in range(nirdl1 + 1, nfrdl0 + 1):
                id1 = np.where((n1 == icov) & (n2 == nirdl1))[0][1]
                Covext = np.append(Covext, CovG[id1 : id1 + lr010])
                id1 = np.where((n1 == icov) & (n2 == nirdl1))[0][3]
                Covext = np.append(Covext, CovG[id1 : id1 + lr010])
        else:  # Only the R_01 present
            if verbose:
                print("Only R_01 present")
            id1 = np.where((n1 == nirdl1) & (n2 == nirdl1))[0][0]
            Covext = CovG[id1 : id1 + lr010]
            for icov in range(nirdl1 + 1, nfrdl0 + 1):
                id1 = np.where((n1 == icov) & (n2 == nirdl1))[0][0]
                Covext = np.append(Covext, CovG[id1 : id1 + lr010])
                id1 = np.where((n1 == icov) & (n2 == nirdl1))[0][1]
                Covext = np.append(Covext, CovG[id1 : id1 + lr010])
    # start l=1, finish l=0
    elif nirdl0 > nirdl1 and nfrdl0 > nfrdl1:
        if verbose:
            print("Start l=1, finish l=0")
        # l=0 exist in the file, take the second pair of nirDl1
        if any((n1 == nirdl1) & (rty1 == b"r01")):
            if verbose:
                print("l=0 exist in the file, take the second pair of nirDl1")
            id1 = np.where((n1 == nirdl1) & (n2 == nirdl1))[0][3]
            Covext = CovG[id1 : id1 + lr010]
            for icov in range(nirdl1 + 1, nfrdl0):
                id1 = np.where((n1 == icov) & (n2 == nirdl1))[0][1]
                Covext = np.append(Covext, CovG[id1 : id1 + lr010])
                id1 = np.where((n1 == icov) & (n2 == nirdl1))[0][3]
                Covext = np.append(Covext, CovG[id1 : id1 + lr010])
            id1 = np.where((n1 == nfrdl0) & (n2 == nirdl1))[0][1]
            Covext = np.append(Covext, CovG[id1 : id1 + lr010])
        else:  # only the R_01 present
            if verbose:
                print("Only the R_01 present")
            id1 = np.where((n1 == nirdl1) & (n2 == nirdl1))[0][0]
            Covext = CovG[id1 : id1 + lr010]
            for icov in range(nirdl1 + 1, nfrdl0):
                id1 = np.where((n1 == icov) & (n2 == nirdl1))[0][0]
                Covext = np.append(Covext, CovG[id1 : id1 + lr010])
                id1 = np.where((n1 == icov) & (n2 == nirdl1))[0][1]
                Covext = np.append(Covext, CovG[id1 : id1 + lr010])
            id1 = np.where((n1 == nfrdl0) & (n2 == nirdl1))[0][0]
            Covext = np.append(Covext, CovG[id1 : id1 + lr010])
    else:
        print("Problem building Covext!")
        pass

    cont = 0
    for in1 in range(len(covar010[0, :])):
        for in2 in range(len(covar010[:, 0])):
            covar010[in1, in2] = Covext[cont]
            cont += 1
    return r010var, covar010


def read_r02(filename, rrange, nottrustedfile):
    """
    Routine to extract the frequency ratios r02 in the desired n-range,
    and the corresponding covariance matrix

    Parameters
    ----------
    filename : str
        Name of file to read
    rrange : list
        Radial range in ratios to be used for different angular degree
    nottrustedfile : str or None
        Name of file containing the (l, n) values of frequencies to be
        omitted in the fit. If None, no modes will be excluded.

    Returns
    -------
    r02var : array
        Value of the frequency ratios, the associated central frequency,
        and the uncertainty in the ratios
    covar02 : array
        Covariances between frequency ratios
    """
    orders, ratios, ratio_types, errors = read_ratios_xml(filename)[:4]
    if b"r02" not in ratio_types:
        r02var, covar02 = None, None
        return r02var, covar02

    # Unpack rrange
    nirdl2, nfrdl2 = rrange[4:]

    # Compute frequency ratios array
    obskey, obs, _ = read_freq(filename, nottrustedfile=nottrustedfile)
    obskey_l0, obs_l0 = su.get_givenl(l=0, osc=obs, osckey=obskey)

    r02ind = ratio_types == b"r02"
    nr02 = orders[r02ind]
    r02dat = np.zeros((2, len(nr02)))
    r02dat[0, :] = ratios[r02ind]
    r02dat[1, :] = errors[r02ind]

    # Extract them by n-range?
    id1 = np.where(nr02 == nirdl2)[0][0]
    id2 = np.where(nr02 == nfrdl2)[0][0] + 1
    r02var = np.zeros((3, len(r02dat[0, id1:id2])))
    r02var[0, :] = r02dat[0, id1:id2]
    r02var[2, :] = r02dat[1, id1:id2]
    nmask = (obskey_l0[1, :] >= nirdl2) & (obskey_l0[1, :] <= nfrdl2)
    r02var[1, :] = obs_l0[0, nmask]

    # Read covariances
    cov, corr, CovG, corG, n1, n2, rty1 = read_cov_xml(filename, allparams=True)[:7]
    r02ind = rty1 == b"r02"
    CovG = CovG[r02ind]
    corG = corG[r02ind]
    n1 = n1[r02ind]
    n2 = n2[r02ind]
    rty1 = rty1[r02ind]
    lr02 = len(r02var[0, :])
    covar02 = np.zeros((lr02, lr02))

    # Extract the matrix if necessary
    id1 = np.where((n1 == nirdl2) & (n2 == nirdl2))[0][0]
    Covext02 = CovG[id1 : id1 + lr02]
    for icov in np.arange(nirdl2 + 1, nfrdl2 + 1):
        id1 = np.where((n1 == icov) & (n2 == nirdl2))[0][0]
        Covext02 = np.append(Covext02, CovG[id1 : id1 + lr02])

    cont = 0
    for in1 in range(len(covar02[0, :])):
        for in2 in range(len(covar02[:, 0])):
            covar02[in1, in2] = Covext02[cont]
            cont += 1

    return r02var, covar02


def read_glh(filename, grtype="glitches"):
    """
    Read glitch parameters.

    Parameters
    ----------
    filename : str
        Name of file to read
    grtype : str
        Glitch-ratio combination (one of glitches, gr02, gr01,
        gr10, gr010, gr012, gr102)

    Returns
    -------
    glhrto : array
        Array of median glitch parameters
    covgr : array
        Covariance matrix
    """
    glhrto, covgr = None, None
    if filename is not None:
        if os.path.isfile(filename):
            with h5py.File(filename, "r") as data:
                if grtype == "glitches":
                    try:
                        glhrto = data["medglh"][()]
                        covgr = data["covglh"][()]
                    except KeyError:
                        glhrto, covgr = None, None

                elif grtype == "gr02":
                    try:
                        glhrto = data["medg02"][()]
                        covgr = data["covg02"][()]
                    except KeyError:
                        glhrto, covgr = None, None

                elif grtype == "gr01":
                    try:
                        glhrto = data["medg01"][()]
                        covgr = data["covg01"][()]
                    except KeyError:
                        glhrto, covgr = None, None

                elif grtype == "gr10":
                    try:
                        glhrto = data["medg10"][()]
                        covgr = data["covg10"][()]
                    except KeyError:
                        glhrto, covgr = None, None

                elif grtype == "gr010":
                    try:
                        glhrto = data["medg010"][()]
                        covgr = data["covg010"][()]
                    except KeyError:
                        glhrto, covgr = None, None

                elif grtype == "gr012":
                    try:
                        glhrto = data["medg012"][()]
                        covgr = data["covg012"][()]
                    except KeyError:
                        glhrto, covgr = None, None

                elif grtype == "gr102":
                    try:
                        glhrto = data["medg102"][()]
                        covgr = data["covg102"][()]
                    except KeyError:
                        glhrto, covgr = None, None

                else:
                    raise ValueError("Unrecognized glitch-ratio type!")

    return glhrto, covgr


def read_rt(
    inputparams,
    filename,
    glhhdf,
    rt,
    numax,
    plotratios,
    getfreqcovar=False,
    nottrustedfile=None,
    verbose=False,
):
    """
    Routine to all necesary data from individual frequencies for the
    desired fit

    Parameters
    ----------
    inputparams : dict
        Dictionary of all controls and input.
    filename : string
        Name of file to read
    glhhdf : str
        Name of file containing glitch parameters and covariances.
    rt : list
        Type of fits available for individual frequencies
    numax : scalar
        Frequency of maximum power
    plotratios : bool
        Whether or not computation of ratios for plots is needed
    getfreqcovar : bool
        Whether to try to read frequency covariances from the input xml
    nottrustedfile : str or None.
        Name of file containing the (l, n) values of frequencies to be
        omitted in the fit. If None, no modes will be excluded.
    verbose : bool, optional
        If True, extra text will be printed to log (for developers).

    Returns
    -------
    datos : array
        Individual frequencies, uncertainties, and combinations read
        directly from the observational input files
    cov : array
        Covariances between individual frequencies and frequency ratios
        read directly from the observational input files
    obs : array
        Individual frequencies, uncertainties, and combinations computed
        from these frequencies
    dnudata : scalar
        Large frequency separation obtained by fitting the radial mode observed
        frequencies. Similar to dnufit, but from data and not from the
        theoretical frequencies in the grid of models
    dnudata_err : scalar
        Uncertainty on dnudata
    frq_sd : array
        Second differences (l, n, v(muHz), err(muHz), dif2(muHz), err(muHz))
    icov_sd : array
        Inverse covariance matrix for second differences
    vmin : float
        Minimum value of the observed frequency (muHz)
    vmax : float
        Maximum value of the observed frequency (muHz)
    num_of_n : array of int
        Number of modes for each l
    """
    # Observed frequencies
    obskey, obs, covf = read_freq(filename, nottrustedfile, covarfre=getfreqcovar)
    datos_f, cov_f = obs, covf

    # Ratios and glitches (observables and corresponding covariances)
    # ---------------------------------------------------------------
    # --> Fit ratios only, choose from {"r010", "r02", "r01", "r10", "r012", "r102"};
    # fit glitches only, choose "glitches";
    # fit both, choose from {"gr010", "gr02", "gr01", "gr10", "gr012", "gr102"}.

    # Define 'frq' to be used in ratio- and glitch-related calculations
    frq = np.zeros((len(obskey[0, obskey[0, :] < 3]), 4))
    frq[:, 0] = obskey[0, obskey[0, :] < 3]
    frq[:, 1] = obskey[1, obskey[0, :] < 3]
    frq[:, 2] = obs[0, obskey[0, :] < 3]
    frq[:, 3] = obs[1, obskey[0, :] < 3]

    # --> Calculate the following to be used in glitch-related calculations:
    # number of harmonic degrees (l), total number of modes,
    # number of modes per l, and minimum and maximum values of frequency
    num_of_l = np.rint(frq[-1, 0] + 1).astype(int)
    nmodes = frq.shape[0]
    num_of_n = np.zeros(num_of_l, dtype=int)
    for i in range(num_of_l):
        num_of_n[i] = len(frq[np.rint(frq[:, 0]).astype(int) == i, 0])
    vmin, vmax = np.amin(frq[:, 2]), np.amax(frq[:, 2])

    # Compute second differences, if necessary
    num_sd, frq_sd, icov_sd = None, None, None
    if any(x in [*freqtypes.glitches, *freqtypes.grtypes] for x in rt) and (
        inputparams["method"].lower() == "sd"
    ):
        num_sd, frq_sd, icov_sd = sg.compDif2(num_of_l, frq, nmodes, num_of_n)

    # Compute large frequency separation (the same way as dnufit)
    FWHM_sigma = 2.0 * np.sqrt(2.0 * np.log(2.0))
    yfitdnu = frq[np.rint(frq[:, 0]).astype(int) == 0, 2]
    xfitdnu = frq[np.rint(frq[:, 0]).astype(int) == 0, 1]  # np.arange(0, len(yfitdnu))
    wfitdnu = np.exp(
        -1.0
        * np.power(yfitdnu - numax, 2)
        / (2 * np.power(0.25 * numax / FWHM_sigma, 2.0))
    )
    fitcoef, fitcov = np.polyfit(xfitdnu, yfitdnu, 1, w=np.sqrt(wfitdnu), cov=True)
    dnudata, dnudata_err = fitcoef[0], np.sqrt(fitcov[0, 0])

    # Initialize ratios and corresponding covariance matrices
    datos010, datos02, datos01, datos10, datos012, datos102 = (
        None,
        None,
        None,
        None,
        None,
        None,
    )
    cov010, cov02, cov01, cov10, cov012, cov102 = (None, None, None, None, None, None)

    # Initialize glitch-ratio combinations and corresponding covariance matrices
    datosg010, datosg02, datosg01, datosg10, datosg012, datosg102 = (
        None,
        None,
        None,
        None,
        None,
        None,
    )
    covg010, covg02, covg01, covg10, covg012, covg102 = (
        None,
        None,
        None,
        None,
        None,
        None,
    )

    r02, r01, r10 = freq_fit.ratios(frq)
    if r02 is None:
        if any(x in [*freqtypes.rtypes, *freqtypes.grtypes] for x in rt):
            print(
                "WARNING: Missing radial orders!"
                " Skipping ratio(-glitch combinations) fitting!"
            )
    else:
        # Ratios and their covariances
        r010, r012, r102 = su.combined_ratios(r02, r01, r10)
        rrange = (
            int(round(r01[0, 0])),
            int(round(r01[-1, 0])),
            int(round(r10[0, 0])),
            int(round(r10[-1, 0])),
            int(round(r02[0, 0])),
            int(round(r02[-1, 0])),
        )

        # R010
        datos010, cov010 = read_r010(filename, rrange, nottrustedfile, verbose=verbose)
        if datos010 is None and "r010" in rt:
            print("* r010 unavailable in xml. Computing it ... ", end="", flush=True)
            nr = r010.shape[0]
            rto, cov010 = su.glitch_and_ratio(
                frq,
                nr,
                grtype="r010",
                num_of_n=None,
                vmin=None,
                vmax=None,
                delta_nu=None,
                icov_sd=None,
                nrealizations=inputparams["nrealizations"],
                method=None,
                tol_grad=None,
                regu_param=None,
                n_guess=None,
                tauhe=None,
                dtauhe=None,
                taucz=None,
                dtaucz=None,
            )
            datos010 = np.zeros((3, nr))
            datos010[0, :] = rto[:]
            datos010[1, :] = r010[:, 3]
            datos010[2, :] = np.sqrt(np.diag(cov010))
            print("done!")

        # R02
        datos02, cov02 = read_r02(filename, rrange, nottrustedfile)
        if datos02 is None and ("r02" in rt or plotratios):
            print("* r02 unavailable in xml. Computing it ... ", end="", flush=True)
            nr = r02.shape[0]
            rto, cov02 = su.glitch_and_ratio(
                frq,
                nr,
                grtype="r02",
                num_of_n=None,
                vmin=None,
                vmax=None,
                delta_nu=None,
                icov_sd=None,
                nrealizations=inputparams["nrealizations"],
                method=None,
                tol_grad=None,
                regu_param=None,
                n_guess=None,
                tauhe=None,
                dtauhe=None,
                taucz=None,
                dtaucz=None,
            )
            datos02 = np.zeros((3, nr))
            datos02[0, :] = rto[:]
            datos02[1, :] = r02[:, 3]
            datos02[2, :] = np.sqrt(np.diag(cov02))
            print("done!")

        # R01
        if datos01 is None and ("r01" in rt or plotratios):
            print("* r01 unavailable in xml. Computing it ... ", end="", flush=True)
            nr = r01.shape[0]
            rto, cov01 = su.glitch_and_ratio(
                frq,
                nr,
                grtype="r01",
                num_of_n=None,
                vmin=None,
                vmax=None,
                delta_nu=None,
                icov_sd=None,
                nrealizations=inputparams["nrealizations"],
                method=None,
                tol_grad=None,
                regu_param=None,
                n_guess=None,
                tauhe=None,
                dtauhe=None,
                taucz=None,
                dtaucz=None,
            )
            datos01 = np.zeros((3, nr))
            datos01[0, :] = rto[:]
            datos01[1, :] = r01[:, 3]
            datos01[2, :] = np.sqrt(np.diag(cov01))
            print("done!")

        # R10
        if datos10 is None and ("r10" in rt or plotratios):
            print("* r10 unavailable in xml. Computing it ... ", end="", flush=True)
            nr = r10.shape[0]
            rto, cov10 = su.glitch_and_ratio(
                frq,
                nr,
                grtype="r10",
                num_of_n=None,
                vmin=None,
                vmax=None,
                delta_nu=None,
                icov_sd=None,
                nrealizations=inputparams["nrealizations"],
                method=None,
                tol_grad=None,
                regu_param=None,
                n_guess=None,
                tauhe=None,
                dtauhe=None,
                taucz=None,
                dtaucz=None,
            )
            datos10 = np.zeros((3, nr))
            datos10[0, :] = rto[:]
            datos10[1, :] = r10[:, 3]
            datos10[2, :] = np.sqrt(np.diag(cov10))
            print("done!")

        # R012
        if datos012 is None and "r012" in rt:
            print("* r012 unavailable in xml. Computing it ... ", end="", flush=True)
            nr = r012.shape[0]
            rto, cov012 = su.glitch_and_ratio(
                frq,
                nr,
                grtype="r012",
                num_of_n=None,
                vmin=None,
                vmax=None,
                delta_nu=None,
                icov_sd=None,
                nrealizations=inputparams["nrealizations"],
                method=None,
                tol_grad=None,
                regu_param=None,
                n_guess=None,
                tauhe=None,
                dtauhe=None,
                taucz=None,
                dtaucz=None,
            )
            datos012 = np.zeros((3, nr))
            datos012[0, :] = rto[:]
            datos012[1, :] = r012[:, 3]
            datos012[2, :] = np.sqrt(np.diag(cov012))
            print("done!")

        # R102
        if datos102 is None and "r102" in rt:
            print("* r102 unavailable in xml. Computing it ... ", end="", flush=True)
            nr = r102.shape[0]
            rto, cov102 = su.glitch_and_ratio(
                frq,
                nr,
                grtype="r102",
                num_of_n=None,
                vmin=None,
                vmax=None,
                delta_nu=None,
                icov_sd=None,
                nrealizations=inputparams["nrealizations"],
                method=None,
                tol_grad=None,
                regu_param=None,
                n_guess=None,
                tauhe=None,
                dtauhe=None,
                taucz=None,
                dtaucz=None,
            )
            datos102 = np.zeros((3, nr))
            datos102[0, :] = rto[:]
            datos102[1, :] = r102[:, 3]
            datos102[2, :] = np.sqrt(np.diag(cov102))
            print("done!")

        # --> Glitch-ratio combinations and their covariances
        # GR010
        ngr = r010.shape[0] + 3
        glhrto, covg010 = read_glh(glhhdf, grtype="gr010")
        if glhrto is not None:
            datosg010 = np.zeros((3, ngr))
            datosg010[0, :] = glhrto[:]
            datosg010[1, 0 : ngr - 3] = r010[:, 3]
            datosg010[2, :] = np.sqrt(np.diag(covg010))

        if datosg010 is None and "gr010" in rt:
            print("* gr010 unavailable in xml. Computing it ... ", end="", flush=True)
            glhrto, covg010 = su.glitch_and_ratio(
                frq,
                ngr,
                grtype="gr010",
                num_of_n=num_of_n,
                vmin=vmin,
                vmax=vmax,
                delta_nu=dnudata,
                icov_sd=icov_sd,
                nrealizations=inputparams["nrealizations"],
                method=inputparams["method"],
                tol_grad=inputparams["atol"],
                regu_param=inputparams["lamda"],
                n_guess=inputparams["nguesses"],
                tauhe=inputparams["tauhe"],
                dtauhe=inputparams["dtauhe"],
                taucz=inputparams["taucz"],
                dtaucz=inputparams["dtaucz"],
            )
            datosg010 = np.zeros((3, ngr))
            datosg010[0, :] = glhrto[:]
            datosg010[1, 0 : ngr - 3] = r010[:, 3]
            datosg010[2, :] = np.sqrt(np.diag(covg010))
            print("done!")

        # GR02
        ngr = r02.shape[0] + 3
        glhrto, covg02 = read_glh(glhhdf, grtype="gr02")
        if glhrto is not None:
            datosg02 = np.zeros((3, ngr))
            datosg02[0, :] = glhrto[:]
            datosg02[1, 0 : ngr - 3] = r02[:, 3]
            datosg02[2, :] = np.sqrt(np.diag(covg02))

        if datosg02 is None and "gr02" in rt:
            print("* gr02 unavailable in xml. Computing it ... ", end="", flush=True)
            glhrto, covg02 = su.glitch_and_ratio(
                frq,
                ngr,
                grtype="gr02",
                num_of_n=num_of_n,
                vmin=vmin,
                vmax=vmax,
                delta_nu=dnudata,
                icov_sd=icov_sd,
                nrealizations=inputparams["nrealizations"],
                method=inputparams["method"],
                tol_grad=inputparams["atol"],
                regu_param=inputparams["lamda"],
                n_guess=inputparams["nguesses"],
                tauhe=inputparams["tauhe"],
                dtauhe=inputparams["dtauhe"],
                taucz=inputparams["taucz"],
                dtaucz=inputparams["dtaucz"],
            )
            datosg02 = np.zeros((3, ngr))
            datosg02[0, :] = glhrto[:]
            datosg02[1, 0 : ngr - 3] = r02[:, 3]
            datosg02[2, :] = np.sqrt(np.diag(covg02))
            print("done!")

        # GR01
        ngr = r01.shape[0] + 3
        glhrto, covg01 = read_glh(glhhdf, grtype="gr01")
        if glhrto is not None:
            datosg01 = np.zeros((3, ngr))
            datosg01[0, :] = glhrto[:]
            datosg01[1, 0 : ngr - 3] = r01[:, 3]
            datosg01[2, :] = np.sqrt(np.diag(covg01))

        if datosg01 is None and "gr01" in rt:
            print("* gr01 unavailable in xml. Computing it ... ", end="", flush=True)
            glhrto, covg01 = su.glitch_and_ratio(
                frq,
                ngr,
                grtype="gr01",
                num_of_n=num_of_n,
                vmin=vmin,
                vmax=vmax,
                delta_nu=dnudata,
                icov_sd=icov_sd,
                nrealizations=inputparams["nrealizations"],
                method=inputparams["method"],
                tol_grad=inputparams["atol"],
                regu_param=inputparams["lamda"],
                n_guess=inputparams["nguesses"],
                tauhe=inputparams["tauhe"],
                dtauhe=inputparams["dtauhe"],
                taucz=inputparams["taucz"],
                dtaucz=inputparams["dtaucz"],
            )
            datosg01 = np.zeros((3, ngr))
            datosg01[0, :] = glhrto[:]
            datosg01[1, 0 : ngr - 3] = r01[:, 3]
            datosg01[2, :] = np.sqrt(np.diag(covg01))
            print("done!")

        # GR10
        ngr = r10.shape[0] + 3
        glhrto, covg10 = read_glh(glhhdf, grtype="gr10")
        if glhrto is not None:
            datosg10 = np.zeros((3, ngr))
            datosg10[0, :] = glhrto[:]
            datosg10[1, 0 : ngr - 3] = r10[:, 3]
            datosg10[2, :] = np.sqrt(np.diag(covg10))

        if datosg10 is None and "gr10" in rt:
            print("* gr10 unavailable in xml. Computing it ... ", end="", flush=True)
            glhrto, covg10 = su.glitch_and_ratio(
                frq,
                ngr,
                grtype="gr10",
                num_of_n=num_of_n,
                vmin=vmin,
                vmax=vmax,
                delta_nu=dnudata,
                icov_sd=icov_sd,
                nrealizations=inputparams["nrealizations"],
                method=inputparams["method"],
                tol_grad=inputparams["atol"],
                regu_param=inputparams["lamda"],
                n_guess=inputparams["nguesses"],
                tauhe=inputparams["tauhe"],
                dtauhe=inputparams["dtauhe"],
                taucz=inputparams["taucz"],
                dtaucz=inputparams["dtaucz"],
            )
            datosg10 = np.zeros((3, ngr))
            datosg10[0, :] = glhrto[:]
            datosg10[1, 0 : ngr - 3] = r10[:, 3]
            datosg10[2, :] = np.sqrt(np.diag(covg10))
            print("done!")

        # GR012
        ngr = r012.shape[0] + 3
        glhrto, covg012 = read_glh(glhhdf, grtype="gr012")
        if glhrto is not None:
            datosg012 = np.zeros((3, ngr))
            datosg012[0, :] = glhrto[:]
            datosg012[1, 0 : ngr - 3] = r012[:, 3]
            datosg012[2, :] = np.sqrt(np.diag(covg012))

        if datosg012 is None and "gr012" in rt:
            print("* gr012 unavailable in xml. Computing it ... ", end="", flush=True)
            glhrto, covg012 = su.glitch_and_ratio(
                frq,
                ngr,
                grtype="gr012",
                num_of_n=num_of_n,
                vmin=vmin,
                vmax=vmax,
                delta_nu=dnudata,
                icov_sd=icov_sd,
                nrealizations=inputparams["nrealizations"],
                method=inputparams["method"],
                tol_grad=inputparams["atol"],
                regu_param=inputparams["lamda"],
                n_guess=inputparams["nguesses"],
                tauhe=inputparams["tauhe"],
                dtauhe=inputparams["dtauhe"],
                taucz=inputparams["taucz"],
                dtaucz=inputparams["dtaucz"],
            )
            datosg012 = np.zeros((3, ngr))
            datosg012[0, :] = glhrto[:]
            datosg012[1, 0 : ngr - 3] = r012[:, 3]
            datosg012[2, :] = np.sqrt(np.diag(covg012))
            print("done!")

        # GR102
        ngr = r102.shape[0] + 3
        glhrto, covg102 = read_glh(glhhdf, grtype="gr102")
        if glhrto is not None:
            datosg102 = np.zeros((3, ngr))
            datosg102[0, :] = glhrto[:]
            datosg102[1, 0 : ngr - 3] = r102[:, 3]
            datosg102[2, :] = np.sqrt(np.diag(covg102))

        if datosg102 is None and "gr102" in rt:
            print("* gr102 unavailable in xml. Computing it ... ", end="", flush=True)
            glhrto, covg102 = su.glitch_and_ratio(
                frq,
                ngr,
                grtype="gr102",
                num_of_n=num_of_n,
                vmin=vmin,
                vmax=vmax,
                delta_nu=dnudata,
                icov_sd=icov_sd,
                nrealizations=inputparams["nrealizations"],
                method=inputparams["method"],
                tol_grad=inputparams["atol"],
                regu_param=inputparams["lamda"],
                n_guess=inputparams["nguesses"],
                tauhe=inputparams["tauhe"],
                dtauhe=inputparams["dtauhe"],
                taucz=inputparams["taucz"],
                dtaucz=inputparams["dtaucz"],
            )
            datosg102 = np.zeros((3, ngr))
            datosg102[0, :] = glhrto[:]
            datosg102[1, 0 : ngr - 3] = r102[:, 3]
            datosg102[2, :] = np.sqrt(np.diag(covg102))
            print("done!")

    # Glitch parameters and the corresponding covariance matrix
    datosg, covg = None, None
    if r02 is None and (inputparams["method"].lower() == "sd"):
        if "glitches" in rt:
            # Glitch fitting also requires continuous radial order for SDs
            print("WARNING: Missing radial orders! Skipping glitch fitting!")
    else:
        ng = 3
        glh, covg = read_glh(glhhdf, grtype="glitches")
        if glh is not None:
            datosg = np.zeros((3, ng))
            datosg[0, :] = glh[:]
            datosg[2, :] = np.sqrt(np.diag(covg))

        if datosg is None and "glitches" in rt:
            print(
                "* glitches unavailable in xml. Computing it ... ", end="", flush=True
            )
            glh, covg = su.glitch_and_ratio(
                frq,
                ng,
                grtype="glitches",
                num_of_n=num_of_n,
                vmin=vmin,
                vmax=vmax,
                delta_nu=dnudata,
                icov_sd=icov_sd,
                nrealizations=inputparams["nrealizations"],
                method=inputparams["method"],
                tol_grad=inputparams["atol"],
                regu_param=inputparams["lamda"],
                n_guess=inputparams["nguesses"],
                tauhe=inputparams["tauhe"],
                dtauhe=inputparams["dtauhe"],
                taucz=inputparams["taucz"],
                dtaucz=inputparams["dtaucz"],
            )
            datosg = np.zeros((3, ng))
            datosg[0, :] = glh[:]
            datosg[2, :] = np.sqrt(np.diag(covg))
            print("done!")

    datos = (
        datos010,
        datos02,
        datos_f,
        datos01,
        datos10,
        datos012,
        datos102,
        datosg,
        datosg010,
        datosg02,
        datosg01,
        datosg10,
        datosg012,
        datosg102,
    )
    cov = (
        cov010,
        cov02,
        cov_f,
        cov01,
        cov10,
        cov012,
        cov102,
        covg,
        covg010,
        covg02,
        covg01,
        covg10,
        covg012,
        covg102,
    )

    return (
        datos,
        cov,
        obskey,
        obs,
        dnudata,
        dnudata_err,
        frq_sd,
        icov_sd,
        vmin,
        vmax,
        num_of_n,
    )


def get_freq_ranges(filename):
    """
    Get the ranges of available radial orders in a frequency xml-file

    Parameters
    ----------
    filename : str
        Name of the xml-file

    Returns
    -------
    nrange : tuple
        Lower and upper bounds on the radial order n for l=0,1,2
    rrange : tuple
        Lower and upper bounds on the radial order n for r01, r10, and r02
    """

    try:
        data = read_freqs_xml(filename)
        orders, ratios, ratio_types = read_ratios_xml(filename)[:3]
    except Exception as err:
        print(err)
        return

    n0 = data[2][data[3] == 0]
    n1 = data[2][data[3] == 1]
    n2 = data[2][data[3] == 2]

    r01 = []
    r10 = []
    r02 = []
    for j, rtype in enumerate(ratio_types):
        if "01" in rtype.decode("utf-8"):
            r01.append(orders[j])
        elif "10" in rtype.decode("utf-8"):
            r10.append(orders[j])
        elif "02" in rtype.decode("utf-8"):
            r02.append(orders[j])
        else:
            raise ValueError("Unknown ratio type encountered")

    nrange = (n0[0], n0[-1], n1[0], n1[-1], n2[0], n2[-1])
    rrange = (r01[0], r01[-1], r10[0], r10[-1], r02[0], r02[-1])

    return nrange, rrange


def freqs_ascii_to_xml(
    directory,
    starid,
    symmetric_errors=True,
    check_radial_orders=False,
    nbeforel=True,
    quiet=False,
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
    symmetric_errors : bool, optional
        If True, the ascii files are assumed to only include symmetric
        errors. Otherwise, asymmetric errors are assumed. Default is
        True.
    check_radial_orders : bool or float, optional
        If True, the routine will correct the radial order printed in the
        xml, based on the calculated epsilon value, with its own dnufit.
        If float, does the same, but uses the inputted float as dnufit.
    nbeforel : bool, optional
        Passed to frequency reading. If True (default), the column
        containing the orders n is [0], and the column containing the
        degrees l is [1]. If False, it is the other way around.
    quiet : bool, optional
        Toggle to silence the output (useful for running batches)
    """

    # (Potential) filepaths
    freqsfile = os.path.join(directory, starid + ".fre")
    covfile = os.path.join(directory, starid + ".cov")
    ratiosfile = os.path.join(directory, starid + ".ratios")
    cov010file = os.path.join(directory, starid + ".cov010")
    cov02file = os.path.join(directory, starid + ".cov02")

    # Flags for existence of ratios and covariances
    cov_flag, ratios_flag, cov010_flag, cov02_flag = 1, 1, 1, 1

    #####################################
    # Flags which are redundant for now #
    #####################################
    cov01_flag, cov10_flag, cov012_flag, cov102_flag = 0, 0, 0, 0

    # Make sure that the frequency file exists, and read the frequencies
    if os.path.exists(freqsfile):
        freqs = read_fre(freqsfile, symmetric_errors, nbeforel)
        # If covariances are available, read them
        if os.path.exists(covfile):
            cov = read_cov_freqs(covfile)
        else:
            cov_flag = 0
    else:
        raise RuntimeError("Frequency file not found")

    # Check the value of epsilon, to estimate if the radial orders
    # are correctly identified. Correct them, if user allows to.
    ncorrection = su.check_epsilon_of_freqs(
        freqs, starid, check_radial_orders, quiet=quiet
    )
    if check_radial_orders:
        freqs["order"] += ncorrection
        if not quiet:
            print("The proposed correction has been implemented.\n")
    elif not quiet:
        print("No correction made.\n")

    # Look for ratios and their covariances, read if available
    if os.path.exists(ratiosfile):
        ratios = read_ratios(ratiosfile, symmetric_errors)
        if os.path.exists(cov010file):
            cov010 = read_cov_ratios(cov010file)
        else:
            cov010_flag = 0
        if os.path.exists(cov02file):
            cov02 = read_cov_ratios(cov02file)
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

    # Frequency correlation elements
    if cov_flag:
        mode1_id = 0
        for i in range(len(cov)):
            id2_index = np.mod(i, len(freqs))
            if id2_index == 0:
                mode1_id += 1
            mode2_id = id2_index + 1

            order1 = cov["n1"][i]
            order2 = cov["n2"][i]
            deg1 = cov["l1"][i]
            deg2 = cov["l2"][i]
            covariance = cov["covariance"][i]
            correlation = cov["correlation"][i]

            freq_corr_element = SubElement(
                main,
                "frequency_corr",
                {
                    "id1": "mode" + str(mode1_id),
                    "id2": "mode" + str(mode2_id),
                    "order1": str(order1),
                    "order2": str(order2),
                    "degree1": str(deg1),
                    "degree2": str(deg2),
                },
            )
            SubElement(freq_corr_element, "covariance", {"value": str(covariance)})
            SubElement(freq_corr_element, "correlation", {"value": str(correlation)})

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
