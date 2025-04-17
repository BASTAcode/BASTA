"""
Running BASTA from XML files. Main wrapper!
"""

import os
import gc
import sys
import copy
import h5py
import traceback
from xml.etree import ElementTree

import numpy as np

from basta.bastamain import BASTA, LibraryError
from basta.constants import sydsun as sydc
from basta.constants import parameters
from basta.constants import freqtypes
from basta.fileio import no_models, read_freq_xml, write_star_to_errfile
from basta.utils_xml import ascii_to_xml
from basta.utils_general import strtobool, unique_unsort
from basta.interpolation_driver import perform_interpolation


def _find_get(root, path, value, *default):
    """
    Error catching of things required to be set in xml. Gives useful
    errormessage instead of stuff like "AttributeError: 'NoneType' object
    has no attribute 'get'"

    Parameters
    ----------
    root : Element
        Element in xml-tree to find parameter in
    path : str
        What path in the xml-tree the wanted value should be at
    value : str
        Name of the value to be extracted, e.g. "value", "path", "error"
    default : str, int, None, bool
        Default to return if not set

    Returns
    -------
    val : str
        Extracted value upon location in xml
    """
    # Protect against calls like
    # _find_get(root, path, value, 0, "foo", 42, None, True, [1, 2])
    assert len(default) <= 1
    tag = path.split("/")[-1]
    place = root.find(path)
    if place == None:
        if default:
            return default[0]
        raise KeyError("Missing tag '{0}' in input!".format(tag))
    val = place.get(value)
    if val == None:
        if default:
            return default[0]
        raise ValueError("Missing '{0}' in tag '{1}'!".format(value, tag))
    return val


def _centroid_and_uncert(root, inputparams):
    """
    Extract the centroid and uncertainty definitions for the fit. These need to
    apply default values if not set, therefore this check exists.

    Parameters
    ----------
    root : Element
        Element of the xml input file
    inputparams : dict
        Dictionary of inputparameters for BASTA extracted from the xml

    Returns
    -------
    inputparams : dict
        Dictionary of inputparameters for BASTA extracted from the xml
    """
    try:
        inputparams["centroid"] = root.find("default/centroid").get("value").lower()
        if inputparams["centroid"] not in ["median", "mean"]:
            raise KeyError(
                "Centroid must be either 'median' or 'mean', "
                + "currently set to '{:s}'".format(inputparams["centroid"])
            )
    except AttributeError:
        inputparams["centroid"] = "median"
    try:
        inputparams["uncert"] = root.find("default/uncert").get("value").lower()
        if inputparams["uncert"] not in ["quantiles", "std"]:
            raise KeyError(
                "Unceartainty must be either 'quantiles' or 'std', "
                + "currently set to '{:s}'".format(inputparams["uncert"])
            )
    except AttributeError:
        inputparams["uncert"] = "quantiles"
    return inputparams


def _get_true_or_list(
    params: list, deflist: list | None = None, check: bool = True
) -> list:
    """
    Several input lists can simply be set to true, in order to follow default
    behaviour, while inputting parameters changes the behaviour to the user
    specified. This function extracts the input of that.

    Parameters
    ----------
    params : list
        The inputted list
    deflist : list
        List to copy, if input in params is simply True
    check : bool
        Whether to check the entrances in the list with available parameters in
        BASTA, defined in basta.constants.parameters

    Returns
    -------
    extract : list
        List depending on the input. The 'False' case is an empty list, for
        simplification of later code. The 'True' case is '[True]'.
    """
    # This is not pretty, but reduces the amount of repeated code
    if len(params) == 0:
        extract = []
    elif len(params) == 1:
        if params[0].tag.lower() == "true" and type(deflist) == type(None):
            extract = [True]
        elif params[0].tag.lower() == "true":
            extract = [par for par in deflist]
        elif params[0].tag.lower() != "false":
            extract = [params[0].tag]
        else:
            extract = []
    else:
        extract = []
        for par in params:
            extract.append(par.tag)
    if check and not type(deflist) == type(None):
        checklist = ["distance", "parallax", *parameters.names]
        mask = [True if par in checklist else False for par in extract]
        extract = list(np.asarray(extract)[mask])
    return extract


def _get_freq_minmax(star, freqpath):
    """
    Extract the frequency interval of the star, using dnu as an
    estimation of the extension of the boundaries.

    Parameters
    ----------
    star : Element
        xml root element of the star
    freqpath : str
        Location of the frequency xml's

    Returns
    -------
    fmin : float
        Minimum frequency value
    fmax : float
        Maximum frequency value
    """
    freqfile = os.path.join(freqpath, star.get("starid") + ".xml")
    f, _, _, _ = read_freq_xml(freqfile)
    dnu = float(star.find("dnu").get("value"))
    fmin = f.min() - 2.0 * dnu
    fmax = f.max() + 2.0 * dnu
    return fmin, fmax


def _get_intpol(root, gridfile, freqpath=None):
    """
    Extract interpolation settings.

    Parameters
    ----------
    root : Element
        Element object of the whole xml inputfile
    gridfile : str
        Name of the inputted gridfile
    freqpath : str or None
        If fitting frequencies, we want to read and limit the interpolated
        frequencies to only be within the observed frequencies.

    Returns
    -------
    allintpol : dict
        Dictionary of interpolation settings for each star. Identical for all
        stars if interpolation construction is 'encompass'.
    """
    # Read input
    intpol = {}
    for param in root.findall("default/interpolation/"):
        if param.tag.lower() == "limits":
            limits = {}
            for par in param:
                limits[par.tag] = [
                    float(par.attrib.get("min", -np.inf)),
                    float(par.attrib.get("max", np.inf)),
                    float(par.attrib.get("abstol", np.inf)),
                    float(par.attrib.get("sigmacut", np.inf)),
                ]
            intpol["limits"] = limits
        elif param.tag.lower() == "trackresolution":
            intpol[param.tag.lower()] = {
                "param": param.attrib.get("param", "freq"),
                "value": float(param.attrib.get("value")),
                "baseparam": param.attrib.get("baseparam", "default"),
            }
        elif param.tag.lower() == "gridresolution":
            intpol[param.tag.lower()] = {
                "age": int(param.attrib.get("age", 0.0)),
                "massini": int(param.attrib.get("massini", 0.0)),
                "FeHini": int(param.attrib.get("FeHini", 0.0)),
                "MeHini": int(param.attrib.get("MeHini", 0.0)),
                "yini": int(param.attrib.get("yini", 0.0)),
                "alphaMLT": int(param.attrib.get("alphaMLT", 0.0)),
                "ove": int(param.attrib.get("ove", 0.0)),
                "alphaFe": int(param.attrib.get("alphaFe", 0.0)),
                "gcut": int(param.attrib.get("gcut", 0.0)),
                "eta": int(param.attrib.get("eta", 0.0)),
                "scale": float(param.attrib.get("scale", 0.0)),
                "baseparam": param.attrib.get("baseparam", "xcen"),
                "extend": strtobool(param.attrib.get("extend", "False")),
            }
        elif param.tag.lower() == "name":
            intpol[param.tag.lower()] = {
                "value": param.attrib.get("value"),
            }
        elif param.tag.lower() == "method":
            intpol[param.tag.lower()] = {
                "case": param.attrib.get("case"),
                "construction": param.attrib.get("construction"),
                "retrace": strtobool(param.attrib.get("retrace", "False")),
            }
        else:
            raise ValueError(
                "Unknown parameter encountered in group 'interpolation': {0}".format(
                    param
                )
            )

    # Read and check construction method
    construct = intpol["method"]["construction"]
    if construct not in ["encompass", "bystar"]:
        raise ValueError(
            "Unknown construction method selected. Must be either 'bystar' or 'encompass'!"
        )

    # Permeate retrace option
    if intpol["method"]["retrace"]:
        if "trackresolution" in intpol:
            intpol["trackresolution"]["retrace"] = True
        if "gridresolution" in intpol:
            intpol["gridresolution"]["retrace"] = True

    # If interpolation in frequencies requested, extract limits
    freqnames = ["freq", "freqs", "frequency", "frequencies", "osc"]
    if freqpath:
        freqminmax = {}
        for star in root.findall("star"):
            fmin, fmax = _get_freq_minmax(star, freqpath)
            freqminmax[star.get("starid")] = [fmin, fmax]
    elif intpol["trackresolution"]["param"] in freqnames:
        errmsg = (
            "Interpolation along resolution in individual frequencies is requested, "
        )
        errmsg += (
            "without requesting interpolation in individual frequencies. As this is "
        )
        errmsg += "extremely expensive, please use a different variable."
        raise KeyError(errmsg)

    # If construct is encompass, only need to determine limits once
    limerrmsg = "Abstol or sigmacut of {0} requested in interpolation"
    limerrmsg += "but missing for given star(s)."
    if construct == "encompass":
        limits = {}
        for param in intpol["limits"]:
            gparam = "dnu" if "dnu" in param else param
            minval, maxval, abstol, nsigma = intpol["limits"][param]
            if abstol != np.inf or nsigma != np.inf:
                try:
                    vals = [
                        float(star.find(gparam).get("value"))
                        for star in root.findall("star")
                    ]

                except:
                    raise ValueError(limerrmsg.format(gparam))
                try:
                    err = max(
                        [
                            float(star.find(gparam).get("error"))
                            for star in root.findall("star")
                        ]
                    )
                except:
                    err = 0

                if err and err * nsigma < abstol / 2.0:
                    abstol = 2 * err * nsigma
                if min(vals) - abstol / 2.0 > minval:
                    minval = min(vals) - abstol / 2.0
                if max(vals) + abstol / 2.0 < maxval:
                    maxval = max(vals) + abstol / 2.0
            if minval != -np.inf or maxval != np.inf:
                limits[param] = [minval, maxval]
        if freqpath:
            mins = [f[0] for _, f in freqminmax.items()]
            maxs = [f[1] for _, f in freqminmax.items()]
            limits["freqs"] = [min(mins), max(maxs)]

    # Reformat intpol for individual stars
    allintpol = {}
    for star in root.findall("star"):
        starid = star.get("starid")
        intpolstar = copy.deepcopy(intpol)

        # Determine output gridname
        if "name" in intpol and construct == "encompass":
            intpolstar["name"]["value"] = "intpol_{0}".format(intpol["name"]["value"])

        elif construct == "encompass":
            gridname = gridfile.split("/")[-1].split(".")[-2]
            intpolstar["name"] = {"value": "intpol_{0}".format(gridname)}

        elif "name" in intpol and construct == "bystar":
            intpolstar["name"]["value"] = "intpol_{0}_{1}".format(
                intpol["name"]["value"], starid
            )

        elif construct == "bystar":
            intpolstar["name"] = {"value": "intpol_{0}".format(starid)}

        # Decide limits for interpolation
        if construct == "encompass":
            intpolstar["limits"] = limits
        else:
            for param in intpolstar["limits"]:
                gparam = "dnu" if "dnu" in param else param
                minval, maxval, abstol, nsigma = intpolstar["limits"][param]
                # If abstol or nsigma defined, find the min and max from all of the stars
                if abstol != np.inf or nsigma != np.inf:
                    try:
                        val = float(star.find(gparam).get("value"))
                    except:
                        raise ValueError(limerrmsg.format(gparam))
                    try:
                        err = float(star.find(gparam).get("error"))
                    except:
                        err = 0

                    if err and err * nsigma < abstol / 2.0:
                        abstol = 2 * err * nsigma
                    if val - abstol / 2.0 > minval:
                        minval = val - abstol / 2.0
                    if val + abstol / 2.0 < maxval:
                        maxval = val + abstol / 2.0
                if minval != -np.inf or maxval != np.inf:
                    intpolstar["limits"][param] = [minval, maxval]
            if freqpath:
                intpolstar["limits"]["freqs"] = freqminmax[starid]

        # Append star settings to list of all stars
        allintpol[starid] = intpolstar
    return allintpol


def _read_glitch_controls(fitfreqs: dict) -> dict:
    """
    If glitches have been pre-computed, read the options used, to
    be used for model computation.

    Parameters
    ----------
    fitfreqs : dict
        Frequency fitting options to write options to

    Returns
    -------
    fitfreqs : dict
        Updated frequency fitting options dictionary
    """

    # Translation dict of options
    translate = {"FQ": "Freq", "SD": "SecDif"}
    # Read file
    gfile = h5py.File(fitfreqs["glitchfile"])

    # Read/translate the options
    fitfreqs["glitchmethod"] = translate[gfile["header/method"][()].decode("UTF-8")]
    fitfreqs["npoly_params"] = gfile["header/npoly_params"][()]
    fitfreqs["nderiv"] = gfile["header/nderiv"][()]
    fitfreqs["tol_grad"] = gfile["header/tol_grad"][()]
    fitfreqs["regu_param"] = gfile["header/regu_param"][()]
    fitfreqs["nguesses"] = gfile["header/n_guess"][()]

    # Release file
    gfile.close()
    return fitfreqs


#####################
# MAIN ROUTINE
#####################
def run_xml(
    xmlfile,
    seed=None,
    debug=False,
    verbose=False,
    developermode=False,
    validationmode=False,
):
    """
    Runs BASTA using an xml file as input. This is how you should run BASTA!

    Parameters
    ----------
    xmlfile : str
        Absolute path to the xml file
    seed : int, optional
        The set seed of randomness
    debug : bool, optional
        Activate additional output for debugging (for developers)
    verbose : bool, optional
        Activate a lot (!) of additional output (for developers)
    developermode : bool, optional
        Activate experimental features (for developers)
    validationmode : bool, optional
        Activate validation mode features (for validation purposes only)
    """

    # Get path and change dir
    if not os.path.exists(xmlfile):
        raise FileNotFoundError("Input file not found!")
    oldpath = os.getcwd()
    xmlpath = "/".join(xmlfile.split("/")[:-1])
    xmlname = xmlfile.split("/")[-1]
    if xmlpath:
        os.chdir(xmlpath)

    # Parse XML file
    tree = ElementTree.parse(xmlname)
    root = tree.getroot()

    # Prepare dict and lists for collection
    inputparams = {}
    defaultfit = {}
    overwriteparams = {}
    fitparams = []
    fitfreqs = {"active": False}
    fitdist = False
    stdout = sys.stdout

    # IO parameters
    grid = _find_get(root, "default/library", "path")
    inputparams["inputfile"] = xmlname
    inputparams["output"] = _find_get(root, "default/output", "path")

    # Solar reference parameters
    inputparams["dnusun"] = float(
        _find_get(root, "default/solardnu", "value", sydc.SUNdnu)
    )
    inputparams["numsun"] = float(
        _find_get(root, "default/solarnumax", "value", sydc.SUNnumax)
    )
    inputparams["solarmodel"] = _find_get(root, "default/solarmodel", "value", "")

    # Format of outputted plots, default png for speed, pdf for vector art
    inputparams["plotfmt"] = _find_get(root, "default/plotfmt", "value", "png")

    # Switch to include star identifier in plots, not just in filename, False is default
    inputparams["nameinplot"] = strtobool(
        _find_get(root, "default/nameinplot", "value", "False")
    )

    # Path for science-cases in isochrones, only active for that case
    bastiparams = root.find("default/basti")
    if bastiparams:
        ove = float(_find_get(root, "default/basti/ove", "value"))
        dif = float(_find_get(root, "default/basti/dif", "value"))
        eta = float(_find_get(root, "default/basti/eta", "value"))
        alphaFe = float(_find_get(root, "default/basti/alphaFe", "value"))
        gridid = (ove, dif, eta, alphaFe)
    else:
        gridid = None

    # Type of reported values for centroid and reported uncertainties
    inputparams = _centroid_and_uncert(root, inputparams)

    # Check for frequency fitting and activate
    if any(x.tag in freqtypes.alltypes for x in root.findall("default/fitparams/")):
        fitfreqs["active"] = True
        fitfreqs["fittypes"] = []

    # Extract and classify fitparameters
    for param in root.findall("default/fitparams/"):
        if param.tag in freqtypes.alltypes:
            fitfreqs["fittypes"].append(param.tag)
        if param.tag == "parallax":
            fitdist = True
        fitparams.append(param.tag)

    # Get global parameters
    for param in root.findall("default/overwriteparams/"):
        if param.tag == "phase" or param.tag == "dif":
            overwriteparams[param.tag] = param.get("value")
        else:
            overwriteparams[param.tag] = (
                float(param.get("value")),
                float(param.get("error")),
            )

    # List of parameters for plots and out, follows fitparams if true
    outparams = _get_true_or_list(root.findall("default/outparams/"), fitparams)
    cornerplots = _get_true_or_list(root.findall("default/cornerplots/"), fitparams)
    kielplots = _get_true_or_list(root.findall("default/kielplots/"))
    freqplots = _get_true_or_list(root.findall("default/freqplots/"), check=False)
    inputparams["asciiparams"] = unique_unsort(outparams)
    inputparams["cornerplots"] = unique_unsort(cornerplots)
    inputparams["kielplots"] = unique_unsort(kielplots)
    inputparams["freqplots"] = freqplots

    # Check if distance output is requested
    if "distance" in [*cornerplots, *outparams]:
        fitdist = True

    # Extract parameters for frequency fitting
    if fitfreqs["active"]:
        fitfreqs["freqpath"] = _find_get(root, "default/freqparams/freqpath", "value")
        fitfreqs["fcor"] = _find_get(
            root, "default/freqparams/fcor", "value", "cubicBG14"
        )
        if fitfreqs["fcor"] == "HK08":
            fitfreqs["bexp"] = float(
                _find_get(root, "default/freqparams/bexp", "value")
            )
        else:
            fitfreqs["bexp"] = None
        fitfreqs["correlations"] = strtobool(
            _find_get(root, "default/freqparams/correlations", "value", "False")
        )
        fitfreqs["nrealizations"] = int(
            _find_get(root, "default/freqparams/nrealizations", "value", 10000)
        )
        fitfreqs["threepoint"] = strtobool(
            _find_get(root, "default/freqparams/threepoint", "value", "False")
        )
        fitfreqs["readratios"] = strtobool(
            _find_get(root, "default/freqparams/readratios", "value", "False")
        )
        fitfreqs["dnufrac"] = float(
            _find_get(root, "default/freqparams/dnufrac", "value", 0.15)
        )
        fitfreqs["dnufit_in_ratios"] = strtobool(
            _find_get(root, "default/freqparams/dnufit_in_ratios", "value", "False")
        )
        fitfreqs["interp_ratios"] = strtobool(
            _find_get(root, "default/freqparams/interp_ratios", "value", "True")
        )
        fitfreqs["nsorting"] = strtobool(
            _find_get(root, "default/freqparams/nsorting", "value", "True")
        )
        fitfreqs["dnuprior"] = strtobool(
            _find_get(root, "default/freqparams/dnuprior", "value", "True")
        )
        fitfreqs["dnubias"] = float(
            _find_get(root, "default/freqparams/dnubias", "value", 0)
        )

        # Read seismic weight quantities
        dof = _find_get(root, "default/freqparams/dof", "value", None)
        N = _find_get(root, "default/freqparams/N", "value", None)
        allowed_sweights = ["1/N", "1/1", "1/N-dof"]
        seisw = _find_get(root, "default/freqparams/seismicweight", "value", "1/N")
        if seisw not in allowed_sweights:
            errmsg = "Tag 'seismicweight' defined as '{0}', but must be either '{1}'!"
            raise KeyError(errmsg.format(seisw, ", ".join(allowed_sweights)))

        # Fill weight related things into one dict  to not spam other routines with
        # redundant input
        fitfreqs["seismicweights"] = {"weight": seisw, "dof": dof, "N": N}

        # Detect glitch fitting activation
        if any(x in freqtypes.glitches for x in fitfreqs["fittypes"]):
            fitfreqs["glitchfit"] = True
            fitfreqs["readglitchfile"] = strtobool(
                _find_get(root, "default/freqparams/readglitchfile", "value", "False")
            )
            if not fitfreqs["readglitchfile"]:
                fitfreqs["glitchmethod"] = _find_get(
                    root, "default/grparams/method", "value", "Freq"
                )
                fitfreqs["npoly_params"] = int(
                    _find_get(root, "default/grparams/npoly_params", "value", 5)
                )
                fitfreqs["nderiv"] = int(
                    _find_get(root, "default/grparams/nderiv", "value", 3)
                )
                fitfreqs["tol_grad"] = float(
                    _find_get(root, "default/grparams/tol_grad", "value", 1e-3)
                )
                fitfreqs["regu_param"] = float(
                    _find_get(root, "default/grparams/regu_param", "value", 7)
                )
                fitfreqs["nguesses"] = int(
                    _find_get(root, "default/grparams/nguesses", "value", 200)
                )
        else:
            fitfreqs["glitchfit"] = False

    # Get bayesian weights
    # --> If not provided by the user, assume them to be active
    usebayw = strtobool(_find_get(root, "default/bayesianweights", "value", "True"))

    # Get optional output files
    optoutput = root.findall("default/optionaloutputfiles/")
    useoptoutput = []
    for param in optoutput:
        if param.tag == "True":
            useoptoutput = True
        elif param.tag == "False":
            useoptoutput = False
        elif param.tag == "":
            useoptoutput = False
        else:
            useoptoutput.append(param.tag)

    # Get priors
    limits = {}
    usepriors = []
    for param in root.findall("default/priors/"):
        if any(
            [limit in param.attrib for limit in ["min", "max", "abstol", "sigmacut"]]
        ):
            limits[param.tag] = [
                float(param.attrib.get("min", -np.inf)),
                float(param.attrib.get("max", np.inf)),
                float(param.attrib.get("abstol", np.inf)),
                float(param.attrib.get("sigmacut", np.inf)),
            ]
        elif param.tag == "IMF":
            usepriors.append("salpeter1955")
        else:
            usepriors.append(param.tag)

    # Get interpolation if requested (and if available!), otherwise empty dictionary
    if root.find("default/interpolation"):
        if fitfreqs["active"]:
            allintpol = _get_intpol(root, grid, fitfreqs["freqpath"])
        else:
            allintpol = _get_intpol(root, grid)
    else:
        allintpol = {}

    # Path to ascii output file
    asciifile = _find_get(root, "default/outputfile", "value", "results.ascii")
    asciifilepath = inputparams["output"] + asciifile
    asciifile_dist = asciifilepath.split(".ascii")[0] + "_distance" + ".ascii"

    # Path to error file
    errfile = asciifile.rsplit(".", 1)[0] + ".err"
    errfilepath = inputparams["output"] + errfile

    # Path to warning file
    warnfile = asciifile.rsplit(".", 1)[0] + ".warn"
    warnfilepath = inputparams["output"] + warnfile

    # Make sure the output path exists
    if not os.path.exists(inputparams["output"]):
        os.makedirs(inputparams["output"])

    # First, delete any existing file, then open file for appending
    # --> This is essential when running on multiple stars
    with (
        open(asciifilepath, "wb"),
        open(asciifilepath, "ab+") as fout,
        open(errfilepath, "w"),
        open(errfilepath, "a+") as ferr,
        open(warnfilepath, "w"),
        open(warnfilepath, "a+") as fwarn,
    ):
        inputparams["asciioutput"] = fout
        inputparams["erroutput"] = ferr
        inputparams["warnoutput"] = fwarn

        # Clear the distance-file if necessary
        # --> Required because this file is not part of the with-statement
        if os.path.exists(asciifile_dist):
            tmp_dist = open(asciifile_dist, "wb+")
            tmp_dist.close()

        # Loop over stars
        for star in root.findall("star"):
            starid = star.get("starid")
            starfitparams = {}
            skipstar = False

            # In case of interpolation, one needs to redefine this per star
            gridfile = grid

            # Get fitparameters for the given star
            for param in fitparams:
                # Entry of parameter for the star
                if "dnu" in param:
                    kid = star.find("dnu")
                else:
                    kid = star.find(param)

                # Skip reading for special fitting keys, dealt with later
                if param in [*overwriteparams, *freqtypes.alltypes]:
                    continue

                # Handle the special phase tag behaviour
                if param == "phase":
                    if kid.get("value") is None:
                        if "phase" in inputparams:
                            inputparams.pop("phase")
                    elif "," in kid.get("value"):
                        inputparams["phase"] = tuple(kid.get("value").split(","))
                    else:
                        inputparams["phase"] = kid.get("value")

                # Handle the special diffusion tag behaviour
                elif param == "dif":
                    if kid.get("value") is None:
                        if "dif" in inputparams:
                            inputparams.pop("dif")
                    else:
                        inputparams["dif"] = kid.get("value")

                # Handle general parameters
                elif kid.get("value") is not None:
                    starfitparams[param] = [
                        float(kid.get("value")),
                        float(kid.get("error")),
                    ]

                # Skip the current star, as it is missing input for param
                else:
                    skipstar = True
                    msg = "Fitparameter '{0}' not provided for"
                    msg += " star {1} and will be skipped"
                    no_models(starid, inputparams, msg.format(param, starid))
                    print(msg.format(param, starid))
                    break

            # Check if any overwriting of fitparameter is requested
            for param in overwriteparams:
                if param in fitparams:
                    # Special phase and diffusion behaviour
                    if param == "phase" or param == "dif":
                        val = overwriteparams[param]
                        if "," in val:
                            inputparams[param] = tuple(val.split(","))
                        else:
                            inputparams[param] = val
                    else:
                        gparams = overwriteparams[param]
                        starfitparams[param] = (float(gparams[0]), float(gparams[1]))

            # Collect by-star frequency fit information
            if fitfreqs["active"]:
                # Input files
                fitfreqs["freqfile"] = os.path.join(
                    fitfreqs["freqpath"], starid + ".xml"
                )
                if fitfreqs["glitchfit"] and fitfreqs["readglitchfile"]:
                    fitfreqs["glitchfile"] = os.path.join(
                        fitfreqs["freqpath"], starid + ".hdf5"
                    )
                else:
                    fitfreqs["glhfile"] = None
                try:
                    fitfreqs["nottrustedfile"] = star.find("nottrustedfile").get(
                        "value"
                    )
                except AttributeError:
                    fitfreqs["nottrustedfile"] = None
                try:
                    fitfreqs["excludemodes"] = star.find("excludemodes").get("value")
                except AttributeError:
                    fitfreqs["excludemodes"] = None
                try:
                    fitfreqs["onlyradial"] = star.find("onlyradial").get("value")
                except AttributeError:
                    fitfreqs["onlyradial"] = None
                try:
                    onlyls = star.find("onlyls").get("value")
                    assert isinstance(onlyls, str)
                    fitfreqs["onlyls"] = list(map(int, onlyls.split(",")))
                except AttributeError:
                    fitfreqs["onlyls"] = None

                # dnufit for prior, numax for scaling
                fitfreqs["dnufit"] = float(_find_get(star, "dnu", "value"))
                fitfreqs["numax"] = float(_find_get(star, "numax", "value"))
                try:
                    fitfreqs["dnufit_err"] = float(_find_get(star, "dnu", "error"))
                except ValueError:
                    pass

            # Add to inputparams
            inputparams["fitfreqs"] = fitfreqs

            # Add parameters to inputparams
            inputparams["fitparams"] = starfitparams
            inputparams["magnitudes"] = {}

            # Add limits to inputparams
            inputparams["limits"] = {}
            for param in limits:
                minval, maxval, abstol, nsigma = limits[param]
                # Decide which limits to use if both min/max and abstol is specified
                if param in inputparams["fitparams"]:
                    val, error = inputparams["fitparams"][param]
                    if error * nsigma < abstol / 2.0:
                        abstol = 2 * error * nsigma
                    if val - abstol / 2.0 > minval:
                        minval = val - abstol / 2.0
                    if val + abstol / 2.0 < maxval:
                        maxval = val + abstol / 2.0
                if minval != -np.inf or maxval != np.inf:
                    inputparams["limits"][param] = [minval, maxval]

            # Add parallax and other distance parameters to dictionary
            if fitdist:
                distanceparams = {}

                # Fitting parallax? Storing in distparams for info
                if "parallax" in fitparams:
                    distanceparams["parallax"] = [
                        float(_find_get(star, "parallax", "value")),
                        float(_find_get(star, "parallax", "error")),
                    ]

                # Handle coordinate values
                for coord in ["lon", "lat", "RA", "DEC"]:
                    fc = star.find(coord)
                    if fc is not None:
                        distanceparams[coord] = float(fc.get("value"))

                # Load extinction or use dustmap
                EBV = star.find("EBV")
                if EBV is not None:
                    val = float(EBV.get("value"))
                    distanceparams["EBV"] = [0, val, 0]

                distanceparams["dustframe"] = _find_get(
                    root, "default/distanceInput/dustframe", "value"
                )

                # Find available filters and load corresponding magnitudes
                distanceparams["filters"] = []
                distanceparams["m"] = {}
                distanceparams["m_err"] = {}
                for f in root.findall("default/distanceInput/filters/"):
                    distanceparams["filters"].append(f.tag)
                for f in distanceparams["filters"]:
                    try:
                        distanceparams["m"][f] = float(star.find(f).get("value"))
                        distanceparams["m_err"][f] = float(star.find(f).get("error"))
                    except Exception:
                        print("WARNING: Could not find values for " + f)

                # Store in inputparams
                inputparams["distanceparams"] = distanceparams

            # Seperate treatment of distance output file
            if "distance" in inputparams["asciiparams"]:
                fout_dist = open(asciifile_dist, "ab+")
                inputparams["asciioutput_dist"] = fout_dist

            # Check if star has been broken along the way
            if skipstar:
                continue

            # Call interpolation routine
            try:
                if starid in allintpol:
                    gridfile = perform_interpolation(
                        gridfile,
                        gridid,
                        allintpol[starid],
                        inputparams,
                        debug=debug,
                    )
            except KeyboardInterrupt:
                print("BASTA interpolation stopped manually. Goodbye!")
                traceback.print_exc()
                return
            except ValueError as e:
                print("\nBASTA interpolation stopped due to a value error!\n")
                traceback.print_exc()
                write_star_to_errfile(
                    starid,
                    inputparams,
                    "Interpolation {}: {}".format(e.__class__.__name__, e),
                )
                continue
            except Exception as e:
                print(
                    "BASTA interpolation failed for star {0} with the error:".format(
                        starid
                    )
                )
                print(traceback.format_exc())
                no_models(
                    starid,
                    inputparams,
                    "Unhandled {}: {}".format(e.__class__.__name__, e),
                )

            # Call BASTA itself!
            try:
                BASTA(
                    starid=starid,
                    gridfile=gridfile,
                    inputparams=inputparams,
                    gridid=gridid,
                    usebayw=usebayw,
                    usepriors=usepriors,
                    optionaloutputs=useoptoutput,
                    seed=seed,
                    debug=debug,
                    verbose=verbose,
                    developermode=developermode,
                    validationmode=validationmode,
                )
            except KeyboardInterrupt:
                print("BASTA stopped manually. Goodbye!")
                traceback.print_exc()
                return
            except LibraryError:
                print("BASTA stopped due to a library error!")
                return
            except Exception as e:
                print("BASTA failed for star {0} with the error:".format(starid))
                print(traceback.format_exc())
                no_models(
                    starid,
                    inputparams,
                    "Unhandled {}: {}".format(e.__class__.__name__, e),
                )

            # Make sure to write to file, and clear memory
            if "distance" in inputparams["asciiparams"]:
                fout_dist.flush()
            sys.stdout = stdout
            fout.flush()
            ferr.flush()
            fwarn.flush()
            gc.collect()

    # Create XML output from results.ascii
    xmlfilepath = asciifilepath.rsplit(".", 1)[0] + ".xml"
    ascii_to_xml(asciifilepath, xmlfilepath, uncert=inputparams["uncert"])

    # If the distance file was opened, close it again
    if "distance" in inputparams["asciiparams"]:
        fout_dist.close()

    # Reset path and return
    os.chdir(oldpath)
