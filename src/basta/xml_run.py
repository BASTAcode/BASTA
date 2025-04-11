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

from typing import Optional, Union, Dict, List, Any, Tuple

from basta.bastamain import BASTA, LibraryError
from basta.constants import sydsun as sydc
from basta.constants import parameters
from basta.constants import freqtypes
from basta.fileio import no_models, read_freq_xml, write_star_to_errfile
from basta.utils_xml import ascii_to_xml
from basta.utils_general import strtobool, unique_unsort, flush_all
from basta.interpolation_driver import perform_interpolation


def _find_get(
    root: ElementTree.Element,
    path: str,
    value: str,
    default: Optional[Union[str, float, int]] = 'no-default',  # Default can be str, int, or float
) -> str:
    """
    Error catching of things required to be set in xml. Gives useful
    error messages instead of AttributeError.

    Parameters
    ----------
    root : Element
        Element in XML tree to find parameter in.
    path : str
        Path in the XML tree where the wanted value should be.
    value : str
        Name of the value to be extracted, e.g., "value", "path", "error".
    default : Optional[Union[str, int, float]]
        Default value to return if not set.

    Returns
    -------
    val : str
        Extracted value from the XML.
    """
    tag = path.split("/")[-1]
    element = root.find(path)

    if element is None:
        if default != 'no-default':
            return default
        raise KeyError(f"Missing tag '{tag}' in input!")

    val = element.get(value)
    if val is None:
        if default != 'no-default':
            return default
        raise ValueError(f"Missing attribute '{value}' in tag '{tag}'!")

    return val


def _define_centroid_and_uncertainties(
    root: ElementTree.Element, inputparams: Dict[str, str]
) -> Dict[str, str]:
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
        Updated dictionary of input parameters
    """
    try:
        centroid_element = root.find("default/centroid")
        if centroid_element is not None:
            inputparams["centroid"] = centroid_element.get("value", "").lower()
        else:
            inputparams["centroid"] = "median"
        if inputparams["centroid"] not in {"median", "mean"}:
            raise ValueError(
                f"Centroid must be either 'median' or 'mean', but got '{inputparams['centroid']}'"
            )
    except (AttributeError, ValueError) as e:
        inputparams["centroid"] = "median"
        print(f"Warning: {e}")

    try:
        uncert_element = root.find("default/uncert")
        if uncert_element is not None:
            inputparams["uncert"] = uncert_element.get("value", "").lower()
        else:
            inputparams["uncert"] = "quantiles"
        if inputparams["uncert"] not in {"quantiles", "std"}:
            raise ValueError(
                f"Uncertainty must be either 'quantiles' or 'std', but got '{inputparams['uncert']}'"
            )
    except (AttributeError, ValueError) as e:
        inputparams["uncert"] = "quantiles"
        print(f"Warning: {e}")

    return inputparams


def _get_true_or_list(
    params: List[ElementTree.Element],
    deflist: Optional[List[str]] = None,
    check: bool = True,
) -> Union[List[str], List[bool]]:
    """
    Handles input lists that may be set to True to follow a default behavior or
    specified parameters for custom behavior.

    Parameters
    ----------
    params : list
        The inputted list
    deflist : list
        List to copy, if input in params is simply True
    check : bool
        Whether to check the entrances in the list with available parameters in
        BASTA, defined in `basta.constants.parameters`

    Returns
    -------
    extract : list
        - Returns an empty list if `params` is empty or 'False'.
        - Returns `[True]` if `params` is 'True' and `deflist` is `None`.
        - Returns `deflist` if `params` is 'True' and `deflist` exists.
        - Otherwise, extracts the tags from `params`.

    Notes
    -----
    - If `check` is True, filters the extracted list based on available parameters.
    """
    if not params:
        return []

    first_tag = params[0].tag.lower()

    if len(params) == 1:
        if first_tag == "true":
            if deflist is None:
                return [True]
            else:
                deflist[:]
        if first_tag == "false":
            return []
        return [first_tag]

    extract = [par.tag for par in params]

    if check and deflist is not None:
        checklist = {"distance", "parallax", *parameters.names}
        extract = [par for par in extract if par in checklist]

    return extract


def _get_freq_minmax(star: ElementTree.Element, freqpath: str) -> Tuple[float, float]:
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
    starid = star.get("starid", "unknowntarget") 
    freqfile = os.path.join(freqpath, f"{starid}.xml")

    if not os.path.exists(freqfile):
        raise FileNotFoundError(f"Frequency file not found: {freqfile}")

    f, _, _, _ = read_freq_xml(freqfile)

    dnu_element = star.find("dnu")
    if dnu_element is None or dnu_element.get("value") is None:
        raise ValueError(
            f"Missing 'dnu' value for star {star.get('starid', 'unknown')}."
        )

    dnu_value = dnu_element.get("value")
    if dnu_value is None:
        raise ValueError(f"Missing 'dnu' value: {dnu_element.get('value')}")
    try:
        dnu = float(dnu_value)
    except ValueError:
        raise ValueError(f"Invalid 'dnu' value: {dnu_element.get('value')}")

    fmin = np.amin(f) - 2.0 * dnu
    fmax = np.amax(f) + 2.0 * dnu

    return fmin, fmax


def _get_intpol(root: ElementTree.Element, gridfile, freqpath=None):
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


def _read_glitch_controls(fitfreqs: Dict[str, Any]) -> Dict[str, Any]:
    """
    Reads precomputed glitch options to be used for model computation.

    Parameters
    ----------
    fitfreqs : dict
        Frequency fitting options to write options to, expected to contain "glitchfile"

    Returns
    -------
    fitfreqs : dict
        Updated frequency fitting options dictionary

    Raises
    ------
    KeyError:
        If the required "glitchfile" key is missing in `fitfreqs`.
    FileNotFoundError:
        If the glitch file does not exist.
    KeyError:
        If the required dataset keys are missing in the HDF5 file.
    """
    if "glitchfile" not in fitfreqs:
        raise KeyError("Missing 'glitchfile' key in fitfreqs dictionary.")

    try:
        with h5py.File(fitfreqs["glitchfile"], "r") as gfile:
            # Translation dictionary of options
            translate = {"FQ": "Freq", "SD": "SecDif"}

            # Check if required keys exist
            required_keys = [
                "header/method",
                "header/npoly_params",
                "header/nderiv",
                "header/tol_grad",
                "header/regu_param",
                "header/n_guess",
            ]

            for key in required_keys:
                if key not in gfile:
                    raise KeyError(f"Missing key in glitch file: {key}")

            # Read and translate the options
            method_key = gfile["header/method"][()].decode("UTF-8")
            fitfreqs["glitchmethod"] = translate.get(method_key, method_key)
            fitfreqs["npoly_params"] = gfile["header/npoly_params"][()]
            fitfreqs["nderiv"] = gfile["header/nderiv"][()]
            fitfreqs["tol_grad"] = gfile["header/tol_grad"][()]
            fitfreqs["regu_param"] = gfile["header/regu_param"][()]
            fitfreqs["nguesses"] = gfile["header/n_guess"][()]

    except FileNotFoundError:
        raise FileNotFoundError(f"Glitch file not found: {fitfreqs['glitchfile']}")
    except KeyError as e:
        raise KeyError(f"Key error while reading glitch file: {e}")

    return fitfreqs


#####################
# MAIN ROUTINE
#####################
def run_xml(
    xmlfile: str,
    seed: int,
    verbose: bool = False,
    flag_debug: bool = False,
    flag_developermode: bool = False,
    flag_validationmode: bool = False,
):
    """
    Parses an XML input file, extracts parameters, and prepares inputs for BASTA.

    Parameters
    ----------
    xmlfile : str
        Absolute path to the xml input file
    seed : int, optional
        Seed value for reproducibility.
    verbose : bool, optional
        Activate a lot (!) of additional output
    flag_debug : bool, optional
        Activate additional output for debugging
    flag_developermode : bool, optional
        Activate experimental features
    flag_validationmode : bool, optional
        Activate validation mode features
    """
    if not os.path.isfile(xmlfile):
        raise FileNotFoundError(f"Error: Input file '{xmlfile}' not found!")

    xmlpath = os.path.dirname(xmlfile)
    xmlname = os.path.basename(xmlfile)

    oldpath = os.getcwd()
    if xmlpath:
        os.chdir(xmlpath)

    # Parse XML file
    tree = ElementTree.parse(xmlname)
    root = tree.getroot()

    # Initialize parameters
    inputparams = {
        "inputfile": xmlname,
        "output": _find_get(root, "default/output", "path"),
        "plotfmt": _find_get(root, "default/plotfmt", "value", "png"),
        "nameinplot": strtobool(
            _find_get(root, "default/nameinplot", "value", "False")
        ),
        "dnusun": float(_find_get(root, "default/solardnu", "value", default=sydc.SUNdnu)),
        "numsun": float(_find_get(root, "default/solarnumax", "value", default=sydc.SUNnumax)),
        "solarmodel": _find_get(root, "default/solarmodel", "value", ""),
    }

    # Retrieve grid path
    grid = _find_get(root, "default/library", "path")

    # Handle BASTI-specific parameters
    bastiparams = root.find("default/basti")
    if bastiparams:
        gridid = (
            float(_find_get(root, "default/basti/ove", "value")),
            float(_find_get(root, "default/basti/dif", "value")),
            float(_find_get(root, "default/basti/eta", "value")),
            float(_find_get(root, "default/basti/alphaFe", "value")),
        )
    else:
        gridid = None

    # Handle centroid and uncertainty settings
    inputparams = _define_centroid_and_uncertainties(root, inputparams)

    # Restore original working directory
    os.chdir(oldpath)

    # Check for frequency fitting and activate
    fitparams = [param.tag for param in root.findall("default/fitparams/")]

    fitfreqs = {}
    fitfreqs["active"] = any(param in freqtypes.alltypes for param in fitparams)
    fitfreqs["fittypes"] = [param for param in fitparams if param in freqtypes.alltypes]
    fitdist = "parallax" in fitparams  # If parallax is included, fit for distance

    stdout = sys.stdout

    # Get global parameters
    overwriteparams = {}
    for param in root.findall("default/overwriteparams/"):
        if param.tag == "phase" or param.tag == "dif":
            overwriteparams[param.tag] = param.get("value")
        else:
            overwriteparams[param.tag] = (
                float(param.get("value")),
                float(param.get("error")),
            )

    # Extract plotting parameters, defaulting to fitparams if "True"
    inputparams.update(
        {
            "asciiparams": unique_unsort(
                _get_true_or_list(root.findall("default/outparams/"), fitparams)
            ),
            "cornerplots": unique_unsort(
                _get_true_or_list(root.findall("default/cornerplots/"), fitparams)
            ),
            "kielplots": unique_unsort(
                _get_true_or_list(root.findall("default/kielplots/"))
            ),
            "freqplots": _get_true_or_list(
                root.findall("default/freqplots/"), check=False
            ),
        }
    )

    # Check if distance output is required
    if (
        "distance" in inputparams["cornerplots"]
        or "distance" in inputparams["asciiparams"]
    ):
        fitdist = True

    # Extract parameters for frequency fitting
    if fitfreqs["active"]:
        fitfreqs.update(
            {
                "freqpath": _find_get(root, "default/freqparams/freqpath", "value"),
                "fcor": _find_get(
                    root, "default/freqparams/fcor", "value", "cubicBG14"
                ),
                "bexp": (
                    float(_find_get(root, "default/freqparams/bexp", "value"))
                    if _find_get(root, "default/freqparams/fcor", "value", "cubicBG14")
                    == "HK08"
                    else None
                ),
                "correlations": strtobool(
                    _find_get(root, "default/freqparams/correlations", "value", "False")
                ),
                "nrealizations": int(
                    _find_get(root, "default/freqparams/nrealizations", "value", 10000)
                ),
                "threepoint": strtobool(
                    _find_get(root, "default/freqparams/threepoint", "value", "False")
                ),
                "readratios": strtobool(
                    _find_get(root, "default/freqparams/readratios", "value", "False")
                ),
                "dnufrac": float(
                    _find_get(root, "default/freqparams/dnufrac", "value", 0.15)
                ),
                "dnufit_in_ratios": strtobool(
                    _find_get(
                        root, "default/freqparams/dnufit_in_ratios", "value", "False"
                    )
                ),
                "interp_ratios": strtobool(
                    _find_get(root, "default/freqparams/interp_ratios", "value", "True")
                ),
                "nsorting": strtobool(
                    _find_get(root, "default/freqparams/nsorting", "value", "True")
                ),
                "dnuprior": strtobool(
                    _find_get(root, "default/freqparams/dnuprior", "value", "True")
                ),
                "dnubias": float(
                    _find_get(root, "default/freqparams/dnubias", "value", 0)
                ),
            }
        )

        # Read seismic weight quantities
        allowed_sweights = ["1/N", "1/1", "1/N-dof"]
        seisw = _find_get(root, "default/freqparams/seismicweight", "value", "1/N")
        if seisw not in allowed_sweights:
            raise KeyError(
                f"Tag 'seismicweight' defined as '{seisw}', but must be one of {allowed_sweights}!"
            )
        fitfreqs["seismicweights"] = {
            "weight": seisw,
            "dof": _find_get(root, "default/freqparams/dof", "value", None),
            "N": _find_get(root, "default/freqparams/N", "value", None),
        }

        # Detect glitch fitting activation
        fitfreqs["glitchfit"] = any(
            x in freqtypes.glitches for x in fitfreqs["fittypes"]
        )
        if fitfreqs["glitchfit"]:
            fitfreqs["readglitchfile"] = strtobool(
                _find_get(root, "default/freqparams/readglitchfile", "value", "False")
            )
            if not fitfreqs["readglitchfile"]:
                fitfreqs.update(
                    {
                        "glitchmethod": _find_get(
                            root, "default/grparams/method", "value", "Freq"
                        ),
                        "npoly_params": int(
                            _find_get(root, "default/grparams/npoly_params", "value", 5)
                        ),
                        "nderiv": int(
                            _find_get(root, "default/grparams/nderiv", "value", 3)
                        ),
                        "tol_grad": float(
                            _find_get(root, "default/grparams/tol_grad", "value", 1e-3)
                        ),
                        "regu_param": float(
                            _find_get(root, "default/grparams/regu_param", "value", 7)
                        ),
                        "nguesses": int(
                            _find_get(root, "default/grparams/nguesses", "value", 200)
                        ),
                    }
                )

    # Get bayesian weights (default: True)
    usebayw = strtobool(_find_get(root, "default/bayesianweights", "value", "True"))

    # Extract optional output files
    optoutput = [
        param.tag for param in root.findall("default/optionaloutputfiles/") if param.tag
    ]
    useoptoutput = optoutput if optoutput else False

    # Get priors
    limits = {}
    usepriors = []
    for param in root.findall("default/priors/"):
        if any(limit in param.attrib for limit in ["min", "max", "abstol", "sigmacut"]):
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
    basepath = os.path.join(inputparams["output"], asciifile.rsplit(".", 1)[0])
    output_paths = {
        "ascii": basepath + ".ascii",
        "xml": basepath + ".xml",
        "ascii_distance": basepath + "_distance.ascii",
        "error": basepath + ".err",
        "warning": basepath + ".warn",
    }

    # Make sure the output path exists
    os.makedirs(inputparams["output"], exist_ok=True)

    # First, delete any existing file, then open file for appending
    # --> This is essential when running on multiple stars
    with (
        open(output_paths["ascii"], "wb"),
        open(output_paths["error"], "w"),
        open(output_paths["warning"], "w"),
    ):
        with (
            open(output_paths["ascii"], "ab+") as fout,
            open(output_paths["error"], "a+") as ferr,
            open(output_paths["warning"], "a+") as fwarn,
        ):
            inputparams.update(
                {"asciioutput": fout, "erroutput": ferr, "warnoutput": fwarn}
            )

            # Ensure distance file is cleared
            if os.path.exists(output_paths["ascii_distance"]):
                open(output_paths["ascii_distance"], "wb").close()

            # Loop over stars
            for star in root.findall("star"):
                starid = star.get("starid")
                starfitparams = {}
                skipstar = False
                gridfile = grid

                # Get fitparameters for the given star
                for param in fitparams:
                    kid = star.find("dnu") if "dnu" in param else star.find(param)

                    if param in [*overwriteparams, *freqtypes.alltypes]:
                        continue  # Skip special fitting keys, handled later

                    val = kid.get("value") if kid is not None else None
                    err = kid.get("error") if kid is not None else None

                    # Handle the special phase tag behaviour
                    if param in ["phase", "dif"]:
                        if val is None:
                            inputparams.pop(param, None)
                        elif "," in val:
                            inputparams[param] = tuple(val.split(","))
                        else:
                            inputparams[param] = val
                    elif val is not None:
                        starfitparams[param] = [
                            float(val),
                            float(err),
                        ]
                    else:
                        skipstar = True
                        msg = f"Fitparameter '{param}' not provided for star {starid} and will be skipped"
                        no_models(starid, inputparams, msg)
                        print(msg)
                        break

                # Check if any overwriting of fitparameter is requested
                for param, gparams in overwriteparams.items():
                    if param in fitparams:
                        if param in ["phase", "dif"]:
                            if "," in val:
                                inputparams[param] = tuple(gparams.split(","))
                            else:
                                inputparams[param] = (
                                    float(gparams[0]),
                                    float(gparams[1]),
                                )
                        else:
                            inputparams[param] = (float(gparams[0]), float(gparams[1]))

                # Collect by-star frequency fit information
                if fitfreqs["active"]:
                    fitfreqs.update(
                        {
                            "freqfile": os.path.join(
                                fitfreqs["freqpath"], f"{starid}.xml"
                            ),
                            "glitchfile": (
                                os.path.join(fitfreqs["freqpath"], f"{starid}.hdf5")
                                if fitfreqs["glitchfit"] and fitfreqs["readglitchfile"]
                                else None
                            ),
                            "dnufit": float(_find_get(star, "dnu", "value")),
                            "numax": float(_find_get(star, "numax", "value")),
                            "dnufit_err": float(
                                _find_get(star, "dnu", "error", default="nan")
                            ),
                        }
                    )
                    for fp in ["nottrustedfile", "excludemodes", "onlyradial"]:
                        if star.find(fp) is not None:
                            fitfreqs[fp] = (star.find(fp).get("value"))
                        else:
                            fitfreqs[fp] = None
                    inputparams["fitfreqs"] = fitfreqs

                # Add fitparams and limits
                inputparams["fitparams"] = starfitparams
                inputparams["magnitudes"] = {}
                inputparams["limits"] = {}
                for param, (minval, maxval, abstol, nsigma) in limits.items():
                    if param in starfitparams:
                        val, err = starfitparams[param]
                        abstol = max(abstol, 2 * err * nsigma)
                        minval, maxval = max(minval, val - abstol / 2), min(
                            maxval, val + abstol / 2
                        )
                    if minval != -np.inf or maxval != np.inf:
                        inputparams["limits"][param] = [minval, maxval]

                # Add parallax and other distance parameters to dictionary
                if fitdist:
                    distanceparams = {
                        "parallax": (
                            [
                                float(_find_get(star, "parallax", "value")),
                                float(_find_get(star, "parallax", "error")),
                            ]
                            if "parallax" in fitparams
                            else None
                        ),
                        "dustframe": _find_get(
                            root, "default/distanceInput/dustframe", "value"
                        ),
                        "filters": [
                            f.tag
                            for f in root.findall("default/distanceInput/filters/")
                        ],
                        "m": {},
                        "m_err": {},
                    }

                    for coord in ["lon", "lat", "RA", "DEC"]:
                        fc = star.find(coord)
                        if fc is not None:
                            distanceparams[coord] = float(fc.get("value"))

                    # Load extinction or use dustmap
                    EBV = star.find("EBV")
                    if EBV is not None:
                        distanceparams["EBV"] = [0, float(EBV.get("value")), 0]

                    # Find available filters and load corresponding magnitudes
                    for f in distanceparams["filters"]:
                        try:
                            distanceparams["m"][f] = float(star.find(f).get("value"))
                            distanceparams["m_err"][f] = float(
                                star.find(f).get("error")
                            )
                        except Exception:
                            print("WARNING: Could not find values for " + f)

                    inputparams["distanceparams"] = distanceparams

                # Seperate treatment of distance output file
                if "distance" in inputparams["asciiparams"]:
                    with open(output_paths["ascii_distance"], "ab+") as fout_dist:
                        inputparams["asciioutput_dist"] = fout_dist

                # Skip star if it was marked as broken earlier
                if skipstar:
                    continue

                # Perform interpolation routine if requested
                try:
                    if starid in allintpol:
                        gridfile = perform_interpolation(
                            gridfile,
                            gridid,
                            allintpol[starid],
                            inputparams,
                            debug=flag_debug,
                        )
                except KeyboardInterrupt:
                    print("BASTA interpolation stopped manually. Goodbye!")
                    traceback.print_exc()
                    return
                except ValueError as e:
                    print("\nBASTA interpolation stopped due to a value error!\n")
                    traceback.print_exc()
                    write_star_to_errfile(
                        starid, inputparams, f"Interpolation Error: {e}"
                    )
                    continue
                except Exception as e:
                    error_msg = f"BASTA interpolation failed for star {starid}: {e}"
                    print(error_msg)
                    print(traceback.format_exc())
                    no_models(starid, inputparams, f"Unhandled Error: {e}")

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
                        debug=flag_debug,
                        verbose=verbose,
                        developermode=flag_developermode,
                        validationmode=flag_validationmode,
                    )
                except KeyboardInterrupt:
                    print("BASTA stopped manually. Goodbye!")
                    traceback.print_exc()
                    return
                except LibraryError:
                    print("BASTA stopped due to a library error!")
                    return
                except Exception as e:
                    error_msg = f"BASTA failed for star {starid} due to: {e}"
                    print(error_msg)
                    print(traceback.format_exc())
                    no_models(starid, inputparams, f"Unhandled Error: {e}")

                # Ensure output files are written and clean up memory
                flush_all(fout, ferr, fwarn)

                # Flush distance output only if it's open
                if "distance" in inputparams["asciiparams"]:
                    with open(output_paths["ascii_distance"], "ab+") as fout_dist:
                        fout_dist.flush()

                gc.collect()

    ascii_to_xml(
        output_paths["ascii"], output_paths["xml"], uncert=inputparams["uncert"]
    )

    # Reset path and return
    if os.getcwd() != oldpath:
        os.chdir(oldpath)
