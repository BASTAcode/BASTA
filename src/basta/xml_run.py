"""
Running BASTA from XML files. Main wrapper!
"""

import copy
import os
import traceback
from io import BufferedIOBase
from typing import Any, Literal, TypedDict, overload
from xml.etree import ElementTree as ET

import h5py  # type: ignore[import]
import numpy as np

from basta import core, constants, imfs
from basta.bastamain import BASTA
from basta.constants import freqtypes, parameters
from basta.constants import sydsun as sydc
from basta.errors import LibraryError
from basta.fileio import no_models, read_freq_xml, write_star_to_errfile
from basta.interpolation_driver import perform_interpolation
from basta.utils_general import strtobool, unique_unsort
from basta.utils_xml import ascii_to_xml


@overload
def _find_get(root: ET.Element, path: str, value: str) -> str: ...
@overload
def _find_get(root: ET.Element, path: str, value: str, default: str) -> str: ...
@overload
def _find_get(root: ET.Element, path: str, value: str, default: None) -> str | None: ...
@overload
def _find_get(
    root: ET.Element, path: str, value: str, default: float
) -> str | float | int: ...


def _find_get(
    root: ET.Element,
    path: str,
    value: str,
    default: str | float | None = "no-default",
) -> str | float | int | None:
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
    default : str | int | float
        Default value to return if not set.

    Returns
    -------
    val : str
        Extracted value from the XML.
    """
    tag = path.split("/")[-1]
    element = root.find(path)

    if element is None:
        if default != "no-default":
            return default
        raise KeyError(f"Missing tag '{tag}' in input!")

    val = element.get(value)
    if val is None:
        if default != "no-default":
            return default
        raise ValueError(f"Missing attribute '{value}' in tag '{tag}'!")

    return val


def _define_centroid_and_uncertainties(
    root: ET.Element, inputparams: dict[str, Any]
) -> dict[str, Any]:
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
    params: list[ET.Element],
    deflist: list[str] | None = None,
    check: bool = True,
) -> list[str] | list[bool]:
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
            deflist[:]
        if first_tag == "false":
            return []
        return [first_tag]

    extract = [par.tag for par in params]

    if check and deflist is not None:
        checklist = {"distance", "parallax", *parameters.names}
        extract = [par for par in extract if par in checklist]

    return extract


def _get_freq_minmax(star: ET.Element, freqpath: str) -> tuple[float, float]:
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


class _IntpolTrackresolution(TypedDict):
    param: str
    value: float
    baseparam: str


class IntpolTrackresolution(_IntpolTrackresolution, total=False):
    retrace: bool


class _IntpolGridresolution(TypedDict):
    age: int
    massini: int
    FeHini: int
    MeHini: int
    yini: int
    alphaMLT: int
    ove: int
    alphaFe: int
    gcut: int
    eta: int
    scale: float
    baseparam: str
    extend: Literal[0, 1]


class IntpolGridresolution(_IntpolGridresolution, total=False):
    retrace: bool


class IntpolName(TypedDict):
    # TODO(Amalie): maybe need assert not None?
    value: str | None


class IntpolMethod(TypedDict):
    # TODO(Amalie): maybe need assert not None?
    case: str | None
    # TODO(Amalie): maybe need assert not None?
    construction: str | None
    retrace: Literal[0, 1]


class Intpol(TypedDict, total=False):
    limits: dict[str, list[float]]
    trackresolution: IntpolTrackresolution
    gridresolution: IntpolGridresolution
    name: IntpolName
    method: IntpolMethod


def _get_intpol(root: ET.Element, gridfile, freqpath=None):
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
    intpol: Intpol = {}
    for param in root.findall("default/interpolation/"):
        if param.tag.lower() == "limits":
            limits: dict[str, list[float]] = {}
            for par in param:
                limits[par.tag] = [
                    float(par.attrib.get("min", -np.inf)),
                    float(par.attrib.get("max", np.inf)),
                    float(par.attrib.get("abstol", np.inf)),
                    float(par.attrib.get("sigmacut", np.inf)),
                ]
            intpol["limits"] = limits
        elif param.tag.lower() == "trackresolution":
            intpol["trackresolution"] = {
                "param": param.attrib.get("param", "freq"),
                "value": float(param.attrib["value"]),
                "baseparam": param.attrib.get("baseparam", "default"),
            }
        elif param.tag.lower() == "gridresolution":
            intpol["gridresolution"] = {
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
            intpol["name"] = {
                # TODO(Amalie): assert value is not None?
                "value": param.attrib.get("value"),
            }
        elif param.tag.lower() == "method":
            intpol["method"] = {
                # TODO(Amalie): assert value is not None?
                "case": param.attrib.get("case"),
                # TODO(Amalie): assert value is not None?
                "construction": param.attrib.get("construction"),
                "retrace": strtobool(param.attrib.get("retrace", "False")),
            }
        else:
            raise ValueError(
                f"Unknown parameter encountered in group 'interpolation': {param}"
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
        for limit in intpol["limits"]:
            gparam = "dnu" if "dnu" in limit else limit
            minval, maxval, abstol, nsigma = intpol["limits"][limit]
            if abstol != np.inf or nsigma != np.inf:
                try:
                    vals: list[float] = []
                    for star in root.findall("star"):
                        star_gparam = star.find(gparam)
                        assert star_gparam is not None
                        vals.append(float(star_gparam.attrib["value"]))

                except Exception:
                    raise ValueError(limerrmsg.format(gparam))
                try:
                    errs: list[float] = []
                    for star in root.findall("star"):
                        star_gparam = star.find(gparam)
                        assert star_gparam is not None
                        errs.append(float(star_gparam.attrib["error"]))
                    err = max(errs)
                except Exception:
                    err = 0

                if err and err * nsigma < abstol / 2.0:
                    abstol = 2 * err * nsigma
                minval = max(minval, min(vals) - abstol / 2.0)
                maxval = min(maxval, max(vals) + abstol / 2.0)
            if minval != -np.inf or maxval != np.inf:
                limits[limit] = [minval, maxval]
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
            intpolstar["name"]["value"] = "intpol_{}".format(intpol["name"]["value"])

        elif construct == "encompass":
            gridname = gridfile.split("/")[-1].split(".")[-2]
            intpolstar["name"] = {"value": f"intpol_{gridname}"}

        elif "name" in intpol and construct == "bystar":
            intpolstar["name"]["value"] = "intpol_{}_{}".format(
                intpol["name"]["value"], starid
            )

        elif construct == "bystar":
            intpolstar["name"] = {"value": f"intpol_{starid}"}

        # Decide limits for interpolation
        if construct == "encompass":
            intpolstar["limits"] = limits
        else:
            for limit in intpolstar["limits"]:
                gparam = "dnu" if "dnu" in limit else limit
                minval, maxval, abstol, nsigma = intpolstar["limits"][limit]
                # If abstol or nsigma defined, find the min and max from all of the stars
                if abstol != np.inf or nsigma != np.inf:
                    try:
                        star_gparam = star.find(gparam)
                        assert star_gparam is not None
                        val = float(star_gparam.attrib["value"])
                    except Exception:
                        raise ValueError(limerrmsg.format(gparam))
                    try:
                        star_gparam = star.find(gparam)
                        assert star_gparam is not None
                        err = float(star_gparam.attrib["error"])
                    except Exception:
                        err = 0

                    if err and err * nsigma < abstol / 2.0:
                        abstol = 2 * err * nsigma
                    minval = max(minval, val - abstol / 2.0)
                    maxval = min(maxval, val + abstol / 2.0)
                if minval != -np.inf or maxval != np.inf:
                    intpolstar["limits"][limit] = [minval, maxval]
            if freqpath:
                intpolstar["limits"]["freqs"] = freqminmax[starid]

        # Append star settings to list of all stars
        allintpol[starid] = intpolstar
    return allintpol


def _read_glitch_controls(fitfreqs: dict[str, Any]) -> dict[str, Any]:
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
) -> None:
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
    tree = ET.parse(xmlname)
    root = tree.getroot()

    # Initialize parameters
    # TODO(Amalie): remove inputparams or use a more precise type
    inputparams: dict[str, Any] = {
        "inputfile": xmlname,
        "output": _find_get(root, "default/output", "path"),
        "plotfmt": _find_get(root, "default/plotfmt", "value", "png"),
        "nameinplot": strtobool(
            _find_get(root, "default/nameinplot", "value", "False")
        ),
        "dnusun": float(
            _find_get(root, "default/solardnu", "value", default=sydc.SUNdnu)
        ),
        "numsun": float(
            _find_get(root, "default/solarnumax", "value", default=sydc.SUNnumax)
        ),
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
    centroid = inputparams["centroid"]
    uncert = inputparams["uncert"]

    # Restore original working directory
    os.chdir(oldpath)

    # Check for frequency fitting and activate
    fitparams = [param.tag for param in root.findall("default/fitparams/")]

    # TODO(Amalie): remove fitfreqs or use a more precise type
    fitfreqs: dict[str, Any] = {}
    fitfreqs["active"] = any(param in freqtypes.alltypes for param in fitparams)
    fitfreqs["fittypes"] = [param for param in fitparams if param in freqtypes.alltypes]
    fitdist = "parallax" in fitparams  # If parallax is included, fit for distance

    # Get global parameters
    overwriteparams: dict[str, tuple[float, float]] = {}
    overwritephasedif: dict[str, str] = {}
    for param in root.findall("default/overwriteparams/"):
        if param.tag in {"phase", "dif"}:
            overwritephasedif[param.tag] = param.attrib["value"]
        else:
            overwriteparams[param.tag] = (
                float(param.attrib["value"]),
                float(param.attrib["error"]),
            )

    # Extract plotting parameters, defaulting to fitparams if "True"
    asciiparams = unique_unsort(
        _get_true_or_list(root.findall("default/outparams/"), fitparams)
    )
    inputparams.update(
        {
            "asciiparams": asciiparams,
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
    if "distance" in inputparams["cornerplots"] or "distance" in asciiparams:
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
    useoptoutput = bool(optoutput)

    # Get priors
    priors: dict[str, Any] = {}
    imf: str | None = None
    for param in root.findall("default/priors/"):
        if any(limit in param.attrib for limit in ["min", "max", "abstol", "sigmacut"]):
            priors[param.tag] = [
                float(param.attrib.get("min", -np.inf)),
                float(param.attrib.get("max", np.inf)),
                float(param.attrib.get("abstol", np.inf)),
                float(param.attrib.get("sigmacut", np.inf)),
            ]
        elif param.tag in [
            "IMF",
            "salpeter1955",
            "millerscalo1979",
            "kennicutt1994",
            "scalo1998",
            "kroupa2001",
            "baldryglazebrook2003",
            "chabrier2003",
        ]:
            assert imf is None
            imf = param.tag
        else:
            raise ValueError

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
    distancefilters = (
        [f.tag for f in root.findall("default/distanceInput/filters/")]
        if fitdist
        else None
    )
    outputoptions = core.OutputOptions(
        asciiparams=asciiparams,
        uncert=uncert,
        centroid=centroid,
        optionaloutputs=useoptoutput,
        debug=flag_debug,
        verbose=verbose,
        developermode=flag_developermode,
        validationmode=flag_validationmode,
    )

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

            if fitdist:
                fout_dist: BufferedIOBase | None = open(
                    output_paths["ascii_distance"], "ab+"
                )
            else:
                fout_dist = None
            runfiles = core.RunFiles(
                runbasepath=basepath,
                summarytable=fout,
                summarytablepath=output_paths["ascii"],
                distancesummarytable=fout_dist,
                distancesummarytablepath=output_paths["ascii_distance"],
                warnoutput=fwarn,
                erroroutput=ferr,
            )

            # Loop over stars
            for star in root.findall("star"):
                starid = star.get("starid")
                assert starid is not None
                assert isinstance(starid, str)
                starfitparams: dict[str, core.Fitparam] = {}
                skipstar = False
                gridfile = grid

                filepaths = core.FilePaths(
                    starid=starid,
                    outputdir=inputparams["output"],
                    inputfile=inputparams["inputfile"],
                    plotfmt=inputparams["plotfmt"],
                )

                # Get fitparameters for the given star
                for param in fitparams:
                    kid = star.find("dnu") if "dnu" in param else star.find(param)

                    if param in [
                        *overwriteparams,
                        *overwritephasedif,
                        *freqtypes.alltypes,
                    ]:
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
                        assert err is not None
                        starfitparams[param] = (
                            float(val),
                            float(err),
                        )
                    else:
                        skipstar = True
                        msg = f"Fitparameter '{param}' not provided for star {starid} and will be skipped"
                        no_models(
                            starid,
                            filepaths,
                            runfiles,
                            outputoptions,
                            distancefilters,
                            msg,
                        )
                        print(msg)
                        break

                # Check if any overwriting of fitparameter is requested
                for param, gparams in overwriteparams.items():
                    if param in fitparams:
                        inputparams[param] = (float(gparams[0]), float(gparams[1]))

                for param, phasedifparam in overwritephasedif.items():
                    if param in fitparams:
                        if "," in phasedifparam:
                            inputparams[param] = tuple(
                                map(float, phasedifparam.split(","))
                            )
                        else:
                            inputparams[param] = float(phasedifparam), float(
                                phasedifparam
                            )

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
                        fp_element = star.find(fp)
                        if fp_element is not None:
                            fitfreqs[fp] = fp_element.get("value")
                        else:
                            fitfreqs[fp] = None
                    inputparams["fitfreqs"] = fitfreqs
                else:
                    inputparams["fitfreqs"] = {
                        "active": False,
                        "fittypes": [],
                        "freqpath": "",
                        "freqfile": "",
                        "dnufit": -9999,
                        "dnufit_err": -9999,
                        "numax": -9999,
                        "fcor": "",
                        "seismicweights": {},
                        "bexp": None,
                        "correlations": False,
                        "nrealizations": 10000,
                        "threepoint": False,
                        "readratios": False,
                        "dnufrac": 0.15,
                        "dnufit_in_ratios": False,
                        "interp_ratios": True,
                        "nsorting": True,
                        "dnuprior": True,
                        "dnubias": 0.0,
                        "glitchfit": False,
                        "glitchfile": None,
                        "nottrustedfile": None,
                        "excludemodes": None,
                        "onlyradial": None,
                    }

                # Add fitparams and limits
                # inputparams["fitparams"] = starfitparams
                inputparams["magnitudes"] = {}
                inputparams["limits"] = {}
                for param, (minval, maxval, abstol, nsigma) in priors.items():
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
                    assert distancefilters is not None
                    # TODO(Amalie): add precise type for this in core
                    distanceparams: dict[str, Any] = {
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
                        "filters": distancefilters,
                        "m": {},
                        "m_err": {},
                    }

                    for coord in ["lon", "lat", "RA", "DEC"]:
                        fc = star.find(coord)
                        if fc is not None:
                            distanceparams[coord] = float(fc.attrib["value"])

                    # Load extinction or use dustmap
                    EBV = star.find("EBV")
                    if EBV is not None:
                        distanceparams["EBV"] = [0, float(EBV.attrib["value"]), 0]
                    else:
                        distanceparams["EBV"] = []

                    # Find available filters and load corresponding magnitudes
                    for f in distanceparams["filters"]:
                        try:
                            star_f = star.find(f)
                            assert star_f is not None
                            distanceparams["m"][f] = float(star_f.attrib["value"])
                            distanceparams["m_err"][f] = float(star_f.attrib["error"])
                        except Exception:
                            print("WARNING: Could not find values for " + f)

                    inputparams["distanceparams"] = distanceparams
                else:
                    inputparams["distanceparams"] = {
                        "parallax": [],
                        "dustframe": "",
                        "filters": [],
                        "m": {},
                        "m_err": {},
                        "RA": -9999,
                        "DEC": -9999,
                        "EBV": [],
                    }

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
                    write_star_to_errfile(starid, runfiles, f"Interpolation Error: {e}")
                    continue
                except Exception as e:
                    error_msg = f"BASTA interpolation failed for star {starid}: {e}"
                    print(error_msg)
                    print(traceback.format_exc())
                    no_models(
                        starid,
                        filepaths,
                        runfiles,
                        outputoptions,
                        distancefilters,
                        f"Unhandled Error: {e}",
                    )

                # Call BASTA itself!
                distparams = core.DistanceParameters(
                    magnitudes={
                        f: (m, m_err)
                        for f, m, m_err in zip(
                            inputparams["distanceparams"]["filters"],
                            inputparams["distanceparams"]["m"].values(),
                            inputparams["distanceparams"]["m_err"].values(),
                        )
                    },
                    coordinates={
                        "frame": inputparams["distanceparams"]["dustframe"],
                        "RA": inputparams["distanceparams"]["RA"],
                        "DEC": inputparams["distanceparams"]["DEC"],
                    },
                    params={
                        "parallax": inputparams["distanceparams"]["parallax"],
                    },
                    EBV=inputparams["distanceparams"]["EBV"],
                )

                prefixes = ("dnu", "numax")
                classicalparams = core.ClassicalParameters(
                    params={
                        k: v
                        for k, v in starfitparams.items()
                        if not k.startswith(prefixes)
                        and k
                        not in [
                            "parallax",
                        ]
                    }
                )
                numaxdnuparams = {
                    k: core.ScaledValueError(original=v, scale=1.0)
                    for k, v in starfitparams.items()
                    if k.startswith(prefixes)
                }
                if not numaxdnuparams:
                    numaxdnuparams = {
                        "dnufit": core.ScaledValueError(
                            original=(fitfreqs["dnufit"], fitfreqs["dnufit_err"]),
                            scale=1.0,
                        ),
                        "numax": core.ScaledValueError(
                            original=(fitfreqs["numax"], 0.05 * fitfreqs["numax"]),
                            scale=1.0,
                        ),
                    }
                globalseismicparams = core.GlobalSeismicParameters(
                    params=numaxdnuparams,
                )

                if inputparams["fitfreqs"]["fcor"] == "":
                    surfacecorrection = None
                else:
                    surfcor = constants.SeismicfitAliases.scalias[
                        inputparams["fitfreqs"]["fcor"].lower()
                    ]
                    surfacecorrection = {
                        surfcor: (
                            {"bexp": inputparams["fitfreqs"]["bexp"]}
                            if surfcor == "KBC08"
                            else {}
                        )
                    }
                star = core.InputStar(
                    starid=starid,
                    classicalparams=classicalparams,
                    globalseismicparams=globalseismicparams,
                    distanceparams=distparams,
                    freqpath=inputparams["fitfreqs"]["freqpath"],
                    freqfile=inputparams["fitfreqs"]["freqfile"],
                    surfacecorrection=surfacecorrection,
                    correlations=inputparams["fitfreqs"]["correlations"],
                    nottrustedfile=inputparams["fitfreqs"]["nottrustedfile"],
                    excludemodes=inputparams["fitfreqs"]["excludemodes"],
                    onlyradial=inputparams["fitfreqs"]["onlyradial"],
                    readratios=inputparams["fitfreqs"]["readratios"],
                    threepoint=inputparams["fitfreqs"]["threepoint"],
                    interp_ratios=inputparams["fitfreqs"]["interp_ratios"],
                    nrealizations=inputparams["fitfreqs"]["nrealizations"],
                    glitchfile=inputparams["fitfreqs"]["glitchfile"],
                    nsorting=inputparams["fitfreqs"]["nsorting"],
                    dnubias=inputparams["fitfreqs"]["dnubias"],
                )
                boxpriors: dict[str, core.PriorEntry] = {}
                for param in root.findall("default/priors/"):
                    param_name = param.tag
                    kwargs = {}

                    # Only include these if they are present
                    for key in ["min", "max", "abstol", "sigmacut"]:
                        if key in param.attrib:
                            kwargs[key] = float(param.attrib[key])
                    if param_name == "IMF":
                        param_name = "salpeter1955"
                    if param_name in imfs.PRIOR_FUNCTIONS:
                        continue
                    boxpriors[param_name] = core.PriorEntry(
                        kwargs=kwargs if kwargs else {}
                    )
                # Add dnufrac to priors
                if inputparams["fitfreqs"]["dnufrac"] is not None:
                    boxpriors["dnufrac"] = core.PriorEntry(
                        kwargs={"dnufit": inputparams["fitfreqs"]["dnufrac"]}
                    )
                # Add the constraint on the anchormode
                if inputparams["fitfreqs"]["dnufrac"] is not None:
                    pass
                    # boxpriors["anchormode"] = core.PriorEntry(
                    #    kwargs={"dnufit": inputparams["fitfreqs"]["dnufrac"]}
                    # )

                inferencesettings = core.InferenceSettings(
                    fitparams=fitparams,
                    seed=seed,
                    gridfile=gridfile,
                    gridid=gridid,
                    solarmodel=inputparams["solarmodel"],
                    solarvalues={
                        "numax": inputparams["numsun"],
                        "dnu": inputparams["dnusun"],
                    },
                    usebayw=bool(usebayw),
                    boxpriors=boxpriors,
                    imf=imf,
                    fit_surfacecorrected_dnu=inputparams["fitfreqs"][
                        "dnufit_in_ratios"
                    ],
                    dnuprior=inputparams["fitfreqs"]["dnuprior"],
                    seismicweights=inputparams["fitfreqs"]["seismicweights"],
                )
                plotconfig = core.PlotConfig(
                    nameinplot=inputparams["nameinplot"],
                    kielplots=inputparams["kielplots"],
                    cornerplots=inputparams["cornerplots"],
                    freqplots=inputparams["freqplots"],
                )
                try:
                    BASTA(
                        star=star,
                        inferencesettings=inferencesettings,
                        filepaths=filepaths,
                        runfiles=runfiles,
                        outputoptions=outputoptions,
                        plotconfig=plotconfig,
                    )
                #                     BASTA(
                #                         starid=starid,
                #                         gridfile=gridfile,
                #                         inputparams=inputparams,
                #                         gridid=gridid,
                #                         usebayw=usebayw,
                #                         usepriors=usepriors,
                #                         optionaloutputs=useoptoutput,
                #                         seed=seed,
                #                         debug=flag_debug,
                #                         verbose=verbose,
                #                         developermode=flag_developermode,
                #                         validationmode=flag_validationmode,
                #                     )
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
                    no_models(
                        starid,
                        filepaths,
                        runfiles,
                        outputoptions,
                        distancefilters,
                        f"Unhandled Error: {e}",
                    )

                fout.flush()
                ferr.flush()
                fwarn.flush()
                if fout_dist is not None:
                    fout_dist.flush()

    ascii_to_xml(
        output_paths["ascii"], output_paths["xml"], uncert=outputoptions.uncert
    )

    # Reset path and return
    if os.getcwd() != oldpath:
        os.chdir(oldpath)
