"""
Utilities for handling of XML files
"""
from builtins import str
from xml.dom import minidom
from xml.etree.ElementTree import Element, SubElement, tostring

import numpy as np

from basta.constants import parameters
from basta.constants import freqtypes


def _get_param(vals, names, param):
    """
    Short abbreviation for checking that requested parameter exists in input,
    and give meaningfull error if not.

    Parameters
    ----------
    vals : list
        List of inputted parameter values
    names : list
        Header with names of the inputted parameters
    param : str
        The requested parameter from the list

    Returns
    -------
    val : float
        The value extracted from the input
    """
    try:
        val = vals[np.where(names == param)[0][0]]
    except IndexError:
        raise IndexError("%s not found in input" % param) from None
    return val


def create_xmltag(
    main, params, paramvals, fitparams, distparams, freqparams, missingval, intpollim
):
    """
    Creates tags for xml input files

    Parameters
    ----------
    main : str
        Name of tag added to the xml file.
    params : str
        Name of the parameters in the ``asciifile`` defined in
        :func:`create_xml`
    paramvals : int or float or str
        Values of the parameters read from the ``asciifile`` in
        :func:`create_xml`
    fitparams : str
        Names of parameters to be fitted in the Bayesian analysis
    distparams : str
        Additional parameters for the distance calculation
    freqparams : str
        Additional parameters for the frequency calculation
    missingval : int or float or str
        Value used to replace missing values in the ascii file
    intpollim : list
        List of parameters that require values from the stars to set limits
        in interpolation.

    Returns
    -------
    main : str
        Name of the tag added to the xml file
    """
    params = np.asarray(params)
    starid = _get_param(paramvals, params, "starid")
    star = SubElement(main, "star", {"starid": str(starid)})

    # Special treatment of dnu and numax for frequency fitting
    nuset = {"dnu": False, "numax": False}

    # Loop over fitting parameters
    for param in fitparams:
        if param in freqtypes.alltypes:
            continue

        # If fitting dnu*, get and save the observed value as simply dnu
        param = "dnu" if "dnu" in param else param
        paramval = _get_param(paramvals, params, param)
        if isinstance(paramval, np.str_) and paramval != missingval:
            SubElement(star, param, {"value": str(paramval)})
        if (not isinstance(paramval, np.unicode_)) and (
            not np.isclose(paramval, missingval)
        ):
            paramerr = _get_param(paramvals, params, param + "_err")
            if not np.isclose(paramerr, missingval):
                SubElement(
                    star, param, {"value": str(paramval), "error": str(paramerr)}
                )
        if param in nuset:
            nuset[param] = True

    # Handle additional parameters (without errors)
    for param in distparams:
        paramval = _get_param(paramvals, params, param)
        if not np.isclose(paramerr, missingval):
            SubElement(star, param, {"value": str(paramval)})

    # Handle interpolation parameters
    for param in intpollim:
        out = {}
        gparam = "dnu" if "dnu" in param else param
        nucheck = not nuset[gparam] if gparam in nuset else False
        if "abstol" in intpollim[param] or nucheck:
            paramval = _get_param(paramvals, params, gparam)
            if not np.isclose(paramval, missingval):
                out["value"] = str(paramval)
            if nucheck:
                nuset[gparam] = True
        if "sigmacut" in intpollim[param]:
            paramerr = _get_param(paramvals, params, gparam + "_err")
            if not np.isclose(paramerr, missingval):
                out["error"] = str(paramerr)
        SubElement(star, gparam, out)

    # Handle the nottrustedfile object
    if freqparams and any(x in fitparams for x in freqtypes.alltypes):
        if "nottrustedfile" in freqparams:
            if isinstance(freqparams["nottrustedfile"], dict):
                if starid in freqparams["nottrustedfile"].keys():
                    ntf = freqparams["nottrustedfile"][starid]
                else:
                    ntf = "None"
                SubElement(star, "nottrustedfile", {"value": ntf})
            elif isinstance(freqparams["nottrustedfile"], str):
                # Error handling
                SubElement(
                    star, "nottrustedfile", {"value": freqparams["nottrustedfile"]}
                )
            else:
                raise ValueError("Nottrustedfile is neither a dict or a str")
        if not nuset["dnu"]:
            dnu = _get_param(paramvals, params, "dnu")
            SubElement(star, "dnu", {"value": str(dnu)})
        if not nuset["numax"]:
            numax = _get_param(paramvals, params, "numax")
            SubElement(star, "numax", {"value": str(numax)})

    return main


def ascii_to_xml(asciifile, outputfile, uncert="quantiles"):
    """
    Converts ascii output to XML output

    Parameters
    ----------
    asciifile : str
        Absolute path to ascii file
    outputfile : str
        Absolute path to the output XML file

    Returns
    -------
    pretty_xml : file
        Formatted XML file stored in the ``outputfile``
    """

    # Load ascii file
    results = np.genfromtxt(asciifile, dtype=None, encoding=None, names=True)
    if uncert == "quantiles":
        params = results.dtype.names[1::3]
    else:
        params = results.dtype.names[1::2]
    if results.ndim == 0:
        results = results.reshape(1)
    # Prepare list of params
    units, shortnames, remarks, _ = parameters.get_keys(params)

    # Make XML magic
    stars = Element("stars")
    for result in results:
        star = SubElement(stars, "star", {"starid": result[0].astype(str)})
        for param, unit, shortname, remark in zip(params, units, shortnames, remarks):
            if uncert == "quantiles":
                resdict = {
                    "value": str(result[param]),
                    "error_plus": str(result[param + "_errp"]),
                    "error_minus": str(result[param + "_errm"]),
                }
            else:
                resdict = {
                    "value": str(result[param]),
                    "error": str(result[param + "_err"]),
                }
            theparam = SubElement(
                star,
                param,
                resdict,
            )
            if unit is not None:
                theparam.attrib["unit"] = unit
            if remark is not None:
                SubElement(theparam, "remarks").text = remark

    pretty_xml = minidom.parseString(tostring(stars)).toprettyxml()

    with open(outputfile, "w") as output:
        print(pretty_xml, file=output)

    return pretty_xml
