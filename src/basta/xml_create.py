"""
Creation of XML input files
"""

from xml.dom import minidom
from xml.etree.ElementTree import Element, SubElement, tostring
import numpy as np
from basta.constants import sydsun as sydc
from basta.constants import freqtypes
from basta.utils_xml import create_xmltag


def generate_xml(
    gridfile: str,
    asciifile: str,
    outputpath: str,
    params: tuple[str, ...],
    fitparams: tuple[str, ...],
    outparams: tuple[str, ...],
    outputfile: str = "results.ascii",
    sunnumax: float = sydc.SUNnumax,
    sundnu: float = sydc.SUNdnu,
    solarmodel: bool = False,
    missingval: float | int = -999.999,
    centroid: str = "median",
    uncert: str = "quantiles",
    plotfmt: str = "png",
    nameinplot: bool = False,
    odea: tuple[str, str, str, str] | None = None,
    intpolparams: dict | None = None,
    bayweights: bool = True,
    priors: dict | None = None,
    overwriteparams: dict | None = None,
    freqparams: dict | None = None,
    glitchparams: dict | None = None,
    filters: tuple[str, ...] | None = None,
    dustframe: str | None = None,
    cornerplots: tuple[str, ...] | bool = False,
    kielplots: tuple[str, ...] | bool = False,
    freqplots: bool = False,
    optionaloutputs: bool = True,
    delimiter: str | None = None,
):
    """
    Converts an ascii table into an xml input file. Defines the properties
    of the fit including the type of grid and parameters to be fitted.

    Parameters
    ----------
    gridfile : str
        Absolute path of the grid used in the fit.
    asciifile: str
        Absolute path of ascii table of input values. The table should
        containing all ``params`` specified as an input (see below) and in
        the same order.
    outputpath : str
        Absolute path for all output of the code.
    params : tuple
        Names of the parameters to be read from the ``asciifile``.
    fitparams : tuple
        Names of the parameters to be fitted. These must exists in the list
        of parameters in the grid. These lists can be found in constants.
    outparams : tuple
        A tuple of parameters for which the results of the Bayesian analysis
        is printed in the ``outputfile``.
    outputfile : str
        Name of the output file where all ``outparams`` will be printed out
    sunnumax : float
        Value of the solar frequency of maximum power used in the scaling
        relations
    sundnu : float
        Value of the solar large frequency separation used in the scaling
        relations
    solarmodel : bool
        Activate solar scaling of asteroseismic quantities (dnu's).
    missingval : int or float or str
        Used to replace a missing value from the ``asciifile``. It can be an
        integer, float, or 'nan'.
    centroid : str
        Which centroid value of the posterior is to be reported as the
        result. The standard option is 'median' (the Bayesian 50'th
        percentile), the other option is 'mean'.
    uncert : str
        The reported type of uncertaities. The standard option is 'quantiles'
        reporting the Bayesian 16'th and 84'th percentiles of the posterior,
        while the other option is 'std' for the standard deviation.
    plotfmt : str
        Format of outputted plots, simply given directly to pyplot.savefig.
        Default is 'png' for quick figures. For detailed plots, 'pdf' is
        recommended.
    nameinplot : bool
        Toggle to include star identifier in plots, not simply in filename
        of plot.
    odea : tuple or None
        Specifies the input physics used to compute the grid.
        `o` : overshoot efficiency, 0.0 = no overshoot.
        `d` : microscopic diffusion, 0.0 = no diffusion.
        `e` : eta reimers for mass-loss, 0.0 = no mass-loss.
        `a` : alpha enhancement with respect to solar value, 0.0 = no alpha
        enhancement.
    intpolparams : dict
        Contains inputted settings for performing interpolation with BASTA.
        A single dictionary is given, but can be applied star-by-star.
    bayweights : bool
        Enable the usage of Bayesian weights across tracks or isochrones to
        properly take into account spacing in the creation of the grid and
        evolutionary speed.
    priors : dict
        A dictionary with the name of priors to be considered in the fitting
        {'IMF': 'string', 'param1': {values}, 'param2': {values}, ...}
        which defines the IMF used, and sets the absolute value of the
        tolerances for the selection of models in the grid to compute the
        likelihood. Accepted values are 'min', 'max', and 'abstol'. If not
        specified, all models in the grid are evaluated.
    overwriteparams : dict
        A dictionary in the format
        {'param1' : (value1, uncert1), 'param2': (value2, uncert2), ...}
        used to set a value and uncertainty for all stars fitted. It will
        overwrite the input parameters for the individual stars!
    freqparams : dict
        A dictionary containing the input for fitting individual frequencies
        or ratios.
    grparams : dict
        A dictionary containing control options for fitting glitches along
        with ratios.
    filters : tuple of strings
        If calculating distances to stars, specify photometric filters
    dustframe : str
        Type of reference frame for the dustmap. Possible options

        `galactic` : The default coordinate system. Input coordinates are
        expected in galactic coordinates longitude (l) and latitude (b).

        `icrs` : Input coordinates are expected in celestial right
        ascension (ra) and declination (dec).
    cornerplots : tuple
        Names of parameters to be included in the output plots. These can be
        any of the available parameters of the grid used, and can be found
        in :func:`make_basti` and :func:`make_garstec`.
    kielplots : tuple
        Names of fitted parameters to be indicated in the Kiel diagram. If
        all fitparams are wanted plotted, set to True. False by default.
    freqplots : bool
        Whether or not to genereate echelle- and ratio-diagrams when fitting
        frequencies. False by default.
    optionaloutputs : bool
        Defines if the additional output files are stored
    delimiter : str or None
        Inputted delimiter if asciifile uses a special delimiter not easily
        recognised by numpy.genfromtxt
    """
    inp = np.genfromtxt(
        asciifile, dtype=None, names=params, encoding=None, delimiter=delimiter
    )
    if inp.ndim == 0:
        inp = inp.reshape(1, -1)[0]

    # If missingval is nan, redefine it
    if missingval == "nan":
        missingval = -999.999
        for i, inp1 in enumerate(inp):
            for j, inp2 in enumerate(inp1):
                try:
                    if np.isnan(inp2):
                        inp[i][j] = missingval
                except Exception:
                    pass

    # Create main element stars
    main = Element("stars")
    default = SubElement(main, "default")

    # Add library path to <default>
    SubElement(default, "library", {"path": gridfile})

    # Add output path to <default>
    if not outputpath.endswith("/"):
        outputpath += "/"
    SubElement(default, "output", {"path": outputpath})

    # Add output file path (relative to output path) to <default>
    SubElement(default, "outputfile", {"value": outputfile})

    # Add solar dnu and numax to <default>
    SubElement(default, "solardnu", {"value": str(sundnu)})
    SubElement(default, "solarnumax", {"value": str(sunnumax)})

    # Add solar model toggle to <default>
    if solarmodel is None or isinstance(solarmodel, bool):
        SubElement(default, "solarmodel", {"value": str(solarmodel)})
    else:
        SubElement(default, "solarmodel", {"value": solarmodel})

    # Add missingvalue to <default>
    SubElement(default, "missingval", {"value": str(missingval)})

    # Add output stats types (centroid and undert) to <default>
    if centroid is not None:
        SubElement(default, "centroid", {"value": str(centroid)})
    if uncert is not None:
        SubElement(default, "uncert", {"value": str(uncert)})

    # Add plotformat to <default> if user provided
    if plotfmt is not None:
        SubElement(default, "plotfmt", {"value": str(plotfmt)})

    # Add nameinplot to <default> if set by user
    if nameinplot:
        SubElement(default, "nameinplot", {"value": "True"})

    # Add ove, eta, diffusion and alphaFe to <basti> if isochrone
    if odea:
        # Add <basti> to <default>
        bastielement = SubElement(default, "basti")
        oeaname = ("ove", "dif", "eta", "alphaFe")
        for i, par in enumerate(odea):
            SubElement(bastielement, oeaname[i], {"value": str(par)})

    # Add subelement <interpolation> to <default> (if defiend)
    intpollim = {}
    if intpolparams:
        intpolelement = SubElement(default, "interpolation")
        SubElement(intpolelement, "method", intpolparams["method"])
        if "name" in intpolparams:
            SubElement(intpolelement, "name", {"value": intpolparams["name"]})
        if "trackresolution" in intpolparams:
            trackres = intpolparams["trackresolution"]
            for tag, val in trackres.items():
                if isinstance(val, (float, int)):
                    trackres[tag] = str(val)
            SubElement(intpolelement, "trackresolution", trackres)
        if "gridresolution" in intpolparams:
            gridres = intpolparams["gridresolution"]
            for tag, val in gridres.items():
                if isinstance(val, (float, int)):
                    gridres[tag] = str(val)
            if "resolution" in gridres:
                for restag, resval in gridres["resolution"].items():
                    gridres[restag] = str(resval)
                del gridres["resolution"]
            SubElement(intpolelement, "gridresolution", gridres)
        if "limits" in intpolparams:
            limelement = SubElement(intpolelement, "limits")
            for name, item in intpolparams["limits"].items():
                for n, v in item.items():
                    item[n] = str(v)
                SubElement(limelement, name, item)
                if name not in fitparams:
                    rules = np.asarray(list(intpolparams["limits"][name].keys()))
                    hits = np.argwhere(
                        [x in ["sigmacut", "abstol"] for x in rules]
                    ).flatten()
                    if len(hits):
                        intpollim[name] = list(rules[hits])

    # Add subelement <bayesianweights> to <default>
    if isinstance(bayweights, (bool, str)):
        SubElement(default, "bayesianweights", {"value": str(bayweights)})

    # Add subelement <fitparams to <default>
    fitelement = SubElement(default, "fitparams")
    if isinstance(fitparams, str):
        fitparams = [fitparams]
    for param in fitparams:
        paramdic = {}
        SubElement(fitelement, param, paramdic)

    # Add subelement priors to <default> (if any priors are included)
    if priors:
        priorelement = SubElement(default, "priors")
        for param in priors.keys():
            if param == "IMF":
                SubElement(priorelement, priors[param])
            elif param == "dif":
                # Restrict diffusion in mixed grids
                if priors[param]:
                    dif_switch = {"min": "0.5"}
                else:
                    dif_switch = {"max": "0.5"}
                SubElement(priorelement, param, dif_switch)
            else:
                # Catch new-style IMF-type priors
                try:
                    SubElement(priorelement, param, priors[param])
                except TypeError:
                    SubElement(priorelement, param)

    # Add subelement overwriteparams to <default> (if any global parameters)
    if overwriteparams:
        globalelement = SubElement(default, "overwriteparams")
        for param in overwriteparams:
            paramdic = {}
            if param == "phase":
                paramdic["value"] = overwriteparams[param]
            else:
                paramdic["value"] = str(overwriteparams[param][0])
                paramdic["error"] = str(overwriteparams[param][1])
            SubElement(globalelement, param, paramdic)

    # Add subelement freqparams to <default> (if any)
    if freqparams and any(x in fitparams for x in freqtypes.alltypes):
        freqelement = SubElement(default, "freqparams")
        for param in freqparams:
            # The following are handled in create_xmltag
            if param in ["excludemodes", "nottrustedfile", "onlyradial", "onlyls"]:
                continue
            SubElement(freqelement, param, {"value": str(freqparams[param])})

    # Add subelement grparams to <default> if specified
    if glitchparams and any(x in fitparams for x in freqtypes.glitches):
        glitchelement = SubElement(default, "glitchparams")
        for param in glitchparams:
            SubElement(glitchelement, param, {"value": str(glitchparams[param])})

    # We need to check these before handling distance input
    if isinstance(cornerplots, (str, bool)) and len(cornerplots):
        cornerplots = [str(cornerplots)]
    if isinstance(outparams, (str, bool)) and len(outparams):
        outparams = [str(outparams)]

    # Handle distance related input
    if (
        ("parallax" in fitparams)
        or ("distance" in outparams)
        or ("distance" in cornerplots)
    ):
        # Convert to tuple if the user specified only one filter as a sting
        if isinstance(filters, str):
            filters = (filters,)

        if len(filters) == 0:
            raise ValueError("No filters were given for parallax/distance fitting")

        # Add to <default>
        distanceelement = SubElement(default, "distanceInput")
        BCfilterelement = SubElement(distanceelement, "filters")
        for param in filters:
            SubElement(BCfilterelement, param)
        SubElement(distanceelement, "dustframe", {"value": str(dustframe)})

        # Add coordinate system to the individual targets
        if dustframe == "galactic":
            distparams = ("lat", "lon")
        elif dustframe in ["icrs"]:
            distparams = ("RA", "DEC")
        else:
            print("Illegal dustframe specified! Not adding coordinates.")
            distparams = ()

        # If EBV is supplied, read for each star
        if "EBV" in params:
            distparams += ("EBV",)

        # Add magnitudes to list of parameters
        starparams = fitparams + filters
    else:
        starparams = fitparams
        distparams = ()

    # Add subelement <cornerplots> to <default>
    cornerplotselement = SubElement(default, "cornerplots")
    for param in cornerplots:
        SubElement(cornerplotselement, param)

    # Add subelement <kielplots> to <default>
    kielplotselement = SubElement(default, "kielplots")
    if isinstance(kielplots, bool):
        if kielplots:
            SubElement(kielplotselement, str(kielplots))
    else:
        if isinstance(kielplots, str) and len(kielplots):
            kielplots = [kielplots]
        for param in kielplots:
            SubElement(kielplotselement, param)

    # Add subelement <freqplots> to <default>
    freqplotselement = SubElement(default, "freqplots")
    if isinstance(freqplots, (bool, str)) and freqplots:
        SubElement(freqplotselement, str(freqplots))
    elif freqplots == "True":
        SubElement(freqplotselement, "True")
    elif isinstance(freqplots, (list, tuple)):
        for plot in freqplots:
            SubElement(freqplotselement, plot)

    # Add subelement <outparams> to <default>
    outparamselement = SubElement(default, "outparams")
    for param in tuple(outparams):
        SubElement(outparamselement, param)

    # Add optional output files to <default>
    optoutputelement = SubElement(default, "optionaloutputfiles")
    if isinstance(optionaloutputs, (bool, str)):
        SubElement(optoutputelement, str(optionaloutputs))

    # Create xml tags for all the targets in the input file
    for i in range(len(inp)):
        main = create_xmltag(
            main,
            params,
            inp[i],
            starparams,
            distparams,
            freqparams,
            missingval,
            intpollim,
        )

    # Make the XML into a string:
    xml = tostring(main)

    # Make the XML pretty
    reparsed = minidom.parseString(xml)
    pretty_xml = reparsed.toprettyxml()
    return pretty_xml
