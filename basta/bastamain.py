"""
Main module for running BASTA analysis
"""
import os
import gc
import sys
import time
from copy import deepcopy

import h5py
import numpy as np
from tqdm import tqdm

from basta import freq_fit, stats, process_output, priors, distances, plot_seismic
from basta import utils_seismic as su
from basta import utils_general as util
from basta._version import __version__
from basta import fileio as fio
from basta.constants import freqtypes

# Import matplotlib after other plotting modules for proper setup
# --> Here in main it is only used for clean-up
import matplotlib.pyplot as plt


# Custom exception
class LibraryError(Exception):
    pass


# The main driver!
def BASTA(
    starid,
    gridfile,
    idgrid=None,
    usebayw=True,
    usepriors=(None,),
    inputparams=False,
    optionaloutputs=False,
    seed=None,
    debug=False,
    verbose=False,
    experimental=False,
    validationmode=False,
):
    """
    The BAyesian STellar Algorithm (BASTA).
    (c) 2021, The BASTA Team

    For a description of how to use BASTA, please explore the documentation (https://github.com/BASTAcode/BASTA).
    This function is typically called by :func:'xmltools.run_xml()'

    Parameters
    ----------
    starid : str
        Unique identifier for this target.
    gridfile : str
        Path and name of the hdf5 file containing the isochrones or tracks
        used in the fitting
    idgrid : bool or tuple
        For isochrones, a tuple containing (overshooting [f],
        diffusion [0 or 1], mass loss [eta], alpha enhancement [0.0 ... 0.4])
        used for selecting a science case / path in the library.
    usebayw : bool or tuple
        If True, bayesian weights are applied in the computation of the
        likelihood. See :func:`interpolation_helpers.bay_weights()` for details.
    usepriors : tuple
        Tuple of strings containing name of priors (e.g., an IMF).
        See :func:`priors` for details.
    inputparams : dict
        Dictionary containing most information needed, e.g. controls, fitparameters,
        output options.
    optionaloutputs : bool, optional
        If True, saves a 'json' file for each star with the global results and the PDF.
    seed : int, optional
        The seed of randomness
    debug : bool, optional
        Activate additional output for debugging (for developers)
    verbose : bool, optional
        Activate a lot (!) of additional output (for developers)
    experimental : bool, optional
        Activate experimental features (for developers)
    validationmode : bool, optional
        Activate validation mode features (for validation purposes only)
    """
    # Set output directory and filenames
    t0 = time.localtime()
    outputdir = inputparams.get("output")
    outfilename = os.path.join(outputdir, starid)

    # Start the log
    stdout = sys.stdout
    sys.stdout = util.Logger(outfilename)

    # Pretty printing a header
    linelen = 88
    print(linelen * "=")
    util.prt_center("BASTA", linelen)
    util.prt_center("The BAyesian STellar Algorithm", linelen)
    print()
    util.prt_center("Version {0}".format(__version__), linelen)
    print()
    util.prt_center("(c) 2021, The BASTA Team", linelen)
    util.prt_center("https://github.com/BASTAcode/BASTA", linelen)
    print(linelen * "=")
    print("\nRun started on {0}.\n".format(time.strftime("%Y-%m-%d %H:%M:%S", t0)))
    if experimental:
        print("RUNNING WITH EXPERIMENTAL FEATURES ACTIVATED!\n")
    print(f"Random numbers initialised with seed {seed}")

    # Load the desired grid and obtain information from the header
    Grid = h5py.File(gridfile, "r")
    try:
        gridtype = Grid["header/library_type"][()]
        gridver = Grid["header/version"][()]
        gridtime = Grid["header/buildtime"][()]

        # Allow for usage of both h5py 2.10.x and 3.x.x
        # --> If things are encoded as bytes, they must be made into standard strings
        if isinstance(gridtype, bytes):
            gridtype = gridtype.decode("utf-8")
            gridver = gridver.decode("utf-8")
            gridtime = gridtime.decode("utf-8")
    except KeyError:
        print("Error: Some information is missing in the header of the grid!")
        print(
            "Please check the entries in the header! It must include all",
            "of the following:\n * header/library_type\n * header/version",
            "\n * header/buildtime",
        )
        Grid.close()
        sys.exit(1)

    # Verbose information on the grid file
    print("\nFitting star id: {0} .".format(starid))
    print("* Using the grid '{0}' of type '{1}'.".format(gridfile, gridtype))
    print(
        "  - Grid built with BASTA version {0}, timestamp: {1}.".format(
            gridver, gridtime
        )
    )

    # Check type of grid (isochrones/tracks) and set default grid path
    if "tracks" in gridtype.lower():
        entryname = "tracks"
        defaultpath = "grid/"
        difsolarmodel = None
    elif "isochrones" in gridtype.lower():
        entryname = "isochrones"
        if idgrid:
            difsolarmodel = int(idgrid[1])
            defaultpath = "ove={0:.4f}/dif={1:.4f}/eta={2:.4f}/alphaFe={3:.4f}/".format(
                idgrid[0], idgrid[1], idgrid[2], idgrid[3]
            )
        else:
            print(
                "Unable to construct path for science case."
                + " Probably missing (ove, dif, eta, alphaFe) in input!"
            )
            raise LibraryError

    else:
        raise OSError(
            "Gridtype {} not supported, only 'tracks' and 'isochrones'!".format(
                gridtype
            )
        )

    # Read available weights if not provided by the user
    if not isinstance(usebayw, tuple):
        if usebayw:
            skipweights = False
            try:
                grid_weights = [
                    x.decode("utf-8") for x in list(Grid["header/active_weights"])
                ]

                # Always append the special weight for isochrones/tracks
                if "isochrones" in gridtype.lower():
                    grid_weights.append("massini")
                else:
                    grid_weights.append("age")
            except KeyError:
                grid_weights = ["massini", "FeHini", "age"]
                print("WARNING: No Bayesian weights specified in grid file!\n")
            bayweights = tuple(grid_weights)
        else:
            skipweights = True
    else:
        bayweights = deepcopy(usebayw)
        skipweights = False

    # Check for the weights which requires special care
    # --> dmass and dage: dweights
    if not skipweights:
        if "isochrones" in gridtype.lower() and "massini" in bayweights:
            apply_dweights = True
            bayweights = list(bayweights)
            bayweights.remove("massini")
            bayweights = tuple(x + "_weight" for x in bayweights)
        elif "tracks" in gridtype.lower() and "age" in bayweights:
            apply_dweights = True
            bayweights = list(bayweights)
            bayweights.remove("age")
            bayweights = tuple(x + "_weight" for x in bayweights)
        else:
            apply_dweights = False
    else:
        apply_dweights = False

    # Get list of parameters
    cornerplots = inputparams["cornerplots"]
    outparams = inputparams["asciiparams"]
    allparams = list(np.unique(cornerplots + outparams))

    #
    # *** BEGIN: Preparation for distance fitting ***
    #
    # Special case if assuming gaussian magnitudes
    if "gaussian_magnitudes" in inputparams:
        use_gaussian_priors = inputparams["gaussian_magnitudes"]
    else:
        use_gaussian_priors = False

    # Add magnitudes and colors to fitparams if fitting distance
    inputparams = distances.add_absolute_magnitudes(
        inputparams,
        debug=debug,
        outfilename=outfilename,
        use_gaussian_priors=use_gaussian_priors,
    )

    # If keyword present, add individual filters
    if "distance" in allparams:
        allparams = list(
            np.unique(allparams + inputparams["distanceparams"]["filters"])
        )
        allparams.remove("distance")
    #
    # *** END: Preparation for distance fitting ***
    #

    # Create list of all available input parameters
    fitparams = inputparams.get("fitparams")
    fitfreqs = inputparams["fitfreqs"]
    distparams = inputparams.get("distanceparams", False)
    limits = inputparams.get("limits")

    # Scale dnu and numax using a solar model or default solar values
    inputparams = su.solar_scaling(Grid, inputparams, diffusion=difsolarmodel)

    # Prepare asteroseismic quantities if required
    if fitfreqs["active"]:
        if not all(x in freqtypes.alltypes for x in fitfreqs["fittypes"]):
            print(fitfreqs["fittypes"])
            raise ValueError("Unrecognized frequency fitting parameters!")

        # Obtain/calculate all frequency related quantities
        (
            obskey,
            obs,
            obsfreqdata,
            obsfreqmeta,
            obsintervals,
        ) = su.prepare_obs(inputparams, verbose=verbose, debug=debug)

    # Check if grid is interpolated
    try:
        Grid["header/interpolation_time"][()]
    except KeyError:
        grid_is_intpol = False
    else:
        grid_is_intpol = True

    # Check if any specified limit in prior is in header, and can be used to
    # skip computation of models, in order to speed up computation
    if "tracks" in gridtype.lower():
        headerpath = "header/"
    elif "isochrones" in gridtype.lower():
        headerpath = "header/" + defaultpath
        if "FeHini" in limits:
            del limits["FeHini"]
            print("Warning: Dropping prior in FeHini, " + "redundant for isochrones!")
    else:
        headerpath = False

    # Gridcut dictionary containing cutting parameters
    gridcut = {}
    if headerpath:
        keys = Grid[headerpath].keys()
        # Compare keys in header and limits
        for key in keys:
            if key in limits:
                gridcut[key] = limits[key]
                # Remove key from limits, to avoid redundant second check
                del limits[key]

    # Apply the cut on header parameters with a special treatment of diffusion
    if headerpath and gridcut:
        print("\nCutting in grid based on sampling parameters ('gridcut'):")
        noofskips = [0, 0]
        for cpar in gridcut:
            if cpar != "dif":
                print("* {0}: {1}".format(cpar, gridcut[cpar]))

        # Diffusion switch printed in a more readable format
        if "dif" in gridcut:
            # As gridcut['dif'] is always either [-inf, 0.5] or [0.5, inf]
            # The location of 0.5 can be used as the switch
            switch = np.where(np.array(gridcut["dif"]) == 0.5)[0][0]
            print(
                "* Only considering tracks with diffusion turned",
                "{:s}!".format(["on", "off"][switch]),
            )

    # Print fitparams
    print("\nFitting information:")
    print("* Fitting parameters with values and uncertainties:")
    for fp in fitparams.keys():
        if fp in ["numax", "dnuSer", "dnuscal", "dnuAsf"]:
            fpstr = "{0} (solar units)".format(fp)
        else:
            fpstr = fp
        print("  - {0}: {1}".format(fpstr, fitparams[fp]))

    # Fitting info: Frequencies
    if fitfreqs["active"]:
        if "glitches" in fitfreqs["fittypes"]:
            print("* Fitting of frequency glitches activated!")
        elif "freqs" in fitfreqs["fittypes"]:
            print("* Fitting of individual frequencies activated!")
        elif any(x in freqtypes.rtypes for x in fitfreqs["fittypes"]):
            print(
                "* Fitting of frequency ratios {0} activated!".format(
                    ", ".join(fitfreqs["fittypes"])
                )
            )
            if "r010" in fitfreqs["fittypes"]:
                print(
                    "  - WARNING: Fitting r01 and r10 simultaniously results in overfitting, and is thus not recommended!"
                )
        elif any(x in freqtypes.epsdiff for x in fitfreqs["fittypes"]):
            print(
                "* Fitting of epsilon differences {0} activated!".format(
                    ", ".join(fitfreqs["fittypes"])
                )
            )

        if fitfreqs["bexp"] is not None:
            bexpstr = " with b = {0}".format(fitfreqs["bexp"])
        else:
            bexpstr = ""
        if fitfreqs["correlations"]:
            corrstr = "Yes"
        else:
            corrstr = "No"
        print(
            "  - Constraining lowest l = 0 (n = {0}) with f = {1:.3f} +/-".format(
                obskey[1, 0], obs[0, 0]
            ),
            "{0:.3f} muHz to within {1:.1f} % of dnu ({2:.3f} microHz)".format(
                obs[1, 0],
                fitfreqs["dnufrac"] * 100,
                fitfreqs["dnufrac"] * fitfreqs["dnufit"],
            ),
        )
        print("  - Correlations: {0}".format(corrstr))
        print("  - Frequency input data: {0}".format(fitfreqs["freqfile"]))
        print(
            "  - Frequency input data (list of ignored modes): {0}".format(
                fitfreqs["nottrustedfile"]
            )
        )
        print("  - Surface effect correction: {0}{1}".format(fitfreqs["fcor"], bexpstr))
        print("  - Value of dnu: {0:.3f} microHz".format(fitfreqs["dnufit"]))
        print("  - Value of numax: {0:.3f} microHz".format(fitfreqs["numax"]))
        print("  - Weighting scheme: {0}".format(fitfreqs["seismicweights"]["weight"]))

    # Fitting info: Distance
    if distparams:
        if ("parallax" in distparams) and "distance" in inputparams["asciiparams"]:
            print("* Parallax fitting and distance inference activated!")
        elif "parallax" in distparams:
            print("* Parallax fitting activated!")
        elif "distance" in inputparams["asciiparams"]:
            print("* Distance inference activated!")

        if distparams["dustframe"] == "icrs":
            print(
                "  - Coordinates (icrs): RA = {0}, DEC = {1}".format(
                    distparams["RA"], distparams["DEC"]
                )
            )
        elif distparams["dustframe"] == "galactic":
            print(
                "  - Coordinates (galactic): lat = {0}, lon = {1}".format(
                    distparams["lat"], distparams["lon"]
                )
            )

        print("  - Filters (magnitude value and uncertainty): ")
        for filt in distparams["filters"]:
            print(
                "    + {0}: [{1}, {2}]".format(
                    filt, distparams["m"][filt], distparams["m_err"][filt]
                )
            )

        if "parallax" in distparams:
            print("  - Parallax: {0}".format(distparams["parallax"]))

    # Fitting info: Phase
    if "phase" in inputparams:
        print("* Fitting evolutionary phase!")

    # Print additional info on given input
    print("\nAdditional input parameters and settings in alphabetical order:")
    noprint = [
        "asciioutput",
        "asciioutput_dist",
        "distanceparams",
        "dnufit",
        "dnufrac",
        "erroutput",
        "fcor",
        "fitfreqs",
        "fitparams",
        "limits",
        "magnitudes",
        "nottrustedfile",
        "numax",
        "warnoutput",
    ]
    for ip in sorted(inputparams.keys()):
        if ip not in noprint:
            print("* {0}: {1}".format(ip, inputparams[ip]))

    # Print weights and priors
    print("\nWeights and priors:")
    if "isochrones" in gridtype.lower() and apply_dweights:
        gtname = "isochrones"
        dwname = "mass"
    elif "tracks" in gridtype.lower() and apply_dweights:
        gtname = "tracks"
        dwname = "age"
    else:
        dwname = ""
    print("* Bayesian weights:")
    print("  - Along {0}: {1}".format(gtname, dwname))
    print(
        "  - Between {0}: {1}".format(
            gtname, ", ".join([q.split("_")[0] for q in bayweights])
        )
    )
    print("* Flat, constrained priors and ranges:")
    for lim in limits.keys():
        print("  - {0}: {1}".format(lim, limits[lim]))
    print("* Additional priors (IMF): {0}".format(", ".join(usepriors)))

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Start likelihood computation
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    # Two loop cases for the outer "metal" loop:
    # - For Garstec and MESA grids, the top level contains only one element ("tracks").
    #   Here the outer loop will run only once.
    # - For BaSTI, the top level is a list of metallicities and the outer loop will run
    #   multiple times.
    metal = util.list_metallicities(Grid, defaultpath, inputparams, limits)

    # We assume Garstec grid structure. The path will be updated in the loop for BaSTI
    group_name = defaultpath + "tracks/"

    # Before running the actual loop, all tracks/isochrones are counted to better
    # estimate the progress.
    trackcounter = 0
    for FeH in metal:
        if "grid" not in defaultpath:
            group_name = defaultpath + "FeH=" + format(FeH, ".4f") + "/"

        group = Grid[group_name]
        trackcounter += len(group.items())

    # Prepare the main loop
    shapewarn = False
    warn = True
    selectedmodels = {}
    noofind = 0
    noofposind = 0
    print(
        "\n\nComputing likelihood of models in the grid ({0} {1}) ...".format(
            trackcounter, entryname
        )
    )

    # Use a progress bar (with the package tqdm; will write to stderr)
    pbar = tqdm(total=trackcounter, desc="--> Progress", ascii=True)
    for FeH in metal:
        if "grid" not in defaultpath:
            group_name = defaultpath + "FeH=" + format(FeH, ".4f") + "/"

        group = Grid[group_name]
        for noingrid, (name, libitem) in enumerate(group.items()):
            # Update progress bar in the start of the loop to count skipped tracks
            pbar.update(1)

            # For grid with interpolated tracks, skip tracks flagged as empty
            if grid_is_intpol:
                if libitem["FeHini_weight"][()] < 0:
                    continue

            # Check for diffusion
            if "dif" in inputparams:
                if int(round(libitem["dif"][0])) != int(
                    round(float(inputparams["dif"]))
                ):
                    continue

            # Check if mass or age is in limits to efficiently skip
            if "grid" not in defaultpath:
                param, val = name.split("=")
                if param == "mass":
                    param += "ini"
                if param in limits:
                    # if age or massini is outside limits, skip this iteration
                    if float(val) < limits[param][0] or float(val) > limits[param][1]:
                        continue

            # Check if track should be skipped from cut in initial parameters
            if gridcut:
                noofskips[1] += 1
                docut = False
                for param in gridcut:
                    if "tracks" in gridtype.lower():
                        value = Grid[headerpath][param][noingrid]
                    elif "isochrones" in gridtype.lower():
                        # For isochrones, metallicity is already cut from the
                        # metal list and lookup of age is simplest and fastest
                        if param == "age":
                            value = float(name[4:])
                    # If value is outside cut limits, skip looking at the rest
                    if not (value >= gridcut[param][0] and value <= gridcut[param][1]):
                        docut = True
                        continue
                # Actually skip this iteration
                if docut:
                    noofskips[0] += 1
                    continue

            # Check which models have parameters within limits
            index = np.ones(len(libitem["age"][:]), dtype=bool)
            for param in limits:
                index &= libitem[param][:] >= limits[param][0]
                index &= libitem[param][:] <= limits[param][1]

            # Check which models have phases as specified
            if "phase" in inputparams:

                # Mapping of verbose input phases to internal numbers
                pmap = {
                    "pre-ms": 1,
                    "solar": 2,
                    "rgb": 3,
                    "flash": 4,
                    "clump": 5,
                    "agb": 6,
                }

                # Fitting multiple phases or just one
                if isinstance(inputparams["phase"], tuple):
                    iphases = [pmap[ip] for ip in inputparams["phase"]]

                    phaseindex = libitem["phase"][:] == iphases[0]
                    for j in range(1, len(iphases)):
                        phaseindex |= libitem["phase"][:] == iphases[j]
                    index &= phaseindex
                else:
                    iphase = pmap[inputparams["phase"]]
                    index &= libitem["phase"][:] == iphase

            # Check which models have l=0, lowest n within tolerance
            if fitfreqs["active"]:
                indexf = np.zeros(len(index), dtype=bool)
                for ind in np.where(index)[0]:
                    rawmod = libitem["osc"][ind]
                    rawmodkey = libitem["osckey"][ind]
                    mod = su.transform_obj_array(rawmod)
                    modkey = su.transform_obj_array(rawmodkey)
                    modkeyl0, modl0 = su.get_givenl(l=0, osc=mod, osckey=modkey)
                    # As mod is ordered (stacked in increasing n and l),
                    # then [0, 0] is the lowest l=0 mode
                    same_n = modkeyl0[1, :] == obskey[1, 0]
                    cl0 = modl0[0, same_n]
                    if len(cl0) > 1:
                        cl0 = cl0[0]

                    # Note to self: This code is pretty hard to read...
                    if (
                        cl0
                        >= (
                            obs[0, 0]
                            - min(
                                (fitfreqs["dnufrac"] / 2 * fitfreqs["dnufit"]),
                                (3 * obs[1, 0]),
                            )
                        )
                    ) and (cl0 - obs[0, 0]) <= (
                        fitfreqs["dnufrac"] * fitfreqs["dnufit"]
                    ):
                        indexf[ind] = True
                index &= indexf

            # If any models are within tolerances, calculate statistics
            if np.any(index):
                chi2 = np.zeros(index.sum())
                paramvalues = {}
                for param in fitparams:
                    paramvals = libitem[param][index]
                    chi2 += (
                        (paramvals - fitparams[param][0]) / fitparams[param][1]
                    ) ** 2.0
                    if param in allparams:
                        paramvalues[param] = paramvals

                # Add parameters not in fitparams
                for param in allparams:
                    if param not in fitparams:
                        paramvalues[param] = libitem[param][index]

                # Frequency (and/or ratio and/or glitch) fitting
                if fitfreqs["active"]:
                    for indd, ind in enumerate(np.where(index)[0]):
                        chi2_freq, warn, shapewarn = stats.chi2_astero(
                            obskey,
                            obs,
                            obsfreqmeta,
                            obsfreqdata,
                            obsintervals,
                            libitem,
                            ind,
                            fitfreqs,
                            warnings=warn,
                            shapewarn=shapewarn,
                            debug=debug,
                            verbose=verbose,
                        )
                        chi2[indd] += chi2_freq

                # Bayesian weights (across tracks/isochrones)
                logPDF = 0.0
                if debug:
                    bayw = 0.0
                    magw = 0.0
                    IMFw = 0.0
                if not skipweights:
                    for weight in bayweights:
                        logPDF += util.inflog(libitem[weight][()])
                        if debug:
                            bayw += util.inflog(libitem[weight][()])

                    # Within a given track/isochrone; these are called dweights
                    if "isochrones" in gridtype.lower() and apply_dweights:
                        logPDF += util.inflog(libitem["dmass"][index])
                        if debug:
                            bayw += util.inflog(libitem["dmass"][index])
                    elif "tracks" in gridtype.lower() and apply_dweights:
                        logPDF += util.inflog(libitem["dage"][index])
                        if debug:
                            bayw += util.inflog(libitem["dage"][index])

                # Multiply by absolute magnitudes, if present
                for f in inputparams["magnitudes"]:
                    mags = inputparams["magnitudes"][f]["prior"]
                    absmags = libitem[f][index]
                    interp_mags = mags(absmags)

                    logPDF += util.inflog(interp_mags)
                    if debug:
                        magw += util.inflog(interp_mags)

                # Multiply priors into the weight
                for prior in usepriors:
                    logPDF += util.inflog(getattr(priors, prior)(libitem, index))
                    if debug:
                        IMFw += util.inflog(getattr(priors, prior)(libitem, index))

                # Calculate likelihood from weights, priors and chi2
                # PDF = weights * np.exp(-0.5 * chi2)
                logPDF -= 0.5 * chi2
                if debug and verbose:
                    print(
                        "DEBUG: Mass with nonzero likelihood:",
                        libitem["massini"][index][~np.isinf(logPDF)],
                    )

                noofind += len(logPDF)
                noofposind += np.count_nonzero(~np.isinf(logPDF))
                if debug and verbose:
                    print(
                        "DEBUG: Index found: {0}, {1}".format(
                            group_name + name, ~np.isinf(logPDF)
                        )
                    )
                if debug:
                    selectedmodels[group_name + name] = stats.priorlogPDF(
                        index, logPDF, chi2, bayw, magw, IMFw
                    )
                else:
                    selectedmodels[group_name + name] = stats.Trackstats(
                        index, logPDF, chi2
                    )
            else:
                if debug and verbose:
                    print(
                        "DEBUG: Index not found: {0}, {1}".format(
                            group_name + name, ~np.isinf(logPDF)
                        )
                    )
        # End loop over isochrones/tracks
        #######################################################################
    # End loop over metals
    ###########################################################################
    pbar.close()
    print(
        "Done! Computed the likelihood of {0} models,".format(str(noofind)),
        "found {0} models with non-zero likelihood!\n".format(str(noofposind)),
    )
    if gridcut:
        print(
            "(Note: The use of 'gridcut' skipped {0} out of {1} {2})\n".format(
                noofskips[0], noofskips[1], gridtype
            )
        )

    # Raise possible warnings
    if shapewarn:
        print(
            "Warning: Found models with fewer frequencies than observed!",
            "These were set to zero likelihood!",
        )
        if "intpol" in gridfile:
            print(
                "This is probably due to the interpolation scheme. Lookup",
                "`interpolate_frequencies` for more details.",
            )
    if noofposind == 0:
        fio.no_models(starid, inputparams, "No models found")
        return

    # Print a header to signal the start of the output section in the log
    print("\n*****************************************")
    print("**                                     **")
    print("**   Output and results from the fit   **")
    print("**                                     **")
    print("*****************************************\n")

    # Find and print highest likelihood model info
    maxPDF_path, maxPDF_ind = stats.get_highest_likelihood(
        Grid, selectedmodels, outparams
    )
    stats.get_lowest_chi2(Grid, selectedmodels, outparams)

    # Generate posteriors of ascii- and plotparams
    # --> Print posteriors to console and log
    # --> Generate corner plots
    # --> Generate Kiel diagrams
    print("\n\nComputing posterior distributions for the requested output parameters!")
    print("==> Summary statistics printed below ...\n")
    process_output.compute_posterior(
        starid=starid,
        selectedmodels=selectedmodels,
        Grid=Grid,
        inputparams=inputparams,
        outfilename=outfilename,
        gridtype=gridtype,
        debug=debug,
        experimental=experimental,
        validationmode=validationmode,
    )

    # Make frequency-related plots
    freqplots = inputparams.get("freqplots")
    if fitfreqs["active"] and len(freqplots):
        # Check which plots to create
        allfplots = freqplots[0] == True
        if any(x == "echelle" for x in freqplots):
            freqplots += ["dupechelle", "echelle"]
        if any(x in freqtypes.rtypes for x in freqplots):
            freqplots += ["ratios"]

        # Naming of plots preparation
        plotfmt = inputparams["plotfmt"]
        plotfname = outfilename + "_{0}." + plotfmt

        rawmaxmod = Grid[maxPDF_path + "/osc"][maxPDF_ind]
        rawmaxmodkey = Grid[maxPDF_path + "/osckey"][maxPDF_ind]
        maxmod = su.transform_obj_array(rawmaxmod)
        maxmodkey = su.transform_obj_array(rawmaxmodkey)
        maxmod = maxmod[:, maxmodkey[0, :] < 2.5]
        maxmodkey = maxmodkey[:, maxmodkey[0, :] < 2.5]
        maxjoins = freq_fit.calc_join(
            mod=maxmod,
            modkey=maxmodkey,
            obs=obs,
            obskey=obskey,
            obsintervals=obsintervals,
        )
        maxjoinkeys, maxjoin = maxjoins
        maxmoddnu = Grid[maxPDF_path + "/dnufit"][maxPDF_ind]

        if allfplots or "echelle" in freqplots:
            plot_seismic.echelle(
                selectedmodels,
                Grid,
                obs,
                obskey,
                mod=maxmod,
                modkey=maxmodkey,
                dnu=fitfreqs["dnufit"],
                join=maxjoin,
                joinkeys=maxjoinkeys,
                pair=False,
                duplicate=False,
                output=plotfname.format("echelle_uncorrected"),
            )
        if allfplots or "pairechelle" in freqplots:
            plot_seismic.echelle(
                selectedmodels,
                Grid,
                obs,
                obskey,
                mod=maxmod,
                modkey=maxmodkey,
                dnu=fitfreqs["dnufit"],
                join=maxjoin,
                joinkeys=maxjoinkeys,
                pair=True,
                duplicate=False,
                output=plotfname.format("pairechelle_uncorrected"),
            )
        if allfplots or "dupechelle" in freqplots:
            plot_seismic.echelle(
                selectedmodels,
                Grid,
                obs,
                obskey,
                mod=maxmod,
                modkey=maxmodkey,
                dnu=fitfreqs["dnufit"],
                join=maxjoin,
                joinkeys=maxjoinkeys,
                duplicate=True,
                pair=True,
                output=plotfname.format("dupechelle_uncorrected"),
            )

        if fitfreqs["fcor"] == "None":
            corjoin = maxjoin
            coeffs = [1]
        elif fitfreqs["fcor"] == "HK08":
            corjoin, coeffs = freq_fit.HK08(
                joinkeys=maxjoinkeys,
                join=maxjoin,
                nuref=fitfreqs["numax"],
                bcor=fitfreqs["bexp"],
            )
        elif fitfreqs["fcor"] == "BG14":
            corjoin, coeffs = freq_fit.BG14(
                joinkeys=maxjoinkeys, join=maxjoin, scalnu=fitfreqs["numax"]
            )
        elif fitfreqs["fcor"] == "cubicBG14":
            corjoin, coeffs = freq_fit.cubicBG14(
                joinkeys=maxjoinkeys, join=maxjoin, scalnu=fitfreqs["numax"]
            )

        if len(coeffs) > 1:
            print("The surface correction coefficients are", *coeffs)
        else:
            print("The surface correction coefficient is", *coeffs)

        if allfplots or "echelle" in freqplots:
            plot_seismic.echelle(
                selectedmodels,
                Grid,
                obs,
                obskey,
                mod=maxmod,
                modkey=maxmodkey,
                dnu=fitfreqs["dnufit"],
                join=corjoin,
                joinkeys=maxjoinkeys,
                freqcor=fitfreqs["fcor"],
                coeffs=coeffs,
                scalnu=fitfreqs["numax"],
                pair=False,
                duplicate=False,
                output=plotfname.format("echelle"),
            )
        if allfplots or "pairechelle" in freqplots:
            plot_seismic.echelle(
                selectedmodels,
                Grid,
                obs,
                obskey,
                mod=maxmod,
                modkey=maxmodkey,
                dnu=fitfreqs["dnufit"],
                join=corjoin,
                joinkeys=maxjoinkeys,
                freqcor=fitfreqs["fcor"],
                coeffs=coeffs,
                scalnu=fitfreqs["numax"],
                pair=True,
                duplicate=False,
                output=plotfname.format("pairechelle"),
            )
        if allfplots or "dupechelle" in freqplots:
            plot_seismic.echelle(
                selectedmodels,
                Grid,
                obs,
                obskey,
                mod=maxmod,
                modkey=maxmodkey,
                dnu=fitfreqs["dnufit"],
                join=corjoin,
                joinkeys=maxjoinkeys,
                freqcor=fitfreqs["fcor"],
                coeffs=coeffs,
                scalnu=fitfreqs["numax"],
                duplicate=True,
                pair=True,
                output=plotfname.format("dupechelle"),
            )
        if "freqcormap" in freqplots or debug:
            plot_seismic.correlation_map(
                "freqs",
                obsfreqdata,
                plotfname.format("freqs_cormap"),
                obskey=obskey,
            )
        if obsfreqmeta["getratios"]:
            for ratseq in obsfreqmeta["ratios"]["plot"]:
                ratnamestr = "ratios_{0}".format(ratseq)
                plot_seismic.ratioplot(
                    obsfreqdata,
                    maxjoinkeys,
                    maxjoin,
                    maxmodkey,
                    maxmod,
                    ratseq,
                    output=plotfname.format(ratnamestr),
                    threepoint=fitfreqs["threepoint"],
                    interp_ratios=fitfreqs["interp_ratios"],
                )
                if fitfreqs["correlations"]:
                    plot_seismic.correlation_map(
                        ratseq,
                        obsfreqdata,
                        output=plotfname.format(ratnamestr + "_cormap"),
                    )

        if obsfreqmeta["getepsdiff"]:
            for epsseq in obsfreqmeta["epsdiff"]["plot"]:
                epsnamestr = "epsdiff_{0}".format(epsseq)
                plot_seismic.epsilon_difference_diagram(
                    mod=maxmod,
                    modkey=maxmodkey,
                    moddnu=maxmoddnu,
                    sequence=epsseq,
                    obsfreqdata=obsfreqdata,
                    output=plotfname.format(epsnamestr),
                )
                if fitfreqs["correlations"]:
                    plot_seismic.correlation_map(
                        epsseq,
                        obsfreqdata,
                        output=plotfname.format(epsnamestr + "_cormap"),
                    )
        if obsfreqmeta["getepsdiff"] and debug:
            if len(obsfreqmeta["epsdiff"]["plot"]) > 0:
                plot_seismic.epsilon_difference_components_diagram(
                    mod=maxmod,
                    modkey=maxmodkey,
                    moddnu=maxmoddnu,
                    obs=obs,
                    obskey=obskey,
                    dnudata=obsfreqdata["freqs"]["dnudata"],
                    obsfreqdata=obsfreqdata,
                    obsfreqmeta=obsfreqmeta,
                    output=plotfname.format("DEBUG_epsdiff_components"),
                )
    else:
        print(
            "Did not get any frequency file input, skipping ratios and echelle plots."
        )

    # Save dictionary with full statistics
    if optionaloutputs:
        pfname = outfilename + ".json"
        fio.save_selectedmodels(pfname, selectedmodels)
        print("Saved dictionary to {0}".format(pfname))

    # Print time of completion
    t1 = time.localtime()
    print(
        "\nFinished on {0}".format(time.strftime("%Y-%m-%d %H:%M:%S", t1)),
        "(runtime {0} s).\n".format(time.mktime(t1) - time.mktime(t0)),
    )

    # Save log and recover standard output
    sys.stdout = stdout
    print("Saved log to {0}.log".format(outfilename))

    # Close grid, close open plots, and try to free memory between multiple runs
    Grid.close()
    plt.close("all")
    gc.collect()
