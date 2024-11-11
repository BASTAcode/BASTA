"""
Auxiliary functions for running the examples
"""

import os
import sys
import argparse
import numpy as np
from basta.constants import freqtypes
from basta.xml_create import generate_xml


def _run_consistency_checks(
    define_io, define_fit, define_output, define_plots, define_intpol
):
    """
    Run consistency checks.

    Parameters
    ----------
    define_* : dict
        User-defined parameters in the specific format used by the example template


    Returns
    -------
    errors : int
        Number of errors found

    len_ascii : int
        Length of the read ascii files (i.e., the number of stars to fit)
    """
    errors = 0

    # TASK 0: Grid
    if not os.path.exists(define_io["gridfile"]):
        print("--> WARNING: Cannot find a grid at '{0}'".format(define_io["gridfile"]))
        print("    Did you want to use on of our example grids? Then run the command")
        print("    'BASTAdownload -h' and pick a grid to be downloaded.")
        errors += 1

    # TASK 1: Ascii file
    try:
        delim = define_io["delimiter"]
    except KeyError:
        delim = None

    try:
        inp = np.genfromtxt(
            define_io["asciifile"],
            dtype=None,
            names=define_io["params"],
            encoding=None,
            delimiter=delim,
        )
    except Exception:
        print(
            "--> WARNING: Issues detected reading the file '{0}'!".format(
                define_io["asciifile"]
            )
        )
        print("    * Is the path correct?")
        print("    * Do you have the correct number of columns?")
        print("    * Do the list of parameters match the columns?")
        len_ascii = 0
        errors += 1
    else:
        len_ascii = inp.ndim + 1

    # TASK 2: Isochrone checking
    if "iso" in define_io["gridfile"]:
        try:
            define_fit["odea"]
        except KeyError:
            print(
                "--> WARNING: 'define_fit[\"odea\"]' not detected but grid seems to",
                "be isochrones!",
            )
            errors += 1

    # TASK 3: Activated special fits or outputs?
    allparams = list(
        set(
            [
                *define_fit["fitparams"],
                *define_output["outparams"],
                *define_plots["cornerplots"],
            ]
        )
    )
    globast = False
    freqfit = False
    distfit = False
    distout = False
    distcor = False
    for param in define_fit["fitparams"]:
        if param == "parallax":
            distfit = True
        elif param in freqtypes.alltypes:
            freqfit = True
        elif "dnu" in param:
            globast = True
    for param in define_output["outparams"]:
        if param == "distance":
            distout = True
    if len(define_plots["cornerplots"]) > 0:
        for param in define_plots["cornerplots"]:
            if param == "distance":
                distcor = True

    # TASK 4a: Asteroseismology | Global
    if globast:
        if not define_fit["solarmodel"]:
            print("--> Note: Solar scaling deactivated for asteroseismic fit?")
        if "iso" in define_io["gridfile"]:
            for param in allparams:
                if "dnufit" in param:
                    print(
                        "--> WARNING: 'dnufit' is not available for isochrones! Use",
                        "another dnu (e.g. 'dnuSer') instead!",
                    )
                    errors += 1

    # TASK 4b: Asteroseismology | Individual
    if freqfit:
        try:
            define_fit["freqparams"]
        except KeyError:
            print(
                "--> WARNING: Frequency fitting is activated, but",
                "'define_fit[\"freqparams\"]' is not defined!",
            )
            errors += 1
        if not define_plots["freqplots"]:
            print("--> Note: No frequency plots requested?")
        if not define_fit["solarmodel"]:
            print("--> Note: Solar scaling deactivated for frequency fitting?")

    # Task 5a: Distance | Filters
    if any([distfit, distout, distcor]):
        try:
            define_fit["filters"]
        except KeyError:
            print(
                "--> WARNING: Distance fitting/prediction requested, but",
                "'define_fit[\"filters\"]' is not defined!",
            )
            errors += 1
        try:
            define_fit["dustframe"]
        except KeyError:
            print(
                "--> WARNING: Distance fitting/prediction requested, but",
                "'define_fit[\"dustframe\"]' is not defined!",
            )
            errors += 1

    # Task 5b: Distance | Fit, out, corner?
    if distfit and not distout:
        print("--> Note: Fitting parallax, but not outputting distance?")
    if distout and not distcor:
        print("--> Note: Outputting distance, but not making distance corner?")

    return errors, len_ascii


def _print_summary(
    numfits,
    define_io,
    define_fit,
    define_output,
    define_plots,
    define_intpol,
):
    """
    Print a human-readable summary of the BASTA run resulting from the given input.

    Parameters
    ----------
    numfits : int
        Number of stars to fit

    define_* : dict
        User-defined parameters in the specific format used by the example template

    Returns
    -------
    None
    """
    # Basic info
    fitlist = ", ".join(define_fit["fitparams"])
    print(
        "A total of {0} star(s) will be fitted with {{{1}}} to the grid '{2}'.".format(
            numfits, fitlist, define_io["gridfile"]
        )
    )
    if len(define_intpol) > 0:
        print(
            "Grid interpolation will be activated! Please check the settings carefully!"
        )

    # Basic: Isochrones?
    try:
        odea = define_fit["odea"]
    except KeyError:
        pass
    else:
        print(
            "Using the isochrone science case (overshooting={0},".format(odea[0]),
            "diffusion={0}, mass loss eta={1}, enhancement alphaFe={2}).".format(
                odea[1], odea[2], odea[3]
            ),
        )

    # Output: Reported stats
    try:
        centroid = define_output["centroid"]
        uncert = define_output["uncert"]
    except KeyError:
        exstr = ""
    else:
        exstr = "USING EXPERIMENTAL OUTPUT with '{0}' and '{1}'.".format(
            centroid, uncert
        )

    # Output: Distance
    outlist = list(define_output["outparams"])
    if "distance" in define_output["outparams"]:
        outlist.remove("distance")
        distout = True
    else:
        distout = False
    print(
        "\nThis will output {{{0}}} to a results file. {1}".format(
            ", ".join(outlist),
            exstr,
        )
    )
    if distout:
        print("A seperate distance results file will also be created.")

    # Output: Optional
    if define_output["optionaloutputs"]:
        print("A .json file per star will be saved with calculated statistics.")

    # Plots: Corner
    print()
    cornerlist = list(define_plots["cornerplots"])
    if "distance" in define_plots["cornerplots"]:
        cornerlist.remove("distance")
        distcorner = True
    else:
        distcorner = False
    if len(define_plots["cornerplots"]) > 0:
        print(
            "Corner plots include {{{0}}} with observational bands on {{{1}}}.".format(
                ", ".join(cornerlist), fitlist
            )
        )
        if distcorner:
            print("A seperate distance corner plot will also be created.")
    else:
        print("Corner plots will not be produced!")

    # Plots: Kiel
    if define_plots["kielplots"]:
        print(
            "Kiel diagrams will be made with observational bands on {{{0}}}.".format(
                fitlist
            )
        )
    else:
        print("Kiel diagrams will not be produced!")

    # Plots: Freqs
    if define_plots["freqplots"]:
        print("Asteroseismic plots will be produced!")

    # Priors
    print()
    flat_priors = []
    imf = None
    for pp in define_fit["priors"].keys():
        if pp != "IMF":
            flat_priors.append(pp)
        else:
            imf = define_fit["priors"][pp]

    if len(flat_priors) > 0:
        print(
            "A restricted flat prior will be applied to: {0}.".format(
                ", ".join(flat_priors)
            )
        )
    if imf is not None:
        print(
            "Additionally, a {0} IMF will be used as a prior.".format(imf.capitalize())
        )


# Main routine for running! Will be imported by specific examples
def make_basta_input(define_user_input):
    """
    Get user input, run consistency checks, produce files, print summary.

    Parameters
    ----------
    define_user_input : func
        Function handle to the user defined input

    Returns
    -------
    None
    """
    # Initialise argument parser
    parser = argparse.ArgumentParser(description="Create input XML for BASTA")
    parser.add_argument(
        "--quiet", action="store_true", help="Suppress summary text output"
    )
    args = parser.parse_args()

    if not args.quiet:
        print(46 * "*")
        print("*** Generating an XML input file for BASTA ***")
        print(46 * "*", "\n")

    # Define dictionaries for settings
    infodict_io = {}
    infodict_fit = {}
    infodict_output = {}
    infodict_plots = {}
    infodict_intpol = {}

    # Fill the dictionaries depending on the users selections
    print("Reading user input ...")
    (
        xmlname,
        infodict_io,
        infodict_fit,
        infodict_output,
        infodict_plots,
        infodict_intpol,
    ) = define_user_input(
        define_io=infodict_io,
        define_fit=infodict_fit,
        define_output=infodict_output,
        define_plots=infodict_plots,
        define_intpol=infodict_intpol,
    )
    print("Done!")

    # Run consistency checks !
    print("\nRunning consistency checks ...")
    errcode, numstars = _run_consistency_checks(
        define_io=infodict_io,
        define_fit=infodict_fit,
        define_output=infodict_output,
        define_plots=infodict_plots,
        define_intpol=infodict_intpol,
    )

    # If no errors, convert info XML tags and write to file
    if errcode != 0:
        print(
            "Done! \n\n!!! Found {0} warning(s)! Will not create XML... ".format(
                errcode
            ),
            "Please fix!!!",
        )
    else:
        print("Done!")
        print("\nCreating XML input file '{0}' ...".format(xmlname))
        try:
            xmldat = generate_xml(
                **infodict_io,
                **infodict_fit,
                **infodict_output,
                **infodict_plots,
                **infodict_intpol,
            )
        except Exception as e:
            print(
                "--> Error! XML generation failed with the following error: {0}".format(
                    e
                )
            )
            print("    * Did you forget a param in the ascii file or misspelled it?")
            print("\nCannot create XML file! Aborting...")
            sys.exit(1)

        # Write structure to file
        with open(xmlname, "w", encoding="utf-8") as xf:
            print(xmldat, file=xf)
        print("Done!")

        # Print summary of the fit
        if not args.quiet:
            print("\n\n   Summary of the requested BASTA run  ")
            print(40 * "-", "\n")
            _print_summary(
                numfits=numstars,
                define_io=infodict_io,
                define_fit=infodict_fit,
                define_output=infodict_output,
                define_plots=infodict_plots,
                define_intpol=infodict_intpol,
            )
        # Final words...!
        print("\n!!! To perform the fit, run the command: BASTArun {0}".format(xmlname))
