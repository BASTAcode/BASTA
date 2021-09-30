"""
Make an input file for BASTA in XML format
"""
# Definition of the path to BASTA, just in case you need it
import os

BASTADIR = os.environ["BASTADIR"]


# This is the function you are looking for!
def define_input(define_io, define_fit, define_output, define_plots, define_intpol):
    """
    Define user input for BASTA. Will fill the dictionaries with required information
    and pass it on to the XML-generation routines.

    PLEASE MODIFY THINGS IN THIS FUNCTION TO SUIT YOUR NEEDS
    """
    # ==================================================================================
    # BLOCK 1: I/O
    # ==================================================================================
    # Name of the XML input file to produce
    # --> Use as input: BASTArun input-example.xml
    xmlfilename = "input_subgiant.xml"

    # The path to the grid to be used by BASTA for the fitting.
    # --> If using isochrones, remember to also specify physics settings in BLOCK 3c
    define_io["gridfile"] = os.path.join(BASTADIR, "grids/") + "Garstec_validation.hdf5"

    # Where to store the output of the BASTA run
    define_io["outputpath"] = os.path.join(BASTADIR, "examples/") + "output/subgiant/"

    # BASTA is designed to fit multiple stars in the same run. To generate the input
    # file, a table in plain ascii with the observed stellar parameters must be
    # supplied. Please specify the path to the table and a tuple with the columns in the
    # table.
    #
    # Note that even when fitting a single star, such a file must be given!
    #
    # Regarding the columns:
    # --> The first entry should always be "starid"
    # --> The names must match entries in the parameter list (basta/constants.py) unless
    #     it is "dnu". BASTA supports different dnu types and will automatically
    #     translate "dnu" from the table into the correct one based on "fitparams" (see
    #     Block 2 below).
    # --> Both value and error should be given for all quantities (add _err for errors)
    # --> It is important that this list matches the file contents!!!
    #
    # Also note that the file can contain more columns than required for the fit!
    # --> Only those relevant are included in the produced input file.

    # Example of the columns in the example file:
    define_io["asciifile"] = os.path.join(BASTADIR, "examples/data/") + "subgiant.ascii"
    define_io["params"] = (
        "starid",
        "Teff",
        "Teff_err",
        "FeH",
        "FeH_err",
        "dnu",
        "dnu_err",
        "numax",
        "numax_err",
    )

    # Special option: Change the assumed delimiter for the ascii table
    # --> By default, it is None, meaning any consecutive whitespace act as delimiter
    # --> Modify to use e.g. a comma-separated file
    # define_io["delimiter"] = ","

    # Special option: Assumed placeholder for missing values in the ascii table
    # --> Generally it is advised to provide BASTA with a complete table with bad stars
    #     removed, however BASTA can ignore missing values using this key. This might be
    #     useful if using a large, pre-computer table, where some auxiliary data are not
    #     available for certain stars.
    # --> PLEASE BE AWARE, that if any of the parameters you are including in the fit
    #     are missing, the star will be skipped!
    # define_io["missingval"] = -999.999

    # Using "overwriteparams" it is possible to assume a given (value, error) of any
    # parameter for all stars in the fit (e.g. assume all stars to have the same
    # temperature or dnu). This will overwrite whatever (value, error) is given for the
    # individual star!
    # define_io["overwriteparams"] = {"dnufit": (100, 2)}

    # ==================================================================================
    # BLOCK 2: Fitting control
    # ==================================================================================
    # A list of the parameters to fit must be given to BASTA in a tuple.
    # --> Must be present in the list of parameters in BLOCK 2 unless it is special
    #     keyword (see below)
    # --> The names must match entries in the parameter list (basta/constants.py)
    #
    # Important note on keywords:
    # --> To fit frequencies add "freqs"
    # --> To fit ratios add "r012" (or whichever type(s) you want to fit)
    # --> To fit parallax/magnitude/distance, add "parallax"
    # --> If activating any special fits, remember to look in the corresponding block
    #     below to set additional required settings!! E.g., for frequencies/ratios, look
    #     in Block 2b and 2d.
    #
    # Important note on dnu:
    # --> BASTA can use different determinations of the large frequency separation
    #     (dnu). The one provided must match the one you add to fitting parameters in
    #     the next block. If present in the grid, "dnufit" is the most reliable one.
    #     The full list is available in REMEMBER TO ADD LINK !
    define_fit["fitparams"] = ("Teff", "FeH", "freqs")

    # ------------------------------------------------------------
    # BLOCK 2a: Fitting control, priors
    # ------------------------------------------------------------
    # A key part of any Bayesian scheme is the specification of priors. By default,
    # BASTA assumes uninformative priors on all fitting parameters (more details in
    # the paper). In the following the priors can be specified further.

    # On class of priors is constrained, flat priors. In other words, it is restriction
    # of any standard grid parameters. This will reduce computation time, as BASTA
    # can safely ignore models in the grid with non-matching parameters.
    # --> These priors can be defined using a fixed absolute tolerance (abstol) or using
    #     a number of sigmas (sigmacut), both of which will be applied based on the
    #     observed value of the individual star. This is only avaiable for parameters in
    #     fitparams.
    # --> Another option is to specify "min" and/or "max" of a parameter. This can be
    #     done for all grid parameters. Remember that this range should encompass all
    #     stars! An example of this is to restrict the mass for isochrones.
    define_fit["priors"] = {"Teff": {"sigmacut": "3"}, "FeH": {"abstol": "0.5"}}

    # If using asteroseismic data, it can be an advantage to apply a cut on dnu, as it
    # significantly reduces computation time.
    # define_fit["priors"] = {**define_fit["priors"], "dnufit": {"sigmacut": "3"}}

    # A different class of priors is to use an initial mass function (IMF). The full
    # list of available IMF's can be seen in basta/priors.py
    define_fit["priors"] = {**define_fit["priors"], "IMF": "salpeter1955"}

    # A key functionality of BASTA is to use so-called Bayesian weights, which take the
    # sampling of the grid into account. These will also accommodate the different
    # evolutionary speed of stars in different phases.
    # --> IT IS NOT RECOMMENDED TO DISABLE THE USE OF THE WEIGHTS, but is can be done
    #     for e.g. testing.
    # define_fit["bayweights"] = False

    # ------------------------------------------------------------
    # BLOCK 2b: Fitting control, solar scaling
    # ------------------------------------------------------------
    # When using asteroseismic observables, it is advantageous to make sure the observed
    # values are compatible with the quantities in the grid. This can be done by using
    # Sun to scale the input; dnu of the solar model in the grid is compared to the
    # assumed dnu of the Sun determined using the observational pipeline.

    # The dnu of the solar model is automatically extracted from the grid. If set to
    # False, the scaling functionality is switched off (not recommended unless testing).
    define_fit["solarmodel"] = True

    # To perform the scaling, the observed values of dnu and numax for the Sun must be
    # assumed. By default BASTA uses the values from the SYD pipeline.
    define_fit["sundnu"] = 135.1
    define_fit["sunnumax"] = 3090.0

    # ------------------------------------------------------------
    # BLOCK 2c: Fitting control, isochrones
    # ------------------------------------------------------------
    # When fitting to (the BaSTI) isochrones, the input physics ("science case") MUST be
    # specified. Set in the tuple "odea":
    # - o: Overshooting efficiency [step] (0 = no overshooting). Activation value: 0.2
    # - d: Diffusion (0 = no diffusion, 1 = diffusion)
    # - e: Mass loss [Reimers eta] (0 = no mass loss). Activation value: 0.3
    # - a: Alpha enhancement [fixed] (0 = no alpha enhancement). Activation value: 0.4
    #
    # Possible science cases:
    # define_fit["odea"] = (0,   0, 0,   0)    # "Normal"
    # define_fit["odea"] = (0.2, 0, 0,   0)    # Overshooting
    # define_fit["odea"] = (0.2, 0, 0.3, 0)    # + mass loss
    # define_fit["odea"] = (0.2, 1, 0.3, 0)    # + diffusion
    # define_fit["odea"] = (0.2, 1, 0.3, 0.4)  # + alpha enhancement

    # ------------------------------------------------------------
    # BLOCK 2d: Fitting control, frequencies
    # ------------------------------------------------------------
    # When fitting individual frequencies or ratios additional information is required.
    # Please set in the dictionary below:
    #
    # - "freqpath": The path to the frequency files. IMPORTANT:: The names of the
    #               frequency files are assumed to match the corresponding StarID's.
    #
    # - "fcor": The assumed correction prescription for the asteroseismic surface
    #           effect. Choose between the following:
    #           * "BG14": Two-term correction from Ball & Gizon (2014)
    #           * "cubicBG14": One-term (cubic term) correction from Ball & Gizon (2014)
    #           * "HK08": Power law from Kjeldsen et al. (2008). IMPORTANT: When using
    #                     this correction, "bexp" must be specified in the dictionary!
    #                     Typically a value of 4.5 is used.
    #           * "None": No correction. PLEASE ONLY USE FOR TESTING/VALIDATION or if
    #                     input frequencies have already been corrected.
    #
    # - "correlations": To include the correlations between the frequencies/ratios in
    #                   the fit. Currently correlations are only supported for ratios.
    #
    # - "dnufrac": Only model matching the lowest observed l=0 within this fraction is
    #              considered. This is useful when fitting ratios. It is also for
    #              computational efficiency purposes, as the model search is restricted.
    #
    # - "seismicweight": To balance the fit (and let the classical observables still
    #                    have an impact), it is customary to weight/scale the
    #                    contribution of the individual frequencies (or ratios).
    #                    Typcally, the chi2 term is divided by a given factor before
    #                    added to the total chi2. Choose between the following:
    #                    * "1/N": The default weighting scheme. The seismic chi2 is
    #                             divided/normalised by the total number of frequencies
    #                             (or ratios). It is possible to manually adjust the
    #                             scaling by specifying "N".
    #                    * "1/1": No scaling. Each frequency counts as one classical
    #                             parameter
    #                    * "1/N-dof": Normalisation by number of frequencies minus the
    #                                 (user-specified) degrees of freedom. This option
    #                                 must be supplemented by a specification of "dof".

    # Example of typical settings for a frequency fit (with default seismic weights):
    # define_fit["freqparams"] = {
    define_fit["freqparams"] = {
        "freqpath": os.path.join(BASTADIR, "examples/data/freqs/"),
        "fcor": "BG14",
        "correlations": False,
        "dnufrac": 0.15,
    }

    # An example of manually forcing the weights with "N", and an example of using "dof"
    # define_fit["freqparams"] = {
    #     **define_fit["freqparams"],
    #     "seismicweight": "1/N",
    #     "N": 2,
    # }
    # define_fit["freqparams"] = {
    #     **define_fit["freqparams"],
    #     "seismicweight": "1/N-dof",
    #     "dof": 3,
    # }

    # ------------------------------------------------------------
    # BLOCK 2e: Fitting control, distances
    # ------------------------------------------------------------
    # If fitting parallax and/or predicting distance, additional information is required
    # below.

    # Which photometric filters to use. Must be avaiable for the stars (i.e. read from
    # the table in Block 1). The filter names must be valid (i.e. match entries in the
    # list of available parameters).
    # define_fit["filters"] = ("Mj_2MASS", "Mh_2MASS", "Mk_2MASS")

    # Assumed coordinate system. IMPORTANT: Remember to add coordinates to all stars in
    # reading in Block 1. Can be:
    # - "galactic": Input coordinates are expected in galactic coordinates longitude (l)
    #              and latitude (b).
    # - "icrs": Input coordinates are expected in celestial right ascension (RA) and
    #           declination (DEC).
    # define_fit["dustframe"] = "icrs"

    # ==================================================================================
    # BLOCK 3: Output control
    # ==================================================================================
    # A list of quantities to output. Will be printed to the log of each individual star
    # and stored in the output/results file(s).
    # --> The names must match entries in the parameter list (basta/constants.py)
    # --> A reasonable choice is to (as a minimum) output the parameters used in the fit
    # --> If you want to predict distance, add the special keyword "distance".
    define_output["outparams"] = ("Teff", "FeH", "radPhot", "massfin", "age")

    # Name of the output file containing the results of the fit in ascii format.
    # --> A version in xml-format will be automatically created
    # --> The same name will be used for the {.err, .warn} files
    define_output["outputfile"] = "results.ascii"

    # A dump of the statistics (chi2, logPDF) for all models in the grids can be saved
    # to a .json file. One file is produced per star.
    define_output["optionaloutputs"] = False

    # BASTA is designed to work with the median and corresponding Bayesian credibility
    # intervals or quantiles (16th and 84th percentile). Thus, by default BASTA will
    # report the median of the posterior distribution the parameter value and the
    # quantiles as the (asymmetric) errors.
    # --> For some applications (e.g. to compare with other results), it can be useful
    #     to instead report the mean and standard deviation of the distributions.
    # --> Note that this only is reasonable for normal distributions! Please inspect
    #     the corner plots beforehand.
    # --> Be aware that switching output mode is not fully tested and might be unstable!
    # define_output["centroid"] = "mean"
    # define_output["uncert"] = "std"

    # ==================================================================================
    # BLOCK 4: Plotting control
    # ==================================================================================
    # BASTA can produce various plots. Include the various lines to produce the plots.

    # Corner plots of posteriors. Specify a list of parameters to plot.
    # --> Typically, plotting the same quantities as written to the output is a
    #     reasonable choice, unless you want to output many parameters to ascii.
    # --> If the keyword "distance" is present, an additional distance corner plot is
    #     produced.
    # --> To disable, use an empty list or tuple.
    define_plots["cornerplots"] = define_output["outparams"]

    # BASTA can produce a Kiel diagram (Teff vs logg) with the observations and the
    # model points from the grid. The latter will be color coded based on the fitting
    # parameters and their uncertainties/constraints.
    define_plots["kielplots"] = True

    # When fitting frequencies or frequency ratios, BASTA can generate echelle diagrams
    # and plots of the ratios. Setting True will produce all plots. Setting "echelle"
    # will skip the plot of ratios, which might be useful to save time when fitting
    # individual frequencies (as ratios can take a while to compute)
    # --> All possible options: ("echelle", "dupechelle", "pairechelle", "ratios",
    #                            True, False)
    define_plots["freqplots"] = "echelle"

    # By default BASTA will save plots in png format, as they are much faster to
    # produce. This can be changed to pdf to obtain higher quality plots at the cost of
    # speed. For fitting a single star (especially with frequencies/ratios), pdf is
    # usually preferred.
    # --> Note that the ratios plot will always be pdf, because it is using a multipage
    #     pdf backend.
    define_plots["plotfmt"] = "pdf"

    # ==================================================================================
    # BLOCK 5: Interpolation
    # ==================================================================================
    # Based on the input grid of models, BASTA can use interpolation to increase the
    # resolution. The new interpolated grid will be saved for easy re-fitting.
    # --> Only parameters in "fitparams" and/or "outparams" will be saved to the
    #     interpolated grid to keep the file size down!
    # --> If an interpolated grid is already present, BASTA will not calculate it anew.
    #     Please remove the old interpolated grid if you want a fresh one.
    # --> BE AWARE that grid interpolation can be a very time consuming process!!

    # Note that this is a special block, as many different settings must be specified
    # when interpolating.
    interpolation = False
    if interpolation:
        define_intpol["intpolparams"] = {}

        # As interpolation is time consuming, it is normal procedure to define a 'box'
        # around the target star and only construct the interpolated grid in this
        # region. This will yield higher resolution in the important region of the grid,
        # while reducing the computation time and keeping the size of the interpolated
        # grid as low as possible.
        # Please set limits for the parameters (same syntax as for the flat priors) to
        # control the grid creation.
        # --> In addition to the standard observed quantities, it is also possible to
        #     specify a limit on all other parameters, e.g., "massini": {"min": 1.05}
        #     to only consider models with a mass above 1.05 solar masses.
        define_intpol["intpolparams"]["limits"] = {
            "Teff": {"sigmacut": 1},
            "FeH": {"abstol": 0.2},
        }

        # The method for interpolation. Possible options:
        # - case = "along": Interpolation along each track
        # - case = "across": Interpolation across/between tracks
        # - case = "combined": Interpolation across/between tracks and then afterwards
        #                      interpolation along each track
        #
        # The method for constructing the interpolated grid *if fitting multiple stars*.
        # Possible options:
        # - construction = "bystar": An interpolated grid will be produced for each star
        # - construction = "encompass": One interpolated grid will be produced to which
        #                               all stars are fitted. The limits specified above
        #                               will be used to find the extreme values for all
        #                               stars to make them all fit in the same grid.
        #                               WARNING: Only do this stars very close together
        #                               in parameter space!
        define_intpol["intpolparams"]["method"] = {
            "case": "combined",
            "construction": "bystar",
        }

        # An identifier to be added included in the naming of the interpolated grids.
        # Assuming bystar-construction, the grids will be automatically named
        # "intpol_<identifier>_STARID.hdf5" and placed in the output folder.
        define_intpol["intpolparams"]["name"] = "testgrid"

        # Settings for interpolation across tracks. Set one of these depending on which
        # type of grid:
        # - "scale": For Sobol sampling only. The increase in number of tracks. E.g. a
        #            scale of 2 will double the number of tracks.
        # - "resolution": For Cartesian sampling only. The increase in resolution in the
        #                 sampling parameters. E.g., setting "resolution": {"FeHini": 2}
        #                 will double number of metallicities. It is a dict!
        #
        # Additionally, it is possible to specify the following:
        # - "baseparam": Parameter forming the base of the interpolation. Two cases:
        #                * tracks: Using central density "rhocen" is recommended if,
        #                          available in the grid. Otherwise use central hydrogen
        #                          content "xcen".
        #                * isochrones: Using final mass "massfin" is recommended.
        define_intpol["intpolparams"]["gridresolution"] = {
            "scale": 1.5,
            "baseparam": "rhocen",
        }

        # Settings for interpolation along a track. Specify the following:
        # - "param": The parameter used to define the resolution. Typically "dnufit" is
        #            a good choice for asteroseismic applications.
        # - "value": The required resolution in the specified parameter. For "dnufit",
        #            the frequency resolution between two consecutive models will be
        #            this value (in microHz) or better. Note that "dnufit" is *not*
        #            available for ishcornes (use e.g. "dnuSer" instead).
        #
        # Additionally, it is possible to specify the following:
        # - "baseparam": Parameter forming the base of the interpolation. Two cases:
        #                * tracks: Using central density "rhocen" is recommended if,
        #                          available in the grid. Otherwise use central hydrogen
        #                          content "xcen".
        #                * isochrones: Using final mass "massfin" is recommended.
        define_intpol["intpolparams"]["trackresolution"] = {
            "param": "dnufit",
            "value": 0.01,
            "baseparam": "rhocen",
        }
        # END OF INTERPOLATION

    # Done! Nothing more to specify.
    return (
        xmlfilename,
        define_io,
        define_fit,
        define_output,
        define_plots,
        define_intpol,
    )


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# ~~~~~~~~~~~~~~~~~~~~~~~~ AUTOMATED PART OF THE SCRIPT BELOW ~~~~~~~~~~~~~~~~~~~~~~~~~
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
def _run_sanity_checks(
    define_io, define_fit, define_output, define_plots, define_intpol
):
    """
    Run sanity checks. Return the number of errors and number of stars to fit.
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
    xmlinputname,
    define_io,
    define_fit,
    define_output,
    define_plots,
    define_intpol,
):
    """
    Print a human-readable summary of the BASTA run resulting from the given input.
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

    # Final words...!
    print(
        "\n\n!!! To perform the fit, run the command: BASTArun {0}".format(xmlinputname)
    )


if __name__ == "__main__":
    import sys
    import numpy as np
    from basta.constants import freqtypes
    from basta.xml_create import generate_xml

    print("~~~ Generating an XML input file for BASTA ~~~")

    # Define dictionaries for settings
    infodict_io = {}
    infodict_fit = {}
    infodict_output = {}
    infodict_plots = {}
    infodict_intpol = {}

    # Fill the dictionaries depending on the users selections
    print("\nReading user input ...")
    (
        xmlname,
        infodict_io,
        infodict_fit,
        infodict_output,
        infodict_plots,
        infodict_intpol,
    ) = define_input(
        define_io=infodict_io,
        define_fit=infodict_fit,
        define_output=infodict_output,
        define_plots=infodict_plots,
        define_intpol=infodict_intpol,
    )
    print("Done!")

    # Run sanity checks !
    print("\nRunning sanity checks ...")
    errcode, numstars = _run_sanity_checks(
        define_io=infodict_io,
        define_fit=infodict_fit,
        define_output=infodict_output,
        define_plots=infodict_plots,
        define_intpol=infodict_intpol,
    )

    # If no errors, convert info XML tags and write to file
    if errcode != 0:
        print(
            "Done! Found {0} warning(s)! Will not create XML... Please fix!".format(
                errcode
            )
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

        with open(xmlname, "w", encoding="utf-8") as xf:
            print(xmldat, file=xf)
        print("Done!")
        print("\n\n" + 42 * "*")
        print("*** Summary of the requested BASTA run ***")
        print(42 * "*", "\n")
        _print_summary(
            numfits=numstars,
            xmlinputname=xmlname,
            define_io=infodict_io,
            define_fit=infodict_fit,
            define_output=infodict_output,
            define_plots=infodict_plots,
            define_intpol=infodict_intpol,
        )
