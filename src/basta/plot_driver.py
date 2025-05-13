import numpy as np

from basta import core, freq_fit, plot_seismic
from basta import utils_seismic as su
from basta.constants import freqtypes


def plot_all_seismic(
    *,
    inputstar: core.InputStar,
    star: core.Star,
    inferencesettings: core.InferenceSettings,
    outputoptions: core.OutputOptions,
    plotconfig: core.PlotConfig,
    filepaths: core.FilePaths,
    Grid,
    # obsfreqmeta,
    # obsfreqdata,
    selectedmodels,
    path,
    ind,
    glitchparams=None,
) -> None:
    """
    Driver for producing all seismic related plots

    Parameters
    ----------
    freqplots : list
        Plots to be produced
    Grid : hdf5 file
        Stellar models, as tracks or isochrones
    obsfreqmeta : dict
        The requested information about which frequency products to fit or
        plot, unpacked for easier access later.
    obsfreqdata : dict
        Requested frequency-dependent data such as glitches, ratios, and
        epsilon difference. It also contains the covariance matrix and its
        inverse of the individual frequency modes.
        The keys correspond to the science case, e.g. `r01`, `glitch`, or
        `e012`.
        Inside each case, you find the data (`data`), the covariance matrix
        (`cov`), and its inverse (`covinv`).
    obskey : array
        Array containing the angular degrees and radial orders of obs
    obs : array
        Individual frequencies and uncertainties.
    obsintervals : array
        Array containing the endpoints of the intervals used in the frequency
        fitting routine in :func:'freq_fit.calc_join'.
    selectedmodels : dict
        Contains information on all models with a non-zero likelihood.
    path : str
        Path to the highest likelihood track/isocohrone in the grid
    ind : int
        Index of the highest likelihood model in the track
    plotfname : str
        Output plotname format
    nameinplot : bool
        Whether to include star identifier in the plots itself
    debug : bool
        Whether to produce debugging output

    """

    freqplots = plotconfig.freqplots
    assert star.modes is not None
    obs = np.asarray([star.modes.modes.frequencies, star.modes.modes.errors])
    obsintervals = star.modes.obsintervals
    allfplots = freqplots[0] == True  # noqa: E712
    if any(x == "allechelle" for x in freqplots):
        freqplots += ["dupechelle", "echelle", "pairechelle"]
    if any(x in freqtypes.rtypes for x in freqplots):
        freqplots += ["ratios"]
    try:
        rawmaxmod = Grid[path + "/osc"][ind]
        rawmaxmodkey = Grid[path + "/osckey"][ind]
        model_modes = core.make_model_modes_from_ln_freqinertia(rawmaxmodkey, rawmaxmod)
        assert star.modes is not None
        maxjoins = freq_fit.calc_join(star.modes, model_modes)
        maxjoinkeys, maxjoin = maxjoins
        maxmoddnu = Grid[path + "/dnufit"][ind]
    except Exception as e:
        print("\nFrequency plots initialisation failed with the error:", e)
        return

    # Extract the original observed dnu for use on the echelle diagrams
    # --> (equivalent to re-scaling if solar scaling activated)

    x = plot_seismic.EchellePlotBase(
        selectedmodels=selectedmodels,
        Grid=Grid,
        obs=obs,
        mod=maxmod,
        modkey=maxmodkey,
        join=maxjoin,
        joinkeys=maxjoinkeys,
        star=star,
        inferencesettings=inferencesettings,
        plotconfig=plotconfig,
        outputoptions=outputoptions,
    )
    if allfplots or "echelle" in freqplots:
        try:
            plot_seismic.echelle(
                x,
                pairmode=False,
                duplicatemode=False,
                outputfilename=filepaths.plotfile("echelle_uncorrected"),
            )
        except Exception as e:
            print("\nUncorrected echelle failed with the error:", e)

    if allfplots or "pairechelle" in freqplots:
        try:
            plot_seismic.echelle(
                x,
                pairmode=True,
                duplicatemode=False,
                outputfilename=filepaths.plotfile("pairechelle_uncorrected"),
            )
        except Exception as e:
            print("\nUncorrected pairechelle failed with the error:", e)

    if allfplots or "dupechelle" in freqplots:
        try:
            plot_seismic.echelle(
                x,
                pairmode=True,
                duplicatemode=True,
                outputfilename=filepaths.plotfile("dupechelle_uncorrected"),
            )
        except Exception as e:
            print("\nUncorrected dupechelle failed with the error:", e)

    if star.modes.surfacecorrection is None:
        corjoin = maxjoin
        coeffs = np.array([1])
    elif star.modes.surfacecorrection.get("KBC08") is not None:
        corjoin, coeffs = freq_fit.HK08(
            joinkeys=maxjoinkeys,
            join=maxjoin,
            nuref=star.globalseismicparams.get_scaled("numax")[0],
            bcor=star.modes.surfacecorrection["KBC08"]["bexp"],
        )
    elif star.modes.surfacecorrection.get("two-term-BG14") is not None:
        corjoin, coeffs = freq_fit.BG14(
            joinkeys=maxjoinkeys,
            join=maxjoin,
            scalnu=star.globalseismicparams.get_scaled("numax")[0],
        )
    elif star.modes.surfacecorrection.get("cubic-term-BG14") is not None:
        corjoin, coeffs = freq_fit.cubicBG14(
            joinkeys=maxjoinkeys,
            join=maxjoin,
            scalnu=star.globalseismicparams.get_scaled("numax")[0],
        )

    print("Surface correction coefficient(s):", *coeffs)

    x = plot_seismic.EchellePlotBase(
        selectedmodels=selectedmodels,
        Grid=Grid,
        obs=obs,
        mod=maxmod,
        modkey=maxmodkey,
        join=corjoin,
        joinkeys=maxjoinkeys,
        coeffs=coeffs,
        star=star,
        inferencesettings=inferencesettings,
        plotconfig=plotconfig,
        outputoptions=outputoptions,
    )
    if allfplots or "echelle" in freqplots:
        try:
            plot_seismic.echelle(
                x,
                pairmode=False,
                duplicatemode=False,
                outputfilename=filepaths.plotfile("echelle"),
            )
        except Exception as e:
            print("\nEchelle failed with the error:", e)

    if allfplots or "pairechelle" in freqplots:
        try:
            plot_seismic.echelle(
                x,
                pairmode=True,
                duplicatemode=False,
                outputfilename=filepaths.plotfile("pairechelle"),
            )
        except Exception as e:
            print("\nPairechelle failed with the error:", e)

    if allfplots or "dupechelle" in freqplots:
        try:
            plot_seismic.echelle(
                x,
                duplicatemode=True,
                pairmode=True,
                outputfilename=filepaths.plotfile("dupechelle"),
            )
        except Exception as e:
            print("\nDupechelle failed with the error:", e)

    if "freqcormap" in freqplots or outputoptions.debug:
        try:
            plot_seismic.correlation_map(
                "freqs",
                star,
                filepaths.plotfile("freqs_cormap"),
            )
        except Exception as e:
            print("\nFrequencies correlation map failed with the error:", e)

    for ratiotype in plotconfig.freqplots:
        if ratiotype not in freqtypes.rtypes:
            continue
        try:
            ratnamestr = f"ratios_{ratiotype}"
            plot_seismic.ratioplot(
                star,
                maxjoinkeys,
                maxjoin,
                maxmodkey,
                maxmod,
                ratiotype,
                outputfilename=filepaths.plotfile(ratnamestr),
                threepoint=inputstar.threepoint,
                interp_ratios=inputstar.interp_ratios,
            )
        except Exception as e:
            print(
                f"\nRatio plot for {ratiotype} sequence failed with the error:",
                e,
            )

        if inputstar.correlations:
            try:
                plot_seismic.correlation_map(
                    ratiotype,
                    star,
                    outputfilename=filepaths.plotfile(ratnamestr + "_cormap"),
                )
            except Exception as e:
                print(
                    f"\nRatio correlation map for {ratiotype} sequence failed with the error:",
                    e,
                )

    for glitchseq in plotconfig.freqplots:
        if glitchseq not in freqtypes.glitches:
            continue
        glitchnamestr = f"glitches_{glitchseq}"
        try:
            plot_seismic.glitchplot(
                star,
                glitchseq,
                glitchparams,
                maxPath=path,
                maxInd=np.argmax(selectedmodels[path].logPDF),
                outputfilename=filepaths.plotfile(glitchnamestr),
            )
        except Exception as e:
            print(
                f"\nGlitch plot for {glitchseq} sequence failed with the error:",
                e,
            )
        if glitchseq != "glitches":
            # TODO(Amalie): Implement different approach
            ratiotype = glitchseq[1:]
            ratnamestr = f"ratios_{ratiotype}"
            if ratiotype not in obsfreqdata:
                mask = np.where(
                    np.isin(obsfreqdata[glitchseq]["data"][2, :], [1.0, 2.0, 10.0])
                )[0]
                obsfreqdata[ratiotype] = {
                    "data": obsfreqdata[glitchseq]["data"][:, mask],
                    "cov": obsfreqdata[glitchseq]["cov"][np.ix_(mask, mask)],
                }
            try:
                plot_seismic.ratioplot(
                    star,
                    maxjoinkeys,
                    maxjoin,
                    maxmodkey,
                    maxmod,
                    ratiotype,
                    outputfilename=filepaths.plotfile(ratnamestr),
                    threepoint=inputstar.threepoint,
                    interp_ratios=inputstar.interp_ratios,
                )
            except Exception as e:
                print(
                    f"\nRatio plot for {ratiotype} sequence failed with the error:",
                    e,
                )
        if inputstar.correlations:
            try:
                plot_seismic.correlation_map(
                    glitchseq,
                    star,
                    outputfilename=filepaths.plotfile(glitchnamestr + "_cormap"),
                )
            except Exception as e:
                print(
                    f"\nGlitch correlation map for {glitchseq} sequence failed with the error:",
                    e,
                )

    for epsseq in plotconfig.freqplots:
        if epsseq not in freqtypes.epsdiff:
            continue
        try:
            epsnamestr = f"epsdiff_{epsseq}"
            plot_seismic.epsilon_difference_diagram(
                mod=maxmod,
                modkey=maxmodkey,
                moddnu=maxmoddnu,
                sequence=epsseq,
                star=star,
                outputfilename=filepaths.plotfile(epsnamestr),
            )
        except Exception as e:
            print(
                f"\nEpsilon difference plot for {epsseq} sequence failed with the error:",
                e,
            )

        if inputstar.correlations:
            try:
                plot_seismic.correlation_map(
                    epsseq,
                    star,
                    outputfilename=filepaths.plotfile(epsnamestr + "_cormap"),
                )
            except Exception as e:
                print(
                    f"\nEpsilon difference correlation map for {epsseq} sequence failed with the error:",
                    e,
                )
