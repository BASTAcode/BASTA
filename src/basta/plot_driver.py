import numpy as np

from typing import Any

from basta import core, freq_fit, plot_seismic, stats, surfacecorrections
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
    selectedmodels: dict[str, stats.priorlogPDF | stats.Trackstats],
    path: str,
    ind: int,
    quantities_at_runtime: dict[str, Any] | None = None,
) -> None:
    """
    Driver for producing all seismic related plots

    Parameters
    ----------
    Grid : hdf5 file
        Stellar models, as tracks or isochrones
    selectedmodels : dict
        Contains information on all models with a non-zero likelihood.
    path : str
        Path to the highest likelihood track/isocohrone in the grid
    ind : int
        Index of the highest likelihood model in the track

    """

    freqplots = plotconfig.freqplots

    assert star.modes is not None

    allfplots = freqplots[0] == True  # noqa: E712
    if "allechelle" in freqplots:
        freqplots += ["dupechelle", "echelle", "pairechelle"]
    if any(x in freqtypes.rtypes for x in freqplots):
        freqplots += ["ratios"]
    try:
        rawmaxmod = Grid[path + "/osc"][ind]
        rawmaxmodkey = Grid[path + "/osckey"][ind]
        model_modes = core.make_model_modes_from_ln_freqinertia(rawmaxmodkey, rawmaxmod)
        assert star.modes is not None
        joinedmodes = freq_fit.calc_join(star.modes, model_modes)
        maxmoddnu = Grid[path + "/dnufit"][ind]
    except Exception as e:
        print("\nFrequency plots initialisation failed with the error:", e)
        return

    # Extract the original observed dnu for use on the echelle diagrams
    # --> (equivalent to re-scaling if solar scaling activated)

    x = plot_seismic.EchellePlotBase(
        selectedmodels=selectedmodels,
        Grid=Grid,
        model_modes=model_modes,
        joinedmodes=joinedmodes,
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

    corrected_joinedmodes, coeffs = surfacecorrections.apply_surfacecorrection(
        joinedmodes=joinedmodes, star=star
    )
    if coeffs is not None:
        print("Surface correction coefficient(s):", *coeffs)

    x = plot_seismic.EchellePlotBase(
        selectedmodels=selectedmodels,
        Grid=Grid,
        model_modes=model_modes,
        joinedmodes=joinedmodes,
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
                joinedmodes,
                model_modes,
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
                quantities_at_runtime["glitches"],
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
                    joinedmodes,
                    model_modes,
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
                model_modes=model_modes,
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
