import numpy as np

from typing import Any

from basta import core, constants, freq_fit, plot_seismic, stats, surfacecorrections
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

    plots = []

    assert star.modes is not None

    if plotconfig.freqplots:
        if isinstance(plotconfig.freqplots[0], list):
            plots.extend(plotconfig.freqplots)
            if "ratios" in plots:
                plots.extend(constants.freqtypes.defaultrtypes)
            if "epsilondifferences" in plots:
                plots.extend(constants.freqtypes.defaultepstypes)
        else:
            if inferencesettings.has_any_seismic_case:
                plots.extend(["dupechelle", "echelle", "pairechelle"])
            if inferencesettings.has_ratios:
                plots.extend(
                    [
                        x
                        for x in constants.freqtypes.rtypes
                        if x in inferencesettings.fitparams
                    ]
                )
            if inferencesettings.has_glitches:
                plots.extend(
                    [
                        x
                        for x in constants.freqtypes.glitches
                        if x in inferencesettings.fitparams
                    ]
                )
            if inferencesettings.has_epsilondifferences:
                plots.extend(
                    [
                        x
                        for x in constants.freqtypes.epsdiff
                        if x in inferencesettings.fitparams
                    ]
                )
            if outputoptions.debug:
                plots.extend(
                    [
                        "cormap",
                    ]
                )

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
    xs = [
        x,
    ]
    labels = [
        "_uncorrected",
    ]

    assert joinedmodes is not None
    corrected_joinedmodes, coeffs = surfacecorrections.apply_surfacecorrection(
        joinedmodes=joinedmodes, star=star
    )
    if coeffs is not None:
        print(f"\nSurface correction coefficient(s):")
        print(np.array2string(coeffs, precision=4, separator=", "))
        print("")

        corr_x = plot_seismic.EchellePlotBase(
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

        xs.append(corr_x)
        labels.append("")

    for x, label in zip(xs, labels):
        if "echelle" in plots:
            plotname = f"echelle{label}"
            try:
                plot_seismic.echelle(
                    x,
                    pairmode=False,
                    duplicatemode=False,
                    outputfilename=filepaths.plotfile(plotname),
                )
            except Exception as e:
                print(f"\n{plotname} failed with the error:", e)

        if "pairechelle" in plots:
            plotname = f"pairechelle{label}"
            try:
                plot_seismic.echelle(
                    x,
                    pairmode=True,
                    duplicatemode=False,
                    outputfilename=filepaths.plotfile(plotname),
                )
            except Exception as e:
                print(f"\n{plotname} failed with the error:", e)

        if "dupechelle" in plots:
            plotname = f"dupechelle{label}"
            try:
                plot_seismic.echelle(
                    x,
                    pairmode=True,
                    duplicatemode=True,
                    outputfilename=filepaths.plotfile(plotname),
                )
            except Exception as e:
                print(f"\n{plotname} failed with the error:", e)

    if "cormap" in plots:
        try:
            plot_seismic.correlation_map(
                "freqs",
                star,
                filepaths.plotfile("freqs_cormap"),
            )
        except Exception as e:
            print("\nFrequencies correlation map failed with the error:", e)

    if any([x in constants.freqtypes.rtypes for x in plots]):
        for sequence in constants.freqtypes.rtypes:
            if not sequence in plots:
                continue
            try:
                ratnamestr = f"ratios_{sequence}"
                plot_seismic.ratioplot(
                    star=star,
                    joinedmodes=joinedmodes,
                    model_modes=model_modes,
                    sequence=sequence,
                    outputfilename=filepaths.plotfile(ratnamestr),
                    kwargs_ratios=inferencesettings.kwargs_ratios,
                    interp_ratios=inferencesettings.interp_ratios,
                )
            except Exception as e:
                print(
                    f"Ratio plot for {sequence} sequence failed with the error:",
                    e,
                )
            if "cormap" in plots:
                try:
                    plot_seismic.correlation_map(
                        sequence,
                        star,
                        outputfilename=filepaths.plotfile(ratnamestr + "_cormap"),
                    )
                except Exception as e:
                    print(
                        f"Ratio correlation map for {sequence} sequence failed with the error:",
                        e,
                    )

    if any([x in constants.freqtypes.glitches for x in plots]):
        for sequence in constants.freqtypes.glitches:
            if not sequence in plots:
                continue
            glitchnamestr = f"glitches_{sequence}"
            assert quantities_at_runtime is not None
            try:
                plot_seismic.glitchplot(
                    star,
                    sequence,
                    quantities_at_runtime["glitches"],
                    maxPath=path,
                    maxInd=np.argmax(selectedmodels[path].logPDF),
                    outputfilename=filepaths.plotfile(glitchnamestr),
                )
            except Exception as e:
                print(
                    f"\nGlitch plot for {sequence} sequence failed with the error:",
                    e,
                )

            # TODO(Amalie) Fix this plot
            """
            ratiotype = sequence[1:]
            ratnamestr = f"ratios_{ratiotype}"
            if ratiotype not in obsfreqdata:
                mask = np.where(
                    np.isin(obsfreqdata[sequence]["data"][2, :], [1.0, 2.0, 10.0])
                )[0]
                obsfreqdata[ratiotype] = {
                    "data": obsfreqdata[sequence]["data"][:, mask],
                    "cov": obsfreqdata[sequence]["cov"][np.ix_(mask, mask)],
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
            """
            if "cormap" in plots:
                try:
                    plot_seismic.correlation_map(
                        sequence,
                        star,
                        outputfilename=filepaths.plotfile(glitchnamestr + "_cormap"),
                    )
                except Exception as e:
                    print(
                        f"\nGlitch correlation map for {sequence} sequence failed with the error:",
                        e,
                    )

    if any([x in constants.freqtypes.epsdiff for x in plots]):
        for sequence in constants.freqtypes.epsdiff:
            if not sequence in plots:
                continue
            try:
                epsnamestr = f"epsdiff_{sequence}"
                plot_seismic.epsilon_difference_diagram(
                    model_modes=model_modes,
                    model_dnu=maxmoddnu,
                    sequence=sequence,
                    star=star,
                    outputfilename=filepaths.plotfile(epsnamestr),
                )
            except Exception as e:
                print(
                    f"\nEpsilon difference plot for {sequence} sequence failed with the error:",
                    e,
                )

            if "cormap" in plots:
                try:
                    plot_seismic.correlation_map(
                        sequence,
                        star,
                        outputfilename=filepaths.plotfile(epsnamestr + "_cormap"),
                    )
                except Exception as e:
                    print(
                        f"\nEpsilon difference correlation map for {sequence} sequence failed with the error:",
                        e,
                    )
