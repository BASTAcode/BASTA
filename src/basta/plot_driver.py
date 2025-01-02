import numpy as np

from basta import freq_fit, plot_seismic
from basta import utils_seismic as su
from basta.constants import freqtypes


def plot_all_seismic(
    freqplots,
    Grid,
    fitfreqs,
    obsfreqmeta,
    obsfreqdata,
    obskey,
    obs,
    obsintervals,
    selectedmodels,
    path,
    ind,
    plotfname,
    nameinplot=False,
    dnusurf=None,
    glitchparams=None,
    debug=False,
):
    """
    Driver for producing all seismic related plots

    Parameters
    ----------
    freqplots : list
        Plots to be produced
    Grid : hdf5 file
        Stellar models, as tracks or isochrones
    fitfreqs : dict
        Input frequency fitting options/controls
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

    # Check which plots to create
    allfplots = freqplots[0] == True
    if any(x == "allechelle" for x in freqplots):
        freqplots += ["dupechelle", "echelle", "pairechelle"]
    if any(x in freqtypes.rtypes for x in freqplots):
        freqplots += ["ratios"]
    try:
        rawmaxmod = Grid[path + "/osc"][ind]
        rawmaxmodkey = Grid[path + "/osckey"][ind]
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
        maxmoddnu = Grid[path + "/dnufit"][ind]
    except Exception as e:
        print("\nFrequency plots initialisation failed with the error:", e)
        return None

    # Extract the original observed dnu for use on the echelle diagrams
    # --> (equivalent to re-scaling if solar scaling activated)
    plotdnu = fitfreqs["dnu_obs"]

    if allfplots or "echelle" in freqplots:
        try:
            plot_seismic.echelle(
                selectedmodels=selectedmodels,
                Grid=Grid,
                obs=obs,
                obskey=obskey,
                mod=maxmod,
                modkey=maxmodkey,
                dnu=plotdnu,
                join=maxjoin,
                joinkeys=maxjoinkeys,
                pairmode=False,
                duplicatemode=False,
                outputfilename=plotfname.format("echelle_uncorrected"),
            )
        except Exception as e:
            print("\nUncorrected echelle failed with the error:", e)

    if allfplots or "pairechelle" in freqplots:
        try:
            plot_seismic.echelle(
                selectedmodels,
                Grid,
                obs,
                obskey,
                mod=maxmod,
                modkey=maxmodkey,
                dnu=plotdnu,
                join=maxjoin,
                joinkeys=maxjoinkeys,
                pairmode=True,
                duplicatemode=False,
                outputfilename=plotfname.format("pairechelle_uncorrected"),
            )
        except Exception as e:
            print("\nUncorrected pairechelle failed with the error:", e)

    if allfplots or "dupechelle" in freqplots:
        try:
            plot_seismic.echelle(
                selectedmodels,
                Grid,
                obs,
                obskey,
                mod=maxmod,
                modkey=maxmodkey,
                dnu=plotdnu,
                join=maxjoin,
                joinkeys=maxjoinkeys,
                duplicatemode=True,
                pairmode=True,
                outputfilename=plotfname.format("dupechelle_uncorrected"),
            )
        except Exception as e:
            print("\nUncorrected dupechelle failed with the error:", e)

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
        try:
            plot_seismic.echelle(
                selectedmodels,
                Grid,
                obs,
                obskey,
                mod=maxmod,
                modkey=maxmodkey,
                dnu=plotdnu,
                join=corjoin,
                joinkeys=maxjoinkeys,
                freqcor=fitfreqs["fcor"],
                coeffs=coeffs,
                scalnu=fitfreqs["numax"],
                pairmode=False,
                duplicatemode=False,
                outputfilename=plotfname.format("echelle"),
            )
        except Exception as e:
            print("\nEchelle failed with the error:", e)

    if allfplots or "pairechelle" in freqplots:
        try:
            plot_seismic.echelle(
                selectedmodels,
                Grid,
                obs,
                obskey,
                mod=maxmod,
                modkey=maxmodkey,
                dnu=plotdnu,
                join=corjoin,
                joinkeys=maxjoinkeys,
                freqcor=fitfreqs["fcor"],
                coeffs=coeffs,
                scalnu=fitfreqs["numax"],
                pairmode=True,
                duplicatemode=False,
                outputfilename=plotfname.format("pairechelle"),
            )
        except Exception as e:
            print("\nPairechelle failed with the error:", e)

    if allfplots or "dupechelle" in freqplots:
        try:
            plot_seismic.echelle(
                selectedmodels,
                Grid,
                obs,
                obskey,
                mod=maxmod,
                modkey=maxmodkey,
                dnu=plotdnu,
                join=corjoin,
                joinkeys=maxjoinkeys,
                freqcor=fitfreqs["fcor"],
                coeffs=coeffs,
                scalnu=fitfreqs["numax"],
                duplicatemode=True,
                pairmode=True,
                outputfilename=plotfname.format("dupechelle"),
            )
        except Exception as e:
            print("\nDupechelle failed with the error:", e)

    if "freqcormap" in freqplots or debug:
        try:
            plot_seismic.correlation_map(
                "freqs",
                obsfreqdata,
                plotfname.format("freqs_cormap"),
                obskey=obskey,
            )
        except Exception as e:
            print("\nFrequencies correlation map failed with the error:", e)

    if obsfreqmeta["getratios"]:
        for ratseq in obsfreqmeta["ratios"]["plot"]:
            try:
                ratnamestr = "ratios_{0}".format(ratseq)
                plot_seismic.ratioplot(
                    obsfreqdata,
                    maxjoinkeys,
                    maxjoin,
                    maxmodkey,
                    maxmod,
                    ratseq,
                    outputfilename=plotfname.format(ratnamestr),
                    threepoint=fitfreqs["threepoint"],
                    interp_ratios=fitfreqs["interp_ratios"],
                )
            except Exception as e:
                print(
                    "\nRatio plot for {} sequence failed with the error:".format(
                        ratseq
                    ),
                    e,
                )

            if fitfreqs["correlations"]:
                try:
                    plot_seismic.correlation_map(
                        ratseq,
                        obsfreqdata,
                        outputfilename=plotfname.format(ratnamestr + "_cormap"),
                    )
                except Exception as e:
                    print(
                        "\nRatio correlation map for {} sequence failed with the error:".format(
                            ratseq
                        ),
                        e,
                    )

    if obsfreqmeta["getglitch"]:
        for glitchseq in obsfreqmeta["glitch"]["plot"]:
            glitchnamestr = "glitches_{0}".format(glitchseq)
            try:
                plot_seismic.glitchplot(
                    obsfreqdata,
                    glitchseq,
                    glitchparams,
                    maxPath=path,
                    maxInd=np.argmax(selectedmodels[path].logPDF),
                    outputfilename=plotfname.format(glitchnamestr),
                )
            except Exception as e:
                print(
                    "\nGlitch plot for {} sequence failed with the error:".format(
                        glitchseq
                    ),
                    e,
                )
            if glitchseq != "glitches":
                ratseq = glitchseq[1:]
                ratnamestr = "ratios_{0}".format(ratseq)
                if ratseq not in obsfreqdata:
                    mask = np.where(
                        np.isin(obsfreqdata[glitchseq]["data"][2, :], [1.0, 2.0, 10.0])
                    )[0]
                    obsfreqdata[ratseq] = {
                        "data": obsfreqdata[glitchseq]["data"][:, mask],
                        "cov": obsfreqdata[glitchseq]["cov"][np.ix_(mask, mask)],
                    }
                try:
                    plot_seismic.ratioplot(
                        obsfreqdata,
                        maxjoinkeys,
                        maxjoin,
                        maxmodkey,
                        maxmod,
                        ratseq,
                        outputfilename=plotfname.format(ratnamestr),
                        threepoint=fitfreqs["threepoint"],
                        interp_ratios=fitfreqs["interp_ratios"],
                    )
                except Exception as e:
                    print(
                        "\nRatio plot for {} sequence failed with the error:".format(
                            ratseq
                        ),
                        e,
                    )
            if fitfreqs["correlations"]:
                try:
                    plot_seismic.correlation_map(
                        glitchseq,
                        obsfreqdata,
                        outputfilename=plotfname.format(glitchnamestr + "_cormap"),
                    )
                except Exception as e:
                    print(
                        "\nGlitch correlation map for {} sequence failed with the error:".format(
                            glitchseq
                        ),
                        e,
                    )

    if obsfreqmeta["getepsdiff"]:
        for epsseq in obsfreqmeta["epsdiff"]["plot"]:
            try:
                epsnamestr = "epsdiff_{0}".format(epsseq)
                plot_seismic.epsilon_difference_diagram(
                    mod=maxmod,
                    modkey=maxmodkey,
                    moddnu=maxmoddnu,
                    sequence=epsseq,
                    obsfreqdata=obsfreqdata,
                    outputfilename=plotfname.format(epsnamestr),
                )
            except Exception as e:
                print(
                    "\nEpsilon difference plot for {} sequence failed with the error:".format(
                        epsseq
                    ),
                    e,
                )

            if fitfreqs["correlations"]:
                try:
                    plot_seismic.correlation_map(
                        epsseq,
                        obsfreqdata,
                        outputfilename=plotfname.format(epsnamestr + "_cormap"),
                    )
                except Exception as e:
                    print(
                        "\nEpsilon difference correlation map for {} sequence failed with the error:".format(
                            epsseq
                        ),
                        e,
                    )

    if obsfreqmeta["getepsdiff"] and debug:
        if len(obsfreqmeta["epsdiff"]["plot"]) > 0:
            try:
                plot_seismic.epsilon_difference_components_diagram(
                    mod=maxmod,
                    modkey=maxmodkey,
                    moddnu=maxmoddnu,
                    obs=obs,
                    obskey=obskey,
                    dnudata=obsfreqdata["freqs"]["dnudata"],
                    obsfreqdata=obsfreqdata,
                    obsfreqmeta=obsfreqmeta,
                    outputfilename=plotfname.format("DEBUG_epsdiff_components"),
                )
            except Exception as e:
                print("\nEpsilon difference compoenent plot failed with the error:", e)
    return None
