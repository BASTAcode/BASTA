import h5py  # type: ignore[import]
import numpy as np

from basta import core
from basta import utils_general as util


def gridlimits(
    grid: h5py.File,
    gridheader: util.GridHeader,
    gridinfo: util.GridInfo,
    inferencesettings: core.InferenceSettings,
    outputoptions: core.OutputOptions,
) -> None:
    """
    Refactor of grid cut section

    Check if any specified limit in prior is in header, and can be used to
    skip computation of models, in order to speed up computation
    """
    limits = list(inferencesettings.boxpriors.keys())

    gridcut = {}

    # Determine header path
    if "tracks" in gridheader["gridtype"]:
        headerpath = "header/"
    elif "isochrones" in gridheader["gridtype"]:
        headerpath = f"header/{gridinfo['defaultpath']}"
        if "FeHini" in limits:
            print("Warning: Dropping prior in FeHini, redundant for isochrones!")
            inferencesettings.boxpriors.pop("FeHini")
    else:
        headerpath = None

    if headerpath:
        header_keys = grid[headerpath].keys()

        # TODO(Amalie) wait, here diffusion is explicitly handled - why is it also in bastamain?
        # Extract gridcut params
        gridcut_keys = set(header_keys) & set(limits)
        gridcut = {key: limits.pop(key) for key in gridcut_keys}

        if gridcut:
            print("\nCutting in grid based on sampling parameters ('gridcut'):")
            for cutpar, cutval in gridcut.items():
                if cutpar != "dif":
                    print(f"* {cutpar}: {cutval}")

            # Special handling for diffusion switch
            if "dif" in gridcut:
                # Expecting value like [-inf, 0.5] or [0.5, inf]
                switch = np.where(np.array(gridcut["dif"]) == 0.5)[0][0]
                print(
                    f"* Only considering tracks with diffusion turned {'on' if switch == 1 else 'off'}!"
                )
    # TODO(Amalie) I should probably change this so it aligns with the application of the prior
    inferencesettings.boxpriors["gridcut"] = core.PriorEntry(
        kwargs={"gridcut": gridcut},
        limits=None,
    )


def dnufrac_prior(
    star: core.Star,
    inferencesettings: core.InferenceSettings,
    outputoptions: core.OutputOptions,
    priorid: str = "dnufrac",
    dnu_name: str = "dnufit",
) -> None:
    if any(np.isin([priorid, dnu_name], inferencesettings.boxpriors)):
        assert dnu_name in star.globalseismicparams.params
        dnufit = star.globalseismicparams.get_scaled(dnu_name)
        dnufrac = inferencesettings.boxpriors["dnufrac"]["dnufit"]
        dnufit_frac = dnufrac * dnufit[0]
        dnuerr = max(3 * dnufit[1], dnufit_frac)
        limits = [
            min(dnufit[0] - dnuerr, 0),
            dnufit[0] + dnuerr,
        ]
        inferencesettings.boxpriors[dnu_name].limits = limits
