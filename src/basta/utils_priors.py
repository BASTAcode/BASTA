"""
This BASTA module handles functions related to priors.
"""

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
    inferencesettings.boxpriors["gridcut"] = core.PriorEntry(
        kwargs={"gridcut": gridcut},
        limits=None,
    )


def get_dnufrac_limits(
    priors: dict[str, core.PriorEntry],
    inputstar: core.InputStar,
    dnutype: str = "dnufit",
) -> dict[str, tuple[float, float]]:
    dnufrac = priors.get("dnufrac")

    dnufrac_limits: dict[str, tuple[float, float]] = {}
    if isinstance(dnufrac, core.PriorEntry):
        if dnutype in inputstar.globalseismicparams.params:
            dnu_value, dnu_error = inputstar.globalseismicparams.get_scaled(dnutype)
            frac = dnufrac.kwargs[dnutype]
            three_sigma = 3 * dnu_error
            delta = min(three_sigma, frac * dnu_value)
            dnufrac_limits = {dnutype: (max(0.0, dnu_value - delta), dnu_value + delta)}
    return dnufrac_limits


def get_anchormodecut(
    priors: dict[str, core.PriorEntry],
    inputstar: core.InputStar,
    dnutype: str = "dnufit",
) -> dict[str, tuple[float, float]]:
    # anchormodecut
    pass


def get_limits(
    inputstar: core.InputStar,
    inferencesettings: core.InferenceSettings,
    distancelimits: dict[str, tuple[float, float]] | None = None,
):
    """
    This function computes the bounds specified as boxpriors by the user or the range of the grid.

    The user input can be specified using the `dnufrac` keyword (percentage of given dnu) to constain `dnufit` or as boxpriors.
    For the boxpriors, a number of keywords can be given and they are interpreted as
    - `min`: an absolute lower bound.
    - `max`: an absolute upper bound.
    - `abstol`: symmetric bound around the observed stellar property
    - `sigmacut`: symmetric bound around the observed stellar property

    As they can overlap, we intersect all apllicable constraints.
    """
    limits: dict[str, tuple[float, float]] = {}
    priors = inferencesettings.boxpriors
    if distancelimits is None:
        distancelimits = {}

    params = {
        **inputstar.classicalparams.params,
        **inputstar.globalseismicparams.params,
    }

    # Unpack special cases such as gridcut and dnufrac
    gridcut = priors.get("gridcut")
    gridcut_limits: dict[str, tuple[float, float]] = (
        gridcut if isinstance(gridcut, dict) else {}
    )

    dnufrac_limits = get_dnufrac_limits(priors=priors, inputstar=inputstar)

    all_dimensions = (
        set(priors.keys())
        | set(distancelimits.keys())
        | set(gridcut_limits.keys())
        | set(dnufrac_limits.keys())
    )
    for dimension in all_dimensions:
        if dimension in [
            "gridcut",
            "dnufrac",
        ]:
            continue  # as we are processing its contents instead.

        star_param = params.get(dimension)
        prior_entry = priors.get(dimension)
        dist_limit = distancelimits.get(dimension)
        gridcut_limit = gridcut_limits.get(dimension)
        dnufrac_limit = dnufrac_limits.get(dimension)

        prior_bounds: list[tuple[float, float]] = []
        if prior_entry:
            # star_param must be Fitparam, since we have priors.
            assert isinstance(star_param, tuple)  # Fitparam is a tuple
            kwargs = prior_entry.kwargs
            if "min" in kwargs:
                prior_bounds.append((kwargs["min"], float("inf")))
            if "max" in kwargs:
                prior_bounds.append((float("-inf"), kwargs["max"]))
            if "abstol" in kwargs:
                tol = kwargs["abstol"]
                if star_param is None:
                    raise ValueError(
                        f"Missing stellar value for dimension '{dimension}'"
                    )
                prior_bounds.append((star_param[0] - tol, star_param[0] + tol))
            if "sigmacut" in kwargs:
                if star_param is None:
                    raise ValueError(
                        f"Missing stellar value for dimension '{dimension}'"
                    )
                sigma = star_param[1]
                sigcut = kwargs["sigmacut"]
                prior_bounds.append(
                    (star_param[0] - sigcut * sigma, star_param[0] + sigcut * sigma)
                )

        prior_limit = None
        if prior_bounds:
            lower = max(b[0] for b in prior_bounds)
            upper = min(b[1] for b in prior_bounds)
            if lower > upper:
                raise ValueError(
                    f"Incompatible prior limits for '{dimension}': {prior_bounds}"
                )
            prior_limit = (lower, upper)

        candidate_limits = [
            l
            for l in [prior_limit, dist_limit, gridcut_limit, dnufrac_limit]
            if l is not None
        ]

        # Combine limits by intersecting ranges if this applies
        if candidate_limits:
            lower = max(l[0] for l in candidate_limits)
            upper = min(l[1] for l in candidate_limits)
            if lower > upper:
                raise ValueError(
                    f"Incompatible combined limits for '{dimension}': {candidate_limits}"
                )
            limits[dimension] = (lower, upper)
        else:
            raise NotImplementedError(
                f"Limit generation not implemented for '{dimension}'"
            )

    return limits
