"""
Parallax fitting and computation of distances
"""

import os
import warnings
from collections.abc import Callable
from pathlib import Path

import dustmaps  # type: ignore[import]
import h5py  # type: ignore[import]
import matplotlib as mpl
import numpy as np
import scipy.stats  # type: ignore[import]
from astropy.coordinates import SkyCoord  # type: ignore[import]
from astropy.utils.exceptions import AstropyWarning  # type: ignore[import]
from healpy import ang2pix  # type: ignore[import]
from scipy.interpolate import interp1d  # type: ignore[import]

import basta.constants as cnsts
import basta.utils_distances as udist
from basta import core, stats

mpl.use("Agg")
import matplotlib.pyplot as plt

# Don't print Astropy warnings (catch error caused by mock'ing astropy in Sphinx)
try:
    warnings.filterwarnings("ignore", category=AstropyWarning)
except AssertionError:
    pass

try:
    from basta._dustpath import __dustpath__
except ModuleNotFoundError:
    print("\nCannot find path to dustmaps. Did you run 'setup.py'?\n")
    raise


def get_EBV_along_LOS(
    distanceparams: core.DistanceParameters,
) -> Callable[[np.ndarray], np.ndarray]:
    """
    Returns color excess E(B-V) for a line of sight, mainly using a
    pre-downloaded 3D extinction map provided by Green et al. 2015/2018 - see
    http://argonaut.skymaps.info/.

    The extinction map is only computed for distance modulus between
    :math:`4 < m-M < 19` in units of magnitude.

    Parameters
    ----------
    distanceparams : dict
        Dictionary with parameters required for constructing likelihood for absolute magnitude constraint, i.e. coordinates, given dustframe, etc..

    Returns
    -------
    EBV_along_LOS : function
       Reddening or excess color function along the given line-of-sight
    """
    # If EBV is given, use that instead of reading it from the dust map
    if len(distanceparams.EBV) > 0:
        return lambda x: np.ones(len(x)) * distanceparams.EBV[1]

    # Convert to galactic coordinates
    if distanceparams.coordinates["frame"].lower() == "icrs":
        c = SkyCoord(
            ra=distanceparams.coordinates["RA"],
            dec=distanceparams.coordinates["DEC"],
            frame="icrs",
            unit="deg",
        )
    elif distanceparams.coordinates["frame"].lower() == "galactic":
        c = SkyCoord(
            l=distanceparams.coordinates["lon"],
            b=distanceparams.coordinates["lat"],
            frame="galactic",
            unit="deg",
        )
    else:
        raise ValueError("Unknown dust map frame for computing reddening!")

    # Load extinction data cube
    pathmap = os.path.join(__dustpath__, "bayestar/bayestar2019.h5")
    dcube = h5py.File(pathmap, "r")

    # Distance modulus bins
    bin_edges = dcube["/pixel_info"].attrs["DM_bin_edges"]
    dmbin = bin_edges + (bin_edges[1] - bin_edges[0]) / 2.0

    # If webquery fails use local copy of dustmap
    try:
        bayestar = dustmaps.BayestarQuery(version="bayestar2019")
        Egr_samples = bayestar(c, mode="samples")
    except Exception:
        # contains positional info
        pinfo = dcube["/pixel_info"][:]
        nsides = np.unique(dcube["/pixel_info"][:]["nside"])

        # Convert coordinates to galactic frame
        lon = c.galactic.l.deg
        lat = c.galactic.b.deg

        # Convert l,b[deg] to theta,phi[rad]
        theta = np.pi / 2.0 - lat * np.pi / 180.0
        phi = lon * np.pi / 180.0

        # To check if we are within the maps coordinates
        Egr_samples = np.array([np.nan])

        # Run through nsides
        for nside in reversed(nsides):
            healpixNside = ang2pix(nside, theta, phi, nest=True)
            # Find the one with the correct nside and the correct healpix
            indNside = [
                i for i, x in enumerate(pinfo) if x[0] == nside and x[1] == healpixNside
            ]
            if indNside:
                index = indNside[0]
                Egr_samples = dcube["/samples"][index]
                break

    # If coordinates outside dust map, use Schegel
    if np.isnan(Egr_samples).any():
        print("WARNING: Coordinates outside dust map boundaries!")
        print("Default to Schegel 1998 dust map")
        sfd = dustmaps.sfd.SFDQuery()
        return lambda x: np.full_like(x, sfd(c))

    Egr_med, Egr_err = [], []
    for i in range(len(dmbin)):
        Egr_med.append(np.nanmedian(Egr_samples[:, i]))
        Egr_err.append(np.nanstd(Egr_samples[:, i]))

    Egr_med_fun = interp1d(
        dmbin, Egr_med, bounds_error=False, fill_value=(0, np.max(Egr_med))
    )
    Egr_err_fun = interp1d(
        dmbin, Egr_err, bounds_error=False, fill_value=np.max(Egr_err)
    )

    dcube.close()

    def EBV_along_LOS(dm):
        Egr = np.asarray(np.random.normal(Egr_med_fun(dm), Egr_err_fun(dm)))
        EBV = cnsts.extinction.Conv_Bayestar * Egr
        return EBV

    return EBV_along_LOS


def get_EBV(
    dist: np.ndarray,
    EBV_along_LOS: Callable[[np.ndarray], np.ndarray],
    debug: bool = False,
    debug_dirpath: Path | str = "",
) -> np.ndarray:
    """
    Estimate E(B-V) by drawing distances from a normal parallax
    distribution with EDSD prior.

    Parameters
    -----
    dist : array
        The drawn distances
    EBV_along_LOS : func
        EBV function.
    debug : bool, optional
        Debug flag.
        If True, this function outputs two plots, one of distance modulus
        vs. E(B-V) and a histogram of the E(B-V).
    debug_dirpath : str, optional
        Name of directory of where to put plots outputted if debug is True.

    Returns
    -------
    EBVs : array
        E(B-V) at distances specified in `dist` along the line-of-sight
    """
    dmod = 5 * np.log10(dist / 10)
    EBVs = EBV_along_LOS(dmod)

    if debug:
        plt.figure()
        plt.plot(dmod, EBVs, ".")
        plt.xlabel("dmod")
        plt.ylabel("E(B-V)")
        plt.savefig(f"{debug_dirpath}_DEBUG_dmod_EBVs.png")
        plt.close()

    return EBVs


def get_absorption(EBV: np.ndarray, fitparams: dict, filt: str) -> np.ndarray:
    """
    Compute extinction coefficient Rzeta for band zeta.
    Using parameterized law from Casagrande & VandenBerg 2014.

    Valid for:
    logg = 4.1
    Teff = 5250 - 7000K
    Fe/H = -2.0 - 0.25
    a/Fe = -0.4 - 0.4

    Assume nominal reddening law with RV=3.1. In a band zeta, Azeta = Rzeta*E(B-V).

    Parameters
    ----------
    EBV : array
        E(B-V) values
    fitparams : dict
        The fitting params in inputparams.
    filt : str
        Name of the given filter

    Returns
    -------
    R*EBV : array
        Extinction coefficient times E(B-V)
    """
    N = len(EBV)
    table = cnsts.extinction.R
    i_filter = table["Filter"] == filt
    if not any(i_filter) or table["RZ_mean"][i_filter] == 0:
        print("WARNING: Unknown extinction coefficient for filter: " + filt)
        print("         Using reddening law coefficient R = 0.")
        return np.zeros(N)

    metal = "MeH" if "MeH" in fitparams else "FeH"
    if "Teff" not in fitparams or metal not in fitparams:
        R = np.ones_like(EBV) * table["RZ_mean"][i_filter].item()
    else:
        Teff_val, Teff_err = fitparams["Teff"]
        metal_val, metal_err = fitparams[metal]
        Teff = np.random.normal(Teff_val, Teff_err, size=N)
        FeH = np.random.normal(metal_val, metal_err, size=N)
        a0 = table["a0"][i_filter].item()
        a1 = table["a1"][i_filter].item()
        a2 = table["a2"][i_filter].item()
        a3 = table["a3"][i_filter].item()
        T4 = 1e-4 * Teff
        R = a0 + T4 * (a1 + a2 * T4) + a3 * FeH
    return R * EBV


def add_absolute_magnitudes(
    star: core.InputStar,
    filepaths: core.FilePaths,
    inferencesettings: core.InferenceSettings,
    outputoptions: core.OutputOptions,
    n: int = 1000,
    k: int = 1000,
) -> tuple[core.AbsoluteMagnitudes, dict[str, (float, float)]]:
    """
    Convert apparent magnitudes to absolute magnitudes using the distance and add it to `inputparams`.
    Extinction E(B-V) is estimated based on Green et al. (2015) dust map.
    Extinction is converted to reddening using Casagrande & VandenBerg 2014.
    The converted colors and magnitudes are added to fitsparams.

    Parameters
    ----------
    inputparams : dict
        Inputparams used in BASTA run.
    n : int
        Number of samples from parallax range
    k : int
        Number of samples from apparent magnitude range.
    debug_dirpath : str, optional
        Name of directory of where to put plots outputted if debug is True.
    debug : bool, optional
        Debug flag. If True, debugging plots will be outputted.
    use_gaussian_priors : bool, optional
        If True, gaussian priors will be used for apparent magnitude in
        the distance computation.

    Returns
    -------
    inputparams : dict
        Modified version of inputparams including absolute magnitudes.
    """
    if len(star.distanceparams.params["parallax"]) < 1:
        return {
            "magnitudes": {},
            "absorption": {},
            "prior_EBV": [],
            "prior_distance": [],
        }, {}

    if outputoptions.verbose:
        print("\nPreparing distance/parallax/magnitude input ...", flush=True)

    distanceparams = star.distanceparams
    fitparams = distanceparams.params

    # Get apparent magnitudes from input data
    # mobs = distanceparams.m
    # mobs_err = distanceparams.m_err
    magnitudes = distanceparams.magnitudes

    if len(magnitudes.keys()) < 1:
        raise ValueError("No filters were given")

    # Convert the inputted parallax in mas to as
    plxobs = fitparams["parallax"][0] * 1e-3
    plxobs_err = fitparams["parallax"][1] * 1e-3
    L = udist.EDSD(None, None) * 1e3

    # Sample distances more densely around the mode of the distance distribution
    # See Bailer-Jones 2015, Eq 19
    coeffs = [1 / L, -2, plxobs / (plxobs_err**2), -1 / (plxobs_err**2)]
    roots = np.roots(coeffs)
    if np.sum(np.isreal(roots)) == 1:
        (mode,) = np.real(roots[np.isreal(roots)])
    else:
        assert np.sum(np.isreal(roots)) == 3
        if plxobs >= 0:
            mode = np.amin(np.real(roots[np.isreal(roots)]))
        else:
            (mode,) = np.real(roots[roots > 0])

    # By sampling linearly in quantiles, the probablity mass is equal for the samples
    bla = scipy.stats.norm.cdf(0, loc=mode, scale=1000) + 0.01
    dist = scipy.stats.norm.ppf(
        np.linspace(bla, 0.96, n - n // 2), loc=mode, scale=1000
    )
    lindist = 10 ** np.linspace(-0.4, 4.4, n // 2)
    dist = np.concatenate([dist, lindist])
    dist = np.sort(dist)

    lldist = udist.compute_distlikelihoods(
        dist,
        plxobs,
        plxobs_err,
        L,
        debug=outputoptions.debug,
        debug_dirpath=filepaths.extradirectory,
    )
    dists = np.repeat(dist, k)
    lldists: np.ndarray = np.repeat(lldist, k)

    # Get EBV values
    EBV_along_LOS = get_EBV_along_LOS(distanceparams=distanceparams)
    EBV = get_EBV(
        dist=dist,
        EBV_along_LOS=EBV_along_LOS,
        debug=outputoptions.debug,
        debug_dirpath=filepaths.extradirectory,
    )
    EBVs = np.repeat(EBV, k)

    new_As = {}
    new_magnitudes: dict[str, core.AbsoluteMagnitude] = {}
    llabsms_joined = np.zeros(n * k)
    for filt in magnitudes.keys():
        # Sample apparent magnitudes over the entire parameter range
        if filt in cnsts.distanceranges.filters:
            m = np.linspace(
                cnsts.distanceranges.filters[filt]["min"],
                cnsts.distanceranges.filters[filt]["max"],
                k - k // 2,
            )
        else:
            m = np.linspace(-10, 25, k - k // 2)
        m = np.concatenate(
            [
                m,
                scipy.stats.norm.ppf(
                    np.linspace(0.04, 0.96, k // 2),
                    loc=magnitudes[filt][0],
                    scale=magnitudes[filt][1],
                ),
            ]
        )
        m = np.sort(m)

        llm = udist.compute_mslikelihoods(m, magnitudes[filt][0], magnitudes[filt][1])
        ms = np.tile(m, n)
        llms: np.ndarray = np.tile(llm, n)
        assert len(dists) == len(ms) == n * k

        A = get_absorption(EBV, fitparams, filt)
        As = np.repeat(A, k)

        absms = udist.compute_absmag(ms, dists, As)

        # Construct likelihood distribution
        llabsms = llms + lldists
        llabsms_joined += llabsms
        llabsms -= np.amax(llabsms)
        labsms = np.exp(llabsms - np.log(np.sum(np.exp(llabsms))))

        # Create prior by interpolating a histogram and add it to inputparams
        # Use bins from non-weighted histogram in the weighted histogram
        bin_edges = np.histogram_bin_edges(absms, bins="auto")
        like, bins = np.histogram(absms, bins=bin_edges, weights=labsms, density=True)
        M_prior = interp1d(
            bins[:-1] + np.diff(bins) / 2.0, like, fill_value=0, bounds_error=False
        )

        absms_qs = stats.quantile_1D(absms, labsms, cnsts.statdata.quantiles)
        new_As[filt] = list(stats.quantile_1D(As, labsms, cnsts.statdata.quantiles))

        # Extended dictionary for use in Kiel diagram, and clarifying it as a prior
        new_magnitudes[filt] = {
            "prior": M_prior,
            "median": absms_qs[0],
            "errp": np.abs(absms_qs[2] - absms_qs[0]),
            "errm": np.abs(absms_qs[0] - absms_qs[1]),
        }

    # # Get an estimate from all filters
    labsms_joined = np.exp(llabsms_joined - np.log(np.sum(np.exp(llabsms_joined))))

    prior_EBV = list(stats.quantile_1D(EBVs, labsms_joined, cnsts.statdata.quantiles))
    prior_distance = list(
        stats.quantile_1D(dists, labsms_joined, cnsts.statdata.quantiles)
    )

    # Constrain metallicity within the limits of the color transformations
    metal = "MeH" if "MeH" in inferencesettings.fitparams else "FeH"
    distancelimits: dict[str, (float, float)] = {}
    distancelimits[metal] = [
        cnsts.metallicityranges.values["metallicity"]["min"],
        cnsts.metallicityranges.values["metallicity"]["max"],
    ]

    if outputoptions.verbose:
        print("Done!")

    return {
        "magnitudes": new_magnitudes,
        "absorption": new_As,
        "prior_EBV": prior_EBV,
        "prior_distance": prior_distance,
    }, distancelimits
