"""
Utility functions for the distance calculation and parallax fitting
"""

from pathlib import Path

import matplotlib as mpl
import numpy as np

mpl.use("Agg")
import matplotlib.pyplot as plt


def compute_absmag(m: np.ndarray, dist: np.ndarray, A: np.ndarray) -> np.ndarray:
    """
    Use distance moduli to compute the absolute magnitudes
    from distances, apparent magnitudes m, and absorption A.

    Parameters
    ----------
    m : np.ndarray
        Apparent magnitude
    dist : np.ndarray
        Distances in parsec
    A : np.ndarray
        Absorption

    Returns
    -------
    M : float
        Absolute magnitude
    """
    return m - 5 * np.log10(dist / 10) - A


def compute_distance_from_mag(
    m: np.ndarray, M: np.ndarray, A: np.ndarray
) -> np.ndarray:
    """
    Compute distance from magnitudes.

    Parameters
    ----------
    m : float
        Apparent magnitude
    M : float
        Absolute magnitude
    A : float
        Absorption

    Returns
    -------
    d : float
        Distance in parsec
    """
    return 10 ** (1 + (m - M - A) / 5.0)


def EDSD(libitem: str | None = None, index: int | None = None) -> float:
    """
    Exponentially decreasing space density prior
    Define characteristic length scale k in kpc

    Parameters
    ----------
    libitem : str, None

    index : int, None

    Returns
    -------
    k : float
        Characteristic length scale in kpc
    """
    k = 1.35
    return k


def loggaussian(x: np.ndarray, mu: float, sigma: float) -> np.ndarray:
    """
    Compute the log of a gaussian.

    Parameters
    ----------
    x : array-like
        Data
    mu : float
        The mean of x
    sigma : float
        Standard deviation of x

    Returns
    -------
    loggaussian : array
        The gaussian data
    """
    lnA = -np.log(sigma * np.sqrt(2 * np.pi))
    return lnA - 0.5 * (((x - mu) / sigma) ** 2)


def compute_distlikelihoods(
    r: np.ndarray,
    plxobs: float,
    plxobs_err: float,
    L: float | None = None,
    debug_dirpath: Path | str = "",
    debug: bool = False,
) -> np.ndarray:
    """
    Compute the likelihood as the product between a gaussian of the parallax
    and the exponentially decreasing volume density prior.
    For the unnormalised posterior, see Eq. 18 in Bailer-Jones 2015.
    This also works for nonpositive parallaxes.

    Parameters
    ----------
    r : float
        The distances in pc
    plxobs : float
        Observed parallax
    plsobs_err : float
        Uncertainty in observed parallax
    L : float
        Characteristic scale length of the galaxy
    debug : bool
        Debug flag, used to trigger certain plots
    debug_dirpath : str
        Path to where to store debug plots

    """
    if L is None:
        L = EDSD(None, None) * 1e3

    lls = loggaussian(1 / r, plxobs, plxobs_err)
    lls += 2 * np.log(r) - r / L - np.log(2) - 3 * np.log(L)
    lls[r <= 0] = -np.inf

    if debug:
        plt.figure()
        plt.plot(r, np.exp(lls), "-")
        plt.xlabel("Distance (pc)")
        plt.ylabel("log PDF")
        plt.savefig(f"{debug_dirpath}_DEBUG_distance_lls.png")
        plt.close()

    # Convert from PDF to probability
    lls += np.log(np.append(np.diff(r), r[-1] - r[-2]))
    assert np.array_equal(r, np.sort(r))

    lls -= np.amax(lls)
    lls -= np.log(np.sum(np.exp(lls)))
    return lls


def compute_mslikelihoods(ms: np.ndarray, mobs: float, mobs_err: float) -> np.ndarray:
    """
    Treat the magnitudes as Gausiians given the observed values and return their likelihoods

    Parameters
    ----------
    ms : array-like
        Data
    mobs : float
        The observed apparent magnitude, treated as the mean of ms
    mobs_err : float
        The observed uncertainty in apparent magnitude, treated as the standard deviation of ms

    Returns
    -------
    lls : array
        The scaled log-likelihood
    """
    lls = loggaussian(ms, mobs, mobs_err)

    # Convert from PDF to probability
    lls += np.log(np.append(np.diff(ms), ms[-1] - ms[-2]))

    lls -= np.amax(lls)
    lls -= np.log(np.sum(np.exp(lls)))
    return lls
