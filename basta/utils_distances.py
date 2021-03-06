"""
Utility functions for the distance calculation and parallax fitting
"""
import numpy as np

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def compute_absmag(m, dist, A):
    """
    Use distance moduli to compute the absolute magnitudes
    from distances, apparent magnitudes m, and absorption A.

    Parameters
    ----------
    m : float
        Apparent magnitude
    dist : float
        Distances in parsec
    A : float
        Absorption

    Returns
    -------
    M : float
        Absolute magnitude
    """
    return m - 5 * np.log10(dist / 10) - A


def compute_distlikelihoods(r, plxobs, plxobs_err, L=None, outfilename="", debug=False):
    """
    Compute the likelihood as the product between a gaussian of the parallax
    and the exponentially decreasing volume density prior.
    For the unnormalised posterior, see Eq. 18 in Bailer-Jones 2015.
    This also works for nonpositive parallaxes.
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
        plt.savefig(outfilename + "_DEBUG_distance_lls.png")
        plt.close()

    # Convert from PDF to probability
    lls += np.log(np.append(np.diff(r), r[-1] - r[-2]))
    assert np.array_equal(r, np.sort(r))

    lls -= np.amax(lls)
    lls -= np.log(np.sum(np.exp(lls)))
    return lls


def compute_mslikelihoods(ms, mobs, mobs_err):
    lls = loggaussian(ms, mobs, mobs_err)

    # Convert from PDF to probability
    lls += np.log(np.append(np.diff(ms), ms[-1] - ms[-2]))

    lls -= np.amax(lls)
    lls -= np.log(np.sum(np.exp(lls)))
    return lls


def distance_from_mag(m, M, A):
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
        distance in parsec
    """
    return 10 ** (1 + (m - M - A) / 5.0)


def EDSD(libitem, index):
    """
    Exponentially decreasing space density prior
    Define characteristic length scale k in kpc
    """
    k = 1.35
    return k


def loggaussian(x, mu, sigma):
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
