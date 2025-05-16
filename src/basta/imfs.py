"""
This module contains the possible IMFs that can be specified and used in BASTA.
"""

import h5py  # type: ignore[import]
import numpy as np

from basta import core

PRIOR_FUNCTIONS = {}


def register_prior(fn):
    PRIOR_FUNCTIONS[fn.__name__] = fn
    return fn


def piecewise_imf(m, ms, alphas, ks):
    for i in range(len(alphas)):
        if ms[i] <= m < ms[i + 1]:
            return ks[i] * m ** alphas[i]
    print("Mass outside range of IMF prior")
    return np.inf


def normfactor(alphas, ms):
    # Algorithm from App. A in Pflamm-Altenburg & Kroupa (2006)
    # https://ui.adsabs.harvard.edu/abs/2006MNRAS.373..295P/abstract
    ks = np.zeros(len(alphas))
    ks[0] = (1 / ms[1]) ** alphas[0]
    ks[1] = (1 / ms[1]) ** alphas[1]
    if len(ks) == 2:
        return ks
    ks[2] = (ms[2] / ms[1]) ** alphas[1] * (1 / ms[2]) ** alphas[2]
    if len(ks) == 3:
        return ks
    if len(ks) == 4:
        ks[3] = (
            (ms[2] / ms[1]) ** alphas[1]
            * (ms[3] / ms[2]) ** alphas[2]
            * (1 / ms[3]) ** alphas[3]
        )
        return ks
    print("Mistake in normfactor")
    return None


@register_prior
def salpeter1955(libitem, index):
    """
    Initial mass function from Salpeter (1955)
    https://ui.adsabs.harvard.edu/abs/1955ApJ...121..161S
    """

    return libitem["massini"][index] ** (-2.35)


@register_prior
def millerscalo1979(libitem, index):
    """
    Initial mass function from Miller & Scalo (1979)
    https://ui.adsabs.harvard.edu/abs/1979ApJS...41..513M
    """
    m = libitem["massini"][index]
    ms = [0.1, 1, 10, 100]
    alphas = [-1.4, -2.5, -3.3]
    ks = normfactor(alphas, ms)
    return piecewise_imf(m, ms, alphas, ks)


@register_prior
def kennicutt1994(libitem, index):
    """
    Initial mass function from Kennicutt et al. 1994
    https://ui.adsabs.harvard.edu/abs/1994ApJ...435...22K
    """
    m = libitem["massini"][index]
    ms = [0.1, 1, 100]
    alphas = [-1.4, -2.5]
    ks = normfactor(alphas, ms)
    return piecewise_imf(m, ms, alphas, ks)


@register_prior
def scalo1998(libitem, index):
    """
    Initial mass function from Scalo (1998)
    https://ui.adsabs.harvard.edu/abs/1998ASPC..142..201S
    """
    m = libitem["massini"][index]
    ms = [0.1, 1, 10, 100]
    alphas = [-1.2, -2.7, -2.3]
    ks = normfactor(alphas, ms)
    return piecewise_imf(m, ms, alphas, ks)


@register_prior
def kroupa2001(libitem, index):
    """
    Initial mass function from Kroupa (2001)
    https://ui.adsabs.harvard.edu/abs/2001MNRAS.322..231K
    https://ui.adsabs.harvard.edu/abs/2002Sci...295...82K
    """
    m = libitem["massini"][index]
    ms = [0.01, 0.08, 0.5, 1, 150]
    alphas = [-0.3, -1.3, -2.3]
    ks = normfactor(alphas, ms)
    return piecewise_imf(m, ms, alphas, ks)


@register_prior
def baldryglazebrook2003(libitem, index):
    """
    Initial mass function from Baldry & Glazebrook (2003)
    https://ui.adsabs.harvard.edu/abs/2003ApJ...593..258B
    """
    m = libitem["massini"][index]
    ms = [0.1, 0.5, 120]
    alphas = [-1.5, -2.2]
    ks = normfactor(alphas, ms)
    return piecewise_imf(m, ms, alphas, ks)


@register_prior
def chabrier2003(libitem, index):
    """
    Initial mass function from Chabrier (2003)
    https://ui.adsabs.harvard.edu/abs/2003PASP..115..763C/abstract
    """
    m = libitem["massini"][index]
    ks = [0.158, 0.0443]
    if m < 1:
        return (
            ks[0]
            * (1 / m)
            * np.exp(-0.5 * ((np.log10(m) - np.log10(0.079)) / 0.69) ** 2)
        )
    return ks[1] * m ** (-2.3)
