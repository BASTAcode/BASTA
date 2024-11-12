"""
Definition of priors

Define any prior functions here that you want to use in BASTA!!

The prior function must be of the form:
PRIOR = PRIORFUN(LIBITEM, INDEX)

Any prior defined here can be used from an .xml input file.
"""

import numpy as np
from basta import utils_general as util


def salpeter1955(libitem, index):
    """
    Initial mass function from Salpeter (1955)
    https://ui.adsabs.harvard.edu/abs/1955ApJ...121..161S
    """
    return libitem["massini"][index] ** (-2.35)


def millerscalo1979(libitem, index):
    """
    Initial mass function from Miller & Scalo (1979)
    https://ui.adsabs.harvard.edu/abs/1979ApJS...41..513M
    The global normalisation is not needed as we normalise later.
    """
    m = libitem["massini"][index]
    ms = [0.1, 1, 10, 100]
    alphas = [-1.4, -2.5, -3.3]
    ks = util.normfactor(alphas, ms)
    if (ms[0] <= m) & (m < ms[1]):
        return ks[0] * m ** alphas[0]
    elif (ms[1] <= m) & (m < ms[2]):
        return ks[1] * m ** alphas[1]
    elif (ms[2] <= m) & (m < ms[3]):
        return ks[2] * m ** alphas[2]
    else:
        print("Mass outside range of IMF prior")
        return 0


def kennicutt1994(libitem, index):
    """
    Initial mass function from Kennicutt et al. 1994
    https://ui.adsabs.harvard.edu/abs/1994ApJ...435...22K
    The global normalisation is not needed as we normalise later.
    """
    m = libitem["massini"][index]
    ms = [0.1, 1, 100]
    alphas = [-1.4, -2.5]
    ks = util.normfactor(alphas, ms)
    if (ms[0] <= m) & (m < ms[1]):
        return ks[0] * m ** alphas[0]
    elif (ms[1] <= m) & (m < ms[2]):
        return ks[1] * m ** alphas[1]
    else:
        print("Mass outside range of IMF prior")
        return 0


def scalo1998(libitem, index):
    """
    Initial mass function from Scalo (1998)
    https://ui.adsabs.harvard.edu/abs/1998ASPC..142..201S
    The global normalisation is not needed as we normalise later.
    """
    m = libitem["massini"][index]
    ms = [0.1, 1, 10, 100]
    alphas = [-1.2, -2.7, -2.3]
    ks = util.normfactor(alphas, ms)
    if (ms[0] <= m) & (m < ms[1]):
        return ks[0] * m ** alphas[0]
    elif (ms[1] <= m) & (m < ms[2]):
        return ks[1] * m ** alphas[1]
    elif (ms[2] <= m) & (m < ms[3]):
        return ks[2] * m ** alphas[2]
    else:
        print("Mass outside range of IMF prior")
        return 0


def kroupa2001(libitem, index):
    """
    Initial mass function from Kroupa (2001)
    https://ui.adsabs.harvard.edu/abs/2001MNRAS.322..231K
    https://ui.adsabs.harvard.edu/abs/2002Sci...295...82K
    The global normalisation is not needed as we normalise later.
    """
    m = libitem["massini"][index]
    ms = [0.01, 0.08, 0.5, 1, 150]
    alphas = [-0.3, -1.3, -2.3]
    ks = util.normfactor(alphas, ms)
    if (ms[0] <= m) & (m < ms[1]):
        return ks[0] * m ** alphas[0]
    elif (ms[1] <= m) & (m < ms[2]):
        return ks[1] * m ** alphas[1]
    # This case and the last case are identical with these values
    elif (ms[2] <= m) & (m < ms[4]):
        return ks[2] * m ** alphas[2]
    else:
        print("Mass outside range of IMF prior")
        return 0


def baldryglazebrook2003(libitem, index):
    """
    Initial mass function from Baldry & Glazebrook (2003)
    https://ui.adsabs.harvard.edu/abs/2003ApJ...593..258B
    The global normalisation is not needed as we normalise later.
    """
    m = libitem["massini"][index]
    ms = [0.1, 0.5, 120]
    alphas = [-1.5, -2.2]
    ks = util.normfactor(alphas, ms)
    if (ms[0] <= m) & (m < ms[1]):
        return ks[0] * m ** alphas[0]
    elif (ms[1] <= m) & (m < ms[2]):
        return ks[1] * m ** alphas[1]
    else:
        print("Mass outside range of IMF prior")
        return 0


def chabrier2003(libitem, index):
    """
    Initial mass function from Chabrier (2003)
    https://ui.adsabs.harvard.edu/abs/2003PASP..115..763C/abstract
    Note that this is in linear mass space, hence the (1/m)
    """
    m = libitem["massini"][index]
    ks = [0.158, 0.0443]
    if m < 1:
        return (
            ks[0]
            * (1 / m)
            * np.exp(-0.5 * ((np.log10(m) - np.log10(0.079)) / 0.69) ** 2)
        )
    else:
        return ks[1] * m ** (-2.3)
