"""
Fitting of glitches (work in progress!)
"""
import numpy as np

# Modules compiled with f2py3
try:
    from basta.glitch_fit import glitch_fit

    importexit = 0
except ImportError:
    # Let BASTA run without this imported for light version
    # The exit if the module is needed
    importexit = 1


def glh_params(freq, nmodes, nu1, nu2, tau0, tauhe, taubcz, verbose=False):
    """
    Fit the glitch signatures

    Parameters
    ----------

    Returns
    -------
    glhParams : array
        Glitch parameters
    nerr :
    """
    if importexit:
        print(
            "Unable to import (external) module 'glitch_fit'!"
            "You need to compile the module with f2py3 using 'setup.py',"
            "and a case different from the light case! Aborting now...",
        )
    # Fit the glitch signatures
    num_of_n = np.zeros(5, dtype=int)
    tmp = freq[0:nmodes, :]
    for i in range(5):
        num_of_n[i] = len(tmp[np.rint(tmp[:, 0]) == i, 0])
        params, chi2, reg, nerr = glitch_fit(freq, num_of_n, tau0, tauhe, taubcz)
        if verbose:
            print(
                chi2,
                reg,
                nerr,
                params[-7],
                params[-6],
                params[-5],
                params[-4],
                params[-3],
                params[-2],
                params[-1],
            )

    # Extract the average amplitude, width and acoustic depth
    glhParams = np.zeros(3)
    glhParams[0] = params[-4] / (4.0e-6 * np.pi * params[-3]) ** 2
    tmp = -8.0e-12 * np.pi**2 * params[-3] ** 2
    glhParams[0] *= (np.exp(tmp * nu1**2) - np.exp(tmp * nu2**2)) / (nu2 - nu1)
    glhParams[1] = params[-3]
    glhParams[2] = params[-2]

    return glhParams, nerr
