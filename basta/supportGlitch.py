"""
Auxiliary functions for glitch fitting
"""
import numpy as np
from basta.sd import sd
from basta.icov_sd import icov_sd
from basta.glitch_fq import fit_fq
from basta.glitch_sd import fit_sd
from basta import utils_seismic as su
from copy import deepcopy


# -----------------------------------------------------------------------------------------
def fit(
    freq,
    num_of_n,
    delta_nu,
    num_of_dif2=None,
    freqDif2=None,
    icov=None,
    method="FQ",
    n_rln=1000,
    npoly_params=5,
    nderiv=3,
    tol_grad=1e-3,
    regu_param=7.0,
    n_guess=200,
    tauhe=None,
    dtauhe=None,
    taucz=None,
    dtaucz=None,
    rtype=None,
):
    """
    Fit glitch signatures

    Parameters
    ----------
    freq : array
        Observed modes (l, n, v(muHz), err(muHz))
    num_of_n : array of int
        Number of modes for each l
    delta_nu : float
        An estimate of large frequncy separation (muHz)
    num_of_dif2 : int
        Number of second differences
        used only for method='SD'
    freqDif2 : array
        Second differences (l, n, v(muHz), err(muHz), dif2(muHz), err(muHz))
        used only for method='SD'
    icov : array
        Inverse covariance matrix for second differences
        used only for method='SD'
    method : str
        Fitting method ('FQ' or 'SD')
    n_rln : int
        Number of realizations. If n_rln = 0, just fit the original frequencies/differences
    npoly_params : int
        Number of parameters in the smooth component (5 and 3 generally work well for 'FQ'
        and 'SD', respectively)
    nderiv : int
        Order of derivative used in the regularization (3 and 1 generally work well for
        'FQ' and 'SD', respectively)
    tol_grad : float
        tolerance on gradients (typically between 1e-2 and 1e-5 depending on quality
        of data and 'method' used)
    regu_param : float
        Regularization parameter (7 and 1000 generally work well for 'FQ' and
        'SD', respectively)
    n_guess : int
        Number of initial guesses in search for the global minimum
    tauhe : float, optional
        Determines the range in acoustic depth (s) of He glitch for global minimum
        search (tauhe - dtauhe, tauhe + dtauhe).
        If tauhe = None, tauhe = 0.17 * acousticRadius + 18
    dtauhe : float, optional
        Determines the range in acoustic depth (s) of He glitch for global minimum
        search (tauhe - dtauhe, tauhe + dtauhe).
        If dtauhe = None, dtauhe = 0.05 * acousticRadius
    taucz : float, optional
        Determines the range in acoustic depth (s) of CZ glitch for global minimum
        search (taucz - dtaucz, taucz + dtaucz).
        If taucz = None, taucz = 0.34 * acousticRadius + 929
    dtaucz : float, optional
        Determines the range in acoustic depth (s) of CZ glitch for global minimum
        search (taucz - dtaucz, taucz + dtaucz).
        If dtaucz = None, dtaucz = 0.10 * acousticRadius
    rtype : str, optional
        Ratio type (one of ["r01", "r10", "r02", "r010", "r012", "r102"])
        If None, ignore ratios calculations

    Return
    ------
    param : array
        Fitted parameters
    chi2 : array
        chi-square values
    reg : array
        Values of regularization term
    ier : array
        Values of error parameter
    ratio : array
        Ratio values
    """
    # -----------------------------------------------------------------------------------------

    np.random.seed(1)

    # Initialize acoutic depths (if they are None)
    acousticRadius = 5.0e5 / delta_nu
    if tauhe is None:
        tauhe = 0.17 * acousticRadius + 18.0
    if dtauhe is None:
        dtauhe = 0.05 * acousticRadius
    if taucz is None:
        taucz = 0.34 * acousticRadius + 929.0
    if dtaucz is None:
        dtaucz = 0.10 * acousticRadius

    # Initialize arrays
    chi2 = np.zeros(n_rln + 1, dtype=float)
    reg = np.zeros(n_rln + 1, dtype=float)
    ier = np.zeros(n_rln + 1, dtype=int)

    # Fit oscillation frequencies
    if method.lower() == "fq":

        # Total number of fitting parameters
        nparams = len(num_of_n[num_of_n > 0]) * npoly_params + 7

        # Fit original data
        # --> Glitches
        tmp, chi2[-1], reg[-1], ier[-1] = fit_fq(
            freq,
            num_of_n,
            acousticRadius,
            tauhe,
            dtauhe,
            taucz,
            dtaucz,
            npoly_fq=npoly_params,
            total_num_of_param_fq=nparams,
            nderiv_fq=nderiv,
            tol_grad_fq=tol_grad,
            regu_param_fq=regu_param,
            num_guess=n_guess,
        )
        param = np.zeros((n_rln + 1, len(tmp)), dtype=float)
        param[-1, :] = tmp
        # --> Ratios
        if rtype is not None:
            _, _, tmp = su.specific_ratio(freq, rtype=rtype)
            ratio = np.zeros((n_rln + 1, len(tmp)), dtype=float)
            ratio[-1, :] = tmp
        else:
            ratio = None

        # Fit realizations
        if n_rln > 0:
            freq_rln = deepcopy(freq)
            for i in range(n_rln):
                freq_rln[:, 2] = np.random.normal(loc=freq[:, 2], scale=freq[:, 3])
                # --> Glitches
                param[i, :], chi2[i], reg[i], ier[i] = fit_fq(
                    freq_rln,
                    num_of_n,
                    acousticRadius,
                    tauhe,
                    dtauhe,
                    taucz,
                    dtaucz,
                    npoly_fq=npoly_params,
                    total_num_of_param_fq=nparams,
                    nderiv_fq=nderiv,
                    tol_grad_fq=tol_grad,
                    regu_param_fq=regu_param,
                    num_guess=n_guess,
                )
                # --> Ratios
                if rtype is not None:
                    _, _, ratio[i, :] = su.specific_ratio(freq_rln, rtype=rtype)

    # Fit second differences
    elif method.lower() == "sd":
        if not all(x is not None for x in [num_of_dif2, freqDif2, icov]):
            raise ValueError("num_of_dif2, freqDif2, icov cannot be None for SD!")

        # Total number of fitting parameters
        nparams = npoly_params + 7

        # Fit original data
        # --> Glitches
        tmp, chi2[-1], reg[-1], ier[-1] = fit_sd(
            freqDif2,
            icov,
            acousticRadius,
            tauhe,
            dtauhe,
            taucz,
            dtaucz,
            npoly_sd=npoly_params,
            total_num_of_param_sd=nparams,
            nderiv_sd=nderiv,
            tol_grad_sd=tol_grad,
            regu_param_sd=regu_param,
            num_guess=n_guess,
        )
        param = np.zeros((n_rln + 1, len(tmp)), dtype=float)
        param[-1, :] = tmp
        # --> Ratios
        if rtype is not None:
            _, _, tmp = su.specific_ratio(freq, rtype=rtype)
            ratio = np.zeros((n_rln + 1, len(tmp)), dtype=float)
            ratio[-1, :] = tmp
        else:
            ratio = None

        # Fit realizations
        if n_rln > 0:
            freq_rln = deepcopy(freq)
            for i in range(n_rln):
                freq_rln[:, 2] = np.random.normal(loc=freq[:, 2], scale=freq[:, 3])
                # --> Glitches
                dif2_rln = sd(freq_rln, num_of_n, num_of_dif2)
                param[i, :], chi2[i], reg[i], ier[i] = fit_sd(
                    dif2_rln,
                    icov,
                    acousticRadius,
                    tauhe,
                    dtauhe,
                    taucz,
                    dtaucz,
                    npoly_sd=npoly_params,
                    total_num_of_param_sd=nparams,
                    nderiv_sd=nderiv,
                    tol_grad_sd=tol_grad,
                    regu_param_sd=regu_param,
                    num_guess=n_guess,
                )
                # --> Ratios
                if rtype is not None:
                    _, _, ratio[i, :] = su.specific_ratio(freq_rln, rtype=rtype)

    else:
        raise ValueError("Unrecognized fitting method %s!" % (method))

    return (param, chi2, reg, ier, ratio)


# -----------------------------------------------------------------------------------------
def compDif2(num_of_l, freq, num_of_mode, num_of_n):
    """
    Compute the second differences

    Parameters
    ----------
    num_of_l : int
        Number of harmonic degrees (starting from l = 0)
    freq : array
        Observed modes (l, n, v(muHz), err(muHz))
    num_of_mode : int
        Number of modes
    num_of_n : array of int
        Number of modes for each l

    Return
    ------
    num_of_dif2 : int
        Number of second differences
    freqDif2 : array
        Second differences (l, n, v(muHz), err(muHz), dif2(muHz), err(muHz))
    icov : array
        Inverse covariance matrix for second differences
    """
    # -----------------------------------------------------------------------------------------

    # Compute second differences of oscillation frequencies
    num_of_dif2 = num_of_mode - 2 * len(num_of_n[num_of_n > 0])
    freqDif2 = sd(freq, num_of_n, num_of_dif2)

    # Compute inverse covariance matrix for second differences
    icov = icov_sd(num_of_l, num_of_n, freq, num_of_dif2)

    return (num_of_dif2, freqDif2, icov)


# -----------------------------------------------------------------------------------------
def glitchSignal(nu, param, glitch="He"):
    """
    Compute He/CZ glitch signature at a given frequency
    He : A_he * numu * exp(-8 * pi^2 * nuhz^2 * delta_he^2) *
         sin(4 * pi * nuhz * tau_he + phi_he)
    CZ : A_cz * sin(4 * pi * nuhz * tau_cz + phi_cz) / numu^2

    Parameters
    ----------
    nu : float
        Frequency (muHz)
    param : array
        Fitted Parameters
    glitch : str
        Glitch ('He' or 'CZ')

    Return
    ------
    signal : float
        Glitch signature
    """
    # -----------------------------------------------------------------------------------------

    nuhz = 1.0e-6 * nu
    n0 = len(param) - 7

    # He glitch
    if glitch.lower() == "he":
        signal = (
            param[n0 + 3]
            * nu
            * np.exp(-8.0 * np.pi ** 2 * nuhz ** 2 * param[n0 + 4] ** 2)
            * np.sin(4.0 * np.pi * nuhz * param[n0 + 5] + param[n0 + 6])
        )

    # CZ glitch
    elif glitch.lower() == "cz":
        signal = (
            param[n0]
            * np.sin(4.0 * np.pi * nuhz * param[n0 + 1] + param[n0 + 2])
            / nu ** 2
        )

    else:
        raise ValueError("Unrecognized glitch %s!" % (glitch))

    return signal


# -----------------------------------------------------------------------------------------
def totalGlitchSignal(nu, param):
    """
    Compute total contribution of glitch signatures (He + CZ)
    He : A_he * numu * exp(8 * pi^2 * nuhz^2 * delta_he^2) *
         sin(4 * pi * nuhz * tau_he + phi_he)
    CZ : A_cz * sin(4 * pi * nuhz * tau_cz + phi_cz) / numu^2

    Parameters
    ----------
    nu : float
        Frequency (muHz)
    param : array
        Fitted Parameters

    Return
    ------
    total : float
        Total glitch signature
    """
    # -----------------------------------------------------------------------------------------

    total = glitchSignal(nu, param, glitch="He") + glitchSignal(nu, param, glitch="CZ")

    return total


# -----------------------------------------------------------------------------------------
def smoothComponent(
    param, l=None, n=None, nu=None, num_of_n=None, npoly_params=None, method="FQ"
):
    """
    Compute smooth component for frequency/second-difference fit

    Parameters
    ----------
    param : array
        Fitted Parameters
    l : int
        Harmonic degree of the mode
        used only for method='FQ'
    n : int
        Radial order of the mode
        used only for method='FQ'
    nu : float
        Frequency of the mode (muHz)
        used only for method='SD'
    num_of_n : array of int
        Number of modes for each l
        used only for method='FQ'
    npoly_params : int
        Number of parameters in the smooth component
    method : str
        Fitting method ('FQ' or 'SD')

    Return
    ------
    smooth : float
       smooth component

    """
    # -----------------------------------------------------------------------------------------

    smooth = 0.0

    # Smooth component for frequency fit
    if method.lower() == "fq":
        if not all(x is not None for x in [l, n, num_of_n, npoly_params]):
            raise ValueError("l, n, num_of_n, npoly_params cannot be None for FQ!")

        ntmp = num_of_n[0 : l + 1]
        n0 = npoly_params * (len(ntmp[ntmp > 0]) - 1)
        for i in range(npoly_params - 1, -1, -1):
            smooth = smooth * n + param[n0 + i]

    # Smooth component for frequency fit
    elif method.lower() == "sd":
        if not all(x is not None for x in [nu, npoly_params]):
            raise ValueError("nu, npoly_params cannot be None for SD!")

        for i in range(npoly_params - 1, -1, -1):
            smooth = smooth * nu + param[i]

    else:
        raise ValueError("Unrecognized method %s!" % (method))

    return smooth


# -----------------------------------------------------------------------------------------
def averageAmplitudes(param, vmin, vmax, delta_nu=None, method="FQ"):
    """
    Compute average amplitude of He and CZ signature

    Parameters
    ----------
    param : array
        Fitted parameters
    vmin : float
        Lower limit on frequency used in averaging (muHz)
    vmax : float
        Upper limit on frequency used in averaging (muHz)
    delta_nu : float
        An estimate of large frequncy separation (muHz)
        used only for method='SD'
    method : str
        Fitting method ('FQ' or 'SD')

    Return
    ------
    Acz : float
        Average amplitude of CZ signature (muHz)
    Ahe : float
        Average amplitude of He signature (muHz)
    """
    # -----------------------------------------------------------------------------------------

    n0 = len(param) - 7

    # Amplitude of CZ signature
    Acz = param[n0] / (vmin * vmax)

    # Amplitude of He signature
    vminhz = 1.0e-6 * vmin
    vmaxhz = 1.0e-6 * vmax
    Ahe = (
        param[n0 + 3]
        * (
            np.exp(-8.0 * np.pi ** 2 * vminhz ** 2 * param[n0 + 4] ** 2)
            - np.exp(-8.0 * np.pi ** 2 * vmaxhz ** 2 * param[n0 + 4] ** 2)
        )
        / (16.0 * np.pi ** 2 * 1.0e-12 * (vmax - vmin) * param[n0 + 4] ** 2)
    )

    # Scale amplitudes from SD to FQ
    if method.lower() == "sd":
        if delta_nu is None:
            raise ValueError("delta_nu cannot be None for SD!")
        Acz /= (2.0 * np.sin(2.0 * np.pi * delta_nu * 1.0e-6 * param[n0 + 1])) ** 2
        Ahe /= (2.0 * np.sin(2.0 * np.pi * delta_nu * 1.0e-6 * param[n0 + 5])) ** 2

    return Acz, Ahe
