"""
Auxiliary functions for glitch fitting
"""
import numpy as np
import sys
from basta.sd import sd
from basta.icov_sd import icov_sd
from basta.glitch_fq import fit_fq
from basta.glitch_sd import fit_sd
from copy import deepcopy


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
    num_of_dif2 = num_of_mode - 2 * num_of_l
    freqDif2 = sd(freq, num_of_n, num_of_dif2)

    # Compute inverse covariance matrix for second differences
    icov = icov_sd(num_of_l, num_of_n, freq, num_of_dif2)

    return (num_of_dif2, freqDif2, icov)


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
    tol_grad=1e-3,
    regu_param=7.0,
    n_guess=200,
    tauhe=None,
    dtauhe=None,
    taucz=None,
    dtaucz=None,
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
        search (tauhe - dtauhe, tauhe + dtauhe). If None, make a guess using acoustic
        radius
    dtauhe : float, optional
        Determines the range in acoustic depth (s) of He glitch for global minimum
        search (tauhe - dtauhe, tauhe + dtauhe). If None, make a guess using acoustic
        radius
    taucz : float, optional
        Determines the range in acoustic depth (s) of CZ glitch for global minimum
        search (taucz - dtaucz, taucz + dtaucz). If None, make a guess using acoustic
        radius
    dtaucz : float, optional
        Determines the range in acoustic depth (s) of CZ glitch for global minimum
        search (taucz - dtaucz, taucz + dtaucz). If None, make a guess using acoustic
        radius

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
    """
    # -----------------------------------------------------------------------------------------

    if tauhe is None:
        tauhe = -1.0
    if dtauhe is None:
        dtauhe = -1.0
    if taucz is None:
        taucz = -1.0
    if dtaucz is None:
        dtaucz = -1.0

    # np.random.seed(1)

    vmin, vmax = np.amin(freq[:, 2]), np.amax(freq[:, 2])

    chi2 = np.zeros(n_rln + 1, dtype=float)
    reg = np.zeros(n_rln + 1, dtype=float)
    ier = np.zeros(n_rln + 1, dtype=int)

    # Fit oscillation frequencies
    if method.lower() == "fq":
        print()
        print("Fitting oscillation frequencies...")

        # Original
        tmp, chi2[-1], reg[-1], ier[-1] = fit_fq(
            freq,
            num_of_n,
            delta_nu,
            tol_grad_fq=tol_grad,
            regu_param_fq=regu_param,
            num_guess=n_guess,
            tauhe=tauhe,
            dtauhe=dtauhe,
            taucz=taucz,
            dtaucz=dtaucz,
        )
        param = np.zeros((n_rln + 1, len(tmp)), dtype=float)
        param[-1, :] = tmp
        Acz, Ahe = averageAmplitudes(
            param[-1, :], vmin, vmax, delta_nu=delta_nu, method=method
        )
        print(
            "Ier  Chi2  Reg  Acz  Tcz  Ahe  The = %5d %12.4f %8.4f %12.4f %6d %12.4f "
            "%6d" % (ier[-1], chi2[-1], reg[-1], Acz, param[-1, -6], Ahe, param[-1, -2])
        )

        # Fit realizations
        if n_rln > 0:
            freq_rln = deepcopy(freq)
            for i in range(n_rln):
                freq_rln[:, 2] = np.random.normal(loc=freq[:, 2], scale=freq[:, 3])
                param[i, :], chi2[i], reg[i], ier[i] = fit_fq(
                    freq_rln,
                    num_of_n,
                    delta_nu,
                    tol_grad_fq=tol_grad,
                    regu_param_fq=regu_param,
                    num_guess=n_guess,
                    tauhe=tauhe,
                    dtauhe=dtauhe,
                    taucz=taucz,
                    dtaucz=dtaucz,
                )
                Acz, Ahe = averageAmplitudes(
                    param[i, :], vmin, vmax, delta_nu=delta_nu, method=method
                )
                print(
                    "N_rln  Ier  Chi2  Reg  Acz  Tcz  Ahe  The = %5d %5d %12.4f %8.4f "
                    "%12.4f %6d %12.4f %6d"
                    % (
                        i + 1,
                        ier[i],
                        chi2[i],
                        reg[i],
                        Acz,
                        param[i, -6],
                        Ahe,
                        param[i, -2],
                    )
                )

    # Fit second differences
    elif method.lower() == "sd":
        print()
        print("Fitting second differences...")
        if not all(x is not None for x in [num_of_dif2, freqDif2, icov]):
            print("num_of_dif2, freqDif2, icov cannot be None. Terminating the run...")
            sys.exit(1)

        # Original
        tmp, chi2[-1], reg[-1], ier[-1] = fit_sd(
            freqDif2,
            icov,
            delta_nu,
            tol_grad_sd=tol_grad,
            regu_param_sd=regu_param,
            num_guess=n_guess,
            tauhe=tauhe,
            dtauhe=dtauhe,
            taucz=taucz,
            dtaucz=dtaucz,
        )
        param = np.zeros((n_rln + 1, len(tmp)), dtype=float)
        param[-1, :] = tmp
        Acz, Ahe = averageAmplitudes(
            param[-1, :], vmin, vmax, delta_nu=delta_nu, method=method
        )
        print(
            "Ier  Chi2  Reg  Acz  Tcz  Ahe  The = %5d %12.4f %8.4f %12.4f %6d %12.4f "
            "%6d" % (ier[-1], chi2[-1], reg[-1], Acz, param[-1, -6], Ahe, param[-1, -2])
        )

        # Fit realizations
        if n_rln > 0:
            freq_rln = deepcopy(freq)
            for i in range(n_rln):
                freq_rln[:, 2] = np.random.normal(loc=freq[:, 2], scale=freq[:, 3])
                dif2_rln = sd(freq_rln, num_of_n, num_of_dif2)
                param[i, :], chi2[i], reg[i], ier[i] = fit_sd(
                    dif2_rln,
                    icov,
                    delta_nu,
                    tol_grad_sd=tol_grad,
                    regu_param_sd=regu_param,
                    num_guess=n_guess,
                    tauhe=tauhe,
                    dtauhe=dtauhe,
                    taucz=taucz,
                    dtaucz=dtaucz,
                )
                Acz, Ahe = averageAmplitudes(
                    param[i, :], vmin, vmax, delta_nu=delta_nu, method=method
                )
                print(
                    "N_rln  Ier  Chi2  Reg  Acz  Tcz  Ahe  The = %5d %5d %12.4f %8.4f "
                    "%12.4f %6d %12.4f %6d"
                    % (
                        i + 1,
                        ier[i],
                        chi2[i],
                        reg[i],
                        Acz,
                        param[i, -6],
                        Ahe,
                        param[i, -2],
                    )
                )

    else:
        print("Fitting method is not recognized. Terminating the run...")
        sys.exit(2)

    return (param, chi2, reg, ier)


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
        print("Glitch is not recognized. Terminating the run...")
        sys.exit(1)

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
def smoothComponent(param, nu=None, l=None, n=None, num_of_l=None, method="FQ"):
    """
    Compute smooth component for frequency/second-difference fit

    Parameters
    ----------
    param : array
        Fitted Parameters
    nu : float
        Frequency of the mode (muHz)
        used only for method='SD'
    l : int
        Harmonic degree of the mode
        used only for method='FQ'
    n : int
        Radial order of the mode
        used only for method='FQ'
    num_of_l : int
        Number of harmonic degree used in the fit
        used only for method='FQ'
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
        if not all(x is not None for x in [l, n, num_of_l]):
            print("l, n, num_of_l cannot be None. Terminating the run...")
            sys.exit(1)

        npoly = (len(param) - 7) // num_of_l
        n0 = npoly * l
        for i in range(npoly - 1, -1, -1):
            smooth = smooth * n + param[n0 + i]

    # Smooth component for frequency fit
    elif method.lower() == "sd":
        if nu is None:
            print("nu cannot be None. Terminating the run...")
            sys.exit(2)

        npoly = len(param) - 7
        for i in range(npoly - 1, -1, -1):
            smooth = smooth * nu + param[i]

    else:
        print("Fitting method is not recognized. Terminating the run...")
        sys.exit(1)

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
            print("delta_nu cannot be None. Terminating the run...")
            sys.exit(1)
        Acz /= (2.0 * np.sin(2.0 * np.pi * delta_nu * 1.0e-6 * param[n0 + 1])) ** 2
        Ahe /= (2.0 * np.sin(2.0 * np.pi * delta_nu * 1.0e-6 * param[n0 + 5])) ** 2

    return Acz, Ahe


# -----------------------------------------------------------------------------------------
def medianAndErrors(param_rln):
    """
    Compute the median and (negative and positive) uncertainties from the realizations

    Parameters
    ----------
    param_rln : array
        Parameter values for different realizations

    Return
    ------
    med : float
        Median value
    nerr : float
        Negative error
    perr : float
        Positive error
    """
    # -----------------------------------------------------------------------------------------

    per16 = np.percentile(param_rln, 16)
    per50 = np.percentile(param_rln, 50)
    per84 = np.percentile(param_rln, 84)
    med, nerr, perr = per50, per50 - per16, per84 - per50

    return med, nerr, perr
