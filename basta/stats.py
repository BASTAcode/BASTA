"""
Key statistics functions
"""
import math
import collections

import numpy as np
from numpy.lib.histograms import _hist_bin_fd
from scipy.interpolate import interp1d
from scipy.ndimage.filters import gaussian_filter1d

from basta import freq_fit
from basta import utils_seismic as su
from basta.constants import freqtypes
import basta.supportGlitch as sg
from basta.sd import sd

# Define named tuple used in selectedmodels
Trackstats = collections.namedtuple("Trackstats", "index logPDF chi2 dnufit glhparams")
priorlogPDF = collections.namedtuple(
    "Trackstats", "index logPDF chi2 bayw magw IMFw dnufit glhparams"
)


def _weight(N, seisw):
    """
    Determine weighting scheme dependent on given method

    Parameters
    ----------
    N : int
        Number of data points
    seisw : dict
        User defined control of the weights, by "1/1", "1/N" or
        "1/N-dof", and user defined "dof" and "N" if given

    Returns
    -------
    w : int
        Normalising factor
    """
    # Overwrite N if specified by the user
    if seisw["N"]:
        N = int(seisw["N"])

    # Select weight case
    if seisw["weight"] == "1/1":
        w = 1
    elif seisw["weight"] == "1/N":
        w = N
    elif seisw["weight"] == "1/N-dof":
        if not seisw["dof"]:
            dof = 0
        else:
            dof = int(seisw["dof"])
        w = N - dof
    return w


def chi2_astero(
    modkey,
    mod,
    obskey,
    obs,
    inpdata,
    tipo,
    covinv,
    obsintervals,
    dnudata,
    dnudata_err,
    method="FQ",
    num_of_n=None,
    vmin=None,
    vmax=None,
    icov_sd=None,
    npoly_params=5,
    nderiv=3,
    tol_grad=1e-3,
    regu_param=7.0,
    n_guess=200,
    tauhe=800.0,
    taubcz=2200.0,
    useint=False,
    numax=None,
    bfit=None,
    fcor="BG14",
    seisw={},
    dnufit_in_ratios=False,
    warnings=True,
    shapewarn=False,
    debug=False,
    verbose=False,
):
    """
    `chi2_astero` calculates the chi2 contributions from the asteroseismic
    fitting e.g., the frequency, ratio, or glitch fitting.

    Parameters
    ----------
    modkey : array
        Array containing the angular degrees and radial orders of `mod`
    mod : array
        Array containing the modes in the model.
    obskey : array
        Array containing the angular degrees and radial orders of obs
    obs : array
        Array containing the modes in the observed data
    inpdata : array
        Individual frequencies, uncertainties, and combinations read from the
        observational input files.
    tipo : str
        Flag determining the kind of fitting. tipo means *type* in Spanish.
        `tipo` can be:
        * 'freqs' for fitting individual frequencies.
        * 'rn' for fitting n ratio sequence (e.g. 012, 01)
        * 'grn' for fitting n glitch-ratio combination
    covinv : array
        Covariances between individual frequencies and frequency ratios read
        from the observational input files.
    obsintervals : array
        Array containing the endpoints of the intervals used in the frequency
        fitting routine in :func:'freq_fit.calc_join'.
        As it is the same in all iterations for the observed frequencies,
        this is computed in su.prepare_obs once and given as an argument
        in order to save time and memory.
    dnudata : float
        Large frequency separation obtained by fitting the radial mode observed
        frequencies (like dnufit, but for the data). Used for fitting ratios.
    dnudata_err : float
        Uncertainty on dnudata
    method : str
        Fitting method ('FQ' or 'SD')
    num_of_n : array of int
        Number of modes for each l
    vmin : float
        Minimum value of the observed frequency (muHz)
    vmax : float
        Maximum value of the observed frequency (muHz)
    icov_sd : array
        Inverse covariance matrix for second differences
        used only for method='SD'
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
        Acoustic depth of the He II ionization zone.
    taubcz : float, optional
        Acoustic depth of the base of the convection zone.
    useint : bool, optional
        If True, interpolated ratios are used.
    numax : float or None, optional
        A value for frequency of maximum power given to the HK08 frequency
        correction.
    bfit : float or None, optional
        A number giving the exponent of the HK08 power law for the HK08
        frequency correction.
    fcor : str
        Type of surface correction (see :func:'freq_fit.py').
    seisw : dict
        Control of user defined seismic weights, in terms of normalization
    dnufit_in_ratios : bool, optional
        Flag determining whether to add a large frequency separation term
        using corrected frequencies after surface correction in the ratios
        computation.
    warnings : bool
        If True, print something when it fails.
    debug : str
        Flag for print control.
    verbose : str
        Flag for print control.

    Returns
    -------
    chi2rut : float
        The calculated chi2 value for the given model to the observed data.
        If the calculated value is less than zero, chi2rut will be np.inf.
    warnings : bool
        See 'warnings' above.
    dnusurf: float
        Large frequency separation
    glhparams: array
        Helium glitch parameters (average amplitude, width and depth)
    """
    dnusurf = 0.0
    glhparams = np.zeros(3)

    # If more observed modes than model modes are in one bin, move on
    joins = freq_fit.calc_join(
        mod=mod, modkey=modkey, obs=obs, obskey=obskey, obsintervals=obsintervals
    )
    if joins is None:
        chi2rut = np.inf
        return chi2rut, warnings, shapewarn, dnusurf, glhparams
    else:
        joinkeys, join = joins
        nmodes = joinkeys[:, joinkeys[0, :] < 3].shape[1]

    # Apply surface correction
    if any(x in freqtypes.alltypes for x in tipo):
        if fcor == "None":
            corjoin = join
        elif fcor == "HK08":
            corjoin, _ = freq_fit.HK08(
                joinkeys=joinkeys, join=join, nuref=numax, bcor=bfit
            )
        elif fcor == "BG14":
            corjoin, _ = freq_fit.BG14(joinkeys=joinkeys, join=join, scalnu=numax)
        elif fcor == "cubicBG14":
            corjoin, _ = freq_fit.cubicBG14(joinkeys=joinkeys, join=join, scalnu=numax)
        else:
            print('ERROR: fcor must be either "None", "HK08" or "BG14" or "cubicBG14"')
            return

    # --> Add the chi-square terms for ratios and/or glitches and/or glitch-ratio
    # combinations
    chi2rut = 0.0
    tmp = [*freqtypes.rtypes, *freqtypes.glitches, *freqtypes.grtypes]
    if any(x in tmp for x in tipo):
        if (nmodes != obskey.shape[1]) or not all(
            joinkeys[1, joinkeys[0, :] < 3] == joinkeys[2, joinkeys[0, :] < 3]
        ):
            chi2rut = np.inf
            return chi2rut, warnings, shapewarn, dnusurf, glhparams

        # Compute large frequency separation
        FWHM_sigma = 2.0 * np.sqrt(2.0 * np.log(2.0))
        yfitdnu = corjoin[0, joinkeys[0, :] == 0]
        xfitdnu = joinkeys[1, joinkeys[0, :] == 0]
        wfitdnu = np.exp(
            -1.0
            * np.power(yfitdnu - numax, 2)
            / (2 * np.power(0.25 * numax / FWHM_sigma, 2.0))
        )
        fitcoef = np.polyfit(xfitdnu, yfitdnu, 1, w=np.sqrt(wfitdnu))
        dnusurf = fitcoef[0]

        # Define 'frq' to be used in ratio- and glitch-related calculations
        frq = np.zeros((nmodes, 4))
        frq[:, 0] = joinkeys[0, joinkeys[0, :] < 3]
        frq[:, 1] = joinkeys[1, joinkeys[0, :] < 3]
        frq[:, 2] = join[0, joinkeys[0, :] < 3]
        frq[:, 3] = join[3, joinkeys[0, :] < 3]

        # Compute r02, r01 and r10
        r02, r01, r10 = freq_fit.ratios(frq)

        if r02 is not None:

            # Add large frequency separation term (using corrected frequencies!)
            # --> Equivalent to 'dnufit', but using the frequencies *after*
            #     applying the surface correction.
            # --> Compared to the observed value, which is 'dnudata'.
            if dnufit_in_ratios:
                chi2rut += (
                    (dnudata - dnusurf) / np.sqrt(dnudata_err ** 2 + 0.3 ** 2)
                ) ** 2

            # Compute r010, r012 and r102, if needed
            tmp = ["r010", "r012", "r102", "gr010", "gr012", "gr102"]
            if any([x in tipo for x in tmp]):
                r010, r012, r102 = su.combined_ratios(r02, r01, r10)

            # R010
            if "r010" in tipo:
                x = inpdata[0][0, :] - r010[:, 1]
                w = _weight(len(x), seisw)
                if x.shape[0] == covinv[0].shape[0]:
                    chi2rut += (x.T.dot(covinv[0]).dot(x)) / w
                else:
                    shapewarn = True
                    chi2rut = np.inf

            # R02
            if "r02" in tipo:
                x = inpdata[1][0, :] - r02[:, 1]
                w = _weight(len(x), seisw)
                if x.shape[0] == covinv[1].shape[0]:
                    chi2rut += (x.T.dot(covinv[1]).dot(x)) / w
                else:
                    shapewarn = True
                    chi2rut = np.inf

            # R01
            if "r01" in tipo:
                x = inpdata[3][0, :] - r01[:, 1]
                w = _weight(len(x), seisw)
                if x.shape[0] == covinv[3].shape[0]:
                    chi2rut += (x.T.dot(covinv[3]).dot(x)) / w
                else:
                    chi2rut = np.inf

            # R10
            if "r10" in tipo:
                x = inpdata[4][0, :] - r10[:, 1]
                w = _weight(len(x), seisw)
                if x.shape[0] == covinv[4].shape[0]:
                    chi2rut += (x.T.dot(covinv[4]).dot(x)) / w
                else:
                    shapewarn = True
                    chi2rut = np.inf

            # R012
            if "r012" in tipo:
                x = inpdata[5][0, :] - r012[:, 1]
                w = _weight(len(x), seisw)
                if x.shape[0] == covinv[5].shape[0]:
                    chi2rut += (x.T.dot(covinv[5]).dot(x)) / w
                else:
                    shapewarn = True
                    chi2rut = np.inf

            # R102
            if "r102" in tipo:
                x = inpdata[6][0, :] - r102[:, 1]
                w = _weight(len(x), seisw)
                if x.shape[0] == covinv[6].shape[0]:
                    chi2rut += (x.T.dot(covinv[6]).dot(x)) / w
                else:
                    shapewarn = True
                    chi2rut = np.inf

            # GR010
            if "gr010" in tipo:
                num_sd, frq_sd = None, None
                if method.lower() == "sd":
                    num_sd = icov_sd.shape[0]
                    frq_sd = sd(frq, num_of_n, num_sd)
                param, chi2, reg, ier, _ = sg.fit(
                    frq,
                    num_of_n,
                    dnusurf,
                    num_of_dif2=num_sd,
                    freqDif2=frq_sd,
                    icov=icov_sd,
                    method=method,
                    n_rln=0,
                    npoly_params=npoly_params,
                    nderiv=nderiv,
                    tol_grad=tol_grad,
                    regu_param=regu_param,
                    n_guess=n_guess,
                    tauhe=tauhe,
                    dtauhe=100.0,
                    taucz=taubcz,
                    dtaucz=200.0,
                )
                if ier != 0:
                    chi2rut = np.inf
                else:
                    nr010 = r010.shape[0]
                    gr010 = np.zeros(nr010 + 3)
                    gr010[0:nr010] = r010[:, 1]
                    Acz, Ahe = sg.averageAmplitudes(
                        param[0, :], vmin, vmax, delta_nu=dnusurf, method=method
                    )
                    gr010[-3] = Ahe
                    gr010[-2] = param[0, -3]
                    gr010[-1] = param[0, -2]
                    glhparams[0], glhparams[1], glhparams[2] = (
                        gr010[-3],
                        gr010[-2],
                        gr010[-1],
                    )
                    x = inpdata[8][0, :] - gr010[:]
                    w = _weight(len(x), seisw)
                    if x.shape[0] == covinv[8].shape[0]:
                        chi2rut += (x.T.dot(covinv[8]).dot(x)) / w
                    else:
                        shapewarn = True
                        chi2rut = np.inf

            # GR02
            if "gr02" in tipo:
                num_sd, frq_sd = None, None
                if method.lower() == "sd":
                    num_sd = icov_sd.shape[0]
                    frq_sd = sd(frq, num_of_n, num_sd)
                param, chi2, reg, ier, _ = sg.fit(
                    frq,
                    num_of_n,
                    dnusurf,
                    num_of_dif2=num_sd,
                    freqDif2=frq_sd,
                    icov=icov_sd,
                    method=method,
                    n_rln=0,
                    npoly_params=npoly_params,
                    nderiv=nderiv,
                    tol_grad=tol_grad,
                    regu_param=regu_param,
                    n_guess=n_guess,
                    tauhe=tauhe,
                    dtauhe=100.0,
                    taucz=taubcz,
                    dtaucz=200.0,
                )
                if ier != 0:
                    chi2rut = np.inf
                else:
                    nr02 = r02.shape[0]
                    gr02 = np.zeros(nr02 + 3)
                    gr02[0:nr02] = r02[:, 1]
                    Acz, Ahe = sg.averageAmplitudes(
                        param[0, :], vmin, vmax, delta_nu=dnusurf, method=method
                    )
                    gr02[-3] = Ahe
                    gr02[-2] = param[0, -3]
                    gr02[-1] = param[0, -2]
                    glhparams[0], glhparams[1], glhparams[2] = (
                        gr02[-3],
                        gr02[-2],
                        gr02[-1],
                    )
                    x = inpdata[9][0, :] - gr02[:]
                    w = _weight(len(x), seisw)
                    if x.shape[0] == covinv[9].shape[0]:
                        chi2rut += (x.T.dot(covinv[9]).dot(x)) / w
                    else:
                        shapewarn = True
                        chi2rut = np.inf

            # GR01
            if "gr01" in tipo:
                num_sd, frq_sd = None, None
                if method.lower() == "sd":
                    num_sd = icov_sd.shape[0]
                    frq_sd = sd(frq, num_of_n, num_sd)
                param, chi2, reg, ier, _ = sg.fit(
                    frq,
                    num_of_n,
                    dnusurf,
                    num_of_dif2=num_sd,
                    freqDif2=frq_sd,
                    icov=icov_sd,
                    method=method,
                    n_rln=0,
                    npoly_params=npoly_params,
                    nderiv=nderiv,
                    tol_grad=tol_grad,
                    regu_param=regu_param,
                    n_guess=n_guess,
                    tauhe=tauhe,
                    dtauhe=100.0,
                    taucz=taubcz,
                    dtaucz=200.0,
                )
                if ier != 0:
                    chi2rut = np.inf
                else:
                    nr01 = r01.shape[0]
                    gr01 = np.zeros(nr01 + 3)
                    gr01[0:nr01] = r01[:, 1]
                    Acz, Ahe = sg.averageAmplitudes(
                        param[0, :], vmin, vmax, delta_nu=dnusurf, method=method
                    )
                    gr01[-3] = Ahe
                    gr01[-2] = param[0, -3]
                    gr01[-1] = param[0, -2]
                    glhparams[0], glhparams[1], glhparams[2] = (
                        gr01[-3],
                        gr01[-2],
                        gr01[-1],
                    )
                    x = inpdata[10][0, :] - gr01[:]
                    w = _weight(len(x), seisw)
                    if x.shape[0] == covinv[10].shape[0]:
                        chi2rut += (x.T.dot(covinv[10]).dot(x)) / w
                    else:
                        shapewarn = True
                        chi2rut = np.inf

            # GR10
            if "gr10" in tipo:
                num_sd, frq_sd = None, None
                if method.lower() == "sd":
                    num_sd = icov_sd.shape[0]
                    frq_sd = sd(frq, num_of_n, num_sd)
                param, chi2, reg, ier, _ = sg.fit(
                    frq,
                    num_of_n,
                    dnusurf,
                    num_of_dif2=num_sd,
                    freqDif2=frq_sd,
                    icov=icov_sd,
                    method=method,
                    n_rln=0,
                    npoly_params=npoly_params,
                    nderiv=nderiv,
                    tol_grad=tol_grad,
                    regu_param=regu_param,
                    n_guess=n_guess,
                    tauhe=tauhe,
                    dtauhe=100.0,
                    taucz=taubcz,
                    dtaucz=200.0,
                )
                if ier != 0:
                    chi2rut = np.inf
                else:
                    nr10 = r10.shape[0]
                    gr10 = np.zeros(nr10 + 3)
                    gr10[0:nr10] = r10[:, 1]
                    Acz, Ahe = sg.averageAmplitudes(
                        param[0, :], vmin, vmax, delta_nu=dnusurf, method=method
                    )
                    gr10[-3] = Ahe
                    gr10[-2] = param[0, -3]
                    gr10[-1] = param[0, -2]
                    glhparams[0], glhparams[1], glhparams[2] = (
                        gr10[-3],
                        gr10[-2],
                        gr10[-1],
                    )
                    x = inpdata[11][0, :] - gr10[:]
                    w = _weight(len(x), seisw)
                    if x.shape[0] == covinv[11].shape[0]:
                        chi2rut += (x.T.dot(covinv[11]).dot(x)) / w
                    else:
                        shapewarn = True
                        chi2rut = np.inf

            # GR012
            if "gr012" in tipo:
                num_sd, frq_sd = None, None
                if method.lower() == "sd":
                    num_sd = icov_sd.shape[0]
                    frq_sd = sd(frq, num_of_n, num_sd)
                param, chi2, reg, ier, _ = sg.fit(
                    frq,
                    num_of_n,
                    dnusurf,
                    num_of_dif2=num_sd,
                    freqDif2=frq_sd,
                    icov=icov_sd,
                    method=method,
                    n_rln=0,
                    npoly_params=npoly_params,
                    nderiv=nderiv,
                    tol_grad=tol_grad,
                    regu_param=regu_param,
                    n_guess=n_guess,
                    tauhe=tauhe,
                    dtauhe=100.0,
                    taucz=taubcz,
                    dtaucz=200.0,
                )
                if ier != 0:
                    chi2rut = np.inf
                else:
                    nr012 = r012.shape[0]
                    gr012 = np.zeros(nr012 + 3)
                    gr012[0:nr012] = r012[:, 1]
                    Acz, Ahe = sg.averageAmplitudes(
                        param[0, :], vmin, vmax, delta_nu=dnusurf, method=method
                    )
                    gr012[-3] = Ahe
                    gr012[-2] = param[0, -3]
                    gr012[-1] = param[0, -2]
                    glhparams[0], glhparams[1], glhparams[2] = (
                        gr012[-3],
                        gr012[-2],
                        gr012[-1],
                    )
                    x = inpdata[12][0, :] - gr012[:]
                    w = _weight(len(x), seisw)
                    if x.shape[0] == covinv[12].shape[0]:
                        chi2rut += (x.T.dot(covinv[12]).dot(x)) / w
                    else:
                        shapewarn = True
                        chi2rut = np.inf

            # GR102
            if "gr102" in tipo:
                num_sd, frq_sd = None, None
                if method.lower() == "sd":
                    num_sd = icov_sd.shape[0]
                    frq_sd = sd(frq, num_of_n, num_sd)
                param, chi2, reg, ier, _ = sg.fit(
                    frq,
                    num_of_n,
                    dnusurf,
                    num_of_dif2=num_sd,
                    freqDif2=frq_sd,
                    icov=icov_sd,
                    method=method,
                    n_rln=0,
                    npoly_params=npoly_params,
                    nderiv=nderiv,
                    tol_grad=tol_grad,
                    regu_param=regu_param,
                    n_guess=n_guess,
                    tauhe=tauhe,
                    dtauhe=100.0,
                    taucz=taubcz,
                    dtaucz=200.0,
                )
                if ier != 0:
                    chi2rut = np.inf
                else:
                    nr102 = r102.shape[0]
                    gr102 = np.zeros(nr102 + 3)
                    gr102[0:nr102] = r102[:, 1]
                    Acz, Ahe = sg.averageAmplitudes(
                        param[0, :], vmin, vmax, delta_nu=dnusurf, method=method
                    )
                    gr102[-3] = Ahe
                    gr102[-2] = param[0, -3]
                    gr102[-1] = param[0, -2]
                    glhparams[0], glhparams[1], glhparams[2] = (
                        gr102[-3],
                        gr102[-2],
                        gr102[-1],
                    )
                    x = inpdata[13][0, :] - gr102[:]
                    w = _weight(len(x), seisw)
                    if x.shape[0] == covinv[13].shape[0]:
                        chi2rut += (x.T.dot(covinv[13]).dot(x)) / w
                    else:
                        shapewarn = True
                        chi2rut = np.inf

        # Add the chi-square terms for glitch parameters
        if r02 is not None or (method.lower() == "fq"):
            if "glitches" in tipo:
                num_sd, frq_sd = None, None
                if method.lower() == "sd":
                    num_sd = icov_sd.shape[0]
                    frq_sd = sd(frq, num_of_n, num_sd)
                param, chi2, reg, ier, _ = sg.fit(
                    frq,
                    num_of_n,
                    dnusurf,
                    num_of_dif2=num_sd,
                    freqDif2=frq_sd,
                    icov=icov_sd,
                    method=method,
                    n_rln=0,
                    npoly_params=npoly_params,
                    nderiv=nderiv,
                    tol_grad=tol_grad,
                    regu_param=regu_param,
                    n_guess=n_guess,
                    tauhe=tauhe,
                    dtauhe=100.0,
                    taucz=taubcz,
                    dtaucz=200.0,
                )
                if ier != 0:
                    chi2rut = np.inf
                else:
                    glh = np.zeros(3)
                    Acz, Ahe = sg.averageAmplitudes(
                        param[0, :], vmin, vmax, delta_nu=dnusurf, method=method
                    )
                    glh[-3] = Ahe
                    glh[-2] = param[0, -3]
                    glh[-1] = param[0, -2]
                    glhparams[0], glhparams[1], glhparams[2] = (
                        glh[-3],
                        glh[-2],
                        glh[-1],
                    )
                    x = inpdata[7][0, :] - glh[:]
                    w = _weight(len(x), seisw)
                    if x.shape[0] == covinv[7].shape[0]:
                        chi2rut += (x.T.dot(covinv[7]).dot(x)) / w
                    else:
                        shapewarn = True
                        chi2rut = np.inf

    if "freqs" in tipo:
        # The frequency correction moved up before the ratios fitting!
        # --> If fitting frequencies, just add the already calculated things
        x = corjoin[0, :] - corjoin[2, :]
        w = _weight(len(corjoin[0, :]), seisw)
        if x.shape[0] == covinv[2].shape[0]:
            chi2rut += (x.T.dot(covinv[2]).dot(x)) / w
        else:
            shapewarn = True
            chi2rut = np.inf

        if ~np.isfinite(chi2rut):
            chi2rut = np.inf
            if debug and verbose:
                print("DEBUG: Computed non-finite chi2, setting chi2 to inf")
        elif chi2rut < 0:
            chi2rut = np.inf
            if debug and verbose:
                print("DEBUG: chi2 less than zero, setting chi2 to inf")

    return chi2rut, warnings, shapewarn, dnusurf, glhparams


def most_likely(selectedmodels):
    """
    Find the index of the model with the highest probability

    Parameters
    ----------
    selectedmodels : dict
        Contains information on all models with a non-zero likelihood.

    Returns
    -------
    maxPDF_path : str
        The path to the model with the highest probability.
    maxPDF_ind : int
        Index of the model with the highest probability.
    """
    maxPDF = -np.inf
    for path, trackstats in selectedmodels.items():
        i = np.argmax(trackstats.logPDF)
        if trackstats.logPDF[i] > maxPDF:
            maxPDF = trackstats.logPDF[i]
            maxPDF_path = path
            maxPDF_ind = trackstats.index.nonzero()[0][i]
    if maxPDF == -np.inf:
        print("The logPDF are all -np.inf")
    return maxPDF_path, maxPDF_ind


def lowest_chi2(selectedmodels):
    """
    Find the index of the model with the lowest chi2 (only gaussian fitting parameters)

    Parameters
    ----------
    selectedmodels : dict
        Contains information on all models with a non-zero likelihood.

    Returns
    -------
    minchi2_path : str
        Path to the model with the lowest chi2.
    minchi2_ind : int
        Index of the model with the lowest chi2.
    """
    minchi2 = np.inf
    for path, trackstats in selectedmodels.items():
        i = np.argmin(trackstats.chi2)
        if trackstats.chi2[i] < minchi2:
            minchi2 = trackstats.chi2[i]
            minchi2_path = path
            minchi2_ind = trackstats.index.nonzero()[0][i]
    return minchi2_path, minchi2_ind


def chi_for_plot(selectedmodels):
    """
    FOR VALIDATION PLOTTING!

    Could this be combined with the functions above in a wrapper?

    Return chi2 value of HLM and LCM.
    """
    # Get highest likelihood model (HLM)
    maxPDF = -np.inf
    minchi2 = np.inf
    for _, trackstats in selectedmodels.items():
        i = np.argmax(trackstats.logPDF)
        j = np.argmin(trackstats.chi2)
        if trackstats.logPDF[i] > maxPDF:
            maxPDF = trackstats.logPDF[i]
            maxPDFchi2 = trackstats.chi2[i]
        if trackstats.chi2[j] < minchi2:
            minchi2 = trackstats.chi2[j]

    return maxPDFchi2, minchi2


def get_highest_likelihood(Grid, selectedmodels, outparams):
    """
    Find highest likelihood model and print info.

    Parameters
    ----------
    Grid : hdf5 object
        The already loaded grid, containing the tracks/isochrones.
    selectedmodels : dict
        Contains information on all models with a non-zero likelihood.
    outparams : dict
        Dict containing the wanted output of the run.

    Returns
    -------
    maxPDF_path : str
        Path to model with maximum likelihood.
    maxPDF_ind : int
        Index of model with maximum likelihood.
    """
    print("\nHighest likelihood model:")
    maxPDF_path, maxPDF_ind = most_likely(selectedmodels)
    print(
        "Weighted, non-normalized log-probability:",
        np.max(selectedmodels[maxPDF_path].logPDF),
    )
    print(maxPDF_path + "[" + str(maxPDF_ind) + "]")

    # Print name if it exists
    if "name" in Grid[maxPDF_path]:
        print("Name:", Grid[maxPDF_path + "/name"][maxPDF_ind].decode("utf-8"))

    # Print parameters
    for param in outparams:
        if param == "distance":
            continue
        print(param + ":", Grid[maxPDF_path + "/" + param][maxPDF_ind])
    return maxPDF_path, maxPDF_ind


def get_lowest_chi2(Grid, selectedmodels, outparams):
    """
    Find model with lowest chi2 value and print info.

    Parameters
    ----------
    Grid : hdf5 object
        The already loaded grid, containing the tracks/isochrones.
    selectedmodels : dict
        Contains information on all models with a non-zero likelihood.
    outparams : dict
        Dict containing the wanted output of the run.

    Returns
    -------
    minchi2_path : str
        Path to model with lowest chi2
    minchi2_ind : int
        Index of model with lowest chi2
    """
    print("\nLowest chi2 model:")
    minchi2_path, minchi2_ind = lowest_chi2(selectedmodels)
    print("chi2:", np.min(selectedmodels[minchi2_path].chi2))
    print(minchi2_path + "[" + str(minchi2_ind) + "]")

    # Print name if it exists
    if "name" in Grid[minchi2_path]:
        print("Name:", Grid[minchi2_path + "/name"][minchi2_ind].decode("utf-8"))

    # Print parameters
    for param in outparams:
        if param == "distance":
            continue
        print(param + ":", Grid[minchi2_path + "/" + param][minchi2_ind])
    return minchi2_path, minchi2_ind


def quantile_1D(data, weights, quantile):
    """
    Compute the weighted quantile of a 1D numpy array.
    The function is borrowed from the Python package wquantiles

    Parameters
    ----------
    data : ndarray
        Input array (one dimension).
    weights : ndarray
        Array with the weights of the same size of `data`.
    quantile : float
        Quantile to compute. It must have a value between 0 and 1.

    Returns
    -------
    result : float
        The output value.
    """
    # Sort the data
    ind_sorted = np.argsort(data)
    sorted_data = data[ind_sorted]
    sorted_weights = weights[ind_sorted]

    # Compute the auxiliary arrays
    Sn = np.cumsum(sorted_weights)
    Pn = (Sn - 0.5 * sorted_weights) / np.sum(sorted_weights)

    result = np.interp(quantile, Pn, sorted_data)
    assert not np.any(np.isnan(result)), "NaN encounted in quantile_1D."

    return result


def posterior(x, nonzeroprop, sampled_indices, nsigma=0.25):
    """
    Compute posterior of x as a weighted histogram of x with weights y.
    The bin width is determined by the Freedman Diaconis Estimator
    The histogram is smoothed using a Gaussian kernel with a width
    of nsigma*sigma

    Parameters
    ----------
    x : array
        samples
    y : array
        likelihood of samples
    nonzeroprop : array
        indices of x with non-zero likelihood
    sampled_indices :
        indices of weighted draws from y, used for bin estimation
    nsigma : float
        fractional standard deviation used for smoothing

    Returns
    -------
    like : function
        interpolated posterior
    """
    samples = x[nonzeroprop][sampled_indices]
    xmin, xmax = np.nanmin(samples), np.nanmax(samples)

    # Check if all entries of x are equal
    if np.all(x == x[0]) or math.isclose(xmin, xmax, rel_tol=1e-5):
        xvalues = np.array([x[0], x[0]])
        like = np.array([1.0, 1.0])
        like = interp1d(xvalues, like, fill_value=0, bounds_error=False)
        return like

    # Compute bin width and number of bins
    N = len(samples)
    bin_width = _hist_bin_fd(samples, None)
    if math.isclose(bin_width, 0, rel_tol=1e-5):
        nbins = int(np.ceil(np.sqrt(N)))
    else:
        nbins = int(np.ceil((xmax - xmin) / bin_width)) + 1
    if nbins > N:
        nbins = int(np.ceil(np.sqrt(N)))
    bin_edges = np.linspace(xmin, xmax, nbins)
    xvalues = bin_edges[:-1] + np.diff(bin_edges) / 2.0
    sigma = np.std(samples)
    like = np.histogram(samples, bins=bin_edges, density=True)[0]
    if sigma > 0 and bin_width > 0:
        like = gaussian_filter1d(like, (nsigma * sigma / bin_width))
    like = interp1d(xvalues, like, fill_value=0, bounds_error=False)
    return like


def calc_key_stats(x, centroid, uncert, weights=None):
    """
    Calculate and report the wanted format of centroid value and
    uncertainties.

    Parameters
    ----------
    x : list
        Parameter values of the models with non-zero likelihood.
    centroid : str
        Options for centroid value definition, 'median' or 'mean'.
    uncert : str
        Options for reported uncertainty format, 'quantiles' or
        'std' for standard deviation.
    weights : list
        Weights needed to be applied to x before extracting statistics.

    Returns
    -------
    xcen : float
        Centroid value
    xm : float
        16'th percentile if quantile selected, 1 sigma for standard
        deviation.
    xp : float, None
        84'th percentile if quantile selected, None for standard
        deviation.
    """
    # Definition of Bayesian 16, 50, and 84 percentiles
    qs = [0.5, 0.158655, 0.841345]
    xp = None

    # Handling af all different combinations of input
    if uncert == "quantiles" and not type(weights) == type(None):
        xcen, xm, xp = quantile_1D(x, weights, qs)
    elif uncert == "quantiles":
        xcen, xm, xp = np.quantile(x, qs)
    else:
        xm = np.std(x)
    if centroid == "mean" and not type(weights) == type(None):
        xcen = np.average(x, weights=weights)
    elif centroid == "mean":
        xcen = np.mean(x)
    elif uncert != "quantiles":
        xcen = np.median(x)
    return xcen, xm, xp
