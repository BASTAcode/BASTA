"""
Auxiliary functions for frequency analysis
"""
import os
from copy import deepcopy
from distutils.util import strtobool

import numpy as np
import h5py
from sklearn.covariance import MinCovDet

from basta import freq_fit
from basta import fileio as fio
from basta.constants import sydsun as sydc
from basta.constants import freqtypes
import basta.supportGlitch as sg
from basta.glitch_fq import fit_fq
from basta.glitch_sd import fit_sd
from basta.sd import sd


def combined_ratios(r02, r01, r10):
    """
    Routine to combine r02, r01 and r10 ratios to produce ordered ratios r010,
    r012 and r102

    Parameters
    ----------
    r02 : array
        radial orders, r02 ratios,
        scratch for uncertainties (to be calculated), frequencies
    r01 : array
        radial orders, r01 ratios,
        scratch for uncertainties (to be calculated), frequencies
    r10 : array
        radial orders, r10 ratios,
        scratch for uncertainties (to be calculated), frequencies

    Returns
    -------
    r010 : array
        radial orders, r010 ratios,
        scratch for uncertainties (to be calculated), frequencies
    r012 : array
        radial orders, r012 ratios,
        scratch for uncertainties (to be calculated), frequencies
    r102 : array
        radial orders, r102 ratios,
        scratch for uncertainties (to be calculated), frequencies
    """

    # Number of ratios
    n02 = r02.shape[0]
    n01 = r01.shape[0]
    n10 = r10.shape[0]
    n010 = n01 + n10
    n012 = n01 + n02
    n102 = n10 + n02

    # R010 (R01 followed by R10)
    r010 = np.zeros((n010, 4))
    r010[0:n01, :] = r01[:, :]
    r010[n01 : n01 + n10, 0] = r10[:, 0] + 0.1
    r010[n01 : n01 + n10, 1:4] = r10[:, 1:4]
    r010 = r010[r010[:, 0].argsort()]
    r010[:, 0] = np.round(r010[:, 0])

    # R012 (R01 followed by R02)
    r012 = np.zeros((n012, 4))
    r012[0:n01, :] = r01[:, :]
    r012[n01 : n01 + n02, 0] = r02[:, 0] + 0.1
    r012[n01 : n01 + n02, 1:4] = r02[:, 1:4]
    r012 = r012[r012[:, 0].argsort()]
    r012[:, 0] = np.round(r012[:, 0])

    # R102 (R10 followed by R02)
    r102 = np.zeros((n102, 4))
    r102[0:n10, :] = r10[:, :]
    r102[n10 : n10 + n02, 0] = r02[:, 0] + 0.1
    r102[n10 : n10 + n02, 1:4] = r02[:, 1:4]
    r102 = r102[r102[:, 0].argsort()]
    r102[:, 0] = np.round(r102[:, 0])

    return r010, r012, r102


def glitch_and_ratio(
    frq,
    ngr,
    grtype="r012",
    num_of_n=None,
    vmin=None,
    vmax=None,
    delta_nu=None,
    icov_sd=None,
    nrealizations=10000,
    method="FQ",
    tol_grad=1e-3,
    regu_param=7.0,
    n_guess=200,
    tauhe=None,
    dtauhe=None,
    taucz=None,
    dtaucz=None,
):
    """
    Routine to compute glitch-ratio combination of a given type together with the
    corresponding full covariance matrix

    Parameters
    ----------
    frq : array
        Harmonic degrees, radial orders, frequencies, uncertainties
    ngr : integer
        Total number of ratios and glitch parameters
    grtype : str
        Glitch-ratio combination (one of r02, r01, r10, r010, r012, r102, glitches,
        gr02, gr01, gr10, gr010, gr012, gr102)
    num_of_n : array of int
        Number of modes for each l
    vmin : float
        Minimum value of the observed frequency (muHz)
    vmax : float
        Maximum value of the observed frequency (muHz)
    delta_nu : float
        An estimate of large frequncy separation (muHz)
    icov_sd : array
        Inverse covariance matrix for second differences
        used only for method='SD'
    nrealizations : integer, optional
        Number of realizations used in covariance calculation
    method : str
        Fitting method ('FQ' or 'SD')
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

    Returns
    -------
    med : array
        ratios + helium glitch parameters (average amplitude, width and acoustic depth)
    cov : array
        corresponding full covariance matrix
    """

    # Compute and store different realizations of ratios and glitch parameters
    rln_data = np.zeros((nrealizations, ngr))
    perturb_frq = deepcopy(frq)
    n = 0
    nfailed = 0
    np.random.seed(1)
    for i in range(nrealizations):

        # Perturb frequency
        perturb_frq[:, 2] = np.random.normal(frq[:, 2], frq[:, 3])

        # Compute glitch parameters, if necessary
        if grtype in [*freqtypes.glitches, *freqtypes.grtypes]:

            if not i % 100:
                print("%d realizations completed..." % (i))

            if method.lower() == "fq":
                param, chi2, reg, ier = fit_fq(
                    perturb_frq,
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
            elif method.lower() == "sd":
                frq_sd = sd(perturb_frq, num_of_n, icov_sd.shape[0])
                param, chi2, reg, ier = fit_sd(
                    frq_sd,
                    icov_sd,
                    delta_nu,
                    tol_grad_sd=tol_grad,
                    regu_param_sd=regu_param,
                    num_guess=n_guess,
                    tauhe=tauhe,
                    dtauhe=dtauhe,
                    taucz=taucz,
                    dtaucz=dtaucz,
                )
            else:
                raise ValueError("Invalid glitch-fitting method!")

            # Continue if failed
            if ier != 0:
                nfailed += 1
                continue

            # Compute average amplitude
            Acz, Ahe = sg.averageAmplitudes(
                param, vmin, vmax, delta_nu=delta_nu, method=method
            )
            rln_data[n, -3] = Ahe
            rln_data[n, -2] = param[-3]
            rln_data[n, -1] = param[-2]

        # Compute ratios, if necessary
        if grtype in [*freqtypes.rtypes, *freqtypes.grtypes]:
            tmp_r02, tmp_r01, tmp_r10 = freq_fit.ratios(perturb_frq)
            tmp_r010, tmp_r012, tmp_r102 = combined_ratios(tmp_r02, tmp_r01, tmp_r10)

            # 02
            if grtype == "gr02":
                rln_data[n, 0 : ngr - 3] = tmp_r02[:, 1]
            elif grtype == "r02":
                rln_data[n, :] = tmp_r02[:, 1]

            # 01
            if grtype == "gr01":
                rln_data[n, 0 : ngr - 3] = tmp_r01[:, 1]
            elif grtype == "r01":
                rln_data[n, :] = tmp_r01[:, 1]

            # 10
            if grtype == "gr10":
                rln_data[n, 0 : ngr - 3] = tmp_r10[:, 1]
            elif grtype == "r10":
                rln_data[n, :] = tmp_r10[:, 1]

            # 010
            if grtype == "gr010":
                rln_data[n, 0 : ngr - 3] = tmp_r010[:, 1]
            elif grtype == "r010":
                rln_data[n, :] = tmp_r010[:, 1]

            # 012
            if grtype == "gr012":
                rln_data[n, 0 : ngr - 3] = tmp_r012[:, 1]
            elif grtype == "r012":
                rln_data[n, :] = tmp_r012[:, 1]

            # 102
            if grtype == "gr102":
                rln_data[n, 0 : ngr - 3] = tmp_r102[:, 1]
            elif grtype == "r102":
                rln_data[n, :] = tmp_r102[:, 1]

        n += 1

    rln_data = rln_data[0:n, :]

    if nfailed / nrealizations > 0.3:
        print("Warning: More than 30% of the total realizations failed!")
        print("Failed %d out of %d realizations" % (nfailed, nrealizations))

    # Compute the covariance matrix
    j = int(round(n / 2))
    covtmp = MinCovDet().fit(rln_data[0:j, :]).covariance_
    cov = MinCovDet().fit(rln_data).covariance_

    # Test the convergence (change in standard deviations below a relative tolerance)
    rdif = np.amax(
        np.abs(
            np.divide(
                np.sqrt(np.diag(covtmp)) - np.sqrt(np.diag(cov)), np.sqrt(np.diag(cov))
            )
        )
    )
    if rdif > 0.1:
        print("Warning: Covariance failed to converge!")
        print("Maximum relative difference = %.2e (>0.1)" % (rdif))

    # Compute the median values
    med = np.zeros(ngr)
    for i in range(ngr):
        med[i] = np.median(rln_data[:, i])

    # Write data to a hdf5 file
    # with h5py.File("./16CygA.hdf5", "w") as f:
    #    f.create_dataset("medg012", data=med)
    #    f.create_dataset("covg012", data=cov)

    return med, cov


def solar_scaling(Grid, inputparams, diffusion=None):
    """
    Scale numax and dnu using either a solar model stored in the grid, or using default
    solar values.

    Parameters
    ----------
    Grid : hdf5 object
        The already loaded grid, containing the tracks/isochrones.

    inputparams : tuple
        Tuple of strings with the variables to be fitted.

    diffusion : None or int, optional
        Selection of solar model with/without diffusion for e.g. the BaSTI isochrones.
        None signals grids with only one available solar model. For isochrones (with
        odea), a value of 0 signals no diffusion; 1 is diffusion.

    Returns
    -------
    inputparams : tuple
        Modified version of `inputparams` with the added scaled values.
    """
    # Check for solar values, if not set then use default
    dnusun = inputparams.get("dnusun", sydc.SUNdnu)
    numsun = inputparams.get("numsun", sydc.SUNnumax)

    # Extract solar model toggle from the grid
    try:
        solarmodel = strtobool(inputparams.get("solarmodel", ""))
    except ValueError:
        print(
            "Warning: Invalid value given for activation of solar scaling!",
            "Must be True/False! Will now assume False ...",
        )
        solarmodel = False

    avail_models = list(Grid["solar_models"])
    if solarmodel and len(avail_models) > 0:
        # Isochrone library with diffusion toggle or standard grid
        if diffusion is not None:

            # Note: Hardwired names of the BaSTI solar models!
            if diffusion == 0:
                sunmodname = "bastisun_new"
            else:
                sunmodname = "bastisun_new_diff"
        else:
            if len(avail_models) == 1:
                sunmodname = avail_models[0]
            else:
                raise NotImplementedError("More than one solar model found in grid!")

        # Get all solar model dnu values
        sunmodpath = os.path.join("solar_models", sunmodname)
        print("\nSolar model scaling activated!")
        print("* Solar model found in grid at {0}".format(sunmodpath))
        sunmoddnu = {
            param: Grid.get(os.path.join(sunmodpath, param))[()]
            for param in Grid[sunmodpath]
            if param.startswith("dnu")
        }

    # If no model found
    elif len(avail_models) == 0:
        print("No solar model found!\n--> Dnu will not be scaled.")
        sunmoddnu = {}

    # If no solar model is given (no scaling)
    else:
        print("Solar model scaling not activated!\n--> Dnu will not be scaled.")
        sunmoddnu = {}

    # Libraries use solar units for numax and dnu (from scaling relations)
    # The input values (given in muHz) are converted into solar units
    fitparams = inputparams.get("fitparams")
    outparams = inputparams.get("asciiparams")
    scaling_input = [
        par for par in fitparams if (par.startswith("dnu") or par.startswith("numax"))
    ]
    for param in scaling_input:
        # dnufit is given in muHz in the grid since it is not based on scaling relations
        if param in ["dnufit", "dnufitMos12"]:
            continue

        # numax and any other dnu are scaled
        if param.startswith("numax"):
            scale_factor = numsun
        else:
            scale_factor = dnusun

        # Print the scaling of numax
        print(
            "* {0} converted from {1} microHz to ".format(param, fitparams[param][0]),
            end="",
        )
        fitparams[param] = [p / scale_factor for p in fitparams[param]]
        print(
            "{0} solar units (solar value: {1} microHz)".format(
                round(fitparams[param][0], 3), scale_factor
            )
        )

    # Scale any input dnu's to the solar value of the grid
    dnu_scales = {}
    for dnu in sunmoddnu:
        if dnu in outparams:
            if dnu in ["dnufit", "dnufitMos12"]:
                dnu_rescal = sunmoddnu[dnu] / dnusun
            else:
                # They are already scaled to the solar value.
                dnu_rescal = sunmoddnu[dnu]

            if dnu in fitparams:
                if dnu in ["dnufit", "dnufitMos12"]:
                    print(
                        "* {0} scaled by the factor {1} according to the".format(
                            dnu, round(dnu_rescal, 5)
                        ),
                        "grid solar value of {0} microHz".format(
                            round(sunmoddnu[dnu], 3)
                        ),
                        "and the real solar value of {0} microHz".format(dnusun),
                    )
                    fitparams[dnu] = [(dnu_rescal) * p for p in fitparams[dnu]]
                else:
                    print(
                        "{0} scaled by the factor {1}".format(
                            dnu, round(sunmoddnu[dnu], 5)
                        ),
                        "according to the grid solar value.",
                    )
                    fitparams[dnu] = [(dnu_rescal) * p for p in fitparams[dnu]]
            dnu_scales[dnu] = dnu_rescal
    inputparams["dnu_scales"] = dnu_scales
    return inputparams


def prepare_obs(inputparams, verbose=False):
    """
    Prepare frequencies and ratios for fitting

    Parameters
    ----------
    inputparams : dict
        Inputparameters for BASTA
    verbose : bool
        Flag that if True adds extra text to the log (for developers).

    Returns
    -------
    obskey : array
        Array containing the angular degrees and radial orders of obs
    obs : array
        Array containing the modes in the observed data
    numax : float
        numax as found in `inputparams`.
    dnufrac : float
        The allowed fraction of the large frequency separation that defines
        when the l=0 mode in the model is close enough to the lowest l=0 mode
        in the observed set that the model can be considered.
    datos : array
        Individual frequencies, uncertainties, and combinations read
        directly from the observational input files
    covinv : array
        Covariances between individual frequencies and frequency ratios read
        from the observational input files.
    fcor : strgs
        Type of surface correction (see :func:'freq_fit.py').
    obsintervals : array
        Array containing the endpoints of the intervals used in the frequency
        fitting routine in :func:'freq_fit.calc_join'.
        As it is the same in all iterations for the observed frequencies,
        this is computed in util.prepare_obs once and given as an argument
        in order to save time and memory.
    dnudata : float
        Large frequency separation obtained by fitting the radial mode observed
        frequencies (like dnufit, but for the data). Used for fitting ratios.
    dnudata_err : float
        Uncertainty on dnudata
    frq_sd : array
        Second differences (l, n, v(muHz), err(muHz), dif2(muHz), err(muHz))
    icov_sd : array
        Inverse covariance matrix for second differences
    vmin : float
        Minimum value of the observed frequency (muHz)
    vmax : float
        Maximum value of the observed frequency (muHz)
    num_of_n : array of int
        Number of modes for each l
    """
    print("\nPreparing asteroseismic input ...")

    fitfreqs = inputparams.get("fitfreqs")
    (freqxml, glhhdf, correlations, bexp, rt, seisw) = fitfreqs

    # Get frequency correction method
    fcor = inputparams.get("fcor", "BG14")
    if fcor not in ["None", "HK08", "BG14", "cubicBG14"]:
        raise ValueError(
            'ERROR: fcor must be either "None", "HK08", "BG14" or "cubicBG14"'
        )

    # Get numax
    if inputparams.get("numax", False) is False:
        numaxerr = (
            "ERROR: numax must be specified when fitting individual"
            + " frequencies or ratios!"
        )
        raise ValueError(numaxerr)
    numax = inputparams.get("numax")  # *numsun #NOTE SOLAR UNITS

    # Just check if 'dnufit' is specified, will be used otherwhere
    if inputparams.get("dnufit", False) is False:
        raise ValueError("ERROR: We need a deltanu value!")

    # Read dnu-constraint value
    dnufrac = inputparams.get("dnufrac", 0.15)

    if "freqs" in rt and correlations:
        getfreqcovar = True
    else:
        getfreqcovar = False

    # Get the nottrustedfile for excempted modes
    nottrustedfile = inputparams.get("nottrustedfile")

    # Check if it is unnecesarry to compute ratios
    getratios = False
    freqplots = inputparams.get("freqplots")
    if any(x in [*freqtypes.rtypes, *freqtypes.grtypes] for x in rt):
        getratios = True
    elif len(freqplots):
        if any([freqplots[0] == True, "ratios" in freqplots]):
            getratios = True

    if getratios:
        print(
            "Frequency ratios required for fitting and/or plotting. This may take",
            "a little while...",
        )

    # Load or calculate ratios (requires numax)
    # --> datos and cov are 14-tuples
    (
        datos,
        cov,
        obskey,
        obs,
        dnudata,
        dnudata_err,
        frq_sd,
        icov_sd,
        vmin,
        vmax,
        num_of_n,
    ) = fio.read_rt(
        inputparams,
        freqxml,
        glhhdf,
        rt,
        numax,
        getratios,
        getfreqcovar,
        nottrustedfile=nottrustedfile,
        verbose=verbose,
    )

    # If correlations = False, ignore correlations
    if not correlations:
        cov = list(cov)
        for i in range(len(cov)):
            if cov[i] is None:
                continue
            cov[i] = np.identity(cov[i].shape[0]) * np.diagonal(cov[i])
        cov = tuple(cov)

    # Computing inverse of covariance matrices...
    # ***Why 15 elements instead of 14 inverse covariance matrix-KV***
    covinv = [
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
    ]
    for i in range(14):
        if cov[i] is not None:
            covinv[i] = np.linalg.pinv(cov[i], rcond=1e-12)

    # Compute the intervals used in frequency fitting
    if "freqs" in rt:
        obsintervals = freq_fit.make_intervals(obs, obskey, dnu=inputparams["dnufit"])
    else:
        obsintervals = None

    # Return 'dnudata' as well, which is determined in a similar way as
    # 'dnufit', but based on the observed frequencies of the data (where as
    # 'dnufit' is from model frequencies in the grid)
    print("Done!")
    return (
        obskey,
        obs,
        numax,
        dnufrac,
        datos,
        covinv,
        fcor,
        obsintervals,
        dnudata,
        dnudata_err,
        frq_sd,
        icov_sd,
        vmin,
        vmax,
        num_of_n,
    )


def get_givenl(l, osc, osckey):
    """
    Returns frequencies, radial orders, and inertias or uncertainties of a
    given angular degree l.

    Parameters
    ----------
    l : int
        Angular degree between l=0 and l=2
    osc : array
        An array containing the individual frequencies in [0,:] and the
        inertias/uncertainties in [1, :].
        This can also be a joined array between observations and model modes
        as long as the sorting corresponds to the sorting in osckey.
    osckey : array
        An array containing the n and l of the values in osc.
        They are ordered in the same way.

    Returns
    -------
    osckey_givenl : array
        An array containing the n and l of the values in osc for the input l.
    osc_givenl : array
        An array containing the individual frequencies, inertias for the
        input l.
    """
    givenl = osckey[0, :] == l
    return osckey[:, givenl], osc[:, givenl]


def transform_obj_array(objarr):
    """
    Transform a two-column nested array with dtype=object into a corresponding
    2D array with a dtype matching the indivial columns.

    Useful for reading frequency information from the BASTA HDF5 files stored
    in variable sized arrays.
    """
    return np.vstack([objarr[0], objarr[1]])


def calculate_epsilon(n, l, f, dnu):
    """
    Calculates epsilon by fitting that and dnu to the White et al. 2012,
    equation (1)

    Parameters
    ----------
    n: array
        Radial order of the modes
    l : array
        Angular degree of the modes
    f : array
        Frequency of the modes
    dnu : float or None
        The large frequency parameter.
        If none is provided, a dnu will be estimated from a fit to l=0 modes.

    Returns
    -------
    epsilon : float
        epsilon parameter
    """
    # Fit a dnu, if none is provided
    if not isinstance(dnu, float):
        dnucoeff = np.polyfit(range(len(f[l == 0])), f[l == 0], 1)
        dnu = dnucoeff[0]

    # The l=0 and l=l mode for each frequency for small separation
    f0 = np.array([f[l == 0][n[l == 0] == fn][0] for fn in n])
    fl = np.array([f[l == fl][n[l == fl] == fn][0] for fn, fl in zip(n, l)])

    # Fitting to: nu_nl + deltanu_0l - dnufit (n + l/2) = dnufit * epsilon
    xvalues = range(len(f))
    yvalues = f + (f0 - fl) - dnu * (n + l / 2)
    fitcoeff = np.polyfit(xvalues, yvalues, 0)
    return fitcoeff[0] / dnu


def check_epsilon_of_freqs(freqs, starid, dnu, quiet=False):
    """
    Calculates the offset epsilon of the frequencies, to check if
    the radial order has been labeled correctly.
    Value range of epsilon from White et al. 2012

    Parameters
    ----------
    freqs : dict
        Dictionary of frequencies and their modes, as read in read_fre
    starid : str
        Unique identifier of star, for printing with the alert
    dnu : float or None
        The large frequency parameter.
    quiet : bool, optional
        Toggle to silence the output (useful for running batches)

    Returns
    -------
    ncor : float
        Correction to the ordering of modes.
    """
    epsilon_limits = [0.6, 1.8]

    # Read in the data
    n = freqs["order"]
    l = freqs["degree"]
    f = freqs["frequency"]

    # Can only do this for orders with l=0 modes
    filt = np.isin(n, n[l == 0])
    n = n[filt]
    l = l[filt]
    f = f[filt]

    epsilon = calculate_epsilon(n, l, f, dnu)
    eps_status = lambda eps: eps >= epsilon_limits[0] and eps <= epsilon_limits[1]

    ncor = 0
    if not eps_status(epsilon):
        if not quiet:
            print(
                "\nStar {:s} has an odd epsilon".format(starid),
                "value of {:.1f},".format(epsilon),
            )
        while not eps_status(epsilon):
            if epsilon > epsilon_limits[1]:
                ncor += 1
            else:
                ncor -= 1
            epsilon = calculate_epsilon(n + ncor, l, f, dnu)

        if not quiet:
            print(
                "Correction of n-order by {:d}".format(ncor),
                "gives epsilon value of {:.1f}.".format(epsilon),
            )
        return ncor
    else:
        if not quiet:
            print(
                "Star {:s} has an".format(starid), "epsilon of: {:.1f}.".format(epsilon)
            )
        return 0


def scale_by_inertia(osckey, osc):
    """
    This function outputs the scaled sizes of the modes scaled inversly by the
    normalized inertia of the mode.

    Parameters
    ----------
    osckey : array
        Mode identification of the modes
    osc : array
        Frequencies and either inertia in the case of modelled modes or
        uncertainties in the case of observed modes.

    Returns
    -------
    s : list
        List containing 3 array, one per angular degree l.
        The arrays contain the sizes of the modes scaled by inertia.
    """
    s = []
    el0min = np.min(osc[1, :])
    _, oscl0 = get_givenl(l=0, osckey=osckey, osc=osc)
    _, oscl1 = get_givenl(l=1, osckey=osckey, osc=osc)
    _, oscl2 = get_givenl(l=2, osckey=osckey, osc=osc)

    if len(oscl0) != 0:
        s0 = [10 * (1 / (np.log10(2 * n / (el0min)))) ** 2 for n in oscl0[1, :]]
        s.append(np.asarray(s0))
    if len(oscl1) != 0:
        s1 = [10 * (1 / (np.log10(2 * n / (el0min)))) ** 2 for n in oscl1[1, :]]
        s.append(np.asarray(s1))
    if len(oscl2) != 0:
        s2 = [10 * (1 / (np.log10(2 * n / (el0min)))) ** 2 for n in oscl2[1, :]]
        s.append(np.asarray(s2))
    return s
