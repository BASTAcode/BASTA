"""
Collection of all constants used in BASTA
"""

from dataclasses import dataclass  # Python 3.7+ !
import numpy as np


@dataclass
class sydsun:
    """
    Default solar values from the SYD asteroseismic pipeline.
    """

    SUNdnu = 135.1
    SUNnumax = 3090.0


@dataclass
class freqtypes:
    """
    Different possibilities of fitting frequencies, for global access
    """

    rtypes = ["r010", "r02", "r01", "r10", "r012", "r102"]
    freqs = ["freqs"]
    glitches = ["glitches", "gr010", "gr02", "gr01", "gr10", "gr012", "gr102"]
    epsdiff = ["e01", "e02", "e012"]
    alltypes = [*freqs, *glitches, *rtypes, *epsdiff]
    defaultrtypes = ["r01"]
    defaultepstypes = ["e012"]

    surfeffcorrs = ["HK08", "BG14", "cubicBG14"]


@dataclass
class statdata:
    """
    Constant values for statistics, to ensure consistensy across code

    Contains
    --------
    quantiles : list
        Median, lower and upper percentiles of Bayesian posterior
        distributions to draw
    nsamples : int
        Number of samples to draw when sampling
    nsigma : float
        Fractional standard deviation used for smoothing
    """

    quantiles = [0.5, 0.158655, 0.841345]
    nsamples = 100000
    nsigma = 0.25


@dataclass
class parameters:
    """
    All the different parameters in the form:
    (name, unit, pname, remark, color)

    - Note some parameters are only available for certain tracks.
    - Color is for the Kiel diagram

    The list is available in table format in the
    :ref:`documentation <controls_params>`.
    """

    pcol = "#DDDDDD"  # Placeholder color for non-designated variables

    # Here we disable the Black-formatter and accept the long lines
    # fmt: off
    params = [
              ('modnum', None, r'Model', r'Model number', pcol),
              ('ove', None, r'$\xi_\mathrm{ove}$', r'Overshooting efficiency', pcol),
              ('gcut', None, r'$g_\mathrm{cut}$', r'Geometric cutoff', pcol),
              ('eta', None, r'$\eta$', r'Reimers mass loss', '#858FC2'),
              ('alphaMLT', None, r'$\alpha_\mathrm{MLT}$', r'Mixing length efficiency', '#E4632D'),
              ('Gconst', r'cm3/gs2', r'G', r'Gravitational constant', pcol),
              ('LPhot', r'solar', r'$L$ (L$_\odot$)', r'Photospheric luminosity', '#CCBB44'),
              ('radPhot', r'solar', r'$R_\mathrm{phot}$ (R$_\odot$)', r'Photospheric radius', '#EE6677'),
              ('radTot', r'solar', r'$R_\mathrm{tot}$ (R$_\odot$)', r'Total radius', '#EE6677'),
              ('massini', r'solar', r'$M_\mathrm{ini}$ (M$_\odot$)', r'Initial mass', '#549EB3'),
              ('massfin', r'solar', r'$M$ (M$_\odot$)', r'Current mass', '#4E96BC'),
              ('age', r'Myr', r'Age (Myr)', r'Current age in Myr',  '#999933'),
              ('Teff', r'K', r'$T_\mathrm{eff}$ (K)', r'Effective temperature', '#88CCEE'),
              ('rho', r'g/cm3', r'$\rho$ (g/cm$^3$)', r'Mean stellar density', '#AA4499'),
              ('rhocen', r'g/cm3', r'$\rho_\mathrm{cen}$ (g/cm$^3$)', r'Central density', pcol),
              ('logg', r'log10(cm/s2)', r'$\log \, g$ (dex)', r'Surface gravity', '#DDCC77'),
              ('FeHini', r'dex', r'[Fe/H]$_\mathrm{ini}$ (dex)', r'Initial iron abundance', pcol),
              ('MeHini', r'dex', r'[M/H]$_\mathrm{ini}$ (dex)', r'Initial metallicity', pcol),
              ('MeH', r'dex', r'[M/H] (dex)', r'Metallicity', '#A778B4'),
              ('FeH', r'dex', r'[Fe/H] (dex)', r'Iron abundance', '#6F4C98'),
              ('alphaFe', r'dex', r'[$\alpha$/Fe] (dex)', r'Alpha enhancement', '#60AB9E'),
              ('xsur', None, r'X$_\mathrm{sur}$', r'Surface hydrogen fraction', '#77B77D'),
              ('ysur', None, r'Y$_\mathrm{sur}$', r'Surface helium fraction', '#A6BE54'),
              ('zsur', None, r'Z$_\mathrm{sur}$', r'Surface heavy elements fraction', '#D18541'),
              ('xcen', None, r'X$_\mathrm{cen}$', r'Central hydrogen fraction', '#77B77D'),
              ('ycen', None, r'Y$_\mathrm{cen}$', r'Central helium fraction', '#A6BE54'),
              ('zcen', None, r'Z$_\mathrm{cen}$', r'Central heavy elements fraction', '#D18541'),
              ('xini', None, r'X$_\mathrm{ini}$', r'Initial hydrogen fraction', '#77B77D'),
              ('yini', None, r'Y$_\mathrm{ini}$', r'Initial helium fraction', '#A6BE54'),
              ('zini', None, r'Z$_\mathrm{ini}$', r'Initial heavy elements fraction', '#D18541'),
              ('Mbcz', None, r'M$_\mathrm{bcz}$ (m/M)', r'Mass coordinate of base of the convective zone', '#E49C39'),
              ('Rbcz', None, r'R$_\mathrm{bcz}$ (r/R$_\mathrm{phot}$)', r'Radius coordinate of base of the convective zone', '#DF4828'),
              ('Mcore', None, r'M$_\mathrm{core}$ (m/M)', r'Mass coordinate of the convective core', '#CC6677'),
              ('Rcore', None, r'R$\mathrm{core}$ (r/R$_\mathrm{phot}$)', r'Radius coordination of the convective core', '#882255'),
              ('McoreX', None, r'M$_\mathrm{core}$ (m/M)', r'Mass coordinate of the convective core (old diagnostic)', '#CC6677'),
              ('RcoreX', None, r'R$\mathrm{core}$ (r/R$_\mathrm{phot}$)', r'Radius coordination of the convective core (old diagnostic)', '#882255'),
              ('MMaxNucE', None, r'M$_\mathrm{max}(\epsilon)$ (m/M)', r'Mass coordinate of maximum energy generation', pcol),
              ('RMaxNucE', None, r'R$_\mathrm{max}(\epsilon)$ (r/R)$_\mathrm{phot}$', r'Radius coordinate of maximum energy generation', pcol),
              ('ZAMSTeff', r'K', r'ZAMS $T_\mathrm{eff}$ (K)', r'Effective temperature at the ZAMS', pcol),
              ('ZAMSLPhot', r'solar', r'ZAMS $L$ (L$_odot$)', r'Luminosity at the ZAMS', pcol),
              ('TAMS', None, r'TAMS', r'Age scaled by TAMS (terminal age of main sequence, X$_\mathrm{cen}$ <1e-5)', pcol),
              ('numax', r'solar', r'$\nu_\mathrm{max}$ ($\mu$Hz)', r'Frequency of maximum oscillation power', '#4477AA'),
              ('dnuscal', r'solar', r'$\Delta \nu_\mathrm{scaling}$ ($\mu$Hz)', r'Large frequency separation from scaling relations', '#228833'),
              ('dnufit', r'microHz', r'$\Delta \nu_\mathrm{fit}$ ($\mu$Hz)', r'Large frequency separation from linear fit to individual $\ell=0$ modes', '#228833'),
              ('epsfit', None, r'$\epsilon_\mathrm{fit}$', r'Dimensionless frequency offset', '#B8221E'),
              ('dnufitMos12', r'microHz', r'$\Delta \nu_\mathrm{fit}$ ($\mu$Hz)', r'Large frequency separation from linear fit to individual $\ell=0$ modes (Mosser et al. 2012)', '#117733'),
              ('epsfitMos12', None, r'$\epsilon_\mathrm{fit}$', r'Dimensionless frequency offset (Mosser et al. 12)', '#44AA99'),
              ('dnuAsf', r'solar', r'$\Delta \nu_\mathrm{Asfgrid}$ ($\mu$Hz)', r'Large frequency separation corrected with Asfgrid following Sharma et al. 2016, Stello and Sharma 2022', '#228833'),
              ('numaxAsf', r'solar', r'$\nu_\mathrm{max,\,Asfgrid}$ ($\mu$Hz)', r'Frequency of maximum oscillation power corrected with Asfgrid following Sharma et al. 2016, Stello and Sharma 2022', '#4477AA'),
              ('fdnuAsf', None, r'f$_{\Delta \nu}$ (Asfgrid)', r'Correction factor for large frequency separation with Asfgrid following Sharma et al. 2016, Stello and Sharma 2022', pcol),
              ('fdnuSer', None, r'f$_\Delta \nu$ (Serenelli 17)', r'Correction factor for large frequency separatoin from Serenelli et al. 2017', pcol),
              ('nummodSer', None, r'N$_\mathrm{modes}$ (Serenelli 17)', r'Number of modes used in the corrections from Serenelli et al. 2017', pcol),
              ('errflagSer', None, r'error$_\mathrm{flag}$ (Serenelli 17)', r'Error output of the corrections from Serenelli et al. 2017', pcol),
              ('dnuSer', r'solar', r'$\Delta \nu_\mathrm{Serenelli 17}$', r'Large frequency separation corrected following Serenelli et al. 2017', '#228833'),
              ('TPS', r's', r't', r'to be completed', pcol),
              ('PS', r's', r'$\Delta \Pi$ (s)', r'Asymptotic period spacing', '#332288'),
              ('d02fit', r'microHz', r'$d_{02,{\rm fit}}$ ($\mu$Hz)', r'Weighted mean small frequency separation', '#D36E70'),
              ('d02mean', r'microHz', r'$d_{02,{\rm mean}}$ ($\mu$Hz)', r'Simple mean small frequency separation', '#D36E70'),
              ('tau0', r's', r'$\tau$ (s)', r'Acoustic radius', pcol),
              ('taubcz', r's', r'$\tau_\mathrm{bcz,\,integration}$ (s)', r'Acoustic depth of the base the convective envelope by integration', pcol),
              ('tauhe', r's', r'$\tau_\mathrm{He,\,integration}$ (s)', r'Acoustic depth of the helium ionization zone by integration', pcol),
              ('dage', r'Myr', r'Age$_\mathrm{weight}$ (Myr)', r'Bayesian age weight', pcol),
              ('dmass', r'solar', r'$M_\mathrm{weight}$', r'Bayesian mass weight', pcol),
              ('phase', None, r'Phase', r'Evolutionary phase: 1) hydrogen or 2) helium burning', pcol),
              ('Mu_JC', r'mag', r'$U$', r'$U$ magnitude in the Johnson/Cousins photometric system', '#D1BBD7'),
              ('Mbx_JC', r'mag', r'$Bx$', r'$Bx$ magnitude in the Johnson/Cousins photometric system', '#AE76A3'),
              ('Mb_JC', r'mag', r'$B$', r'$B$ magnitude in the Johnson/Cousins photometric system', '#882E72'),
              ('Mv_JC', r'mag', r'$V$', r'$V$ magnitude in the Johnson/Cousins photometric system', '#1965B0'),
              ('Mr_JC', r'mag', r'$R$', r'$R$ magnitude in the Johnson/Cousins photometric system', '#5289C7'),
              ('Mi_JC', r'mag', r'$I$', r'$I$ magnitude in the Johnson/Cousins photometric system', '#7BAFDE'),
              ('Mj_JC', r'mag', r'$J$', r'$J$ magnitude in the Johnson/Cousins photometric system', '#4EB265'),
              ('Mh_JC', r'mag', r'$H$', r'$H$ magnitude in the Johnson/Cousins photometric system', '#CAE0AB'),
              ('Mk_JC', r'mag', r'$K$', r'$K$ magnitude in the Johnson/Cousins photometric system', '#F7F056'),
              ('Mlp_JC', r'mag', r'$Lp$', r'$Lp$ magnitude in the Johnson/Cousins photometric system', '#F4A736'),
              ('Ml_JC', r'mag', r'$L$', r'$L$ magnitude in the Johnson/Cousins photometric system', '#E8601C'),
              ('Mm_JC', r'mag', r'$M$', r'$M$ magnitude in the Johnson/Cousins photometric system', '#DC050C'),
              ('Mu_SAGE', r'mag', r'$u$', r'$u$ magnitude in the SAGE photometric system', '#882E72'),
              ('Mv_SAGE', r'mag', r'$v$', r'$v$ magnitude in the SAGE photometric system', '#1965B0'),
              ('Mg_SAGE', r'mag', r'$g$', r'$g$ magnitude in the SAGE photometric system', '#7BAFDE'),
              ('Mr_SAGE', r'mag', r'$r$', r'$r$ magnitude in the SAGE photometric system', '#4EB265'),
              ('Mi_SAGE', r'mag', r'$i$', r'$i$ magnitude in the SAGE photometric system', '#CAE0AB'),
              ('DDO51_SAGE', r'mag', r'DDO51', r'DDO51 magnitude in the SAGE photometric system', '#F7F056'),
              ('Han_SAGE', r'mag', r'H$\alpha_\mathrm{n}$', r'H$\alpha_\mathrm{n}$ magnitude in the SAGE photometric system', '#EE8026'),
              ('Haw_SAGE', r'mag', r'H$\alpha_\mathrm{w}$', r'H$\alpha_\mathrm{w}$ magnitude in the SAGE photometric system', '#DC050C'),
              ('Mj_2MASS', r'mag', r'$J$', r'$J$ magnitude in the 2MASS photometric system', '#1965B0'),
              ('Mh_2MASS', r'mag', r'$H$', r'$H$ magnitude in the 2MASS photometric system', '#F7F056'),
              ('Mk_2MASS', r'mag', r'$K$', r'$K$ magnitude in the 2MASS photometric system', '#DC050C'),
              ('G_GAIA', r'mag', r'$G$', r'$G$ magnitude in the Gaia photometric system', '#1965B0'),
              ('BP_GAIA', r'mag', r'$G_\mathrm{BP}$', r'$G_\mathrm{BP}$ magnitude in the Gaia photometric system', '#F7F056'),
              ('RP_GAIA', r'mag', r'$G_\mathrm{RP}$', r'$G_\mathrm{RP}$ magnitude in the Gaia photometric system', '#DC050C'),
              ('F070W_JWST', r'mag', r'F070W', r'F070W magnitude in the JWST photometric system', '#882E72'),
              ('F090W_JWST', r'mag', r'F090W', r'F090W magnitude in the JWST photometric system', '#1965B0'),
              ('F115W_JWST', r'mag', r'F115W', r'F115W magnitude in the JWST photometric system', '#7BAFDE'),
              ('F150W_JWST', r'mag', r'F150W', r'F150W magnitude in the JWST photometric system', '#4EB265'),
              ('F200W_JWST', r'mag', r'F200W', r'F200W magnitude in the JWST photometric system', '#CAE0AB'),
              ('F277W_JWST', r'mag', r'F277W', r'F277W magnitude in the JWST photometric system', '#F7F056'),
              ('F356W_JWST', r'mag', r'F356W', r'F356W magnitude in the JWST photometric system', '#EE8026'),
              ('F444W_JWST', r'mag', r'F444W', r'F444W magnitude in the JWST photometric system', '#DC050C'),
              ('Mu_SLOAN', r'mag', r'$u\prime$', r'$u\prime$ magnitude in the Sloan photometric system', '#1965B0'),
              ('Mg_SLOAN', r'mag', r'$g\prime$', r'$g\prime$ magnitude in the Sloan photometric system', '#7BAFDE'),
              ('Mr_SLOAN', r'mag', r'$r\prime$', r'$r\prime$ magnitude in the Sloan photometric system', '#4EB265'),
              ('Mi_SLOAN', r'mag', r'$i\prime$', r'$i\prime$ magnitude in the Sloan photometric system', '#F7F056'),
              ('Mz_SLOAN', r'mag', r'$z\prime$', r'$z\prime$ magnitude in the Sloan photometric system', '#DC050C'),
              ('Mu_STROMGREN', r'mag', r'$u$', r'$u$ magnitude in the Stromgren photometric system', '#1965B0'),
              ('Mv_STROMGREN', r'mag', r'$v$', r'$v$ magnitude in the Stromgren photometric system', '#7BAFDE'),
              ('Mb_STROMGREN', r'mag', r'$b$', r'$b$ magnitude in the Stromgren photometric system', '#4EB265'),
              ('My_STROMGREN', r'mag', r'$y$', r'$y$ magnitude in the Stromgren photometric system', '#CAE0AB'),
              ('m1_STROMGREN', r'mag', r'$m_{1}$', r'Index m1 in the Stromgren photometric system', '#F7F056'),
              ('c1_STROMGREN', r'mag', r'$c_{1}$', r'Index c1 in the Stromgren photometric system', '#DC050C'),
              ('Mz_VISTA', r'mag', r'$Z$', r'$Z$ magnitude in the VISTA photometric system', '#1965B0'),
              ('My_VISTA', r'mag', r'$Y$', r'$Y$ magnitude in the VISTA photometric system', '#7BAFDE'),
              ('Mj_VISTA', r'mag', r'$J$', r'$J$ magnitude in the VISTA photometric system', '#4EB265'),
              ('Mh_VISTA', r'mag', r'$H$', r'$H$ magnitude in the VISTA photometric system', '#F7F056'),
              ('Mk_VISTA', r'mag', r'$K$', r'$K$ magnitude in the VISTA photometric system', '#DC050C'),
              ('F160W_WFC2', r'mag', r'F160W', r'F160W in the WFC2 photometric system', '#D1BBD7'),
              ('F170W_WFC2', r'mag', r'F170W', r'F170W in the WFC2 photometric system', '#BA8DB4'),
              ('F185W_WFC2', r'mag', r'F185W', r'F185W in the WFC2 photometric system', '#AA6F9E'),
              ('F218W_WFC2', r'mag', r'F218W', r'F218W in the WFC2 photometric system', '#994F88'),
              ('F255W_WFC2', r'mag', r'F255W', r'F255W in the WFC2 photometric system', '#882E72'),
              ('F300W_WFC2', r'mag', r'F300W', r'F300W in the WFC2 photometric system', '#1965B0'),
              ('F336W_WFC2', r'mag', r'F336W', r'F336W in the WFC2 photometric system', '#5289C7'),
              ('F380W_WFC2', r'mag', r'F380W', r'F380W in the WFC2 photometric system', '#7BAFDE'),
              ('F439W_WFC2', r'mag', r'F439W', r'F439W in the WFC2 photometric system', '#4EB265'),
              ('F450W_WFC2', r'mag', r'F450W', r'F450W in the WFC2 photometric system', '#90C987'),
              ('F555W_WFC2', r'mag', r'F555W', r'F555W in the WFC2 photometric system', '#CAE0AB'),
              ('F606W_WFC2', r'mag', r'F606W', r'F606W in the WFC2 photometric system', '#F7F056'),
              ('F622W_WFC2', r'mag', r'F622W', r'F622W in the WFC2 photometric system', '#F6C141'),
              ('F675W_WFC2', r'mag', r'F675W', r'F675W in the WFC2 photometric system', '#F1932D'),
              ('F702W_WFC2', r'mag', r'F702W', r'F702W in the WFC2 photometric system', '#E8601C'),
              ('F791W_WFC2', r'mag', r'F791W', r'F791W in the WFC2 photometric system', '#DC050C'),
              ('F814W_WFC2', r'mag', r'F814W', r'F814W in the WFC2 photometric system', '#72190E'),
              ('F435W_ACS', r'mag', r'F435W', r'F435W in the ACS photometric system', '#882E72'),
              ('F475W_ACS', r'mag', r'F475W', r'F475W in the ACS photometric system', '#1965B0'),
              ('F555W_ACS', r'mag', r'F555W', r'F555W in the ACS photometric system', '#7BAFDE'),
              ('F606W_ACS', r'mag', r'F606W', r'F606W in the ACS photometric system', '#4EB265'),
              ('F625W_ACS', r'mag', r'F625W', r'F625W in the ACS photometric system', '#CAE0AB'),
              ('F775W_ACS', r'mag', r'F775W', r'F775W in the ACS photometric system', '#F7F056'),
              ('F814W_ACS', r'mag', r'F814W', r'F814W in the ACS photometric system', '#DC050C'),
              ('F218W_WFC3', r'mag', r'F218W', r'F218W in the WFC3 UVIS/IR photometric system', '#D1BBD7'),
              ('F225W_WFC3', r'mag', r'F225W', r'F225W in the WFC3 UVIS/IR photometric system', '#BA8DB4'),
              ('F275W_WFC3', r'mag', r'F275W', r'F275W in the WFC3 UVIS/IR photometric system', '#AA6F9E'),
              ('F336W_WFC3', r'mag', r'F336W', r'F336W in the WFC3 UVIS/IR photometric system', '#994F88'),
              ('F390W_WFC3', r'mag', r'F390W', r'F390W in the WFC3 UVIS/IR photometric system', '#882E72'),
              ('F438W_WFC3', r'mag', r'F438W', r'F438W in the WFC3 UVIS/IR photometric system', '#1965B0'),
              ('F475W_WFC3', r'mag', r'F475W', r'F475W in the WFC3 UVIS/IR photometric system', '#5289C7'),
              ('F555W_WFC3', r'mag', r'F555W', r'F555W in the WFC3 UVIS/IR photometric system', '#7BAFDE'),
              ('F606W_WFC3', r'mag', r'F606W', r'F606W in the WFC3 UVIS/IR photometric system', '#4EB265'),
              ('F625W_WFC3', r'mag', r'F625W', r'F625W in the WFC3 UVIS/IR photometric system', '#90C987'),
              ('F775W_WFC3', r'mag', r'F775W', r'F775W in the WFC3 UVIS/IR photometric system', '#CAE0AB'),
              ('F814W_WFC3', r'mag', r'F814W', r'F814W in the WFC3 UVIS/IR photometric system', '#F7F056'),
              ('F105W_WFC3', r'mag', r'F105W', r'F105W in the WFC3 UVIS/IR photometric system', '#F6C141'),
              ('F110W_WFC3', r'mag', r'F110W', r'F110W in the WFC3 UVIS/IR photometric system', '#F1932D'),
              ('F125W_WFC3', r'mag', r'F125W', r'F125W in the WFC3 UVIS/IR photometric system', '#E8601C'),
              ('F140W_WFC3', r'mag', r'F140W', r'F140W in the WFC3 UVIS/IR photometric system', '#DC050C'),
              ('F160W_WFC3', r'mag', r'F160W', r'F160W in the WFC3 UVIS/IR photometric system', '#72190E'),
              ('Mu_DECAM', r'mag', r'$u$', r'$u$ in the DECAM photometric system', '#1965B0'),
              ('Mg_DECAM', r'mag', r'$g$', r'$g$ in the DECAM photometric system', '#7BAFDE'),
              ('Mr_DECAM', r'mag', r'$r$', r'$r$ in the DECAM photometric system', '#4EB265'),
              ('Mi_DECAM', r'mag', r'$i$', r'$i$ in the DECAM photometric system', '#CAE0AB'),
              ('Mz_DECAM', r'mag', r'$z$', r'$z$ in the DECAM photometric system', '#F7F056'),
              ('My_DECAM', r'mag', r'$y$', r'$y$ in the DECAM photometric system', '#DC050C'),
              ('Mu_SKYMAPPER', r'mag', r'$u$', r'$u$ in the SkyMapper photometric system', '#882E72'),
              ('Mv_SKYMAPPER', r'mag', r'$v$', r'$v$ in the SkyMapper photometric system', '#1965B0'),
              ('Mg_SKYMAPPER', r'mag', r'$g$', r'$g$ in the SkyMapper photometric system', '#7BAFDE'),
              ('Mr_SKYMAPPER', r'mag', r'$r$', r'$r$ in the SkyMapper photometric system', '#4EB265'),
              ('Mi_SKYMAPPER', r'mag', r'$i$', r'$i$ in the SkyMapper photometric system', '#CAE0AB'),
              ('Mz_SKYMAPPER', r'mag', r'$z$', r'$z$ in the SkyMapper photometric system', '#F7F056'),
              ('Mule_SKYMAPPER', r'mag', r'$u_\mathrm{le}$', r'$u_\mathrm{le}$ in the SkyMapper photometric system', '#DC050C'),
              ('Mkp_KEPLER', r'mag', r'$K_{p}$', r'Magnitude in the Kepler photometric system', '#1965B0'),
              ('Mhp_TYCHO', r'mag', r'$H_{p}$', r'Hipparcos magnitude in the Tycho photometric system', '#1965B0'),
              ('Mb_TYCHO', r'mag', r'$B_{t}$', r'$B$ magnitude in the Tycho photometric system', '#F7F056'),
              ('Mv_TYCHO', r'mag', r'$V_{t}$', r'$V$ magnitude in the Tycho photometric system', '#DC050C'),
              ('Mt_TESS', r'mag', r'$T_{\mathrm{mag}}$', r'Magnitude in the TESS photometric system', '#1965B0'),
              ('distance', r'pc', r'$d$ (pc)', r'Stellar distance', pcol),
              ('dif', None, r'Diffusion', r'Atomic diffusion: 0) no and 1) yes', pcol)
              ]
    # fmt: on

    names = [i[0] for i in params]

    def exclude_params(excludeparams):
        """
        Takes a list of input parameters (or a
        single parameter) as strings and returns
        the entire params list, except for the
        params given as input.
        """
        classParams = parameters.params
        parnames = [x for x, y, z, v, c in classParams]

        if type(excludeparams) is not list:
            excludeparams = [excludeparams]

        for par in excludeparams:
            if type(par) is not str:
                print("Parameters should be strings!")
                exit()

            if par in parnames:
                parnames.remove(par)
            else:
                print(f"Parameter {par} is not in params!")
                exit()

        return parnames

    def get_keys(inputparams):
        """
        Takes a list of input parameters (or a
        single parameter) as strings and returns
        the correspding units, names shown on a
        plot and remarks for the params.
        """
        paramsunits = []
        paramsplots = []
        paramsremarks = []
        paramscolors = []
        classParams = parameters.params

        if type(inputparams) is not list:
            inputparams = list(inputparams)

        for par in inputparams:
            entry = [i for i in classParams if i[0] == par]
            paramsunits.append(entry[0][1])
            paramsplots.append(entry[0][2])
            paramsremarks.append(entry[0][3])
            paramscolors.append(entry[0][4])

        return paramsunits, paramsplots, paramsremarks, paramscolors


@dataclass
class extinction:
    """
    Reddening law coefficients of the form Az = Rz*E(B-V).
    the coefficients are from Table 6 of Schlafly & Finkbeiner (2011)
    where available. The entries are in a polynomial format for Rz defined as:
    Rz = a0 + T4*(a1 + a2*T4) + a3*FeH with T4 = 1e-4*Teff.
    They are kept like this for backward compatibility reasons with
    Casagrande & VandenBerg (2014).

    Coefficients were extracted from the following references:
    G19: Green et al. 2019
    SF11: Schlafly & Finkbeiner 2011
    SD18: Sanders & Das 2018
    CV14: Casagrande & Vandenberg 2014
    CV18: Casagrande & Vandenberg 2018
    Y13: Yuan et al. 2013

    We aim for homogeneity and prioritise those of SF11, and for systems not
    available in that compilation we use SD18 and CV14/18.
    """

    # The Green extinction map returns E(g-r), which is transformed to E(B-V)
    # using the following coefficient
    Conv_Bayestar = 0.884

    R = np.array(
        [
            # Johnson/Cousins photometric system (CV14)
            ("Mu_JC", 4.814, 4.3241, 1.6005, -1.3063, -0.0073),
            ("Mbx_JC", 4.032, 3.2999, 2.0123, -1.3425, -0.0140),
            ("Mb_JC", 4.049, 3.3155, 2.0119, -1.3400, -0.0145),
            ("Mv_JC", 3.129, 2.9256, 0.5205, -0.3078, -0.0022),
            ("Mr_JC", 2.558, 2.4203, 0.3009, -0.1220, 0),
            ("Mi_JC", 1.885, 1.8459, 0.0741, -0.0151, 0),
            ("Mj_JC", 0, 0, 0, 0, 0),
            ("Mh_JC", 0, 0, 0, 0, 0),
            ("Mk_JC", 0, 0, 0, 0, 0),
            ("Mlp_JC", 0, 0, 0, 0, 0),
            ("Ml_JC", 0, 0, 0, 0, 0),
            ("Mm_JC", 0, 0, 0, 0, 0),
            # SAGE photometric system
            ("Mu_SAGE", 0, 0, 0, 0, 0),
            ("Mv_SAGE", 0, 0, 0, 0, 0),
            ("Mg_SAGE", 0, 0, 0, 0, 0),
            ("Mr_SAGE", 0, 0, 0, 0, 0),
            ("Mi_SAGE", 0, 0, 0, 0, 0),
            ("DDO51_SAGE", 0, 0, 0, 0, 0),
            ("Han_SAGE", 0, 0, 0, 0, 0),
            ("Haw_SAGE", 0, 0, 0, 0, 0),
            # 2MASS photometric system. The provided coefficient relates E(g-r) and Az.
            # To relate to E(B-V), it needs to be multiplied by E(g-r)/E(B-v) = 1/Conv_Bayestar
            ("Mj_2MASS", 0.7927 / Conv_Bayestar, 0.7927 / Conv_Bayestar, 0, 0, 0),
            ("Mh_2MASS", 0.4690 / Conv_Bayestar, 0.4690 / Conv_Bayestar, 0, 0, 0),
            ("Mk_2MASS", 0.3026 / Conv_Bayestar, 0.3026 / Conv_Bayestar, 0, 0, 0),
            # Gaia photometric system eDR3, following the description of CV18 and using Fitzpatrick renormalized as
            # per Schlafly (they should be consistent with Schlafy & Finkbeiner 2011)
            ("G_GAIA", 2.312, 1.132, 2.700, -1.271, -0.010),
            ("BP_GAIA", 2.884, 1.684, 3.098, -1.879, -0.020),
            ("RP_GAIA", 1.633, 1.471, 0.369, -0.167, 0.002),
            # Gaia photometric system DR2 (SD18)
            # ("BP_GAIA", 3.046, 3.046, 0, 0, 0),
            # ("G_GAIA", 2.294, 2.294, 0, 0, 0),
            # ("RP_GAIA", 1.737, 1.737, 0, 0, 0),
            # ("RVS_GAIA", 1.393, 1.393, 0, 0, 0),
            # # Gaia photometric system DR2 (CV18)
            # ('G_GAIA', 2.740, 1.4013, 3.1406, -1.5626, -0.0101),
            # ('BP_GAIA', 3.374, 1.7895, 4.2355, -2.7071, -0.0253),
            # ('RP_GAIA', 2.035, 1.8593, 0.3985, -0.1771, 0.0026),
            # JWST-NIRCam photometric system (CV18)
            ("F070W_JWST", 2.314, 2.2385, 0.1738, -0.0803, 0.0010),
            ("F090W_JWST", 1.514, 1.4447, 0.1833, -0.1125, 0),
            ("F115W_JWST", 1.011, 0.9910, 0.0313, 0.0018, 0),
            ("F150W_JWST", 0.663, 0.6425, 0.0454, -0.0189, 0.0006),
            ("F200W_JWST", 0.425, 0.4159, 0.0261, -0.0195, 0),
            ("F277W_JWST", 0.253, 0.2554, -0.0086, 0.0085, 0),
            ("F356W_JWST", 0.166, 0.1699, -0.0102, 0.0075, 0),
            ("F444W_JWST", 0.119, 0.1270, -0.0246, 0.0200, 0),
            # SDSS photometric system (SF11)
            ("Mu_SLOAN", 4.239, 4.239, 0, 0, 0),
            ("Mg_SLOAN", 3.303, 3.303, 0, 0, 0),
            ("Mr_SLOAN", 2.285, 2.285, 0, 0, 0),
            ("Mi_SLOAN", 1.698, 1.698, 0, 0, 0),
            ("Mz_SLOAN", 1.263, 1.263, 0, 0, 0),
            # Strömgren photometric system (SF11)
            ("Mu_STROMGREN", 4.305, 4.305, 0, 0, 0),
            ("Mb_STROMGREN", 3.350, 3.350, 0, 0, 0),
            ("Mv_STROMGREN", 3.793, 3.793, 0, 0, 0),
            ("My_STROMGREN", 2.686, 2.686, 0, 0, 0),
            ("m1_STROMGREN", 0, 0, 0, 0, 0),
            ("c1_STROMGREN", 0, 0, 0, 0, 0),
            # VISTA photometric system
            ("Mz_VISTA", 0, 0, 0, 0, 0),
            ("My_VISTA", 0, 0, 0, 0, 0),
            ("Mj_VISTA", 0, 0, 0, 0, 0),
            ("Mh_VISTA", 0, 0, 0, 0, 0),
            ("Mk_VISTA", 0, 0, 0, 0, 0),
            # HST-WFC2 photometric system (SF11)
            ("F160W_WFC2", 0, 0, 0, 0, 0),
            ("F170W_WFC2", 0, 0, 0, 0, 0),
            ("F185W_WFC2", 0, 0, 0, 0, 0),
            ("F218W_WFC2", 0, 0, 0, 0, 0),
            ("F255W_WFC2", 0, 0, 0, 0, 0),
            ("F300W_WFC2", 4.902, 4.902, 0, 0, 0),
            ("F336W_WFC2", 0, 0, 0, 0, 0),
            ("F380W_WFC2", 0, 0, 0, 0, 0),
            ("F439W_WFC2", 0, 0, 0, 0, 0),
            ("F450W_WFC2", 3.410, 3.410, 0, 0, 0),
            ("F555W_WFC2", 2.755, 2.755, 0, 0, 0),
            ("F606W_WFC2", 2.415, 2.415, 0, 0, 0),
            ("F622W_WFC2", 0, 0, 0, 0, 0),
            ("F675W_WFC2", 0, 0, 0, 0, 0),
            ("F702W_WFC2", 1.948, 1.948, 0, 0, 0),
            ("F791W_WFC2", 0, 0, 0, 0, 0),
            ("F814W_WFC2", 1.549, 1.549, 0, 0, 0),
            # HST-ACS photometric system (SF11)
            ("F435W_ACS", 3.610, 3.610, 0, 0, 0),
            ("F475W_ACS", 3.268, 3.268, 0, 0, 0),
            ("F555W_ACS", 2.792, 2.792, 0, 0, 0),
            ("F606W_ACS", 2.471, 2.471, 0, 0, 0),
            ("F625W_ACS", 2.219, 2.219, 0, 0, 0),
            ("F775W_ACS", 1.629, 1.629, 0, 0, 0),
            ("F814W_ACS", 1.526, 1.526, 0, 0, 0),
            # HST-WFC3 photometric system (SF11)
            ("F105W_WFC3", 0.969, 0.969, 0, 0, 0),
            ("F110W_WFC3", 0.881, 0.881, 0, 0, 0),
            ("F125W_WFC3", 0.726, 0.726, 0, 0, 0),
            ("F140W_WFC3", 0.613, 0.613, 0, 0, 0),
            ("F160W_WFC3", 0.512, 0.512, 0, 0, 0),
            ("F218W_WFC3", 7.760, 7.760, 0, 0, 0),
            ("F225W_WFC3", 6.989, 6.989, 0, 0, 0),
            ("F275W_WFC3", 5.487, 5.487, 0, 0, 0),
            ("F336W_WFC3", 4.453, 4.453, 0, 0, 0),
            ("F390W_WFC3", 3.896, 3.896, 0, 0, 0),
            ("F438W_WFC3", 3.623, 3.623, 0, 0, 0),
            ("F475W_WFC3", 3.248, 3.248, 0, 0, 0),
            ("F555W_WFC3", 2.855, 2.855, 0, 0, 0),
            ("F606W_WFC3", 2.488, 2.488, 0, 0, 0),
            ("F625W_WFC3", 2.259, 2.259, 0, 0, 0),
            ("F775W_WFC3", 1.643, 1.643, 0, 0, 0),
            ("F814W_WFC3", 1.536, 1.536, 0, 0, 0),
            # DECam photometric system (SF11)
            ("Mu_DECAM", 0, 0, 0, 0, 0),
            ("Mg_DECAM", 3.237, 3.237, 0, 0, 0),
            ("Mr_DECAM", 2.176, 2.176, 0, 0, 0),
            ("Mi_DECAM", 1.595, 1.595, 0, 0, 0),
            ("Mz_DECAM", 1.217, 1.217, 0, 0, 0),
            ("My_DECAM", 1.058, 1.058, 0, 0, 0),
            # Skymapper photometric system (CV18)
            ("Mu_SKYMAPPER", 4.900, 3.3743, 4.5098, -3.2967, -0.0193),
            ("Mv_SKYMAPPER", 4.550, 4.3395, 0.7243, -0.6196, -0.0028),
            ("Mg_SKYMAPPER", 3.446, 2.9349, 1.2782, -0.7275, -0.0054),
            ("Mr_SKYMAPPER", 2.734, 2.6011, 0.2952, -0.1284, 0),
            ("Mi_SKYMAPPER", 1.995, 1.9686, 0.0394, 0.0069, 0),
            ("Mz_SKYMAPPER", 1.468, 1.3831, 0.2551, -0.1886, 0),
            ("Mule_SKYMAPPER", 0, 0, 0, 0, 0),
            # Kepler band
            ("Mkp_KEPLER", 0, 0, 0, 0, 0),
            # TESS band
            ("Mt_TESS", 0, 0, 0, 0, 0),
            # Tycho photometric system (CV18)
            ("Mhp_TYCHO", 3.239, 2.0611, 2.9605, -1.6990, -0.0133),
            ("Mb_TYCHO", 4.222, 3.6609, 1.6185, -1.1570, -0.0126),
            ("Mv_TYCHO", 3.272, 3.0417, 0.5745, -0.3231, -0.0015),
            # WISE photometric system (Y13)
            ("Mw1_WISE", 0.19, 0.19, 0, 0, 0),
            ("Mw2_WISE", 0.15, 0.15, 0, 0, 0),
        ],
        dtype=[
            ("Filter", np.str_, 16),
            ("RZ_mean", float),
            ("a0", float),
            ("a1", float),
            ("a2", float),
            ("a3", float),
        ],
    )


@dataclass
class photsys:
    """
    Available photometric systems and mapping to internal names
    """

    # Mapping to IDs expected by the Fortran code
    # --> v0.25: GAIA (id 4) replaced by the updated GAIA DR2 (id 15)
    # --> v0.29: GAIA DR2 (id 15) replaced by the updated GAIA DR3 (id 18)
    available = {
        "jc": 1,
        "sage": 2,
        "2mass": 3,
        "jwst": 5,
        "sloan": 6,
        "uvby": 7,
        "vista": 8,
        "wfpc2": 9,
        "acs": 10,
        "wfc3": 11,
        "decam": 12,
        "skymap": 13,
        "kepler": 14,
        "tycho": 16,
        "tess": 17,
        "gaia": 18,
    }

    # Remap old names and synonyms
    synonyms = {
        "ubvri": "jc",
        "stromgren": "uvby",
        "wfc3-uvis": "wfc3",
        "sdss": "sloan",
    }

    # Mapping between user-friendly and internal names of photometric systems
    rename = {
        "jc": "JC",
        "sage": "SAGE",
        "2mass": "2MASS",
        "gaia": "GAIA",
        "jwst": "JWST",
        "sloan": "SLOAN",
        "uvby": "STROMGREN",
        "vista": "VISTA",
        "wfpc2": "WFC2",
        "acs": "ACS",
        "wfc3": "WFC3",
        "decam": "DECAM",
        "skymap": "SKYMAPPER",
        "kepler": "KEPLER",
        "tycho": "TYCHO",
        "tess": "TESS",
    }

    # List of default filters
    default = ["2mass", "jc"]


@dataclass
class distanceranges:
    """
    Limits or ranges of different surveys
    """

    # 2MASS.max: https://old.ipac.caltech.edu/2mass/releases/sampler/index.html
    # 2MASS.min: Brightest star in 2mass All-Sky Release PSC is Betelgeuse,
    # https://old.ipac.caltech.edu/2mass/releases/allsky/doc/sec1_6b.html#satr1
    # TODO!
    filters = {
        "Mj_2MASS": {"max": 16.5, "min": -2.99},
        "Mh_2MASS": {"max": 16.0, "min": -4.01},
        "Mk_2MASS": {"max": 15.5, "min": -4.38},
    }


@dataclass
class metallicityranges:
    """
    Limits in metallictity for colors
    """

    values = {
        "metallicity": {"max": 0.50, "min": -4.0},
    }
