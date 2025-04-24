"""
The BASTA core module contains Python dataclasses that define internal
configurations, shared data models, and structures used for function inputs
and outputs within the BASTA framework.

The classes defined here are not necessarily user-facing, but serve as
foundational components for the application's logic, inference processes,
and internal communication.
"""

from collections.abc import Callable
from dataclasses import dataclass, field
from io import BufferedIOBase, TextIOBase
from pathlib import Path
from typing import Any, TypedDict, Literal, Optional


Fitparam = tuple[float, float]


@dataclass(kw_only=True)
class ClassicalParameters:
    """
    Container for classical stellar parameters used in the BASTA inference.

    These are typically fundamental observational values such as effective temperature
    or metallicity, each associated with a value and an uncertainty.

    Parameters
    ----------
    params : dict of str to tuple of float
        Dictionary mapping parameter names (e.g., 'teff', 'feh') to their
        value and uncertainty as a (value, error) tuple.
    """
    params: dict[str, Fitparam]


@dataclass
class ScaledValueError:
    """
    Representation of a parameter value and its error, along with a scaling factor.

    This class is useful for applying transformations (e.g., unit conversions)
    while preserving the original value and error.

    Parameters
    ----------
    original : tuple of float
        Original (value, error) before scaling.
    scale : float
        Multiplicative factor to scale the original value and error.
    
    Attributes
    ----------
    scaled : tuple of float
        Returns the scaled (value, error).
    """
    original: tuple[float, float]
    scale: float

    @property
    def scaled(self) -> tuple[float, float]:
        return (self.original[0] * self.scale, self.original[1] * self.scale)


@dataclass(kw_only=True)
class GlobalSeismicParameters:
    """
    Container for global asteroseismic parameters and their scale transformations.

    This class manages both original and scaled versions of global parameters like
    `numax` and `dnu`, enabling on-the-fly conversion for inference or model comparison.
    
    Example use:
    ```
    numax = globalseismic.get_original("numax")
    scaled_numax = globalseismic.get_scaled("numax")
    ```

    Parameters
    ----------
    params : dict of str to ScaledValueError
        Dictionary mapping seismic parameter names to their original values and scaling.
    scalefactors : dict of str to float
        Scaling factors for each seismic parameter (applied during set_scaled()).

    Attributes
    ----------
    scaled_params : dict of str to ScaledValueError, optional
        Stores scaled parameter values once `set_scaled()` has been called.
    
    Methods
    -------
    set_scalefactor(scalefactors)
        Sets the scaling factors to be used.
    set_scaled()
        Applies the scaling factors and populates `scaled_params`.
    get_scaled(key)
        Retrieves the scaled (value, error) for the given parameter key.
    get_original(key)
        Retrieves the original (value, error) for the given parameter key.
    """
    params: dict[str, ScaledValueError]
    scalefactors: Optional[dict[str, float]] = None
    scaled_params: Optional[dict[str, ScaledValueError]] = field(
        default=None, init=False
    )

    def set_scalefactor(self, scalefactors: dict[str, float]) -> None:
        if self.scalefactors is None:
            self.scalefactors = {}
        self.scalefactors = scalefactors

    def set_scaled(self) -> None:
        self.scaled_params = {
            key: ScaledValueError(
                original=param.original,
                scale=param.scale * (
                    self.scalefactors[key] if self.scalefactors and key in self.scalefactors else 1.0
                ),
            )
            for key, param in self.params.items()
        }

    def get_scaled(self, key: str) -> tuple[float, float]:
        if self.scaled_params is None:
            raise ValueError("Scaled fitparams have not been set.")
        return self.scaled_params[key].scaled

    def get_original(self, key: str) -> tuple[float, float]:
        return self.params[key].original


@dataclass(kw_only=True, frozen=True)
class IndividualFrequencies:
    # TODO(Amalie) clean and add context
    # TODO(Amalie) get_frequencies: dict = {freqpath:, freqfile}
    freqpath: str
    freqfile: str
    # TODO(Amalie) surfacecorrection: use typeddicts per surfacecorrection for numax/bexp
    surfacecorrection: str  # fcor
    bexp: None | float = None
    # TODO(Amalie) Rewrite so these are only read from fitfreqs/GlobalSeismic instead of duplicated
    # dnufit: float
    # dnufit_err: float
    # numax: float
    correlations: bool | int = False
    seismicweights: dict[str, Any]
    # TODO(Amalie)
    # remove_frequencies: dict = {nottrustedfile: nottrustedfile, excludemodes, onlyradial, onlyls}
    nottrustedfile: str | None = None
    excludemodes: bool | None = None
    onlyradial: bool | None = None


@dataclass(kw_only=True, frozen=True)
class Ratios:
    fittypes: list[Literal["r01", "r010", "r012", "r02", "r10", "r102"]]

    readratios: bool | int = False
    threepoint: bool | int = False
    interp_ratios: bool | int = True
    dnufit_in_ratios: bool | int = False


@dataclass(kw_only=True, frozen=True)
class Glitches:
    fittypes: list[Literal["gr01", "gr010", "gr012", "gr02", "gr10", "gr102"]]
    glitchfit: bool = False
    glitchfile: str | None = None
    nrealizations: int = 10000


@dataclass(kw_only=True, frozen=True)
class EpsilonDifferences:
    fittypes: list[Literal["e01", "e012", "e02"]]
    nsorting: bool | int = True


@dataclass(kw_only=True, frozen=True)
class SeismicParameters:
    frequencies: IndividualFrequencies | None = None
    ratios: Ratios | None = None
    glitches: Glitches | None = None
    epsilondifferences: EpsilonDifferences | None = None

    @property
    def has_frequencies(self) -> bool:
        return self.frequencies is not None

    @property
    def has_ratios(self) -> bool:
        return self.ratios is not None

    @property
    def has_glitches(self) -> bool:
        return self.glitches is not None

    @property
    def has_epsilondifferences(self) -> bool:
        return self.epsilondifferences is not None

    @property
    def has_any_case(self) -> bool:
        return any(
            (
                self.has_frequencies,
                self.has_ratios,
                self.has_glitches,
                self.has_epsilondifferences,
            )
        )


class AbsoluteMagnitude(TypedDict):
    """
    Dictionary specifying the prior and uncertainties for an absolute magnitude in a given filter.

    Attributes
    ----------
    prior : Callable[[float], float]
        A prior function that takes a magnitude value and returns a log-probability or weight.
    median : float
        The median absolute magnitude value.
    errp : float
        The positive uncertainty on the absolute magnitude.
    errm : float
        The negative uncertainty on the absolute magnitude.
    """
    prior: Callable[[float], float]
    median: float
    errp: float
    errm: float


class AbsoluteMagnitudes(TypedDict):
    """
    Container for absolute magnitudes and extinction-related prior information.

    Attributes
    ----------
    magnitudes : dict of str to AbsoluteMagnitude
        Dictionary mapping filter names (see `basta.constants`) to absolute magnitude info.
    absorption : dict of str to list
        Dictionary containing absorption in different bands.
    prior_EBV : list of float
        Prior information on reddening (E(B-V)).
    prior_distance : list of float
        Prior information on distance (in parsecs).
    """
    magnitudes: dict[str, AbsoluteMagnitude]
    absorption: dict[str, list[Any]]
    prior_EBV: list[float]
    prior_distance: list[float]


@dataclass(kw_only=True, frozen=True)
class DistanceParameters:
    """
    Contains distance-related observational data for a star.

    This includes apparent magnitudes, sky coordinates, parallax measurements, and extinction.

    Parameters
    ----------
    magnitudes : dict of str to tuple of float
        Dictionary mapping filter names (see `basta.constants`) to (value, uncertainty) tuples.
    coordinates : dict of str to Any
        Coordinate data such as RA/Dec or galactic coordinates. Format may vary.
    parallax : list of float
        List containing parallax value and (optionally) uncertainty, in milliarcseconds.
    EBV : list of Any
        Reddening values or prior information about extinction, format may vary.
    """
    params: dict[str, Fitparam]
    magnitudes: dict[str, tuple[float, float]]
    coordinates: dict[str, Any]
    EBV: list[Any]


@dataclass(kw_only=True)
class Star:
    """
    Main container for all relevant observational and input data for a single star.

    This class bundles classical parameters, global and detailed seismic parameters,
    and distance-related data for use in a BASTA inference run.

    Parameters
    ----------
    starid : str
        Unique identifier for the star.
    classicalparams : ClassicalParameters
        Classical observational parameters such as effective temperature or metallicity.
    globalseismicparams : GlobalSeismicParameters
        Global seismic observables like `numax` and `dnu`.
    seismicparams : SeismicParameters
        Detailed seismic diagnostics including frequencies, ratios, glitches, and more.
    distanceparams : DistanceParameters
        Information related to the star's distance, magnitudes, and extinction.
    """

    starid: str
    # Allowed parameter keys can be found `basta.constants` in `parameters.params`
    classicalparams: ClassicalParameters
    globalseismicparams: GlobalSeismicParameters
    seismicparams: SeismicParameters
    distanceparams: DistanceParameters


@dataclass(kw_only=True, frozen=True)
class RunFiles:
    """
    Stores references to files that collect information on the entire BASTA run.

    Parameters
    ----------
    runbasepath : str
        Base path used to identify and group output files.
    summarytable : BufferedIOBase
        File-like object used for writing the summary table for all stars.
    summarytablepath : str
        File path for the main summary table (typically in ASCII format).
    distancesummarytable : BufferedIOBase or None, optional
        File-like object for distance-related summary output.
    distancesummarytablepath : str
        File path for the distance summary table.
    warnoutput : TextIOBase or None, optional
        File stream for warnings or logging non-critical issues.
    erroroutput : TextIOBase or None, optional
        File stream for logging critical errors.
    """
    runbasepath: str
    summarytable: BufferedIOBase
    summarytablepath: str
    distancesummarytable: BufferedIOBase | None = None
    distancesummarytablepath: str
    warnoutput: TextIOBase | None = None
    erroroutput: TextIOBase | None = None


@dataclass(kw_only=True, frozen=True)
class FilePaths:
    """
    Manages all file paths related to a single target in a BASTA run.
    This generates structured paths for output files, plots and logs.

    Parameters
    ----------
    starid : str
        Unique identifier for the target star.
    outputdir : str
        Directory where output files will be saved.
    inputfile : str
        Path to the original input file used for this star.
    plotfmt : str
        Format string for plot files (e.g., 'png', 'pdf').

    Attributes
    ----------
    base : Path
        Base path (outputdir/starid) used to build all other paths.
    extradirectory : Path
        Directory used to store debug files.
    logfile : Path
        Path to the main log file for this star.
    jsonfile : Path
        Path to the saved results in JSON format.
    plotfile_template : str
        Template string for generating filenames for different types of plots.

    Methods
    -------
    plotfile(kind)
        Returns the full path to a specific type of plot file.
    save_plot(fig, kind, **kwargs)
        Saves a matplotlib figure using the appropriate plot path.
    """

    starid: str
    outputdir: str
    inputfile: str
    plotfmt: str

    def __post_init__(self):
        Path(self.outputdir).mkdir(parents=True, exist_ok=True)

    @property
    def base(self) -> Path:
        return Path(self.outputdir) / self.starid

    @property
    def extradirectory(self) -> Path:
        path = Path(self.outputdir) / "debug"
        path.mkdir(parents=True, exist_ok=True)
        return path

    @property
    def logfile(self) -> Path:
        return self.base.with_suffix(".log")

    @property
    def jsonfile(self) -> Path:
        return self.base.with_suffix(".json")

    @property
    def plotfile_template(self) -> str:
        return str(self.base) + "_{0}." + self.plotfmt

    def plotfile(self, kind: str) -> Path:
        return Path(self.plotfile_template.format(kind))

    def save_plot(self, fig, kind: str, **kwargs) -> Path:
        """
        Saves a matplotlib figure to the appropriate plot path.

        Parameters
        ----------
        fig : matplotlib.figure.Figure
            The figure to save.
        kind : str
            Identifier to insert into the filename (e.g., 'corner', 'pdf').
        **kwargs : dict
            Additional arguments passed to `fig.savefig()`.

        Returns
        -------
        Path
            The full path the figure was saved to.
        """
        path = self.plotfile(kind)
        fig.savefig(path, **kwargs)
        print(f"Saved plot to {path}")
        return path

@dataclass(kw_only=True)
class Priors:


@dataclass(kw_only=True, frozen=True)
class InferenceSettings:
    """
    Contains configuration settings used to control a BASTA inference run.

    Parameters
    ----------
    gridfile : str
        Path to the stellar model grid file.
    seed : int
        Random seed used for reproducibility.
    limits : dict of str to Any
        Parameter limits or constraints used during the inference.
    solarvalues : dict of str to float
        Solar reference values (e.g., solar numax, dnu).
    solarmodel : str, optional
        Identifier or path to the model considered as the solar reference.
    gridid : tuple of float, optional
        Version or metadata identifier for the model grid.
    usebayw : bool, optional
        Whether to use Bayesian weights during inference.
        Can also be a tuple for custom weighting.
    priors : tuple or list of str, optional
        Priors to apply during the inference (e.g. an IMF, metallicity priors).
    """

    gridfile: str
    seed: int
    limits: dict[str, Any]

    solarvalues: dict[
        str, float
    ]  #  = {"numax": constants.sydsun.SUNnumax, "dnu": constants.sydsun.SUNdnu}
    solarmodel: str = ""
    gridid: tuple[float, float, float, float] | None = None

    # TODO(Amalie) Consider removing entirely
    dnuprior: bool | int = True
    dnubias: float = 0.0
    # TODO(Amalie) dnufrac should be in 'priors'
    dnufrac: float = 0.15

    usebayw: bool = True

    priors: tuple[str, ...] | list[str] | None = None


@dataclass(kw_only=True, frozen=True)
class OutputOptions:
    """
    Configuration for optional output and logging behavior in BASTA.

    Parameters
    ----------
    asciiparams : list of str
        List of parameters to write in ASCII summary files.
    uncert : str
        Method for uncertainty estimation
    centroid : str
        Method used for computing the PDF centroid.
    optionaloutputs : bool, optional
        If True, save additional output like per-star JSON files and full PDFs.
    debug : bool, optional
        Enables debug mode with extra logging and intermediate files.
    verbose : bool, optional
        Enables detailed console output during execution.
    developermode : bool, optional
        Enables experimental or in-development features.
    validationmode : bool, optional
        Enables strict checks or modes for validation/testing purposes.
    """

    asciiparams: list[str]  # dict[str, Any]
    uncert: str
    centroid: str

    optionaloutputs: bool = False
    debug: bool = False
    verbose: bool = False
    developermode: bool = False
    validationmode: bool = False


@dataclass(kw_only=True, frozen=True)
class PlotConfig:
    """
    Configuration for plots outputted by BASTA but not the inference in itself.

    Parameters
    ----------

    """

    nameinplot: str
    kielplots: list
    cornerplots: list
    freqplots: list
