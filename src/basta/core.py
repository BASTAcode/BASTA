"""
The BASTA core module contains Python dataclasses that define internal configurations, shared data models, and structures used for function inputs and outputs within the BASTA framework.

The classes defined here are not necessarily user-facing, but serve as foundational
components for the application's logic, inference processes, and internal communication.

    - Star: Star-specific information
    - InferenceOptions: Settings used for the inference in a given BASTA run.
    - OptionalFlags: Settings affecting only the output given by BASTA (e.g. plots or additions to the logfile) but not how the inference in a given run will be performed.

"""

from collections.abc import Callable
from dataclasses import dataclass
from io import BufferedIOBase, TextIOBase
from pathlib import Path
from typing import Any, TypedDict, Literal

Fitparam = tuple[float, float]


@dataclass
class ScaledValueError:
    original: tuple[float, float]
    scale: float

    @property
    def scaled(self) -> tuple[float, float]:
        return (self.original[0] * self.scale, self.original[1] * self.scale)


@dataclass(kw_only=True)
class ClassicalParameters:
    params: dict[str, Fitparam]


@dataclass(kw_only=True)
class GlobalSeismicParameters:
    params: dict[str, ScaledValueError]


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
    # TODO(Amalie) dnufrac should be in 'priors'
    dnufrac: float = 0.15
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

    # TODO(Amalie) Consider removing entirely
    dnuprior: bool | int = True
    dnubias: float = 0.0

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
    prior: Callable[[float], float]
    median: float
    errp: float
    errm: float


class AbsoluteMagnitudes(TypedDict):
    magnitudes: dict[str, AbsoluteMagnitude]
    absorption: dict[str, list[Any]]
    prior_EBV: list[float]
    prior_distance: list[float]


@dataclass(kw_only=True, frozen=True)
class DistanceParameters:
    # could just be keys in obs that are not 'parallax' or 'EBV
    # filters: List[str]
    # m: Dict[str, float]
    # m_err: Dict[str, float]
    # Can be combined in dict called magnitudes
    magnitudes: dict[str, tuple[float, float]]
    coordinates: dict[str, Any]
    # TODO(Amalie) why is parallax here? should it be in star.fitparams?
    parallax: list[float]
    EBV: list[Any]


@dataclass(kw_only=True)
class Star:
    """
    Main class containing star-specific information.

    Parameters
    ----------
    starid : str
        Unique identifier for this target.
    inputparams : dict
        Dictionary containing most information needed, e.g. controls, fitparameters,
        output options.
    """

    starid: str
    # inputparams: dict[str, Any]
    # Allowed keys of fitparams are those in constants.py under parameters.params
    # fitparams: dict[str, Fitparam]  # observed_properties, now minus the global asteroseismic ones
    classicalparams: ClassicalParameters
    globalseismicparams: GlobalSeismicParameters
    seismicparams: SeismicParameters
    distanceparams: DistanceParameters


@dataclass(kw_only=True, frozen=True)
class RunFiles:
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
    Main class containing, handling, and managing all relevant file paths for a single target in a BASTA run.

    Parameters
    ----------

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


@dataclass(kw_only=True, frozen=True)
class InferenceSettings:
    """
    Main class containing settings used for the inference in a given BASTA run.

    Parameters
    ----------
    seed : int, optional
        The seed of randomness
    usebayw : bool or tuple, optional
        If True, Bayesian weights are applied. If tuple, custom weight behavior is applied.
        Default is True.
    priors : tuple, optional
        Tuple of strings containing name of priors (e.g., an IMF).
        See :func:`priors` for details.
    """

    gridfile: str
    seed: int
    limits: dict[str, Any]

    solarvalues: dict[
        str, float
    ]  #  = {"numax": constants.sydsun.SUNnumax, "dnu": constants.sydsun.SUNdnu}
    solarmodel: str = ""
    gridid: tuple[float, float, float, float] | None = None

    usebayw: bool = True

    priors: tuple[str, ...] | list[str] | None = None


@dataclass(kw_only=True, frozen=True)
class OutputOptions:
    """
    Main class containing settings that affect the output files from BASTA but not the inference in itself.

    Parameters
    ----------
    optionaloutputs : bool, optional
        If True, saves a 'json' file for each star with the global results and the PDF.
    debug : bool, optional
        If True, enables debugging. Default is False.
    verbose : bool, optional
        Activate a lot (!) of additional output
    developermode : bool, optional
        Activate experimental features
    validationmode : bool, optional
        Activate validation mode features
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
    Main class containing settings that affect the output files from BASTA but not the inference in itself.

    Parameters
    ----------

    """

    nameinplot: str
    kielplots: list
    cornerplots: list
    freqplots: list
