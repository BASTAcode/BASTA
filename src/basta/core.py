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
from typing import Any, TypedDict, Literal, Sequence
import numpy as np

from basta import constants


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
    Container forglobal asteroseismic parameters and their scale transformations.

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
    scalefactors: dict[str, float] | None = None

    def set_scalefactor(self, scalefactors: dict[str, float]) -> None:
        self.scalefactors = scalefactors

        # Update params to reflect the new total scale (existing scale × new factor)
        for key, factor in scalefactors.items():
            if key in self.params:
                param = self.params[key]
                new_scale = param.scale * factor
                self.params[key] = ScaledValueError(
                    original=param.original, scale=new_scale
                )

    def get_original(self, key: str) -> tuple[float, float]:
        return self.params[key].original

    def get_scale(self, key: str) -> float:
        return self.params[key].scale

    def get_scalefactor(self, key: str) -> float:
        assert self.scalefactors is not None
        return self.scalefactors[key]

    def get_scaled(self, key: str) -> tuple[float, float]:
        return self.params[key].scaled


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


@dataclass(frozen=True)
class ObservedFrequencies:
    data: np.ndarray

    @property
    def l(self) -> np.ndarray:
        return self.data["l"]

    @property
    def n(self) -> np.ndarray:
        return self.data["n"]

    @property
    def frequencies(self) -> np.ndarray:
        return self.data["frequency"]

    @property
    def errors(self) -> np.ndarray:
        return self.data["error"]

    def of_angular_degree(self, given_l: int) -> np.ndarray:
        return self.data[self.l == given_l]

    @property
    def possible_angular_degrees(self) -> np.ndarray:
        return np.unique(self.l)

    @property
    def lowest_observed_radial_frequency(self) -> np.ndarray:
        radial_modes = self.of_angular_degree(0)
        if len(radial_modes) == 0:
            raise ValueError("No radial modes (l=0) found.")
        return radial_modes[np.argmin(radial_modes["n"])]


@dataclass(frozen=True)
class ModelFrequencies:
    data: np.ndarray

    @property
    def l(self) -> np.ndarray:
        return self.data["l"]

    @property
    def n(self) -> np.ndarray:
        return self.data["n"]

    @property
    def frequencies(self) -> np.ndarray:
        return self.data["frequency"]

    @property
    def inertias(self) -> np.ndarray:
        return self.data["inertia"]

    def of_angular_degree(self, given_l: int) -> np.ndarray:
        return self.data[self.l == given_l]


def make_model_modes_from_ln_freqinertia(
    ln: Sequence[np.ndarray], freqinertia: Sequence[np.ndarray]
) -> ModelFrequencies:
    """
    Create ModelFrequencies from two unstructured ndarrays.
    Can also be used with HDF5-based model data.
    """
    return ModelFrequencies(
        data=_pack_structuredarray(
            ln[0],
            ln[1],
            freqinertia[0],
            freqinertia[1],
            ["l", "n", "frequency", "inertia"],
        )
    )


def make_star_modes_from_l_n_freq_error(
    l: np.ndarray,
    n: np.ndarray,
    freq: np.ndarray,
    error: np.ndarray,
) -> ObservedFrequencies:
    """
    Create ObservedFrequencies from four unstructured ndarrays.
    """
    return ObservedFrequencies(
        data=_pack_structuredarray(l, n, freq, error, ["l", "n", "frequency", "error"])
    )


def _pack_structuredarray(
    int0: np.ndarray,
    int1: np.ndarray,
    float0: np.ndarray,
    float1: np.ndarray,
    names: list[str],
) -> np.ndarray:
    "Create structured ndarray from four unstructured ndarrays"
    assert len(int0) == len(int1) == len(float0) == len(float1)
    data = np.zeros(
        len(int0),
        dtype=[(names[0], int), (names[1], int), (names[2], float), (names[3], float)],
    )
    data[names[0]] = int0
    data[names[1]] = int1
    data[names[2]] = float0
    data[names[3]] = float1
    return data


@dataclass(frozen=True)
class JoinedModes:
    data: np.ndarray

    @property
    def l(self) -> np.ndarray:
        return self.data["l"]

    @property
    def observed_n(self) -> np.ndarray:
        return self.data["observed_n"]

    @property
    def model_n(self) -> np.ndarray:
        return self.data["model_n"]

    @property
    def observed_frequencies(self) -> np.ndarray:
        return self.data["observed_frequency"]

    @property
    def observed_error(self) -> np.ndarray:
        return self.data["error"]

    @property
    def model_frequencies(self) -> np.ndarray:
        return self.data["model_frequency"]

    @property
    def inertias(self) -> np.ndarray:
        return self.data["inertia"]

    def of_angular_degree(self, given_l: int) -> np.ndarray:
        return self.data[self.l == given_l]


@dataclass(kw_only=True)
class StarModes:
    modes: ObservedFrequencies
    surfacecorrection: dict[str, Any] | None = None

    obsintervals: np.ndarray | None = None

    # TODO(Amalie) how is this used?
    correlations: bool | int = False

    seismicweights: dict[str, Any]

    inverse_covariance: np.ndarray


@dataclass(frozen=True)
class SeismicSignature:
    """
    Seismic signatures, for example ratios, epsilon differences or glitches.
    """

    # 1-D array
    values: np.ndarray
    # Matrix with inverse covariances of `values`
    inverse_covariance: np.ndarray


@dataclass(kw_only=True)
class Star:
    starid: str

    limits: dict[str, tuple[float, float]]

    classicalparams: ClassicalParameters
    globalseismicparams: GlobalSeismicParameters
    distanceparams: DistanceParameters

    phase: tuple[str] | str | None = None

    absolutemagnitudes: AbsoluteMagnitudes | None = None

    modes: StarModes | None = None
    surfacecorrection: dict[str, Any] | None = None
    ratios: dict[str, dict[str, SeismicSignature]] | None = None
    glitches: dict[str, SeismicSignature] | None = None
    epsilondifferences: dict[str, SeismicSignature] | None = None

    @property
    def has_modes(self) -> bool:
        return self.modes is not None

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
    def has_any_seismic_case(self) -> bool:
        return any(
            (
                self.has_modes,
                self.has_ratios,
                self.has_glitches,
                self.has_epsilondifferences,
            )
        )


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


@dataclass(kw_only=True, frozen=True)
class InputStar:
    starid: str

    classicalparams: ClassicalParameters
    globalseismicparams: GlobalSeismicParameters
    distanceparams: DistanceParameters

    freqpath: str
    freqfile: str

    surfacecorrection: dict[str, Any] | None = None

    correlations: bool | int = False

    nottrustedfile: str | None = None
    excludemodes: str | None = None
    onlyradial: bool | None = None
    # fittypes: list[Literal["r01", "r010", "r012", "r02", "r10", "r102"]]

    readratios: bool | int = False
    threepoint: bool | int = False
    interp_ratios: bool | int = True

    # fittypes: list[Literal["gr01", "gr010", "gr012", "gr02", "gr10", "gr102"]]
    glitchfile: str | None = None
    nrealizations: int = 10000

    # fittypes: list[Literal["e01", "e012", "e02"]]
    nsorting: bool | int = True

    dnubias: float = 0.0


@dataclass(kw_only=True)
class PriorEntry:
    kwargs: dict[str, Any]
    limits: list[float] | None = None


@dataclass(kw_only=True)
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
    boxpriors : tuple or list of str, optional
        Priors to apply during the inference (e.g. an IMF, metallicity priors).
    """

    fitparams: list[str]

    seed: int

    gridfile: str
    gridid: tuple[float, float, float, float] | None = None
    solarvalues: dict[str, float]
    solarmodel: str = ""

    usebayw: bool = True
    boxpriors: dict[str, PriorEntry]
    imf: str | None = "salpeter1955"

    # dnufit_in_ratios: bool | int = False
    fit_surfacecorrected_dnu: bool | int = False
    # TODO(Amalie) Consider removing entirely
    dnuprior: bool | int = True
    seismicweights: dict[str, Any]

    @property
    def has_frequencies(self) -> bool:
        return any([x in constants.freqtypes.freqs for x in self.fitparams])

    @property
    def has_ratios(self) -> bool:
        return any([x in constants.freqtypes.rtypes for x in self.fitparams])

    @property
    def has_glitches(self) -> bool:
        return any([x in constants.freqtypes.glitches for x in self.fitparams])

    @property
    def has_epsilondifferences(self) -> bool:
        return any([x in constants.freqtypes.epsdiff for x in self.fitparams])

    @property
    def has_any_seismic_case(self) -> bool:
        return any(
            (
                self.has_frequencies,
                self.has_ratios,
                self.has_glitches,
                self.has_epsilondifferences,
            )
        )

    @property
    def has_distance_case(self) -> bool:
        return any([x in ["parallax", "distance"] for x in self.fitparams])


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
    kielplots: list[str]
    cornerplots: list[str]
    freqplots: list[str]

    style: str = "poster"
    figuresize: tuple[float, float] = (12.8, 8.8)

    seismic_twinax: bool = True
    seismic_legend_on_top: bool = True

    @property
    def mpl_style(self) -> dict:
        if self.style == "mnras":
            return {
                "figure.figsize": (3.5, 2.5),
                "figure.constrained_layout.use": True,
                "axes.labelsize": 18,
                "axes.titlesize": 18,
                "legend.fontsize": 16,
                "xtick.labelsize": 14,
                "ytick.labelsize": 14,
                "savefig.dpi": 300,
                "figure.dpi": 100,
                "savefig.bbox": "tight",
                "savefig.pad_inches": 0.02,
            }
        return {
            "figure.figsize": self.figuresize,
            "axes.labelsize": 17.6,
            "axes.titlesize": 14,
            "xtick.labelsize": 14,
            "ytick.labelsize": 14,
            "legend.fontsize": 12,
        }
