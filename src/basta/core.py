"""
The BASTA core module contains Python dataclasses that define internal configurations, shared data models, and structures used for function inputs and outputs within the BASTA framework.

The classes defined here are not necessarily user-facing, but serve as foundational
components for the application's logic, inference processes, and internal communication.

    - Star: Star-specific information
    - InferenceOptions: Settings used for the inference in a given BASTA run.
    - OptionalFlags: Settings affecting only the output given by BASTA (e.g. plots or additions to the logfile) but not how the inference in a given run will be performed.

"""

from dataclasses import dataclass
from typing import Any
from pathlib import Path


@dataclass
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
    inputparams: dict[str, Any]


@dataclass
class FilePaths:
    """
    Main class containing, handling, and managing all relevant file paths for a single BASTA run.

    Parameters
    ----------

    """

    star: Star
    outputdir: Path
    plotfmt: str

    def __post_init__(self):
        self.outputdir = Path(self.outputdir)
        self.outputdir.mkdir(parents=True, exist_ok=True)

    @property
    def base(self) -> Path:
        return self.outputdir / self.star.starid

    @property
    def extradirectory(self) -> Path:
        path = self.outputdir / "debug"
        path.mkdir(parents=True, exist_ok=True)
        return path

    @property
    def debugplotfile(self) -> Path:
        return Path(f"{self.extradirectory}_{kind}.{self.plotfmt}")

    @property
    def logfile(self) -> Path:
        return self.base.with_suffix(".log")

    @property
    def plotfile(self, kind: str) -> Path:
        return Path(f"{self.base}_{kind}.{self.plotfmt}")

    @property
    def jsonfile(self) -> Path:
        return self.base.with_suffix(".json")

    @property
    def resultfile(self) -> Path:
        return self.base.with_suffix(".txt")


@dataclass
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

    seed: int
    gridfile: str
    gridid: bool | tuple = False
    usebayw: bool = True
    priors: tuple = (None,)


@dataclass
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

    optionaloutputs: bool = False
    debug: bool = False
    verbose: bool = False
    developermode: bool = False
    validationmode: bool = False
