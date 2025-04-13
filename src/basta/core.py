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
class InferenceSettings:
    """
    Main class containing settings used for the inference in a given BASTA run.

    Parameters
    ----------

    gridfile : str
        Path and name of the hdf5 file containing the isochrones or tracks
        used in the fitting
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
    gridid: bool | tuple = False
    seed: int = 11
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
