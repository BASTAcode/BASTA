"""
The BASTA core module contains Python dataclasses that define internal configurations, shared data models, and structures used for function inputs and outputs within the BASTA framework.

The classes defined here are not necessarily user-facing, but serve as foundational
components for the application's logic, inference processes, and internal communication.

    - Star: Star-specific information
    - InferenceOptions: Settings used for the inference in a given BASTA run.
    - OptionalFlags: Settings affecting only the output given by BASTA (e.g. plots or additions to the logfile) but not how the inference in a given run will be performed.

"""

from dataclasses import dataclass


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
    inputparams: dict


@dataclass
class InferenceOptions:
    """
    Main class containing settings used for the inference in a given BASTA run.

    Parameters
    ----------
    gridfile : str
        Path and name of the hdf5 file containing the isochrones or tracks
        used in the fitting
    seed : int, optional
        The seed of randomness
    usebayw : bool or tuple
        If True, bayesian weights are applied in the computation of the
        likelihood. See :func:`interpolation_helpers.bay_weights()` for details.
    priors : tuple
        Tuple of strings containing name of priors (e.g., an IMF).
        See :func:`priors` for details.
    """

    gridfile: str
    seed: int
    usebayw: bool = True
    usepriors: tuple = (None,)


@dataclass
class OptionalFlags:
    """
    Main class containing settings that affect the output files from BASTA but not the inference in itself.

    Parameters
    ----------
    optionaloutputs : bool, optional
        If True, saves a 'json' file for each star with the global results and the PDF.
    debug : bool, optional
        Activate additional output for debugging
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
