import os
import argparse
import numpy as np
from subprocess import run
import multiprocessing as mp
from contextlib import contextmanager

from basta.xml_run import run_xml


@contextmanager
def cd(newdir: str):
    """Change directory"""
    prevdir = os.getcwd()
    os.chdir(os.path.expanduser(newdir))
    try:
        yield
    finally:
        os.chdir(prevdir)


def _process_xmldir(
    rundir: str, nproc: int = 4, seed: int | None = None, debug: bool = False
):
    """
    Run BASTA on all files in a given directory. Multi-threaded version.

    Parameters
    ----------
    rundir : str
        Path to fully prepared validation run directory

    nproc : int, optional
        Number of cpus to use in the multiprocessing

    debug : bool, optional
        Add --debug option for BASTA

    seed : int, optional
        Initialise using a specific seed for BASTA

    Returns
    -------
    None

    """
    print(
        "~~~~~~ RUNNING BASTA ON {0} WITH {1} THREADS NOW ~~~~~~\n".format(
            rundir, nproc
        )
    )
    with cd(rundir):
        # Construct list of XML files in the directory and then process them in parallel
        bastatasks = []
        for filename in next(os.walk("."))[2]:
            if filename.endswith(".xml"):
                cmd = ["BASTArun", filename]
                if debug:
                    cmd.append("--debug")
                if seed:
                    cmd.append("--seed")
                    cmd.append(str(seed))
                bastatasks.append((cmd,))
        with mp.Pool(processes=nproc) as pool:
            pool.starmap(run, bastatasks)
    print("\n~~~~~~ DONE! ~~~~~~\n")


def main():
    """Main"""
    # Setup argument parser
    helptext = "BASTA -- Run the BAyesian STellar Algorithm"
    parser = argparse.ArgumentParser(description=helptext)

    # Add positional argument (name of inputfile)
    parser.add_argument("inputfile", help="The XML input file to run")

    # Add optional argument: Debugging output
    parser.add_argument(
        "--debug", action="store_true", help="Additional output for debugging."
    )

    # Add optional argument: Extra text output for debugging
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print additional text output. A lot!" " Combine with the debug flag.",
    )

    # Add optional argument: Experimental features
    parser.add_argument(
        "--developermode", action="store_true", help="Enable experimental features."
    )

    # Add optional argument: Validation mode
    parser.add_argument(
        "--validation",
        action="store_true",
        help="DO NOT USE unless making validation runs.",
    )

    # Add optional argument: Set random seed
    parser.add_argument(
        "-s",
        "--seed",
        type=int,
        help="Set random seed to ensure deterministic behavior for debugging",
    )

    # Parse the arguments
    args = parser.parse_args()

    # Set the random seed if given
    if args.seed is not None:
        seed = args.seed
    else:
        seed = np.random.randint(5000)

    np.random.seed(seed)

    # Flag the user if running in validation mode
    if args.validation:
        print("\n*** Running in VALIDATION MODE ", end="")

    # Call BASTA
    run_xml(
        vars(args)["inputfile"],
        seed=seed,
        debug=args.debug,
        verbose=args.verbose,
        developermode=args.developermode,
        validationmode=args.validation,
    )


def multi():
    """
    Run BASTA on multiple input files
    """
    # Initialise parser and gather arguments
    parser = argparse.ArgumentParser(description=("Run BASTA on multiple input files."))
    parser.add_argument(
        "xmlpath", help=("Path to the directory with xml files to process.")
    )
    parser.add_argument(
        "--parallel",
        help="Specify number of threads used in multiprocessing."
        " If not set, will use max available on system.",
        type=int,
    )
    parser.add_argument(
        "--debug", action="store_true", help="Additional output for debugging."
    )
    parser.add_argument(
        "-s",
        "--seed",
        type=int,
        help="Set random seed to ensure deterministic behavior for debugging",
    )
    args = parser.parse_args()

    if args.parallel:
        numthread = args.parallel
    else:
        numthread = os.cpu_count()

    _process_xmldir(
        rundir=os.path.abspath(args.xmlpath),
        nproc=numthread,
        seed=args.seed,
        debug=args.debug,
    )
