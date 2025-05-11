import argparse
import multiprocessing as mp
import os
from contextlib import contextmanager
from subprocess import run
import numpy as np

from basta.xml_run import run_xml


@contextmanager
def cd(newdir: str):
    """
    Context manager for changing the current working directory.

    Parameters
    ----------
    newdir : str
        Target directory to switch to.
    """
    prevdir = os.getcwd()
    os.chdir(os.path.expanduser(newdir))
    try:
        yield
    finally:
        os.chdir(prevdir)


def _process_xmldir(
    rundir: str, nproc: int = 4, seed: int | None = None, debug: bool = False
) -> None:
    """
    Run BASTA on all XML files in the specified directory using multiprocessing.

    Parameters
    ----------
    rundir : str
        Path to the directory containing XML files.

    nproc : int, optional
        Number of processes to use (default is 4).

    seed : int, optional
        Random seed for deterministic behavior.

    debug : bool, optional
        If True, adds the '--debug' flag to BASTA runs.
    """
    print(f"~~~~~~ RUNNING BASTA ON {rundir} WITH {nproc} THREADS NOW ~~~~~~\n")
    with cd(rundir):
        xml_files = [f for f in os.listdir(".") if f.endswith(".xml")]

        bastatasks = []
        for filename in xml_files:
            cmd = ["BASTArun", filename]
            if debug:
                cmd.append("--debug")
            if seed:
                cmd.extend(["--seed", str(seed)])
            bastatasks.append((cmd,))

        with mp.Pool(processes=nproc) as pool:
            pool.starmap(run, bastatasks)
    print("\n~~~~~~ DONE! ~~~~~~\n")


def main() -> None:
    """
    Run BASTA on a single XML input file via command-line arguments.
    """
    parser = argparse.ArgumentParser(
        description="BASTA -- Run the BAyesian STellar Algorithm"
    )
    parser.add_argument("inputfile", help="The XML input file to run")
    parser.add_argument("--debug", action="store_true", help="Enable debugging output")
    parser.add_argument(
        "--verbose", action="store_true", help="Verbose mode for detailed output"
    )
    parser.add_argument(
        "--developermode", action="store_true", help="Enable developer/test features"
    )
    parser.add_argument(
        "--validation",
        action="store_true",
        help="Run in validation mode (for internal use)",
    )
    parser.add_argument(
        "-s", "--seed", type=int, help="Random seed for reproducibility"
    )

    args = parser.parse_args()

    seed = args.seed if args.seed is not None else np.random.randint(5000)
    np.random.seed(seed)

    if args.validation:
        print("\n*** Running in VALIDATION MODE ***\n")

    # Call BASTA
    run_xml(
        args.inputfile,
        seed=seed,
        debug=args.debug,
        verbose=args.verbose,
        developermode=args.developermode,
        validationmode=args.validation,
    )


def multi() -> None:
    """
    Run BASTA on multiple XML input files in a directory.
    Supports multiprocessing and optional debug/seed configuration.
    """
    parser = argparse.ArgumentParser(description="Run BASTA on multiple input files.")
    parser.add_argument("xmlpath", help="Path to the directory containing XML files.")
    parser.add_argument(
        "--parallel", type=int, help="Number of parallel processes to use."
    )
    parser.add_argument("--debug", action="store_true", help="Enable debugging output")
    parser.add_argument(
        "-s", "--seed", type=int, help="Random seed for reproducibility"
    )

    args = parser.parse_args()

    num_threads = args.parallel if args.parallel else os.cpu_count()

    _process_xmldir(
        rundir=os.path.abspath(args.xmlpath),
        nproc=num_threads,
        seed=args.seed,
        debug=args.debug,
    )
