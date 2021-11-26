"""
Check settings, compile external modules, download grids, and fetch dustmaps
"""
import os
import sys
import time
import argparse
from contextlib import contextmanager
from subprocess import call, check_output, CalledProcessError

from basta.downloader import get_grid

# Import dusmaps configuration (the maps themselves are slow, import later)
from dustmaps.config import config

DUSTMAPFILE = "_dustpath.py"


@contextmanager
def cd(newdir):
    """Context manager for changing directory"""
    prevdir = os.getcwd()
    os.chdir(os.path.expanduser(newdir))
    try:
        yield
    finally:
        os.chdir(prevdir)


def main():
    """
    Run setup
    """
    # Make sure a supported version of Python is not used
    assert sys.version_info >= (3, 7), "Python version is too old! Please use > 3.7"

    # Set-up argument parser
    parser = argparse.ArgumentParser(
        description="Initialize settings for BASTA, compile modules and download assets"
    )
    helpstr = (
        "Installing on the Grendel cluster, on your own system, or the light version?"
        " The light version will not compile any of the Fortran dependencies and will"
        " therefore not be able to fit glitches nor support grid interpolation! On"
        " Grendel the dustmaps will not be downloaded."
    )
    parser.add_argument(
        "case",
        choices=["personal", "grendel", "light"],
        help=helpstr,
    )
    args = parser.parse_args()
    print("Chosen case: {0}".format(args.case))

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # LOGICAL BLOCK: Determine directories
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Obtain location of BASTA
    try:
        home = os.environ["BASTADIR"]
    except KeyError:
        print(
            "Cannot find environment variable 'BASTADIR'! Did you define",
            "it? Aborting now...",
        )
        sys.exit(1)

    # If installing on Grendel, check location of the share
    if args.case == "grendel":
        try:
            grendelshare = os.environ["GRENDELSHARE"]
        except KeyError:
            print(
                "Cannot find environment variable 'GRENDELSHARE'! Did you",
                "define it? Aborting now...",
            )
            sys.exit(1)

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # LOGICAL BLOCK: External modules
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Check if a Fortran compiler is available in path before compiling stuff
    try:
        check_output(["which", "gfortran"]).decode("utf-8")
    except CalledProcessError:
        print("Cannot find any Fortran compiler! Will switch to the 'light' case... ")
        args.case = "light"

    # Exitcode for grabbing problems with importing modules
    exitcode = 0
    if args.case != "light":
        # Initial checks for compilation of external modules
        print("\n*******************************************")
        print("BEGIN: Compiling external modules for BASTA\n")
        try:
            f2pyver = check_output(["which", "f2py3"]).decode("utf-8")
        except CalledProcessError:
            print(
                "Unable to find 'f2py3'! Did you make the symlink?", "Aborting now..."
            )
            sys.exit(1)
        else:
            print(
                "Using the following f2py-tool (should point to the one in the",
                "BASTA venv):\n--> {0}".format(f2pyver),
            )

        # Compile each of the modules
        externals = ["glitch_fit", "sobol_numbers"]
        with cd(os.path.join(home, "basta")):
            for modname in externals:
                print("\n*** Module: {0} ***".format(modname))
                binname = modname + ".f95"
                call(["f2py3", "-c", binname, "-m", modname, "--quiet"])
                print("Compilation: Done!")

        print("\nEND: Compiling external modules for BASTA")
        print("*****************************************\n")

        # Redundant print, just to make sure
        print(
            "Compilation performed with the following f2py-tool (should point",
            "to the one located in the BASTA venv):\n--> {0}".format(f2pyver),
        )

        # Check that the modules can be imported
        try:
            import basta.glitch_fit
            import basta.sobol_numbers
        except ImportError:
            print("Unable to import one or more of the external modules!")
            exitcode = 1
        else:
            print("The external modules can be imported! Done!")
            exitcode = 0

    else:
        print(
            "\nThe case {0} has been invoked, and no modules will be compiled".format(
                args.case
            )
        )
        print("Grid interpolation and glitch fitting will not be possible!\n")

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # LOGICAL BLOCK: Example grids
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    print("\n=========================")
    if args.case in ["personal", "light"]:
        exgrid = "16CygA"
        print("Will download the '{0}' grid for use in the examples!\n".format(exgrid))
        get_grid(exgrid)
        print("\nIf you want more grids, try running 'BASTAdownload -h' !")
    else:
        print("No grids will be downloaded!")

    # ~~~~~~~~~~~~~~~~~~~~~~~
    # LOGICAL BLOCK: Dustmaps
    # ~~~~~~~~~~~~~~~~~~~~~~~
    # Define data folder for dustmaps depending on the installation type
    if args.case in ["personal", "light"]:
        dustfolder = os.path.join(os.path.abspath(home), "dustmaps")
        config["data_dir"] = dustfolder
        print("\n=========================")
        print("Will install dustmaps to: {0}".format(dustfolder))
        if not os.path.exists(dustfolder):
            os.mkdir(dustfolder)
        install_dustmaps = True
    else:
        dustfolder = os.path.join(
            os.path.abspath(grendelshare), "basta_input", "dustmaps"
        )
        config["data_dir"] = dustfolder
        print("\n=========================")
        print("\nWill assume dustmaps to be located in: {0}".format(dustfolder))
        install_dustmaps = False

    # Write dustmap datafolder to file
    with open(os.path.join(home, "basta", DUSTMAPFILE), "w") as f:
        f.write("__dustpath__ = '{0}'\n".format(dustfolder))

    # Install if required
    if install_dustmaps:
        # SFD/Schlegel dustmap
        print("\nFetching the SFD dustmap ...\n", flush=True)
        import dustmaps.sfd

        dustmaps.sfd.fetch()
        print("\nDone!")
        print("----------")
        sys.stderr.flush()
        sys.stdout.flush()

        # Bayestar/Green dustmap
        print("\nFetching the Bayestar dustmap ...\n", flush=True)
        import dustmaps.bayestar

        dustmaps.bayestar.fetch()
        print("\nDone!")
        sys.stderr.flush()
        sys.stdout.flush()

    # Wait before moving on (the progress bar from dustmaps hangs a bit)
    time.sleep(1)
    sys.exit(exitcode)


if __name__ == "__main__":
    main()
