"""
Routines to download examples (for non-GitHub installations)
"""

import os
import gzip
import shutil
import tarfile
import requests
import argparse
from tqdm import tqdm

from basta.__about__ import __version__
from basta.downloader import get_basta_dir

EXAMPLESFILE = "_examplespath.py"


def get_bundle(version: float, force: bool = False):
    """
    Download a bundle file with examples from the BASTAcode examples repository.

    Parameters
    ----------
    case : str
        Which version of the examples to get

    force : bool, optional
        Overwrite existing folder
    """
    # Settings
    block_size = 1024
    tqdm_settings = {
        "unit": "B",
        "unit_scale": True,
        "unit_divisor": 1024,
        "ascii": True,
        "desc": "--> Progress",
    }

    # Mapping to download location
    # --> Switched to anon share link due to (temporary?) issues with ERDA
    #     NB! Direct link to files differs in URL from true share link...
    #     (https://anon.erda.au.dk/sharelink/FVgq2M3mxY)
    # baseurl = "https://www.erda.au.dk/vgrid/BASTA/public-examples/"
    baseurl = "https://anon.erda.au.dk/share_redirect/FVgq2M3mxY"

    # Resolve grid name, location and write to file for easy reference
    getname = "basta-examples_v" + version.replace(".", "-") + ".tar.gz"
    url = os.path.join(baseurl, getname)
    basedir = os.path.abspath(".")
    outpath = os.path.join(basedir, "examples")

    # Obtain the grid if it does not exist
    tmptar = "examples.tar"
    if not os.path.exists(outpath) or force:
        try:
            # Step 1: Download into memory
            print(f"Downloading '{getname}'")
            res = requests.get(url, stream=True)
            res.raise_for_status()
            total_size = int(res.headers.get("content-length", 0))
            with tqdm(total=total_size, **tqdm_settings) as pbar:
                with open(getname, "wb") as fid:
                    for data in res.iter_content(block_size):
                        datasize = fid.write(data)
                        pbar.update(datasize)

            # Step 2: Extract (first decompress, then inflate tar)
            print(
                f"Decompressing and inflating into '{outpath}' ... ",
                end="",
                flush=True,
            )
            with gzip.open(getname, "rb") as fin:
                with open(tmptar, "wb") as fout:
                    shutil.copyfileobj(fin, fout)
            tar = tarfile.open(tmptar, "r:")
            tar.extractall()
            tar.close()
            print("done!")

        except Exception:
            if os.path.exists(getname):
                shutil.rmtree(getname)
            if os.path.exists(tmptar):
                shutil.rmtree(tmptar)
            raise

        finally:
            if os.path.exists(getname):
                os.remove(getname)
            if os.path.exists(tmptar):
                os.remove(tmptar)
    else:
        print(f"The folder '{outpath}' already exists! Will not overwrite.")


def main():
    """
    Run the examples/templates downloader
    """
    helptext = (
        "Download templates and/or examples for BASTA to current working directory."
    )
    parser = argparse.ArgumentParser(description=helptext)

    # Argument: How much to download
    # --> Only these grids are supported
    allowed_cases = [
        "simple",
        "full",
    ]
    parser.add_argument(
        "case",
        help=(
            f"What to download. Allowed cases: {allowed_cases}"
            " -- Selecting the simple case will download the template script for "
            "creating an input file for BASTA. "
            "Selecting the full case will download the entire examples directory"
        ),
    )

    # Optional arguments
    parser.add_argument(
        "--force", action="store_true", help="Overwrite existing file(s)?"
    )

    parser.add_argument(
        "--version",
        type=str,
        help="Get a specific version of the examples (default is to get current)",
    )

    # Parse and check
    args = parser.parse_args()
    if args.case not in allowed_cases:
        raise ValueError(f"Unknown instruction! Select from: {allowed_cases}")

    # Obtain the template input file or the full suite of examples
    if args.case == "simple":
        outfilename = "create_inputfile.py"
        if os.path.isfile(outfilename) and not args.force:
            print(f"The file {outfilename} already exists! Aborting...")
            return

        print(f"Downloading {outfilename} ...", end=" ")
        url = "https://raw.githubusercontent.com/BASTAcode/BASTA/refs/heads/main/examples/create_inputfile.py"
        req = requests.get(url)
        open(outfilename, "wb").write(req.content)
        print("done!")
    else:
        if args.version:
            version = args.version
        else:
            version = __version__
        get_bundle(version=version, force=args.force)
