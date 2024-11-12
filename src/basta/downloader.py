"""
Routines to download assets
"""

import os
import gzip
import shutil
import requests
import argparse
from tqdm import tqdm

# Import dusmaps configuration (the maps themselves are slow)
from dustmaps.config import config

DUSTMAPFILE = "_dustpath.py"


def get_basta_dir() -> str:
    """
    Helper to obtain location of BASTA root directory
    """
    rootdir = os.path.dirname(
        os.path.abspath(os.path.join(os.path.abspath(__file__), "../.."))
    )
    return rootdir


def get_grid(case: str, gridpath=None):
    """
    Download a grid from the BASTAcode grid repository. Will be stored in the default
    location: BASTA/grids/ .

    Parameters
    ----------
    case : str
        Which grid to download. Possible value: "16CygA", "validation", "iso".
    gridpath : str, optional
        Path to user-defined location of where to save grids
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
    baseurl = "https://www.erda.au.dk/vgrid/BASTA/public-grids/"

    # Resolve grid name and location
    if case == "iso":
        gridname = "BaSTI_iso2018.hdf5"
    elif case in ["16CygA", "validation", "validation_new-weights"]:
        gridname = f"Garstec_{case}.hdf5"
    else:
        raise ValueError("Unknown grid!")
    url = os.path.join(baseurl, f"{gridname}.gz")

    # Default or user-defined location?
    if gridpath:
        basedir = os.path.abspath(gridpath)
    else:
        home = get_basta_dir()
        basedir = os.path.join(home, "grids")

    # Make sure target exists
    if not os.path.exists(basedir):
        os.makedirs(basedir)

    # Obtain the grid if it does not exist
    gridpath = os.path.join(basedir, gridname)
    if not os.path.exists(gridpath):
        try:
            # Step 1: Download
            gz_tmp = gridpath + ".gz"
            print(f"Downloading '{gridname}' to '{gz_tmp}'")
            res = requests.get(url, stream=True)
            res.raise_for_status()
            total_size = int(res.headers.get("content-length", 0))
            with tqdm(total=total_size, **tqdm_settings) as pbar:
                with open(gz_tmp, "wb") as fid:
                    for data in res.iter_content(block_size):
                        datasize = fid.write(data)
                        pbar.update(datasize)

            # Step 2: Extract
            print(
                f"Decompressing grid into '{gridpath}' ... ",
                end="",
                flush=True,
            )
            with gzip.open(gz_tmp, "rb") as fin:
                with open(gridpath, "wb") as fout:
                    shutil.copyfileobj(fin, fout)
            print("done!")

        except Exception:
            if os.path.exists(gridpath):
                shutil.rmtree(gridpath)
            raise

        finally:
            if os.path.exists(gz_tmp):
                os.remove(gz_tmp)

    else:
        print(f"The grid '{gridpath}' already exists! Will not download.")


def get_dustmaps(dustpath: str | None = None, skip: bool = False):
    """
    Configure dustmaps and download if necessary

    Parameters
    ----------
    dustpath : str, optional
        Where to store/find dustmaps
    skip : bool, optional
        Skip the download of the dustmaps
    """
    home = get_basta_dir()

    # Default or user-defined location?
    if dustpath:
        dustfolder = os.path.abspath(dustpath)
    else:
        dustfolder = os.path.join(home, "dustmaps")

    # Configure package to use the specified path
    config["data_dir"] = dustfolder
    print("\n=========================")
    print(f"Location of dustmaps: {dustfolder}")
    if not os.path.exists(dustfolder):
        os.mkdir(dustfolder)

    # Write dustmap datafolder to file
    with open(os.path.join(home, DUSTMAPFILE), "w") as f:
        f.write(f"__dustpath__ = '{dustfolder}'\n")

    # Install if required
    if not skip:
        print("Obtaining dustmaps!")
        # SFD/Schlegel dustmap
        print("\nFetching the SFD dustmap ...\n", flush=True)
        import dustmaps.sfd

        dustmaps.sfd.fetch()
        print("\nDone!")
        print("----------")

        # Bayestar/Green dustmap
        print("\nFetching the Bayestar dustmap ...\n", flush=True)
        import dustmaps.bayestar

        dustmaps.bayestar.fetch()
        print("\nDone!")
    else:
        print("Assuming dustmaps to be available without download!")


def main():
    """
    Run the downloader
    """
    helptext = (
        "Download assets for BASTA. Currently, it will download grids and dustmaps."
    )
    parser = argparse.ArgumentParser(description=helptext)

    # Argument: Which grid to download
    # --> Only these grids are supported
    allowed_grids = [
        "16CygA",
        "validation",
        "iso",
        "validation_new-weights",
    ]
    parser.add_argument(
        "grid", help=f"The grid to download. Allowed cases: {allowed_grids}"
    )

    # Optional argument: Where to save the grid
    parser.add_argument(
        "--gridpath", type=str, help="Store grid in non-standard location."
    )

    # Optional argument: Location of dustmaps
    parser.add_argument(
        "--dustpath",
        type=str,
        help="Store dustmaps in non-standard location (will make BASTA search in this location at runtime)",
    )

    # Optional argument: Don't download dustmaps
    parser.add_argument(
        "--no-dustmaps",
        action="store_true",
        help="Skip download of dustmaps. Warning: BASTA will not work if they are not available.",
    )

    # Parse and check
    args = parser.parse_args()
    if args.grid not in allowed_grids:
        raise ValueError(f"Unknown grid requsted! Select from: {allowed_grids}")

    get_grid(case=args.grid, gridpath=args.gridpath)
    get_dustmaps(dustpath=args.dustpath, skip=args.no_dustmaps)
