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


def get_basta_dir():
    """
    Helper to obtain location of BASTA root directory
    """
    rootdir = os.path.dirname(
        os.path.abspath(os.path.join(os.path.abspath(__file__), ".."))
    )
    return rootdir


def get_grid(case):
    """
    Download a grid from the BASTAcode grid repository. Will be stored in the default
    location: $BASTADIR/grids/ .

    Parameters
    ----------
    case : str
        Which grid to download. Possible value: "16CygA", "validation", "iso".

    Returns
    -------
    None
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
        gridname = "Garstec_{0}.hdf5".format(case)
    else:
        raise ValueError("Unknown grid!")
    url = os.path.join(baseurl, "{0}.gz".format(gridname))

    # Obtain location of BASTA
    home = get_basta_dir()

    # Make sure the target folder exists
    basedir = os.path.join(home, "grids")
    if not os.path.exists(basedir):
        os.makedirs(basedir)

    # Obtain the grid if it does not exist
    gridpath = os.path.join(basedir, gridname)
    if not os.path.exists(gridpath):
        try:
            # Step 1: Download
            gz_tmp = gridpath + ".gz"
            print("Downloading '{0}' to '{1}'".format(gridname, gz_tmp))
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
                "Decompressing grid into '{0}' ... ".format(gridpath),
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
        print("The grid '{0}' already exists! Will not download.".format(gridpath))


def get_dustmaps():
    # Obtain location of BASTA
    home = get_basta_dir()

    # More...
    dustfolder = os.path.join(home, "dustmaps")
    config["data_dir"] = dustfolder
    print("\n=========================")
    print("Will install dustmaps to: {0}".format(dustfolder))
    if not os.path.exists(dustfolder):
        os.mkdir(dustfolder)
    install_dustmaps = True

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

        # Bayestar/Green dustmap
        print("\nFetching the Bayestar dustmap ...\n", flush=True)
        import dustmaps.bayestar

        dustmaps.bayestar.fetch()
        print("\nDone!")


def main():
    """Main"""
    helptext = "Download assets for BASTA. Currently, only grids are supported!"
    parser = argparse.ArgumentParser(description=helptext)

    # Only grids are supported!
    allowed_grids = [
        "16CygA",
        "validation",
        "iso",
        "validation_new-weights",
    ]
    parser.add_argument(
        "grid", help="The grid to download. Allowed cases: {0}".format(allowed_grids)
    )
    args = parser.parse_args()

    if args.grid not in allowed_grids:
        raise ValueError(
            "Unknown grid requsted! Select from: {0}".format(allowed_grids)
        )

    get_grid(args.grid)
    get_dustmaps()
