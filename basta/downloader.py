"""
Routines to download assets
"""
import os
import gzip
import shutil
import requests
from tqdm import tqdm


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
    baseurl = "https://cloud.phys.au.dk/nextcloud/index.php/s"
    urlmap = {
        "Garstec_16CygA.hdf5": "dSw3y5adCPMBMJb",
        "Garstec_validation.hdf5": "FMJBbAwxpYjzBsd",
        "BaSTI_iso2018.hdf5": "rtjB4owrgcEN4cg",
        "Garstec_16CygA_v1.hdf5": "yLQBDrAJeinFAMN",
    }

    # Resolve grid name and location
    if case == "iso":
        gridname = "BaSTI_iso2018.hdf5"
    elif case in ["16CygA", "validation", "16CygA_v1"]:
        gridname = "Garstec_{0}.hdf5".format(case)
    else:
        raise ValueError("Unknown grid!")
    url = os.path.join(baseurl, urlmap[gridname], "download", "{0}.gz".format(gridname))

    # Obtain location of BASTA
    try:
        home = os.environ["BASTADIR"]
    except KeyError:
        print("Cannot find environment variable 'BASTADIR'! Cannot download grid...")
        return

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
