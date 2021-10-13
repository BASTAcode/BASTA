"""
Interpolation for BASTA: Helper routines
"""
import os

import numpy as np
from tqdm import tqdm
from scipy import interpolate
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

from basta import utils_seismic as su


# ======================================================================================
# Bayesian weights calcuation
# ======================================================================================
def bay_weights(inputvar):
    """
    Calculates bayesian weights. These are computed as the differences between
    consecutive, unique elements in the input array, and are used to weigh the
    values of a quantity across tracks or isochrones.

    Parameters
    ----------
    inputvar : array_like
        Input array of values of a certain variable across tracks or isochrones

    Returns
    -------
    weights : list
        List of bayesian weights
    """
    # Variables for storage
    diffs = []
    weights = []

    # Produce array of unique values
    univar = np.unique(inputvar)

    # Special case if a single input value is given
    try:
        inputvar[0]
    except TypeError:
        weights = [1.0]
    else:
        # Handle if input contains less then three unique entries
        #   -->  Set all weights to 1
        if len(univar) < 3:
            diffs = np.ones(len(univar))

        # ... otherwise: compute the weights
        else:
            for i in range(1, len(univar) - 1):
                diffs.append((univar[i + 1] - univar[i - 1]) / 2e0)
            diffs.insert(0, diffs[0])
            diffs.append(diffs[-1])

        # Assign the weights
        for i in inputvar:
            weights.append(diffs[np.argmin(np.abs(i - univar))])

    return weights


# ======================================================================================
# Common interpolation functions
# ======================================================================================
def _interpolation_wrapper(x, y, xnew, method="linear", along=True):
    """
    Interpolate a 1-D function using the specified method.

    x and y are arrays of values used to approximate some function f: y = f(x). This
    wrapper constructs the interpolation object and evaluates it on the new points xnew.

    Parameters
    ----------
    x : array_like
        A 1-D array of real values. Absicca.

    y : array_like
        A 1-D array of real values. Ordinate.

    xnew : array_like
        A 1-D array of real values. New absissa.

    along : bool
        Toggling of interpolation method. True for 1D interpolation along a track,
        False for linearND interpolation.

    Returns
    -------
    ynew : array_like
        A 1-D array of real values. New ordinate corresponding to new absissa.

    """
    if method in ["linear", "cubic"] and along:
        interp = interpolate.interp1d(x, y, kind=method)
    elif not along:
        interp = interpolate.LinearNDInterpolator(x, y)
    else:
        raise ValueError("Unavaliable interpolation method requested!")
    return interp(xnew).flatten()


def get_selectedmodels(grid, basepath, limits, cut=True, show_progress=True):
    """
    Determine the 'selectedmodels' dictionary of the grid, given by the limitting
    parameters. For interpolation only, it can be desired to keep models of tracks or
    isochrones that are outside the limits, but in between models that are, to
    ensure smooth distribution of points in the interpolated tracks/isochrones.

    Parameters
    ----------
    grid : h5py file
        Handle of grid file to search through

    basepath : str
        Path in grid file where tracks/isochrones are stored.

    limits : dict
        Constraints on the selection in the grid. Must be valid parameter names in grid.

    cut : bool, optional
        Whether to exclude models in a track/isochrone outside the limits, but in the
        interval of other models in the track/isochrone that are within the limits.

    show_progress : bool, optional
        Show a progress bar.

    Returns
    -------
    selectedmodels : dict
        Dictionary of every track/isochrone with models inside the limits, and the index
        of every model that satisfies this.
    """
    # Before running the actual loop, all tracks/isochrones are counted to better
    # estimate the progress.
    if show_progress:
        trackcount = 0
        for _, tracks in grid[basepath].items():
            trackcount += len(tracks.items())
        pbar = tqdm(total=trackcount, desc="--> Transversing grid", ascii=True)

    # The dictionary with the given information, stored as {name: indexes}
    selectedmodels = {}
    for gname, tracks in grid[basepath].items():
        for name, libitem in tracks.items():
            if show_progress:
                pbar.update(1)
            # If empty, skip
            if not len(libitem) or libitem["FeHini_weight"][()] == -1:
                continue

            # Full list of indexes, set False if model outside limits
            index = np.ones(len(libitem["age"][:]), dtype=bool)
            for param in limits:
                index &= libitem[param][:] >= limits[param][0]
                index &= libitem[param][:] <= limits[param][1]
            # Patch index array if it should not be cut
            if not cut and sum(index) > 2:
                mask = np.argwhere(index).T[0]
                index[mask[0] : mask[-1]] = np.ones(mask[-1] - mask[0], dtype=bool)
            # Save indexes for tracks/isochrones with more than 2 models
            if any(index) and sum(index) > 2:
                selectedmodels[os.path.join(gname, name)] = index

    if show_progress:
        pbar.close()

    return selectedmodels


def interpolate_frequencies(
    fullosc,
    fullosckey,
    agevec,
    newagevec,
    sections=None,
    freqlims=None,
    verbose=False,
    debug=False,
    trackid=None,
):
    """
    Perform interpolation in individual oscillation frequencies in a reduced track.

    Only modes with l = {0, 1, 2} are considered, and only modes with frequencies
    within the minimum and maximum, minus and plus half a large frequency separation
    respectively, will be interpolated.

    Should any mode appear or dissappear in the interpolation interval, or should
    it fall out of or into the before mentioned frequency range, they will be
    interpolated only in the part of the track they appear in.

    Also, to account for some models randomly having fewer modes than the rest of
    the track, modes need to appear in at least 80% of the models in the interval
    between the first and the last model they appear in.

    Parameters
    ----------
    fullosc : array
        Full reshaped array of oscillation frequencies and inertia

    fullosckey : array
        Full reshaped array of oscillation keys, in terms of radial order n, and
        degree l

    agevec : array
        Ages of original reduced track. Used to define the interpolation object.

    newagevec : array
        Mesh of ages where to evaluate the interpolation. Coordinates of new track.

    sections : array or None
        Toggle for 1D or ND interpolation, None for 1D, for ND array with indices that
        gives the individual tracks from the full arrays, for quality check.

    verbose : bool, optional
        Print info.

    debug : bool, optional
        Print extra information. Plot *all* interpolated frequencies. Warning: Slow!

    trackid : int, optional
        Must be given when using full debug mode. ID of current track. Used for plots.

    Returns
    -------
    osckeylist : list
        Nested list with an entry per point in the new, interpolated track. Each entry
        is an array with l and n. Following same definitions as BASTA/make_tracks.

    osclist : list
        Nested list with an entry per point in the new, interpolated track. Each entry
        is an array with frequencies and intertias. Following same definitions as
        BASTA/make_tracks.

    """
    along = True if sections == None else False
    if debug and along:
        print("--> Warning: Slow setting selected! Plotting *all* frequencies.")
        skipplots = False
    else:
        skipplots = True

    # Globally define values of l present in the grid
    available_lvalues = [0, 1, 2]

    #
    # *** BLOCK 1: Extract oscillations from the library into more accesible arrays ***
    #
    # Normally we are only interested in quantities one model at a time. However, that
    # it not sufficient for interpolation purposes. Thus, we need to "unpack" all
    # information of all models to interpolate each individual frequency across models.
    #
    # The arrays are called 'xval', with x \in {l, n, f, i} and can be indexed as
    # >   fvals["l=L"][MODELINDEX]
    # to obtain all frequencies with l=L for entry MODELINDEX.
    #
    # As an example: fvals["l=0"][-1][2] will yield the third (2) radial mode (l=0) for
    # the last point in the track (-1). The corresponding degree of that mode is found
    # as nvals["l=0""][-1][2].
    Ntrack = len(fullosc)
    nvals = {"l=0": [], "l=1": [], "l=2": []}
    fvals = {"l=0": [], "l=1": [], "l=2": []}
    ivals = {"l=0": [], "l=1": [], "l=2": []}
    for modid in range(Ntrack):
        for ll in available_lvalues:
            osckey, osc = su.get_givenl(
                l=ll,
                osc=su.transform_obj_array(fullosc[modid]),
                osckey=su.transform_obj_array(fullosckey[modid]),
            )
            lkey = "l={0}".format(ll)
            nvals[lkey].append(osckey[1])
            fvals[lkey].append(osc[0])
            ivals[lkey].append(osc[1])

    # Check if any lvalues can be omitted
    bad = []
    for ll in available_lvalues:
        lkey = "l={0}".format(ll)
        if not any([len(arr) for arr in nvals[lkey]]):
            bad.append(ll)
    for ll in bad:
        available_lvalues.remove(ll)

    #
    # *** BLOCK 2: Find common values of n for each of the l values ***
    #
    # Not all models have all models available (especially in the high end). To safely
    # interpolate, we need to locate which models each mode appears in. Takes care of
    # modes appearing, dissapearing, and random models without the modes
    #
    # Stored in the array 'goodn' with same indexing syntax as the arrays above.
    goodn = {"l=0": {}, "l=1": {}, "l=2": {}}
    # For across we need to check the base limits doesn't change with frequency limits
    if not along:
        newblims = {"l=0": {}, "l=1": {}, "l=2": {}}
    for ll in available_lvalues:
        lkey = "l={0}".format(ll)
        nrange = range(
            int(np.nanmin([arr.min() if len(arr) else np.nan for arr in nvals[lkey]])),
            int(np.nanmax([arr.max() if len(arr) else np.nan for arr in nvals[lkey]]))
            + 1,
        )

        # For each n, check the following
        for testn in nrange:
            index = np.ones(Ntrack, dtype=bool)
            for modid in range(Ntrack):
                modnvals = np.asarray(nvals[lkey][modid])
                modfvals = fvals[lkey][modid]
                # Check if present in model
                if testn not in modnvals:
                    index[modid] = False
                elif freqlims:
                    # Check that the values are within the frequency limits
                    modf = modfvals[np.where(modnvals == testn)[0][0]]
                    if not (modf > freqlims[0] and modf < freqlims[1]):
                        index[modid] = False

            # Check that there are not too many missing models
            badmode = False
            if not sections:
                sections = [0, -1]
            for s in range(len(sections) - 1):
                section = index[sections[s] : sections[s + 1]]
                # Check that there are enough models present
                if sum(section) <= 2:
                    badmode = True
                    break
                section = section[np.where(section)[0][0] : np.where(section)[0][-1]]
                if sum(section) / len(section) < 0.8:
                    badmode = True
                    break

            # Append modes that passed the checks
            if not badmode:
                goodn[lkey][testn] = index
            # Extract new base limits after frequency cut
            if not badmode and not along:
                secmins = []
                secmaxs = []
                for s in range(len(sections) - 1):
                    section = agevec[:, -1][sections[s] : sections[s + 1]]
                    secind = index[sections[s] : sections[s + 1]]
                    secmins.append(min(section[secind]))
                    secmaxs.append(max(section[secind]))
                newblims[lkey][testn] = [max(secmins), min(secmaxs)]

        if debug:
            try:
                minn = min([n for n, _ in goodn[lkey].items()])
                maxn = max([n for n, _ in goodn[lkey].items()])
                print("    l = {0} | n = {1:3} ... {2:3}".format(ll, minn, maxn))
            except:
                print("Debug print failed for _interpolate_frequencies")

    #
    # *** BLOCK 3: Interpolate in frequencies between the models ***
    #
    # The interpolation must be performed for each individual {l, n} seperately across
    # all models. Frequencies and inertias are interpolated and stored in dicts, which
    # can be indexed as e.g. newfreqs["l=0"]["n=12"].
    newfreqs = {"l=0": {}, "l=1": {}, "l=2": {}}
    newinert = {"l=0": {}, "l=1": {}, "l=2": {}}

    # Prepare for debugging plot(s)
    if debug:
        debugpath = "intpolout"
        plotpath = os.path.join(debugpath, "debug_freqs_track{0}.pdf".format(trackid))
        if not os.path.exists(debugpath):
            os.mkdir(debugpath)
        if os.path.exists(plotpath):
            skipplots = True
        if not skipplots:
            pdf = PdfPages(plotpath)

    # Loop over l and then n
    for ll in available_lvalues:
        lkey = "l={0}".format(ll)
        for nn, index in goodn[lkey].items():
            nkey = "n={0}".format(nn)

            # Extract values for a given n-value for all models
            oldf, oldi = [], []
            for i, ind in enumerate(index):
                if not ind:
                    continue
                oldf.append(fvals[lkey][i][np.where(nvals[lkey][i] == nn)[0][0]])
                oldi.append(ivals[lkey][i][np.where(nvals[lkey][i] == nn)[0][0]])

            # Create mask to avoid extrapolation
            if along:
                mask = np.ones(len(newagevec), dtype=bool)
                mask &= newagevec <= np.max(agevec[index])
                mask &= newagevec >= np.min(agevec[index])
            else:
                blims = newblims[lkey][nn]
                mask = np.ones(len(newagevec[:, -1]), dtype=bool)
                mask &= newagevec[:, -1] <= blims[1]
                mask &= newagevec[:, -1] >= blims[0]

            # Interpolate on the same mesh as the other quantities in the main routine.
            newf = _interpolation_wrapper(
                agevec[index], oldf, newagevec[mask], along=along
            )
            newi = _interpolation_wrapper(
                agevec[index], oldi, newagevec[mask], along=along
            )
            newfreqs[lkey][nkey] = newf
            newinert[lkey][nkey] = newi

            # Replace information for new vector
            goodn[lkey][nn] = mask

            # Plot the interpolated frequencies if desired
            if not skipplots:
                if along:
                    xold = agevec[index]
                    xnew = newagevec[mask]
                else:
                    xold = agevec[:, -1][index]
                    xnew = newagevec[:, -1][mask]
                _, ax = plt.subplots()
                plt.title(
                    "TrackID: {0}   |   l = {1}  ,  n = {2}".format(trackid, ll, nn)
                )
                ax.plot(
                    xold,
                    oldf,
                    ".",
                    color="#6C9D34",
                    label="Original",
                    alpha=0.9,
                )
                ax.plot(
                    xnew,
                    newf,
                    ".",
                    color="#482F76",
                    label="Interpolated",
                    alpha=0.5,
                )
                ax.set_xlabel("Base parameter")
                ax.set_ylabel("Frequency")
                ax.legend(loc="best", facecolor="none", edgecolor="none")
                pdf.savefig(bbox_inches="tight")
                plt.close()

    if not skipplots:
        pdf.close()
    elif debug:
        print("    The plot '{0}' already exist! Skipping...".format(plotpath))

    #
    # *** BLOCK 4: Pack the interpolated frequencies for grid storage ***
    #
    # For them to be stored in the HDF5 format, the frequencies must be restored
    # to a per-model structure. For each point in the interpolated track, we transverse
    # all l and then n values. They are stacked in one list per quantity and then stored
    # as nested arrays in lists to conform to the specific format defined by the BASTA
    # routines enabling HDF5 writing.
    osclist = []
    osckeylist = []
    Nnew = len(newagevec) if along else len(newagevec[:, -1])
    for modid in range(Nnew):
        # Arrays corresponding to the read-out from an .obs-file (in BASTA notation)
        lf = []
        nf = []
        ff = []
        ef = []
        for ll in available_lvalues:
            lkey = "l={0}".format(ll)
            modn, indmod = [], []
            for nn, mask in goodn[lkey].items():
                if mask[modid]:
                    modn.append(nn)
                    indmod.append(sum(mask[:modid]))
            # Do not append lists, but append the elemements of the lists
            tmp_lf = len(modn) * [ll]
            lf = [*lf, *tmp_lf]
            nf = [*nf, *modn]
            ff = [
                *ff,
                *[newfreqs[lkey]["n={0}".format(q)][m] for q, m in zip(modn, indmod)],
            ]
            ef = [
                *ef,
                *[newinert[lkey]["n={0}".format(q)][m] for q, m in zip(modn, indmod)],
            ]

        # Add the stacked ".obs-style" lists with all freq information to storage arrays
        osckeylist.append(np.array([lf, nf], dtype=np.int))
        osclist.append(np.array([ff, ef], dtype=np.float))

    return osckeylist, osclist


# ======================================================================================
# Management of header and weights
# ======================================================================================
def _extend_header(outfile, basepath, headvars):
    """
    Rewrites the header with the information from the new tracks.

    Parameters
    ----------
    outfile : hdf5 file
        New grid file to write to.

    basepath : str, optional
        Path in the grid where the tracks are stored. The default value applies to
        standard grids of tracks.

    headvars : list
        Variables to be written to header

    Returns
    -------
    outfile : hdf5 file
        New grid file to write to.

    """

    # Number of tracks in grid
    ltracks = sum([len(lib) for _, lib in outfile[basepath].items()])

    # For every variable in header, collect value from track and write to header
    for var in headvars:
        if "grid" in basepath:
            headpath = os.path.join("header", var)
        else:
            headpath = os.path.join("header", basepath, var)
        if var not in ["tracks", "isochs"]:
            values = np.zeros(ltracks)
            for _, group in outfile[basepath].items():
                for n, (_, libitem) in enumerate(group.items()):
                    if libitem["FeHini_weight"][()] != -1:
                        values[n] = libitem[var][0]
            del outfile[headpath]
            outfile[headpath] = values.tolist()
        else:
            del outfile[headpath]
            outfile[headpath] = [b"Interpolated"] * ltracks
    return outfile


def _write_header(grid, outfile, basepath):
    """
    Write the header of the new grid. Basically copies the old header.

    Parameters
    ----------
    grid : h5py file
        Handle of grid to process

    outfile : h5py file
        Handle of output grid to write to

    basepath : str, optional
        Path in the grid where the tracks are stored. The default value applies to
        standard grids of tracks.

    Returns
    -------
    grid : h5py file
        Handle of grid to process

    outfile : h5py file
        Handle of output grid to write to

    """
    # Needs to run before across interpolation for access to header lists during.
    # Duplicate the header to the new grid file.
    for key in grid["header"].keys():
        # Ignore isochrone-specific header entries (treated just below)
        if "=" not in key:
            outfile[os.path.join("header", key)] = grid[os.path.join("header", key)][()]

    # Treat isochrones with a path-dependent header
    if not "grid" in basepath:
        isochhead = os.path.join("header", basepath)
        for key in grid[isochhead].keys():
            outfile[os.path.join(isochhead, key)] = grid[os.path.join(isochhead, key)][
                ()
            ]

    # Duplicate solar model(s) to the new file, if present in the original grid
    try:
        grid["solar_models"]
    except KeyError:
        print("\nNote: No solar model to add!")
        pass
    else:
        for topkey in grid["solar_models"].keys():
            for key in grid[os.path.join("solar_models", topkey)].keys():
                keystr = os.path.join("solar_models", topkey, key)
                outfile[keystr] = grid[keystr][()]

    return grid, outfile


def _recalculate_weights(outfile, basepath, headvars):
    """
    Recalculates the weights of the tracks/isochrones, for the new grid.
    Tracks not transferred from old grid has FeHini_weight = -1.

    Parameters
    ----------
    outfile : hdf5 file
        New grid file to write to.

    basepath : str
        Path in the grid where the tracks are stored. The default value given in
        parent functions applies to standard grids of tracks.

    headvars : list
        Variables in the header of original grid

    Returns
    -------
    outfile : hdf5 file
        New grid file to write to.

    """
    isomode = False if "grid" in basepath else True

    # Collect the relevant tracks/isochrones
    mask = []
    names = []
    for nogroup, (gname, group) in enumerate(outfile[basepath].items()):
        # Determine which tracks are actually present
        for name, libitem in group.items():
            mask.append(libitem["FeHini_weight"][()])
            names.append(os.path.join(gname, name))
    mask = np.where(np.array(mask) > 0)[0]
    active = np.asarray(names)[mask]

    # For each parameter, collect values, recalculate weights, and replace old weight
    for key in headvars:
        if key in ["tracks", "isochs"]:
            continue
        if not isomode:
            headpath = os.path.join("header", key)
        else:
            headpath = os.path.join("header", basepath, key)
        values = np.array([outfile[headpath][i] for i in mask])
        weights = bay_weights(values)
        for i, name in enumerate(active):
            weight_path = os.path.join(basepath, name, key + "_weight")
            try:
                outfile[weight_path]
            except:
                outfile[weight_path] = weights[i]
            else:
                del outfile[weight_path]
                outfile[weight_path] = weights[i]
    return outfile
