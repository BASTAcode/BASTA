"""
Interpolation for BASTA: Helper routines
"""

import os
import warnings

import numpy as np
import bottleneck as bn
from tqdm import tqdm
from scipy import interpolate
from scipy.stats import qmc
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
def interpolation_wrapper(x, y, xnew, method="linear", along=False):
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


def sobol_wrapper(ndim: int, nsamples: int, seed: int, debug=False):
    """
    Wrapper for the common Sobol-sampling routine, to disable UserWarning. This
    usually highlights that the sequence has to be sampled as 2^n points, to be
    perfectly balanced, which we can not enforce in this framework.

    Parameters
    ----------
    ndim : int
        Number of dimensions/sampling parameters.
    nsamples : int
        Number of points to be sampled.
    seed : int
        Seed to be used for the random sampler.

    Returns
    -------
    numbers : numpy array
        Array of (ndim, nsamples) sampled points.
    """

    with warnings.catch_warnings():
        if not debug:
            # Suppress warning, catch by substring (regex?)
            warnings.filterwarnings("ignore", message="^.*balance properties.*$")
        # Initialize sampler
        sampler = qmc.Sobol(ndim, scramble=False, seed=seed)
        # Sample points
        numbers = sampler.random(nsamples)

    return numbers


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
            if not len(libitem):
                continue
            # If previously interpolated, and track failed
            if "IntStatus" in libitem and libitem["IntStatus"][()] < 0:
                continue

            # Full list of indexes, set False if model outside limits
            index = np.ones(len(libitem["age"][:]), dtype=bool)
            for param in limits:
                if "freq" in param:
                    continue
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


def calc_along_points(intbase, sections, minmax, point, envres=None, resvalue=None):
    """
    Creates new along vector by interpolating from along vectors of
    enveloping tracks from model numbers to mimic spacing in parameter.
    E.g. if the enveloping tracks have a high resolution at the start,
    but low at the end, this function attempts to reproduce that for the
    interpolated track. It does so by normalising the model numbers to
    an interval between 0 and 1 for each track, and use that as the
    interpolation base.

    Parameters
    ----------
    intbase : numpy array
        Interpolation base vector of the enveloping tracks
    sections : list
        Indexes corresponding to the enveloping tracks to sub-divide the single array
    minmax : list
        Minimum and maximum values of along variable in the enveloping
        tracks
    point : list
        Base parameter values for the interpolated track
    envres : numpy array
        Copy of intbase, but with along resolution variable of enveloping
        tracks
    resvalue : float
        Along resolution to achieve

    Returns
    -------
    out : numpy array
        New 1D along variable vector for the interpolated track
    """

    # For counting and collecting during loop over tracks
    mods = []
    yvec = []
    newl = []

    if resvalue:
        envminmax = [[], []]
        for s in range(len(sections) - 1):
            envminmax[0].append(min(envres[sections[s] : sections[s + 1], -1]))
            envminmax[1].append(max(envres[sections[s] : sections[s + 1], -1]))
        envminmax[0] = max(envminmax[0])
        envminmax[1] = min(envminmax[1])
        Nres = int(abs(np.ceil((envminmax[1] - envminmax[0]) / resvalue)))

    if not resvalue:
        for i, s in enumerate(range(len(sections) - 1)):
            # Construct individual track bases
            base = intbase[sections[s] : sections[s + 1], -1]
            # Determine the models within limits
            mask = np.ones(len(base), dtype=bool)
            mask &= np.array(base) >= minmax[0]
            mask &= np.array(base) <= minmax[1]
            lbase = sum(mask)
            newl.append(lbase)

            if lbase < 1:
                raise ValueError

            # Variables for normalisation and to keep edges
            dists = [0, 0]
            offset = 0

            # If the base does not dictate the limits, make appropriate edges
            # If the start is not the limit of the interval, the first model
            # within the interval will not be 0, but a little over, and the
            # previous model will be included as a negativily numbered model.
            if np.sum(mask) != len(base):
                if not mask[0]:
                    ind = np.where(base == base[mask][0])[0][0]
                    ref = minmax[0 if base[0] < base[1] else 1]
                    dists[0] += (base[ind] - ref) / (base[ind] - base[ind - 1])
                    mask[np.where(mask)[0][0] - 1] = True
                    offset = -1
                if not mask[-1]:
                    ind = np.where(base == base[mask][-1])[0][0]
                    ref = minmax[1 if base[0] < base[1] else 0]
                    dists[1] += (ref - base[ind]) / (base[ind + 1] - base[ind])
                    mask[np.where(mask)[0][-1] + 1] = True

            # Reform base to be within interval
            newbase = base[mask]

            # Normalisation of model "numbers" from 0 to 1 in interval
            # "Ghost" models outside interval are kept for interpolation
            mod = (np.arange(sum(mask)) + offset + dists[0]) / (lbase - 1 + sum(dists))
            # Construct new interpolation base
            fmod = intbase[sections[s] : sections[s + 1], :][mask]
            fmod[:, -1] = mod
            mods.append(fmod)

            # Compile list of along variable values
            yvec += list(newbase)

        # Reconstruct base
        intbase = mods[0]
        for m in mods[1:]:
            intbase = np.vstack((intbase, m))
    else:
        for i, s in enumerate(range(len(sections) - 1)):
            base = intbase[sections[s] : sections[s + 1], -1]
            mask = np.ones(len(base), dtype=bool)
            mask &= np.array(base) >= minmax[0]
            mask &= np.array(base) <= minmax[1]
            newl.append(sum(mask))

            yvec += list(base)
        intbase = envres

    # Determine/adjust number of points
    N = int(np.mean(newl))
    if resvalue and N < Nres:
        N = Nres
    elif resvalue:
        print("Warning: Reduced resolution from {0} to {1}".format(N, Nres))
        N = Nres

    # Making new interpolation base
    # Putting them exactly on top creates boundary issues
    if resvalue:
        lin = np.linspace(envminmax[0], envminmax[1], N + 2)[1:-1]
    else:
        lin = np.linspace(0, 1, N + 2)[1:-1]
    inp = np.ones((len(lin), len(point) + 1))
    for i, p in enumerate(point):
        inp[:, i] *= p
    inp[:, -1] = lin

    # Make interpolator
    out = interpolation_wrapper(intbase, yvec, inp)

    return out


def interpolate_frequencies(
    fullosc,
    fullosckey,
    sections,
    triangulation,
    newvec,
    freqlims=None,
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

    sections : list
        Indexes indicating where each track starts and begins within the full
        frequency arrays. Needed to quality check for missing frequencies in tracks.

    triangulation : object
        Delaunay triangulation of the base

    newvec : array
        Base vector of the new track.

    freqlims : list
        Minimum and maximum frequencies to interpolate

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

    available_lvalues = [0, 1, 2]
    along = type(triangulation) == np.ndarray
    #
    # *** BLOCK 1: Determine and allocate matrix sizes ***
    #
    # For computation efficiency, a matrix is allocated for each l-value,
    # with rows corresponding to each n-value, columns corresponding to
    # to each model, and a layer each for frequency and inertia of the mode.
    # With a filler value of nan's for non-existing modes, no resizing or
    # continuous checking of the modes is needed, as all other functions
    # neatly passes nan on.
    bad = []
    freqs = {}
    nranges = {}
    Ntrack = len(fullosc)
    for ll in available_lvalues:
        # Get n-value range
        nmin = 999
        nmax = -999
        for modid in range(Ntrack):
            mask = fullosckey[modid][0] == ll
            if not len(mask):
                continue
            nmin = min(nmin, fullosckey[modid][1][mask][0])
            nmax = max(nmax, fullosckey[modid][1][mask][-1])

        # If none were found, this l should be skipped
        if nmin > nmax:
            bad.append(ll)
            continue

        # Allocate matrix for frequencies and inertia
        matrix = np.empty((2, nmax - nmin + 1, Ntrack))
        matrix[:] = np.nan

        freqs[ll] = matrix
        nranges[ll] = [nmin, nmax]

    # Remove bad modes
    for mode in bad:
        available_lvalues.remove(mode)

    #
    # *** BLOCK 2: Fill out matrices ***
    #
    # Read in frequencies from each model into the matrices.
    # Note that row 0 corresponds to the lowest n-value.
    # The reading is done by l-value of each model (fewest loop
    # iterations).

    for modid in range(Ntrack):
        osc = fullosc[modid]
        osckey = fullosckey[modid]
        for ll in available_lvalues:
            nmin, _ = nranges[ll]
            lmask = osckey[0, :] == ll
            freqs[ll][:, osckey[1, lmask] - nmin, modid] = osc[:, lmask]

    #
    # *** BLOCK 3: Check for bad modes ***
    #
    # The source information for each mode might turn out to be too
    # sparse. Here two checks can exclude a mode:
    #  - A source track have 2 or fewer models with the mode
    #  - From the first model within the track the mode appears in till
    #    the last, 20% or more of the models does not contain the mode.
    # A mode that fails the check will have all frequencies switched to
    # nan's, and is thus ignored going forward.

    for ll in available_lvalues:
        matrix = freqs[ll]

        # Apply frequency limits
        mask = matrix[0] > freqlims[0]
        mask &= matrix[0] < freqlims[1]

        # Check for sections with too many missing modes
        for nn, subm in enumerate(mask):
            badmode = False
            for s in range(len(sections) - 1):
                # If already flagged, skip checking rest of sections
                if badmode:
                    continue
                # Section corresponding to one enveloping track
                section = subm[sections[s] : sections[s + 1]]
                where = np.where(section)[0]
                # Bad mode if not present in track
                if len(where) <= 2:
                    badmode = True
                    continue

                # Bad mode if 20% of models in tracks are missing mode
                section = section[where[0] : where[-1]]
                if bn.nansum(section) / len(section) < 0.8:
                    badmode = True
                    continue
            # Flag the mode by filling with nan's
            if badmode:
                subm[:] = False

        # Write nan's to failed models
        matrix[0][~mask] = np.nan
        matrix[1][~mask] = np.nan

    #
    # *** BLOCK 4: Interpolate to new frequencies ***
    #
    # Now we can interpolate each mode to the new track,
    # by looping over each row in the matrices. If any source
    # is nan, the interpolation method returns a nan, which is
    # filled in the new matrix.

    Nnew = len(newvec)
    newfreqs = {}
    for ll in available_lvalues:
        matrix = freqs[ll]

        # Create collection matrix
        newmatrix = np.empty((2, matrix.shape[1], Nnew))
        newmatrix[:] = np.nan

        # Loop over both frequencies and inertia
        for fi in range(2):
            # Loop over n-values
            for nn in range(matrix.shape[1]):
                # No need if all are nan
                if bn.allnan(matrix[fi][nn][:]):
                    continue
                # Interpolate !!!
                newmatrix[fi][nn][:] = interpolation_wrapper(
                    triangulation,
                    matrix[fi][nn][:],
                    newvec,
                    along=along,
                )

        # Store new frequencies
        newfreqs[ll] = newmatrix

    #
    # *** BLOCK 5: Pack to per-model structure ***
    #
    # Pack the new frequencies to the format used in the hdf5.
    # Here the nan's are finally filtered out, as the output format
    # is not dependent on the order of elements. If no frequencies
    # are obtained for a model in the new track (likely due to inputted
    # frequency limits), it will produce an empty array instead, and is
    # thus skipped over in the fit.

    osclist, osckeylist = [], []

    for modid in range(Nnew):
        osc, osckey = None, None
        for ll in available_lvalues:
            matrix = newfreqs[ll]
            nmin, _ = nranges[ll]

            # Construct osc for this l
            fres = np.zeros((2, matrix.shape[1]), dtype=float)
            fres[0][:] = matrix[0][:, modid]
            fres[1][:] = matrix[1][:, modid]

            # Construct osckeys for this l
            keys = np.zeros((2, matrix.shape[1]), dtype=int)
            keys[0][:] = ll
            keys[1][:] = np.arange(matrix.shape[1]) + nmin

            # Create or stack list
            if type(osc) != np.ndarray:
                osc = fres
                osckey = keys
            else:
                osc = np.hstack((osc, fres))
                osckey = np.hstack((osckey, keys))

        # Remove nan modes
        nanmask = np.isnan(osc[0][:])
        osc = osc[:, ~nanmask]
        osckey = osckey[:, ~nanmask]

        osclist.append(osc)
        osckeylist.append(osckey)

    return osckeylist, osclist


# ======================================================================================
# Management of header and weights
# ======================================================================================
def update_header(outfile, basepath, headvars):
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
    None
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
                    if libitem["IntStatus"][()] >= 0:
                        values[n] = libitem[var][0]
            del outfile[headpath]
            outfile[headpath] = values.tolist()
        else:
            del outfile[headpath]
            outfile[headpath] = [b"Interpolated"] * ltracks


def write_header(grid, outfile, basepath):
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
    None
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


def recalculate_param_weights(outfile, basepath):
    """
    Recalculates the weights of the tracks/isochrones, for the new grid.
    Tracks not transferred from old grid has IntStatus = -1.

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
    None
    """
    isomode = False if "grid" in basepath else True
    headvars = outfile["header/active_weights"][()]

    # Collect the relevant tracks/isochrones
    mask = []
    names = []
    for nogroup, (gname, group) in enumerate(outfile[basepath].items()):
        # Determine which tracks are actually present
        for name, libitem in group.items():
            mask.append(libitem["IntStatus"][()])
            names.append(os.path.join(gname, name))
    mask = np.where(np.array(mask) >= 0)[0]
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


def recalculate_weights(outfile, basepath, sobnums, extend=False, debug=False):
    """
    Recalculates the weights of the tracks/isochrones, for the new grid.
    Tracks not transferred from old grid has IntStatus = -1.

    Parameters
    ----------
    outfile : hdf5 file
        New grid file to write to.

    basepath : str
        Path in the grid where the tracks are stored. The default value given in
        parent functions applies to standard grids of tracks.

    sobnums : array
        Sobol numbers used to generate new base

    extend : bool
        Whether the old tracks have been preserved, for which the Sobol numbers
        needs to be recovered.

    Returns
    -------
    None
    """
    # Collect the relevant tracks/isochrones
    IntStatus = []
    names = []
    for gname, group in outfile[basepath].items():
        # Determine which tracks are actually present
        for name, libitem in group.items():
            IntStatus.append(libitem["IntStatus"][()])
            names.append(os.path.join(gname, name))

    # Reconstruct approximate Sobol numbers for original tracks
    if extend:
        index = np.where(np.array(IntStatus) == 1)[0]
        bpars = outfile["header/pars_sampled"]
        # Collect basis parameters
        base = np.zeros((len(index), len(bpars)))
        gid = 0
        lid = 0
        for gname, group in outfile[basepath].items():
            iter = zip(IntStatus[gid : gid + len(group)], group.items())
            for i, (ist, (_, libitem)) in enumerate(iter):
                if ist != 1:
                    continue
                for j, par in enumerate(bpars):
                    base[lid, j] = libitem[par][0]
                lid += 1
            gid += len(group)

        for i, par in enumerate(bpars):
            mm = [min(base[:, i]), max(base[:, i])]
            base[:, i] -= mm[0]
            base[:, i] /= mm[1] - mm[0]

        sobnums = np.vstack((base, sobnums))

    mask = np.where(np.array(IntStatus) >= 0)[0]
    active = np.asarray(names)[mask]

    # Import necessary packages only used here
    import bottleneck as bn

    # Use IntStatus mask, read key numbers
    sobnums = sobnums[mask, :]
    ntracks = sobnums.shape[0]
    ndim = sobnums.shape[1]

    # Generate oversampled grid
    osfactor = 100
    osntracks = osfactor * ntracks
    osbase = sobol_wrapper(ndim, osntracks, 2, debug=debug)

    # For every track in the grid, gather all the points from the
    # oversampled grid which are closest to the track at hand
    npoints = np.zeros(ntracks, dtype=int)
    for i in range(osntracks):
        diff = osbase[i, :] - sobnums
        distance2 = bn.nansum(diff**2, axis=-1)
        nclosest = bn.nanargmin(distance2)
        npoints[nclosest] += 1

    # Transform to volume weight per track
    weights = npoints / bn.nansum(npoints)

    # Write the weights
    for i, name in enumerate(active):
        weight_path = os.path.join(basepath, name, "volume_weight")
        try:
            outfile[weight_path] = weights[i]
        except:
            del outfile[weight_path]
            outfile[weight_path] = weights[i]

    # Write the active weights as only volume
    del outfile["header/active_weights"]
    outfile["header/active_weights"] = ["volume"]


# ======================================================================================
# Miscellaneous helper functions
# ======================================================================================
def lowest_l0(grid, basepath, track, selmod):
    """
    Determine the lowest n of l=0 frequency modes present along the whole track.

    Parameters
    ----------
    grid : hdf5 file
        Inputted grid
    basepath : str
        Path in the grid where the tracks are stored.
    track : str
        The track to extract from.
    selmod : boolean numpy array
        The selected models within the track to consider

    Returns
    -------
    maxofmin : int
        Lowest n of mode present in all models
    """
    keypath = os.path.join(basepath, track, "osckey")
    min_n_l0 = np.zeros((sum(selmod)))
    for i, osckey in enumerate(grid[keypath][()][selmod]):
        min_n_l0[i] = min(osckey[1][osckey[0] == 0])
    maxofmin = max(min_n_l0)
    return maxofmin


def get_l0_freqs(track, selmod, N):
    """
    Extract the frequency along the track of the given (n=N,l=0)
    frequency mode.

    Parameters
    ----------
    track : hdf5 group
        All data of the track being processed
    selmod : boolean numpy array
        The selected models within the track to consider
    N : int
        Lowest n of mode present in all models across enveloping tracks

    Returns
    -------
    freqs : numpy array
        Frequencies of the given mode along the track
    """
    allkeys = track["osckey"][()][selmod]
    alloscs = track["osc"][()][selmod]
    freqs = np.zeros((sum(selmod)))
    for i, (key, osc) in enumerate(zip(allkeys, alloscs)):
        ind = np.where(key[1][key[0] == 0] == N)[0]
        freqs[i] = osc[0][key[0] == 0][ind]
    return freqs
