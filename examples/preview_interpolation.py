"""
Preview the change in resolution and final distribution of tracks/isochrones
for running BASTA with interpolation.
"""

from basta.downloader import get_basta_dir

# Definition of the path to BASTA, just in case you need it
BASTADIR = get_basta_dir()


def define_preview(define_input, define_along, define_across):
    """
    Define information for previewing an interpolation run. Fill the relevant
    dictionaries and pass it to the automatic routines. Full explanation of options
    is also given in `create_inputfile.py`, block 5.
    """

    # ==================================================================================
    # BLOCK 1: Grid and limits for subgrid
    # ==================================================================================
    # The path to the grid to be used by BASTA
    define_input["gridfile"] = os.path.join(BASTADIR, "grids", "Garstec_16CygA.hdf5")

    # If the inputted grid is BaSTI isochrones, specify the science case. See
    # `create_inputfile.py` block 2c for available cases by standard
    # define_input["odea"] = (0, 0, 0, 0)

    # Construction of interpolated grid(s). There are to options:
    # - "bystar" for an interpolated grid for each star in the input file
    # - "encompass" for a single grid spanning all of the stars
    define_input["construction"] = "bystar"

    # Define limits of subgrid to be interpolated, within the full grid.
    # Primarely to avoid spending a large amount of time interpolating in regions that
    # are not close to the fitted star(s).
    define_input["limits"] = {
        "Teff": {"abstol": 150},
        "FeH": {"abstol": 0.2},
        "dnufit": {"abstol": 8},
    }

    # Take ascii-file input in order to do "abstol" and "sigmacut" in limits
    # If no limits are given in terms of "abstol" or "sigmacut" this can be ignored
    define_input["asciifile"] = os.path.join("data", "16CygA.ascii")
    define_input["params"] = (
        "starid",
        "RA",
        "DEC",
        "numax",
        "numax_err",
        "dnu",
        "dnu_err",
        "Teff",
        "Teff_err",
        "FeH",
        "FeH_err",
        "logg",
        "logg_err",
    )

    # Output-path
    outpath = os.path.join("output", "preview_interp_MS")

    # ==================================================================================
    # BLOCK 2: Controls for along interpolation
    # ==================================================================================
    # To compare the current resolution along a track/isochrone with an inputted value,
    # switch this option on
    along_interpolation = True
    if along_interpolation:
        # Resolution parameters to preview current resolution for.
        # Input any list of parameters, "freqs" for viewing the spacing in the l=0 modes
        # in the models. Compares to the inputted value.
        define_along["resolution"] = {
            "freqs": 0.5,
            # "dnufit": 0.04,
            # "age": 20,
        }

        # Location, name and format of the outputted figure (histogram)
        # Use either .png (fast) or .pdf (high resolution) format, .png by default
        define_along["figurename"] = os.path.join(
            outpath, "interp_preview_along_resolution.pdf"
        )

    # ==================================================================================
    # BLOCK 3: Controls for across interpolation
    # ==================================================================================
    # To compare the current gridresolution with what would be obtained given the input
    across_interpolation = True
    if across_interpolation:
        # Definition of the increase in resolution. "scale" for Sobol sampling with the
        # given multiplicative increase in the number of tracks/isochrones. For
        # Cartesian sampling, define the increase in number of tracks between current
        # points, e.g. "FeHini": 2 will result three times the number of tracks
        define_across["resolution"] = {
            "scale": 6,
        }

        # Location, name and format of the outputted figure (histogram)
        # Use either .png (fast) or .pdf (high resolution) format, .png by default
        define_across["figurename"] = os.path.join(
            outpath, "interp_preview_across_resolution.pdf"
        )

    # Done! Nothing more to specify.
    return define_input, define_along, define_across


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# ~~~~~~~~~~~~~~~~~~~~~~~~ AUTOMATED PART OF THE SCRIPT BELOW ~~~~~~~~~~~~~~~~~~~~~~~~~
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
def determine_l0_spacing(grid, bpath, index, track):
    """
    Determine the separation of l=0 modes in the given track, and only for
    the modes that are continuous throughout the track. It currently differs from
    the full interpolation scheme, that also accounts for modes appearing and
    dissapearing within the subgrid, but works as an overview.

    Parameters
    ----------
    grid : h5py file
        Input grid

    bpath : str
        Base path for accessing the tracks in the grid

    index : list
        Booleans of the indices in the track within the subgrid

    track : str
        Path to the track parameters

    Returns
    -------
    diffs : list
        Absolute spacings between l=0 modes in the subgrid

    """
    # Extract full list of mode keys and frequencies
    index2d = np.array(np.transpose([index, index]))
    fullosc = grid[os.path.join(bpath, track, "osc")][index2d].reshape((-1, 2))
    fullosckey = grid[os.path.join(bpath, track, "osckey")][index2d].reshape((-1, 2))

    # Prepare dat collection of restructured data
    Ntrack = len(fullosc)
    nvals = []
    fvals = []
    dnu = []

    # For each model, extract every radial order of l=0 modes
    for modid in range(Ntrack):
        osckey, osc = su.get_givenl(
            l=0,
            osc=su.transform_obj_array(fullosc[modid]),
            osckey=su.transform_obj_array(fullosckey[modid]),
        )
        nvals.append(osckey[1])
        fvals.append(osc[0])

    # Extract modes that are continuous
    trialn = list(range(40))
    goodn = []
    for testn in trialn:
        nstatus = []
        for modid in range(Ntrack):
            if testn in nvals[modid]:
                nstatus.append(1)
        if len(nstatus) == Ntrack:
            goodn.append(testn)

    # Determine and extract all diffs in the continuous frequency modes
    diffs = []
    for inn, nn in enumerate(goodn):
        fvector = [fvals[q][np.where(nvals[q] == nn)[0][0]] for q in range(Ntrack)]
        diffs.extend(abs(np.diff(fvector)))
    return diffs


def test_across_interpolation(
    grid, selectedmodels, acopt, bpath="grid/", outname="across.png"
):
    """
    Runs a test of a set of input parameters for interpolation across in a grid

    Parameters
    ----------
    gridname : h5py file
        Input grid

    selectedmodels : dict
        Selected models from the original grid, forming the subgrid

    acopt : dict
        The given options for across interpolation

    baseparams : list
        Parameters forming the base in the grid. If none provided, extract them
        from the grid as all varied parameters.

    bpath : str
        Base path to the tracks in the grid

    outname : str
        Name and destination of the outputted figure

    Returns
    -------
    None
    """
    resolution = acopt["resolution"]
    if "baseparams" in acopt:
        baseparams = acopt["baseparams"]
    else:
        baseparams = []

    if "scale" in resolution:
        assert resolution["scale"] > 1.0
        sobol = resolution["scale"]

    if len(baseparams) == 0:
        baseparams = [par.decode("UTF-8") for par in grid["header/pars_sampled"]]

    base = np.zeros((len(selectedmodels), len(baseparams)))
    for i, name in enumerate(selectedmodels):
        for j, bpar in enumerate(baseparams):
            parm = grid[os.path.join(bpath, name)][bpar][0]
            base[i, j] = parm
    tri = spatial.Delaunay(base)
    newbase, _, _ = iac._calc_sobol_points(base, baseparams, tri, sobol, outname)
    success = ip.base_corner(baseparams, base, newbase, tri, sobol, outname)


def test_along_interpolation(
    grid, selectedmodels, alopt, bpath="grid/", outname="along.png"
):
    """
    Plot histograms of the distribution of spacing of the resolution parameter(s)
    along the tracks in the subgrid, in order to get and idea of the increase in
    resolution it will provide.

    Parameters
    ----------
    grid : h5py file
        Input grid

    selectedmodels : dict
        Selected models from the original grid, forming the subgrid

    alopt : dict
        The given options for along interpolation

    bpath : str
        Location of tracks/isochrones in gridfile

    outname : str
        Name and destination of the outputted figure

    Returns
    -------
    None
    """
    # Load standard plotstyle
    plt.style.use(os.path.join(BASTADIR, "src/basta/plots.mplstyle"))
    freqres = ["freq", "freqs", "frequency", "frequencies", "osc"]

    # Unpack options
    resolution = alopt["resolution"]

    # Start list for collecting spacing results, l=0 frequency is special
    labpars = ["age"]
    output = {}
    freqs = False
    for i, par in enumerate(resolution):
        output[par] = []
        if par not in freqres:
            labpars.append(par)
        else:
            freqs = True
            inserti = i + 1

    # Loop over each track, and extract spacing for each parameter
    for nt, track in enumerate(selectedmodels):
        for par in resolution:
            index = selectedmodels[track]
            if par not in freqres:
                vals = grid[os.path.join(bpath, track, par)][index]
                output[par].extend(abs(np.diff(vals)))
            else:
                # l=0 spacing handled in separate routine
                diffs = determine_l0_spacing(grid, bpath, index, track)
                output[par].extend(diffs)

    # Get parameter labels and colors for plot
    _, parlab, _, parcol = bc.parameters.get_keys(labpars)
    if freqs:
        parlab.insert(inserti, r"$\nu_{\mathrm{l}=0}\,(\mu\mathrm{Hz})$")
        parcol.insert(inserti, "#EE6677")
    parlab.pop(0)
    parcol.pop(0)

    # Plotting
    K = len(resolution)
    fig, axes = plt.subplots(K, 1, figsize=(6, K * 3))
    for i, par in enumerate(resolution):
        if K == 1:
            ax = axes
        else:
            ax = axes[i]

        res = resolution[par]
        col = parcol[i] if parcol[i] != "#DDDDDD" else "#88CCEE"
        # Plot within 3.5 standard deviations, to get manageable ranges
        x = output[par]
        md = np.mean(x)
        std = np.std(x)
        xlim = [max(md - 3.5 * std, 0), md + 3.5 * std]
        # Make sure desired resolution is in plot
        if res < xlim[0]:
            delta = xlim[1] - res
            xlim[0] = max(res - delta * 0.1, 0)
        elif res > xlim[1]:
            delta = res - xlim[0]
            xlim[1] = res + delta * 0.1

        # Determine bins as in cornerplots
        bins = np.histogram_bin_edges(x, bins="auto")
        ax.hist(x, bins=bins, color=col, label=r"Grid")

        # Plot the inputted 'desired' resolution
        ylim = ax.get_ylim()
        ax.plot([res, res], list(ylim), color="k", label=r"Desired")

        # Set labelling stuff
        ax.set_xlim(xlim)
        ax.set_ylim(ylim)
        ax.set_yticks([])
        ax.set_xlabel(parlab[i])
        ax.legend(title=r"Resolution:", title_fontsize=13)
    fig.tight_layout()
    fig.savefig(outname)
    plt.close()


def _unpack_input(inputinf):
    """
    Determines the limits "bystar" while for encompass, replaces list by
    a single entry. Also provides prefix for plots, given "bystar" method.

    Parameters
    ----------
    inputinf : dict
        Inputted generel info

    Returns
    -------
    limits_list : dict
        Limits "bystar" or for all in case of encompass construction

    names : dict
        Prefix for plots, depending on construction

    """
    construct = inputinf["construction"]
    # First, read input
    limopts = np.array(["min", "max", "abstol", "sigmacut"])
    read_asc = False
    limits = {}
    for par, dlim in input_info["limits"].items():
        parlim = [-np.inf, np.inf, np.inf, np.inf]
        for lim, val in dlim.items():
            # Check and find the limit as an allowed option
            try:
                opt = np.where(limopts == lim)[0][0]
            except:
                errmsg = "Limit given as '{0}', but must be given as '{1}'!"
                raise KeyError(errmsg.format(lim, "', '".join(limopts)))

            # If abstol or sigmacut requested, turn on ascii read
            if opt > 1:
                read_asc = True
            parlim[opt] = val

        # Collect inputted values
        limits[par] = parlim

    # If required, read the asc file
    if read_asc:
        asciifile = inputinf["asciifile"]
        params = inputinf["params"]
        inp = np.genfromtxt(asciifile, dtype=None, names=params, encoding=None)
        params = np.asarray(params)
        if inp.ndim == 0:
            inp = inp.reshape(1, -1)[0]

    # If by star, and star-dependent limits determine limits for each star
    limits_list = {}
    names = {}
    if construct == "bystar" and read_asc:
        for i in range(len(inp)):
            starlimits = {}
            for par, vals in limits.items():
                gpar = "dnu" if "dnu" in par else par
                minv, maxv, abst, nsig = vals
                # Check abstol
                if abst != np.inf:
                    starval = xu._get_param(inp[i], params, gpar)
                    if starval - abst / 2.0 > minv:
                        minv = starval - abst / 2.0
                    if starval + abst / 2.0 < maxv:
                        maxv = starval + abst / 2.0
                # Check sigmacut
                if nsig != np.inf:
                    starval = xu._get_param(inp[i], params, gpar)
                    starerr = xu._get_param(inp[i], params, gpar + "_err")
                    if starval - starerr * nsig > minv:
                        minv = starval - starerr * nsig
                    if starval + starerr * nsig < maxv:
                        maxv = starval + starerr * nsig

                # Add parameter limits
                starlimits[par] = [minv, maxv]
            starid = xu._get_param(inp[i], params, "starid")
            limits_list[starid] = starlimits
            names[starid] = str(starid) + "_"
    # If overall or no asc read, do one overall limits
    elif construct == "encompass" or not read_asc:
        all_limits = {}
        for par, vals in limits.items():
            gpar = "dnu" if "dnu" in par else par
            minv, maxv, abst, nsig = vals
            if not read_asc:
                all_limits[par] = [minv, maxv]
            else:
                # Check abstol limits
                if abst != np.inf:
                    starvals = [xu._get_param(i, params, gpar) for i in inp]
                    if min(starvals) - abst / 2.0 > minv:
                        minv = min(starvals) - abst / 2.0
                    if max(starvals) + abst / 2.0 < maxv:
                        maxv = max(starvals) + abst / 2.0
                # Check sigmacut limits
                if nsig != np.inf:
                    starvals = [xu._get_param(i, params, gpar) for i in inp]
                    starerr = max(
                        [xu._get_param(i, params, gpar + "_err") for i in inp]
                    )
                    if min(starvals) - starerr * nsig > minv:
                        minv = min(starvals) - starerr * nsig
                    if max(starvals) + starerr * nsig < maxv:
                        maxv = max(starvals) + starerr * nsig

                # Add parameter
                all_limits[par] = [minv, maxv]
        # Seems odd, but works
        limits_list["all"] = all_limits
        names["all"] = ""

    return limits_list, names


if __name__ == "__main__":
    import os
    import sys
    import h5py
    import numpy as np
    import matplotlib.pyplot as plt
    from scipy import spatial
    from basta import constants as bc
    from basta import utils_seismic as su
    from basta import utils_xml as xu
    from basta import interpolation_across as iac
    from basta import interpolation_helpers as ih
    from basta import plot_interp as ip

    input_info = {}
    along_info = {}
    across_info = {}
    settings = define_preview(input_info, along_info, across_info)
    input_info, along_info, across_info = settings

    # Check that the user has actually toggled anything on
    if len(along_info.keys()) < 1 and len(across_info.keys()) < 1:
        raise KeyError("You have yet to toggle any preview on....")

    # Unpack grid access variables
    grid = h5py.File(input_info["gridfile"], "r")
    if "odea" not in input_info.keys():
        basepath = "grid/"
    else:
        odea = input_info["odea"]
        basepath = "ove={0:.4f}/dif={1:.4f}/eta={2:.4f}/alphaFe={3:.4f}/".format(
            odea[0], odea[1], odea[2], odea[3]
        )

    # Unpack subgrid related
    all_lim, all_names = _unpack_input(input_info)
    i = 0
    for star, limits in all_lim.items():
        prefix = all_names[star]
        if star == "all":
            print("Preparing...")
        else:
            print("Preparing star '{0}'...".format(star))
        print("Using the following limits for subgrid:")
        for lim, vals in limits.items():
            print("\t" + lim + ": [{0:.3f} , {1:.3f}]".format(vals[0], vals[1]))
        selectedmodels = ih.get_selectedmodels(grid, basepath, limits)

        # Run along preview
        if len(along_info.keys()) > 0:
            print("Plotting along interpolation preview...")
            figname = along_info["figurename"].split("/")[-1]
            if len(along_info["figurename"].split("/")[:-1]):
                outdir = along_info["figurename"][: -len(figname)]
                if not os.path.exists(outdir):
                    print(
                        "Output dir '{0}' does not exist, creating it...".format(outdir)
                    )
                    os.makedirs(outdir)
                outname = os.path.join(outdir, prefix + figname)
            else:
                outname = prefix + figname
            if outname[-4:] not in [".png", ".pdf"]:
                outname += ".png"
            test_along_interpolation(
                grid, selectedmodels, along_info, basepath, outname
            )
            print("Shown in " + outname + "\n")
        # Run along preview
        if len(across_info.keys()) > 0:
            print("Plotting across interpolation preview...")
            figname = across_info["figurename"].split("/")[-1]
            if len(across_info["figurename"].split("/")[:-1]):
                outdir = across_info["figurename"][: -len(figname)]
                if not os.path.exists(outdir):
                    print(
                        "Output dir '{0}' does not exist, creating it...".format(outdir)
                    )
                    os.makedirs(outdir)
                outname = os.path.join(outdir, prefix + figname)
            else:
                outname = prefix + figname
            if outname[-4:] not in [".png", ".pdf"]:
                outname += ".png"
            test_across_interpolation(
                grid, selectedmodels, across_info, basepath, outname
            )
            print("Shown in " + outname + "\n")

        if star == "all":
            print("\nAll done!\n")
        else:
            i += 1
            print(
                "\nDone with star '{0}' ({1}/{2})\n".format(
                    star, i, len(all_lim.keys())
                )
            )

# DONE !
