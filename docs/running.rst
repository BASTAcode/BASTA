.. _running:

Running BASTA
#############

Two steps to running BASTA
**************************

The following tutorial is a general introduction to fitting observational data and determining stellar properties with
BASTA. It is meant as a guide to get started with simple fits based of various sets of input, and should not be
considered as the default settings for producing scientific results. We encourage the users to explore the options
offered by the code and build their input files according to their needs. As a final remark, we note that *BASTA is designed to fit any input given without ensuring that the solution is physically viable*. In other words, it is *up to the user to correctly interpret the results of any fit done with the code*.

The input to the code is given using an ``.xml`` file containing all the relevant information to perform the fit.

A Python routine to produce such ``.xml`` files in a semi-automatic way is included in the code repository and can be found in ``BASTA/examples/create_inputfile.py`` if you cloned the code from GitHub or installed the full examples suite with `BASTAexamples full`. You can also obtain this file with `BASTAexamples simple`. This is meant as a basic version just to check that the code is running. It also serves as a template with all options documented. For examples of 'real' fits, have a look at the examples sections in this tutorial (:ref:`examples`).

The basic procedure to run BASTA can be reduced to the two following steps:

1. Create the input file
========================

The following commands will create a file named ``input_myfit.xml`` in the directory ``BASTA/examples/``.

.. code-block:: bash

    cd BASTA
    source bastaenv/bin/activate
    cd examples/
    python create_inputfile.py

2. Run BASTA
============

Once the ``input_myfit.xml`` has been correctly created, BASTA is simply run as follows (with the virtual environment activated):

.. code-block:: bash

    BASTArun input_myfit.xml

The output of the fit is located in ``BASTA/examples/output/myfit`` and we encourage the user to inspect it and ensure that BASTA is correctly running while getting familiar with the type of output and figures produced.


Common blocks
*************

Regardless of the type of fit, a standard I/O block must be specified containing the name of the ``.xml`` file to be
created, the grid of models to be used, and the output directory for the results. Additionally, an ascii file with the parameters of the star(s) to be fitted must be given (more details in the full file). The following comprises (most of) Block 1 of :py:meth:`create_inputfile.define_input` (note that some comments have been removed compared to the full file):

.. code-block:: python

    # ==================================================================================
    # BLOCK 1: I/O
    # ==================================================================================
    # Name of the XML input file to produce
    xmlfilename = "input_myfit.xml"

    # The path to the grid to be used by BASTA for the fitting.
    define_io["gridfile"] = os.path.join(BASTADIR, "grids", "Garstec_16CygA.hdf5")

    # Where to store the output of the BASTA run
    define_io["outputpath"] = os.path.join("output", "myfit")

    # Location of the input file with the star(s) to be fitted and the columns included
    define_io["asciifile"] = os.path.join("data", "16CygA.ascii")
    define_io["params"] = (
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

Note that BASTA uses the `numpy.genfromtxt <https://numpy.org/doc/stable/reference/generated/numpy.genfromtxt.html>`_
function to read the input ascii file, allowing the presence of additional columns that will not be
used by the code as long as the appropriate number of entries is given in ``define_io["params"]``.

The other common blocks to all fits are the ones corresponding to the fitting, output, and plotting controls. The main components (again some comments and auxiliary things are removed compared to the file):

.. code-block:: python

    # ==================================================================================
    # BLOCK 2: Fitting control
    # ==================================================================================
    # A list of the parameters to fit must be given to BASTA in a tuple.
    define_fit["fitparams"] = ("Teff", "FeH", "logg")

    # ==================================================================================
    # BLOCK 3: Output control
    # ==================================================================================
    # A list of quantities to output.
    define_output["outparams"] = ("Teff", "FeH", "logg", "radPhot", "massfin", "age")

    # Name of the output file containing the results of the fit in ascii format.
    define_output["outputfile"] = "results.ascii"

    # A dump of the statistics (chi2, logPDF) for all models in the grids can be saved
    # to a .json file.
    define_output["optionaloutputs"] = True

    # ==================================================================================
    # BLOCK 4: Plotting control
    # ==================================================================================
    # Corner plots of posteriors. Specify a list of parameters to plot.
    define_plots["cornerplots"] = define_output["outparams"]

    # BASTA can produce a Kiel diagram (Teff vs logg) with the observations and the
    # model points from the grid. The latter will be color coded based on the fitting
    # parameters and their uncertainties/constraints.
    define_plots["kielplots"] = True


Please note that Block 2 contains five sub-blocks with different controls depending on the specific type of fit. Also note that  in the above example, the same quantities being output to ``results.ascii`` are included in the corner plot, but these can be specified independently. Finally, some options have been omitted for clarity, e.g., the entry ``define_plots["freqplots"]`` in Block 4 as it is only relevant when fitting :ref:`example_freqs`.

**Important** The summary statistics for all stars included in the input ascii will be written to ``results.ascii``,
while figures and details of the run for each individual target will be stored as., ``starid_XXX.png`` and
``starid.json``. If another run is made for the same stars varying some of the fit parameters, it **must** be stored
in a different folder otherwise BASTA will overwrite the previous output.
