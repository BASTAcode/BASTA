.. _running:

Running BASTA
=============

Two steps to running BASTA
--------------------------

The following tutorial is a general introduction to fitting observational data and determining stellar properties with
BASTA. It is meant as a guide to get started with simple fits based of various sets of input, and should not be
considered as the default settings for producing scientific results. We encourage the users to explore the options
offered by the code and build their input files according to their needs. As a final remark, we note that BASTA is
designed to fit any input given without ensuring that the solution is physically viable. In other words, it is up to
the user to correctly interpret the results of any fit done with the code.

The input to the code is given using an ``.xml`` file containing all the relevant information to perform the fit. A
python routine to produce these ``.xml`` files in a semi-automatic way is included in the code repository and can be
found in ``${BASTADIR}/examples/create_inputfile.py``. The basic procedure to run BASTA is reduced to the following
steps.

    * **Create the input file**

    The following commands will create a file named ``input-example.xml`` in the directory ``${BASTADIR}/examples/``.
    The user can modify the name of the file and output directory at will in the
    :py:meth:`create_inputfile.define_input` routine of the example ``create_inputfile.py`` file provided with the code.

    .. code-block:: bash

        cd ${BASTADIR}
        source venv/bin/activate
        cd examples/
        python create_inputfile.py

    * **Run BASTA**

    Once the ``input-example.xml`` has been correctly created, BASTA is simply run as follows (with the virtual
    environment activated):

    .. code-block:: bash

        BASTArun input-example.xml

The output of the fit is located in ``${BASTADIR}/examples/output/`` and we encourage the user to inspect it and ensure
that BASTA is correctly running while getting familiar with the type of output and figures produced.

Common blocks
-------------

Regardless of the type of fit, a standard I/O block must be specified containing the name of the ``.xml`` file to be
created, the grid of models to be used, and the output directory for the results. This comprises Block 1 of
:py:meth:`create_inputfile.define_input`:

.. code-block:: python

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # BLOCK 1: I/O
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Name of the XML input file to produce
    xmlfilename = "input_myfit.xml"

    # The path to the grid to be used by BASTA for the fitting.
    define_io["gridfile"] = "grids/mygrid.hdf5"

    # Where to store the output of the BASTA run
    define_io["outputpath"] = "output"

    # Location of the input file with the star(s) to be fitted and the columns included
    define_io["asciifile"] = "data/myfile.ascii"
    define_io["params"] = (
        "starid",
        "Teff",
        "Teff_err",
        "FeH",
        "FeH_err",
        "logg",
        "logg_err",
    )

Note that BASTA uses the `numpy.genfromtxt <https://numpy.org/doc/stable/reference/generated/numpy.genfromtxt.html>`_
function to read the input ascii file ``myfile.ascii``, allowing the presence of additional columns that will not be
used by the code as long as the appropriate number of entries is giving in ``define_io["params"]``.

The other common blocks to all fits are the ones corresponding to the output and plotting controls, which includes the
following features:

.. code-block:: python

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # BLOCK 3: Output control
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # A list of quantities to output. Will be printed to the log of each individual star
    # and stored in the output/results file(s).
    define_output["outparams"] = ("Teff", "FeH", "radPhot", "massfin", "age")

    # Name of the output file containing the results of the fit in ascii format.
    define_output["outputfile"] = "results.ascii"

    # A dump of the statistics (chi2, logPDF) for all models in the grids can be saved
    # to a .json file.
    define_output["optionaloutputs"] = True

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # BLOCK 4: Plotting control
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Corner plots of posteriors. Specify a list of parameters to plot.
    define_plots["cornerplots"] = define_output["outparams"]

    # BASTA can produce a Kiel diagram (Teff vs logg) with the observations and the
    # model points from the grid. The latter will be color coded based on the fitting
    # parameters and their uncertainties/constraints.
    define_plots["kielplots"] = True

Note that the final entry in Block 4 ``define_plots["freqplots"] = echelle`` has been omitted as it is only relevant
when fitting :ref:`example_freqs`. In the above example, the same quantities being output to ``results.ascii`` are
included in the corner plot, but these can be specified independently.

**Important** The summary statistics for all stars included in ``myfile.ascii`` will be written to ``results.ascii``,
while figures and details of the run for each individual target will be stored as., ``starid_XXX.png`` and
``starid.json``. If another run is made for the same stars varying some of the fit parameters, it **must** be stored
in a different folder otherwise BASTA will overwrite the previous output.
