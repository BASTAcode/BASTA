.. _controls_io:

I/O controls
************

The following lists all possible control options for the I/O control block in the :py:func:`define_input`
function in the ``create_inputfile.py`` scripts. To see the recommended/default usage of these
controls for specific fitting cases, see the :ref:`examples <examples>` section, or the provided
example scripts ``BASTA/examples/xmlinput/create_inputfile_*.py``.

.. _Name of input file:

Name of input file
==================
.. code-block:: python

    xmlfilename = "input_myfit.xml"

Name and location of the produced input file to be run by BASTA. The only requirement
is for it to end on ``xml``. From there the user can freely define name and location
of the file, e.g. when producing multiple input file for different frequency fitting methods,
whereby the user might want to store them as ``freqfits/input03_r012.xml``.

Name of grid file
=================
.. code-block:: python

    define_io["gridfile"] = os.path.join(BASTADIR, "grids", "Garstec_16CygA.hdf5")

Name and location of the grid of stellar models to fit the observed stars to. It must be provided
as a :py:func:`hdf5` file, otherwise there are no requirements on location or name.

The default points to the standard location of grids downloaded
using the :py:func:`BASTAdownload` command (see :ref:`grids`), that being the ``grids/`` directory
within the main BASTA directory. The default grid used (``Garstec_16CygA.hdf5``) is the small grid
computed for the star 16 Cyg A, specifically for the examples given in :ref:`the examples <examples>`.

.. _controls_io_outputdir:

Output directory
================
.. code-block:: python

    define_io["outputpath"] = os.path.join("output", "myfit")

The directory into which the output from the BASTA run will be stored, relative to the location of
the :ref:`input file<Name of input file>`. If the directory does not exist, it will be created on
startup of BASTA.

.. _controls_io_paramfile:

Observed parameters file
========================
.. code-block:: python

    define_io["asciifile"] = os.path.join("data", "16CygA.ascii")

The name and location of the user-provided stellar parameters to be fitted. It must
be provided in ASCII format, readable by the
`numpy.genfromtxt <https://numpy.org/doc/stable/reference/generated/numpy.genfromtxt.html>`_
routine. Location is provided relative to the location from which the python script is run.
This constitutes the list of stars for which the defined fit is run.
For a basic example, see the ``BASTA/examples/data/16CygA.ascii`` file, for a single star.

.. code-block:: python

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

The by-column definition of which parameters are in the file. The first entry must be the
``starid``, which is the identifying string/name of the star(s). It must be unique and compatible
with the filesystem on which BASTA is run, as by-star output files uses this identifier for the
naming of the files.

The remaining names in the ``params`` tuple *must follow the order* in which they are provided in
the ASCII file, all columns must be named in the tuple, and the names must match the parameter names
in BASTA's :ref:`parameter list <controls_params>`, where
the units assumed by BASTA is also available. The associated error of a parameter must be provided
as a distinct column with the same name of the parameter followed by ``_err``.

The only exception is the large frequency separation, :math:`\Delta\nu`, which here should simply
be ``dnu``, as it has multiple purposes depending on the methods employed. If the parameter is
fitted directly, the grid value it is compared to is defined in the :ref:`list of fit parameters <controls_fit_fitparams>`.

Note that the provided parameters can exceed the parameters needed by BASTA, as it simply searches
this list for what it needs. Therefore, the user can supply a single file with all available information
for the given star(s), and use it in multiple different runs of BASTA that needs different parameters,
as long as the necessary parameters are provided.


Format options
--------------
If the user has a parameter file in a specific format, the following options can be passed to
the `numpy.genfromtxt <https://numpy.org/doc/stable/reference/generated/numpy.genfromtxt.html>`_
routine, to allow BASTA to load it.

.. code-block:: python

    define_io["delimiter"] = ","

This can be set to change the assumed delimiter between columns in the ASCII file. It is ``None``
by default, which means any consecutive whitespace act as a delimiter.

.. code-block:: python

    define_io["missingval"] = -999.999

Placeholder value to indicate missing values. It is generally advised to provide BASTA with a
complete table with bad stars removed, but using this key, missing values can be ignored.
This might be useful if a large pre-computed table is provided, where some data is not available
for all stars.

Be aware that if a :ref:`parameter to be fitted<controls_fit_fitparams>` is missing, the star will be skipped!

.. code-block:: python

    define_io["overwriteparams"] = {"dnufit": (100, 2)}

Overwrite the value and error of a given parameter, for every star. Given as dictionary entries
in the form ``{paramerer: (value, error)}``.
