.. _controls_io:

I/O controls
============

Name of input file
------------------
.. code-block:: python

    xmlfilename = "input_myfit.xml"

Name and location of the produced input file to be run by BASTA. The only requirement
is for it to end on :py:func:`xml`. From there the user can freely define name and location
of the file, e.g. when producing multiple input file for different frequency fitting methods,
whereby the user might want to store them as :py:func:`freqfits/input03_r012.xml`.

Name of grid file
-----------------
.. code-block:: python

    define_io["gridfile"] = os.path.join(BASTADIR, "grids", "Garstec_16CygA.hdf5")

Name and location of the grid of stellar models to fit the observed stars to. It must be provided
as a :py:func:`hdf5` file, otherwise there are no requirements on location or name.

The default points to the standard location of grids downloaded
using the :py:func:`BASTAdownload` command (see :ref:`grids`), that being the ``grids/`` directory
within the main BASTA directory. The default grid used (``Garstec_16CygA.hdf5``) is the small grid
computed for the star 16 Cyg A, specifically for the examples given in :ref:`examples`.

Output directory
----------------
.. code-block:: python

    define_io["outputpath"] = os.path.join("output", "myfit")
