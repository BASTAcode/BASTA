.. _grids:

Grids of models
###############

BASTA uses grids of stellar tracks or isochrones stored in hierarchical data format ``hdf5``. The list of
parameters included in our grids can be seen in the :ref:`parameter list <controls_params>`, but one should keep in mind that
not all parameters are included in all grids. As an example, isochrones do not contain individual frequencies of
oscillations while stellar tracks do, and fits to e.g., frequency ratios can only be performed with grids of stellar
tracks. If in doubt about a particular entry in the :ref:`parameter list <controls_params>` you are encouraged to contact one of
the core developers of BASTA.

A small grid is downloaded automatically if you follow the recommended installation instructions and is (as a default) stored in
``BASTA/grids/Garstec_16CygA.hdf5``. As the name suggest, this grid is built with the GARching STellar Evolution
Code around the observed parameters of the *Kepler* target 16 Cyg A. Many of the :ref:`examples` are built using
this grid, while others are constructed from larger grids that can be downloaded.

For this purpose, BASTA includes a grid download tool that can be easily accessed from the command line. After
installing the code, go to the BASTA folder and run the following:

    .. code-block:: bash

        BASTAdownload -h

and follow the instructions to download the grids. Currently the following grids are available for download:

* The validation grid of stellar tracks described in Section 6 of `The BASTA paper II <https://arxiv.org/abs/2109.14622>`_ (called `validation`)
* A complete grid of BaSTI stellar isochrones including all science cases described in the `Solar scaled paper <https://ui.adsabs.harvard.edu/abs/2018ApJ...856..125H/abstract>`_ and the `Alpha-enhanced paper <https://ui.adsabs.harvard.edu/abs/2021ApJ...908..102P/abstract>`_. (called `iso`)

Additional grids can be built by the BASTA core development team upon reasonable request. Currently we are working on
the following additions to the list of grids:

* The complete grid of BaSTI stellar tracks described in the `Solar scaled paper <https://ui.adsabs.harvard.edu/abs/2018ApJ...856..125H/abstract>`_ and the `Alpha-enhanced paper <https://ui.adsabs.harvard.edu/abs/2021ApJ...908..102P/abstract>`_.
* A tool to build your own MESA grids

Stay tuned to the repository updates if you are interested in these grids.
