.. _controls_outplots:

Output controls
=========================

In the following, an overview of the all the controls related to what will be outputed
when running BASTA is given, which corresponds to the fitting controls block in the :py:func:`define_output`
function in the ``create_inputfile.py`` scripts. To see the recommended/default usage of these
controls for specific fitting cases, see the :ref:`examples <examples>` section, or the provided
example scripts ``BASTA/examples/xmlinput/create_inputfile_*.py``.

Output file
-----------

.. code-block:: python

    define_output["outputfile"] = "results.ascii"

Name of the outputted ASCII file containing the inferred parameters (listed below here).
Will be placed in the given :ref:`output directory <controls_io_outputdir>`, and can be
any format that can be created using `numpy.savetxt <https://numpy.org/doc/stable/reference/generated/numpy.savetxt.html>`_.

.. _controls_outplots_outparams:

Outputted/inferred parameters
-----------------------------
.. code-block:: python

    define_output["outparams"] = ("Teff", "FeH", "logg", "radPhot", "massfin", "age")

List of parameters to be inferred and printed to output. Can be any parameter
contained in the grid, and listed in the :meth:`parameter list <constants.parameters>`.
These will be printed to the output file in the same order as provided. It is
recommended to **always** include the :ref:`parameters being fitted <controls_fit_fitparams>`,
to compare with the observed values.

A special keyword that can be included is ``"distance"``, which will try to
infer the distance to the star given observed magnitudes and coordinates, as
shown in :ref:`the example <example_dist_estimate>` or explained in
:ref:`the method section <methods_general_distance>`.

Optional outputs
----------------
.. code-block:: python

    define_output["optionaloutputs"] = True

Will produce a dump (`.json` file) of the :math:`\chi^2` and logarithmic likelihood of
each model considered in the fit (not excluded by :ref:`priors <controls_fit_priors>`),
by default ``False``. This can be read into python using :meth:`filio.load_selectedmodels`,
and used to determine and compare the N'th best fitting model, or similar statistics.


Outputted statistics
--------------------
By default, BASTA reports/outputs the median, 16th and 84th quantiles of the
posterior distribution of the given parameter. However, should the user want
to change this (e.g. to compare against other methods), it is done through
the following options.

.. code-block:: python

    define_output["centroid"] = "mean"
    define_output["uncert"] = "std"

The ``centroid`` can be changed between reporting the ``median`` of the distribution
(default) and reporting the ``mean`` value. The unceartainty (``uncert``) can be changed
between reporting the ``quantiles`` (default) or the standard deviation (``std``).

Plotting controls
=================

Corner plot
-----------
.. code-block:: python

    define_plots["cornerplots"] = define_output["outparams"]

The list of parameters to display the posterior distributions and correlations of in
a corner diagram. This is typically set to the same as the :ref:`outputted parameter <controls_outplots_outparams>`,
but can be set with a separate tuple of parameters. If ``"distance"`` is present in
the list/tuple, an additional corner diagram with the distance-related parameters is
produced.


Kiel diagram
------------
.. code-block:: python

    define_plots["kielplots"] = True

Toggle for outputting a Kiel (HR) diagram of the resulting fit. This displays the
tracks/isochrones considered in the fit, and overlays the observed parameters using
different colours, to give a visual representation of the convergence of observed
parameters across the models.

Individual frequencies plots
----------------------------
.. code-block:: python

    define_plots["freqplots"] = False



Plot format
-----------
.. code-block:: python

    define_plots["plotfmt"] = "pdf"
