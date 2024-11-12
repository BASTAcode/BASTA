.. _controls_outplots:

Output controls
***************

In the following, an overview of the all the controls related to what will be outputted
when running BASTA is given, which corresponds to the fitting controls block in the :py:func:`define_input`
function in the ``create_inputfile.py`` scripts. To see the recommended/default usage of these
controls for specific fitting cases, see the :ref:`examples <examples>` section, or the provided
example scripts ``BASTA/examples/xmlinput/create_inputfile_*.py``.

Output file
===========

.. code-block:: python

    define_output["outputfile"] = "results.ascii"

Name of the outputted ASCII file containing the inferred parameters (listed below here).
Will be placed in the given :ref:`output directory <controls_io_outputdir>`, and can be
any format that can be created using `numpy.savetxt <https://numpy.org/doc/stable/reference/generated/numpy.savetxt.html>`_.

.. _controls_outplots_outparams:

Output/inferred parameters
==========================
.. code-block:: python

    define_output["outparams"] = ("Teff", "FeH", "logg", "radPhot", "massfin", "age")

List of parameters to be inferred and printed to output. Can be any parameter
contained in the grid, and listed in the :meth:`parameter list <controls_params>`.
These will be printed to the output file in the same order as provided. It is
recommended to **always** include the :ref:`parameters being fitted <controls_fit_fitparams>`,
to compare with the observed values.

A special keyword that can be included is ``"distance"``, which will try to
infer the distance to the star given observed magnitudes and coordinates, as
shown in :ref:`the example <example_dist_estimate>` or explained in
:ref:`the method section <methods_general_distance>`.

Optional outputs
================
.. code-block:: python

    define_output["optionaloutputs"] = True

Will produce a dump (`.json` file) of the :math:`\chi^2` and logarithmic likelihood of
each model considered in the fit (not excluded by :ref:`priors <controls_fit_priors>`),
by default ``False``. This can be read into python using :meth:`filio.load_selectedmodels`,
and used to determine and compare the N'th best fitting model, or similar statistics.


Output statistics
=================
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

In the following, an overview of the plotting related controls are given. These
control which of the automatically generatable plots should be produced when
running BASTA.

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

.. _controls_outplots_freqplots:

Individual frequencies plots
----------------------------
.. code-block:: python

    define_plots["freqplots"] = False

Controls for which plots to be produced, from the individual frequencies and/or
quantities derived therefrom, of the best fitting model compared to the observations.
This requires the individual frequencies to be supplied (see :ref:`controls_fit_freqparams`),
and be available in the grid. If set to ``False`` none of the plots will be produced,
while ``True`` will produce *all* figures (with default choices of sequences for the
derived quantities) for each star being fitted. They are placed in the
:ref:`output directory <controls_io_outputdir>` following the syntax ``<starid>_<plotname>.<plotfmt>``.

The plots can enabled individually by instead providing a tuple with the names of
plots to be produced. The options are:

* ``echelle``: Produces two échelle diagrams of the provided observed individual frequencies against the models, one being with the surface-corrected model frequencies, and the other the uncorrected model frequencies, whereby the ``_uncorrected`` is added to the filename. Using different keys, varied versions of the échelle diagrams are produced. The options are:

   * ``echelle``: Simplest version of the diagram, as described above.
   * ``pairechelle``: Adds a line between the observed frequencies and the matched model frequency.
   * ``dupechelle``: Same as ``pairechelle``, but adds a duplicated panel, so sequences crossing the axis can be visualized in a clearer way.
   * ``allechelle``: Produces *all* the above versions.
* ``ratios``: Produces a plot of the observed frequency ratios against the best fitting model. If ratios are being fitted, it will plot the sequence being fitted. If not fitted, the default ``r01`` sequence will be plotted. Instead of ``ratios``, specific sequences can be set in the list to produce plots for specific sequences. Multiple can be defined at the same time.
* ``epsdiff``: Same as for ratios, but for the phase shift differences. Default is the ``e012`` sequence.

If ``correlations`` in the :ref:`freqparams <controls_fit_freqparams>` input is set
to ``True``, a correlation map of the individual frequencies or derived quantities
will also be produced, following the syntax ``<starid>_<plotname>_cormap.<plotfmt>``.

Plot format
-----------
.. code-block:: python

    define_plots["plotfmt"] = "pdf"

Defines the format of which figures are created. Default is ``png`` which is a
small format, so preferable when creating many figures/fitting multiple stars.
However, if high resolution/vector graphics is desirable, ``pdf`` is recommended.
Otherwise, it can be any file format compatible with
`matplotlib.pyplot.savefig <https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.savefig.html>`_.


Star identifier in plots
------------------------
.. code-block:: python

    define_plots["nameinplot"] = True

The star identifier is normally only contained in the name of plot files.
However, depending on the preferred post-processing procedure of the user,
it can be beneficial to include in the plots itself, which can be turned
on using this key. Currently, only implemented for Kiel diagram and corner plots.
