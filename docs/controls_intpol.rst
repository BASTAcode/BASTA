.. _controls_intpol:

Interpolation controls
**********************

In the following, an overview of all control options related to the BASTA
interpolation routine is given, which corresponds to the interpolation controls
block in the :py:func:`define_input` function in the ``create_inputfile.py``
scripts. To see the recommended/default usage of these
controls for specific fitting cases, see the :ref:`examples <examples>` section, or the provided
example scripts ``BASTA/examples/xmlinput/create_inputfile_*.py``.
For an overview of the routine, see the :ref:`method section <methods_intpol>`.

Overall, the interpolation routine is a large, almost separate, module for BASTA.
It improves the resolution of the grid of models by interpolating between/along
tracks. This allows for the posterior distributions to be smoother, and gaps in
the parameter space to be filled. The routine is however based on many
user-defined choices, and has not been tested in extensive details. It is thus
an **experimental feature**, and should be used with care.

This feature is toggled on/off with the boolean

.. code-block:: python

    interpolation = False


which by default is turned off. The following description is based
around interpolating a grid of stellar evolutionary tracks. The
method is however also applicable to grids of stellar isochrones, whereby
the words tracks and isocohrones are interchangeable in the descriptions.


Sub-grid from limits
====================
.. code-block:: python

    define_intpol["intpolparams"]["limits"] = {
        "Teff": {"sigmacut": 1},
        "FeH": {"abstol": 0.2},
    }

The part of the original grid to be interpolated within (sub-grid),
is defined through these limits. The application/usage is identical to
the :ref:`flat priors <controls_fit_priors>`, and it is generally
recommended using a wider prior/limit here than when fitting, to ensure
the interpolated grid actually covers the desired parameter range of the
following fit.


Definition of applied method
============================
.. code-block:: python

    define_intpol["intpolparams"]["method"] = {
        "case": "combined",
        "construction": "bystar",
    }

Defines the overarching methods by which interpolation should be applied.
The keyword ``case`` controls which interpolation method should be used
among the following

* ``along``: Interpolation purely along the tracks of the sub-grid, to increase the resolution within the tracks, but not place new between the tracks.
* ``across``: Interpolation purely across the tracks of the sub-grid. New tracks will be placed in between the originals, while the resolution along the track will attempt to mimic the original tracks used for interpolation.
* ``combined``: Interpolation will be applied both across and along the tracks in the sub-grid, in a combined approach. When this mode is selected, it therefore needs the control blocks of both methods to be provided.

The keyword ``construction`` controls how interpolation is applied for a
BASTA run with multiple stars. If set to ``"bystar"``, a sub-grid based on
provided limits will be determined and interpolated within for each star
being fitted. If set to ``"encompass"``, a single sub-grid based on the
proivded limits applied to the full range of observed parameters across
all inputted stars will be determined and interpolated within, and then
used when fitting all of the stars. For example, if an absolute tolerance
limit of 300K is set for a sample of stars ranging from 5500K to 6100K in
effective temperature, the sub-grid will span fram 5350K to 6250K.

Name of interpolated grid
=========================

.. code-block:: python

    define_intpol["intpolparams"]["name"] = "testgrid"

Optional control of the name of the outputted interpolated sub-grid.
The name will always be preceded by ``intpol_``, and if the method of
construction is ``"bystar"``, the identifier of the star will be appended
to the name. If not provided, the name will be the same as the original
grid.

Across tracks resolution
========================

.. code-block:: python

    define_intpol["intpolparams"]["gridresolution"] = {
        "scale": 1.5,
        "baseparam": "rhocen",
        "extend": False,
    }

Control group for how to scale resolution across the tracks.

The ``scale`` indicates the minimum multiplicative factor by which the
number of tracks should be increased. For example, if the sub-grid
contains 10 tracks, and ``scale`` is set to 1.5, *at least* 15 new tracks
will be interpolated to.

The ``baseparam`` defines what quantity along the tracks should be used
as a base for interpolation. This should be a continuous, monotonic function
that scales with evolutionary phase. Preliminary testing determined central
density (:math:`\rho_{\text{cen}}`, ``"rhocen"``) to generally be a good choice,
central hydrogen abundance (:math:`X_{\text{cen}}`, ``"xcen"``) to be good for
exclusively main-sequence grids, and the large frequency separation (:math:`\Delta\nu`,
``"dnufit"``) to be good when interpolating individual frequencies.

The ``extend`` key toggles whether the original tracks in the sub-grid should be
copied to the interpolated grid. This is by default ``False``, as this interferes
with the desired homogeneity of the distribution of models in the grid. For the
``combined`` case of interpolation they will also have a different resolution along
the tracks. However, the impact on the derived posterior is alleviated by the
weighting of models/tracks according to their occupied volume in the parameter space,
and thus including the original tracks should not cause discrepancies, and allows
for a denser occupation of the parameter space.


Along tracks resolution
=======================
.. code-block:: python

    define_intpol["intpolparams"]["trackresolution"] = {
        "param": "dnufit",
        "value": 0.01,
        "baseparam": "rhocen",
    }

Control group for how to scale resolution along the tracks.

The ``param`` key defines the parameter for which a certain resolution is desired,
and can be any continuous, monotonic parameter. The ``value`` key defines the target
resolution in this parameter, in the units provided in the :ref:`parameter list <controls_params>`.

The ``baseparam`` key defines what parameter is used as a base for interpolation.
A transformation between this and the above resolution parameter is then made, as
the resolution parameter might not be an appropriate choice to use as a base for
interpolation (see :ref:`description in methods <methods_intpol>`).
