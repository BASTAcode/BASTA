.. _controls:

All controls
============

The following sections give the full overview and documentation of the control options in BASTA.
For the recommended set of control options for each fitting case BASTA is used for, see instead :ref:`examples`.

The methods employed are described in the :ref:`BASTA papers<ref_refs>`, while the following is
the documentation of how the controls for these methods are adjusted. The documentation follows
the structure of the :py:func:`define_input` template function in ``BASTA/examples/create_inputfile.py``,
separated into individual sections for each control group (python dictionary) being:

.. toctree::
    :maxdepth: 1

    controls_parameters
    controls_io
    controls_fit
    controls_outplots
    controls_intpol

For examples of input files for each fitting case, with the recommended options, see the template
files ``BASTA/examples/xmlinput/create_inputfile_*.py`` (which can be obtained with the `BASTAexamples` tools, if you do not already have it).

When the control options have been set, the fit is performed (as explained in :ref:`running`) by
creating the ``xml`` input file, which is then run by BASTA. This is done using the commands

.. code-block:: bash

    source bastaenv/bin/activate
    python create_inputfile.py
    BASTArun input_myfit.xml
