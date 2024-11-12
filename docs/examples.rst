.. _examples:

Examples of fits
################

*Important:* If you did not clone the code from GitHub and did not download the examples yet, please run:

.. code-block:: bash

    cd ~/BASTA
    BASTAexamples full


The following sections give practical examples of commonly used fits with BASTA, and explain the options in the code following the template given in :py:func:`define_input` in ``BASTA/examples/create_inputfile.py`` (in the following this will be denoted as :py:meth:`create_inputfile.define_input`).

The fitting examples are located in ``BASTA/examples/xmlinput/`` and a reference set of output can be found in ``BASTA/examples/reference/``, which includes all fits described in the following sections. These can be compared to the output you obtain when running the code as a further sanity check.

Overview of fitting examples:

.. toctree::
   :maxdepth: 1

   examples_global
   examples_freqs
   examples_isochrones
   examples_interpolation


Additional (non-fitting) examples
=================================

In ``BASTA/examples`` two additional files are present:

    * ``process_jsonfile.py``, which shows how to use the ``.json`` files from BASTA. These files are a full dump of the likelihood information for the full grid and can be used  to re-create e.g. posterior distributions for a completed fit without re-running BASTA. `Note:` For this example to work, one of the two fitting examples producing a ``.json`` file (either the default template or the one denoted ``_json``) must be completed prior to running this.

    * ``preview_interpolation.py``, which can be used to visualise a specific set of settings for interpolation in a grid. This is useful for deciding on limits and resolution before running the fit, as the full interpolation itself can be rather time consuming. More details are given in the :ref:`example_interp` example.
