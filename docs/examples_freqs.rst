.. _example_freqs:

Individual frequencies, ratios, glitches
========================================

Using grids that include theoretically computed oscillation frequencies (see :ref:`grids`) BASTA can fit these
individual frequencies with a surface correction, as well as combination of frequencies. In the following we show
examples of the blocks that must be modified in :py:meth:`create_inputfile.define_input` to produce these types of fits.

Individual frequencies: main sequence
-------------------------------------

BASTA is shipped with individual frequencies for the Kepler main-sequence target 16 Cyg A derived by
`Davies et al. 2015 <https://ui.adsabs.harvard.edu/abs/2015MNRAS.446.2959D/abstract>`_. These are included in the
``ascii`` file ``${BASTADIR}/examples/data/freqs/16CygA.fre`` and listed in columns of
(order, degree, frequency, error, flag). To fit these frequencies to a grid of models, the ascii input must be converted
to a suitable ``xml`` file with the routine :py:meth:`fileio.freqs_ascii_to_xml`. Simply run the following commands in
the terminal:

.. code-block:: bash

    cd ${BASTADIR}
    source venv/bin/activate
    cd examples/data/freqs
    python
    from basta import fileio
    fileio.freqs_ascii_to_xml('.','16CygA',check_radial_orders=False,nbeforel=True)

Which gives the oputput:

.. code-block:: bash

    Star 16CygA has an epsilon of: 0.8.
    No correction made.

This will produce the file ``16CygA.xml``. You can provide ``ascii`` files with the columns (order,degree) swapped and
simply use the argument ``nbeforel=False``. Note that when the argument ``check_radial_orders=True`` is given, you
will get the following output when running the command:

.. code-block:: bash

    Star 16CygA has an epsilon of: 0.8.
    The proposed correction has been implemented.

BASTA calculates the epsilon (:math:`\epsilon`) term as described in
`White et al. 2012 <https://ui.adsabs.harvard.edu/abs/2012ApJ...751L..36W/abstract>`_ and correct the radial order of
the frequencies accordingly. In this case the radial order of the input frequencies is already appropriate, and thus
BASTA does not change the values of the radial order.

With the input file in the appropriate ``xml`` format, the following blocks must be modified in ``create_inputfile.py``:

.. code-block:: python

    # ==================================================================================
    # BLOCK 2: Fitting control
    # ==================================================================================
    define_fit["fitparams"] = ("Teff", "FeH", "freqs")

    # ------------------------------------------------------------
    # BLOCK 2d: Fitting control, frequencies
    # ------------------------------------------------------------
    define_fit["freqparams"] = {
        "freqpath": "${BASTADIR}/examples/data/freqs",
        "fcor": "BG14",
        "correlations": False,
        "dnufrac": 0.15,
    }

    # ==================================================================================
    # BLOCK 4: Plotting control
    # ==================================================================================
    define_plots["freqplots"] = "echelle"

This defines ``freqs`` as a ``fitparam`` in block 2. Block 2d gives the path to the frequency ``xml``-file, sets the
frequency correction from `Ball & Gizon 2014 <https://ui.adsabs.harvard.edu/abs/2014A%26A...568A.123B/abstract>`_,
ignores correlations between individual frequencies, and restricts the lowest l=0 mode from each model to be within 15%
of the value of the observed one.

We have provided an ready-to-run example for this star:

.. code-block:: bash

    cd ${BASTADIR}
    source venv/bin/activate
    cd examples/xmlinput
    python create_inputfile_freqs.py
    BASTArun input_freqs.xml

The fit should take less than a minute and the output is stored in ``${BASTADIR}/examples/output/freqs``. Besides the
corner plot and Kiel diagrams, the code produces output of the fit to the individual frequencies in form of echelle
diagrams for both corrected and uncorrected frequencies:

.. figure:: ../examples/reference/freqs/16CygA_pairechelle_uncorrected.pdf
   :alt: Echelle diagram showing the uncorrected frequencies of the best fit model to 16 Cyg A in the grid.

   Echelle diagram showing the uncorrected frequencies of the best fit model to 16 Cyg A in the grid.

.. figure:: ../examples/reference/freqs/16CygA_pairechelle.pdf
   :alt: Echelle diagram after the BG14 frequency correction to the best fit model to 16 Cyg A in the grid.

   Echelle diagram after the BG14 frequency correction to the best fit model to 16 Cyg A in the grid.


Frequency ratios
----------------

BASTA also has the option to fit the frequency ratios (:math:`r_{01}, r_{10}, r_{02}, r_{010}, r_{012}`). To do this,
one simply adds the following ``fitparam`` (for the case of :math:`r_{012}` as an example):

.. code-block:: python

    # ==================================================================================
    # BLOCK 2: Fitting control
    # ==================================================================================
    define_fit["fitparams"] = ("Teff", "FeH", "r012")

    # ==================================================================================
    # BLOCK 4: Plotting control
    # ==================================================================================
    define_plots["freqplots"] = "ratios"

The variable ``freqplots`` can also be set to ``True``, which will produce plots of the ratios and corresponding echelle
diagrams even though individual frequencies are not fitted. We provide an example to run this fit in
``${BASTADIR}/examples/xmlinput/create_inputfiles_ratios.py`` which produces the file ``input_ratios.xml``. Running
this file stores the results of the fit in ``${BASTADIR}/examples/output/ratios/``, and the resulting ratios should look
as follows:

.. figure:: ../examples/reference/ratios/16CygA_ratios.pdf
   :alt: Frequency ratios of the best fit model to 16 Cyg A in the grid.

   Frequency ratios of the best fit model to 16 Cyg A in the grid.

Frequency glitches
------------------

Another feature of BASTA is the fit of frequency glitches related to the base of the convective envelope and the He
ionisation zones. The glitch information must be provided in a file with the ``.glh`` extension that contains the
following information in columns:

#. Amplitude of the base of the convection zone (BCZ) glitch signature [muHz^3]
#. Acoustic depth of the BCZ glitch signature [sec]
#. Phase of the BCZ glitch signature [dimensionless]
#. Amplitude of the helium (He) glitch signature [dimensionless]
#. Acoustic width of the He glitch signature [sec]
#. Acoustic depth of the He glitch signature [sec]
#. Phase of the He glitch signature [dimensionless]
#. Average amplitude of the BCZ glitch signature [muHz]
#. Average amplitude of the He glitch signature [muHz]

An example file with this format can be found in ``${BASTADIR}/examples/data/freqs/16CygA.glh`` containing the glitch
information derived from 1000 MC realisations of the observed individual frequencies of 16 Cyg A. Each realisation
corresponds to one row of the file.

To produce the fit one simply needs to include the appropriate parameter in ``fitparams``

.. code-block:: python

    # ==================================================================================
    # BLOCK 2: Fitting control
    # ==================================================================================
    define_fit["fitparams"] = ("Teff", "FeH", "glitches")

Since the ``.glh`` file is located in the same folder as the individual frequencies, block 2d remains unchanged:

.. code-block:: python

    # ------------------------------------------------------------
    # BLOCK 2d: Fitting control, frequencies
    # ------------------------------------------------------------
    define_fit["freqparams"] = {
        "freqpath": "${BASTADIR}/examples/data/freqs",
        "fcor": "BG14",
        "correlations": False,
        "dnufrac": 0.15,
    }

You can find the corresponding python script to produce the input file for this fit in
``${BASTADIR}/examples/xmlinput/create_inputfiles_glitches.py``. The output should look as follows:

.. figure:: ../examples/reference/glitches/16CygA_corner.pdf
   :alt: Corner plot of the 16 Cyg A fit using glitches.

   Corner plot of the 16 Cyg A fit using glitches.

Individual frequencies: subgiants
---------------------------------

Reproducing the frequency spectrum of subgiant stars is a challenging task from a technical point of view, as the radial
order of the observed mixed-modes does not correspond to the theoretical values used to label them in models. We have
developed an algorithm that deals with this automatically, and we refer to section 4.1.5 of
`The BASTA paper II <https://arxiv.org/abs/2109.14622>`_ for further details.

In practice, you simply need to provide an ``ascii`` file with the individual frequencies in the same format as in the
main-sequence case (order, degree, frequency, error, flag). The radial order given is basically irrelevant, as BASTA
will use the epsilon (:math:`\epsilon`) method to correct the radial order of the l=0 modes, and use only the frequency
values for the l=1,2 modes to find the correct match.

We include an example of frequencies for a subgiant in the file ``${BASTADIR}/examples/data/freqs/Valid_245.fre``. It
corresponds to one of the artificial stars used for the validation of the code as described in section 6 of
`The BASTA paper II <https://arxiv.org/abs/2109.14622>`_. Quick exploration of the file
reveals that it has a number of mixed-modes of l=1 that have radial orders labelled in ascending order. You need to
transform the ``.fre`` file into a ``.xml`` file following the usual procedure:

.. code-block:: bash

    cd ${BASTADIR}
    source venv/bin/activate
    cd examples/data/freqs
    python
    from basta import fileio
    fileio.freqs_ascii_to_xml('.','Valid_245',check_radial_orders=True,nbeforel=True)

You should see the following output:

.. code-block:: bash

    Star Valid_245 has an odd epsilon value of 1.9,
    Correction of n-order by 1 gives epsilon value of 0.9.
    The proposed correction has been implemented.

The input is now ready. The global parameters of the star are contained in ``${BASTADIR}/examples/data/subgiant.ascii``.
To run the example, a few modifications to :py:meth:`create_inputfile.define_input` are necessary (related to input
files and grid to be used). The following blocks are now changed:

.. code-block:: python

    # ==================================================================================
    # BLOCK 1: I/O
    # ==================================================================================
    xmlfilename = "input_subgiant.xml"

    define_io["gridfile"] = "${BASTADIR}/grids/Garstec_validation.hdf5"

    define_io["asciifile"] = "${BASTADIR}/examples/data/subgiant.ascii"
    define_io["params"] = (
        "starid",
        "Teff",
        "Teff_err",
        "FeH",
        "FeH_err",
        "dnu",
        "dnu_err",
        "numax",
        "numax_err",
    )

A ready-to-run file is provided in ``${BASTADIR}/examples/xmlinput/create_inputfile_subgiant.py`` and as usual it can
simply be run as

.. code-block:: bash

    cd ${BASTADIR}
    source venv/bin/activate
    cd examples/xmlinput
    python create_inputfile_subgiant.py
    BASTArun input_subgiant.xml

The resulting duplicated echelle diagram should look as like the following.

.. figure:: ../examples/reference/subgiant/Valid_245_dupechelle.pdf
   :alt: Echelle diagram after the BG14 frequency correction to the best fit model to Validation star 245.

   Echelle diagram after the BG14 frequency correction to the best fit model to Validation star 245.

The corner plot present peaks revealing the underlying sampling in the code. Once again we refer you to the section on
:ref:`example_interp` to refine the grid as desired.
