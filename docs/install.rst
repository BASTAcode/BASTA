.. _install:

Installation
================

**Please read all subsections on this page carefully to make sure all aspects of the installation are performed correctly!**


.. _ref_code:

Obtaining the code and virtual environment
------------------------------------------

*Important note: BASTA is developed for Python 3.9, but Python >= 3.7 should work as well.*


Start out by obtaining a copy of BASTA; either by cloning the GitHub repository or downloadning a source code release. As a default, we recommend that you install basta in the folder ``~/BASTA``. If you have a user on GitHub and use an ssh-keypair, you can simply run:

.. code-block:: bash

    git clone git@github.com:BASTAcode/BASTA.git

If you prefer to enter username and password instead of a key-pair run:

.. code-block:: bash

    git clone https://github.com/BASTAcode/BASTA.git


Now, assuming you have downloaded the code, run the following to setup the virtual environment:

.. code-block:: bash

    cd BASTA
    python3 -m venv venv
    source venv/bin/activate
    pip install --upgrade pip
    deactivate
    source venv/bin/activate
    pip install wheel
    pip install -r requirements.txt
    deactivate

It is important to deactivate and re-activate the virtual environment after upgrading ``pip`` to ensure the proper
installation of packages afterwards, and likewise to install ``wheel`` prior to the other requirements.


.. _ref_shell:

Configuring the shell/environment
---------------------------------

Add the following to your ``~/.bashrc`` (or equivalent) and run ``source ~/.bashrc``:

.. code-block:: bash

    export BASTADIR=${HOME}/BASTA
    export PYTHONPATH=${PYTHONPATH}:${BASTADIR}
    export PATH=${PATH}:${BASTADIR}/bin

If your installation of BASTA is in a folder different than the default ``~/BASTA`` please update the path in the first line accordingly!


.. _ref_f2py:

Consistent f2py
---------------

BASTA needs the executable ``f2py3`` to exist in the path to be able to compile external Fortran routines required for
different parts of the code. In order to avoid issues, this tool *must* match the version of NumPy used to run BASTA.
It is possible to ensure this by utilizing the version installed in the virtual environment (since the executable is
shipped with NumPy). To do this, execute the following commands **after** the creation of the virtual environment and
installation of packages described in :ref:`ref_code`

.. code-block:: bash

    mkdir ~/bin; cd ~/bin
    ln -s ${BASTADIR}/venv/bin/f2py f2py3

Add the following to your ``~/.bashrc`` (or equivalent) and run ``source ~/.bashrc`` to put your personal bin-folder
first in the search path:

.. code-block:: bash

    # My own bin first!
    export PATH=${HOME}/bin:${PATH}

Please note, that now it is only possible to use the ``f2py3`` tool when the virtual environment is activated. You can check that the symlink works and the location is correct, by running

.. code-block:: bash

    which f2py3


.. _ref_dust:

External routines and dustmaps
------------------------------

To automatically compile the external routines with ``f2py3`` (described above) and setup the dustmaps, use the installation file shipped with BASTA (please deactivate and re-activate the venv, if you just installed it):

.. code-block:: bash

    cd ${BASTADIR}
    deactivate
    source venv/bin/activate
    python setup.py CASE

Here ``CASE`` should be ``personal`` unless you are running BASTA natively on a M1/M1X Mac, in which case it should be ``light``. Setting the latter will disable the functionality to fit glitches and to use grid interpolation. Support for the new Mac systems are currently work-in-progress. Internally on AU, it is also possible to use the case ``grendel`` on the Grendel-S cluster, in which case BASTA will use the dustmaps from our shared project folder.

Please note that quite a lot of output might be produced, including some warnings. However, these warnings (e.g. the deprecated NumPy API) are harmless and cannot be avoided until the Scipy-people update ``f2py``. Unless the compilation fails, just ignore the warnings.

The path to ``f2py3`` is printed by the script -- make sure this is correctly pointing to the BASTA virtual environment! The script will try to import the compiled modules to check the compiled files.


.. _ref_hooks:

Git hooks
---------

*If you don't want to contribute to BASTA, you can safely skip this section!*

BASTA uses ``pre-commit`` to manage git hooks, and the final setup task is to
activate them:

.. code-block:: bash

    source venv/bin/activate
    pre-commit install


It might take a minute or two to complete. Now, to ensure everything is
correctly setup, run the command:

.. code-block:: bash

    pre-commit run --all-files


It should pass all checks. BASTA is now ready to go.
