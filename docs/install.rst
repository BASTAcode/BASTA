.. _install:

Installation
================

.. _ref_code:

Obtaining the code and virtual environment
------------------------------------------

*Important note: BASTA is developed for Python 3.11.*


Start out by obtaining a copy of BASTA; either by cloning the GitHub repository or downloadning a source code release. As a default, we recommend that you install basta in the folder ``~/BASTA``. If you have a user on GitHub and use an ssh-keypair, you can simply run:

.. code-block:: bash

    git clone git@github.com:BASTAcode/BASTA.git

If you prefer to enter username and password instead of a key-pair run:

.. code-block:: bash

    git clone https://github.com/BASTAcode/BASTA.git

Now, assuming you have downloaded the code, run the following to setup a virtual environment (feel free to do it any other way you prefer; we strongly recommend to install the code in a fresh virtual environment):

.. code-block:: bash

    cd BASTA
    python3 -m venv bastaenv
    source bastaenv/bin/activate

Then you can install the code and dependencies into the virtual environment:

.. code-block:: bash

    pip install -e .


.. _ref_dust:

Before first use
------------------------------

To finalise the setup, download a example grid, and obtain the dustmaps, simply run the following from the BASTA directory:

.. code-block:: bash

    BASTAdownload


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
