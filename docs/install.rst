.. _install:

Installation
############

.. _ref_code:

Obtaining the code and virtual environment
******************************************

*Important note: BASTA requires Python 3.10 or newer; it is currently developed for Python 3.12.*


Start out by obtaining a copy of BASTA; either from GitHub by cloning the GitHub repository or downloadning a source code release, or from The Python Package Index (PyPI).


.. _ref_pypi:

Python Package Index
====================
We strongly recommend to install the code in a fresh virtual environment. You can do this by running (but feel free to do this anyway you prefer):

.. code-block:: bash

    mkdir ~/venvs; cd venvs
    python3 -m venv bastaenv
    source ~/venvs/bastaenv/bin/activate

When the virtual environment in place, you can obtain BASTA and all dependencies:

.. code-block:: bash

    pip install basta


.. _ref_github:

GitHub repository
=================

Installing from GitHub has the advantage that it is easier to modify the source code if you wish to do so.

As a default, we recommend that you install BASTA in the folder ``~/BASTA`` if you install from GitHub but it is not a requirement. If you have a user on GitHub and use an ssh-keypair, you can simply run:

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
****************

To finalise the setup, download a example grid, and obtain the dustmaps, simply run the following:

.. code-block:: bash

    BASTAdownload


If you installed BASTA from PyPI and wish to obtain the examples and template input file(s), take a look at:

.. code-block:: bash

    BASTAexamples -h

BASTA is now ready to go. If you need to fit acoustic glitches or wish to contribute to the code, please continue reading on this page. If not, then proceed in the menu to the next item.


.. _ref_fortran:

Glitch-fitting and Fortran modules
==================================

*If you don't want to contribute fit glitches, you can safely skip this section!*

In case you need to fit glitches (and only in that case), you must compile the external Fortran-modules. Firstly, activate your virtual environment and then:

.. code-block:: bash

    pip install meson ninja


Assuming you cloned the repository from GitHub to the suggested location run the following:

.. code-block:: bash

    cd ~/BASTA/src/basta
    f2py -c glitch_fq.f95 -m glitch_fq
    f2py -c glitch_sd.f95 -m glitch_sd
    f2py -c icov_sd.f95 -m icov_sd
    f2py -c sd.f95 -m sd



.. _ref_hooks:

Git hooks
=========

*If you don't want to contribute to BASTA, you can safely skip this section!*

BASTA uses ``pre-commit`` to manage git hooks, and the final setup task is to
activate them:

.. code-block:: bash

    source bastaenv/bin/activate
    pre-commit install


It might take a minute or two to complete. Now, to ensure everything is
correctly setup, run the command:

.. code-block:: bash

    pre-commit run --all-files


It should pass all checks.
