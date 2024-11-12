.. _install:

Installation
############

.. _ref_code:

Obtaining the code and virtual environment
******************************************

*Important note: BASTA requires Python 3.10 or newer; it is currently developed for Python 3.12.*


Start out by obtaining a copy of BASTA; either from GitHub by cloning the GitHub repository or downloadning a source code release, or from the Python Package Index (PyPI).


.. _ref_venv:

Virtual environment
===================
To ensure functionality, we suggest to install the code in a fresh virtual environment. You can do this by running (but feel free to do this anyway you prefer):

.. code-block:: bash

    mkdir ~/venvs; cd venvs
    python3 -m venv bastaenv
    source ~/venvs/bastaenv/bin/activate
    pip install --upgrade pip setuptools wheel
    deactivate
    source ~/venvs/bastaenv/bin/activate


.. _ref_pip:

Using pip (from PyPI or GitHub)
===============================
As mentioned above, we strongly recommend to install the code in a fresh virtual environment.

With the environment installed and activated, you can obtain BASTA and all dependencies:

.. code-block:: bash

    pip install basta


It is also possible to instead obtain the code package directly from GitHub:

.. code-block:: bash

    pip install https://github.com/BASTAcode/BASTA/archive/refs/heads/main.zip

Note that by changing `main.zip` to `devel.zip` in the line above, you will get the most recent (and perhaps unstable!) development version of the code (*not recommended*).



.. _ref_github_dev:

Clone the GitHub repository (for developers)
============================================

Cloning from GitHub has the advantage that it is easier to modify the source code if you wish to do so.

As a default, we recommend that you install BASTA in the folder ``~/BASTA`` if you install from GitHub but it is not a requirement. If you have a user on GitHub and use an ssh-keypair, you can simply run:

.. code-block:: bash

    git clone git@github.com:BASTAcode/BASTA.git

If you prefer to enter username and password instead of a key-pair run:

.. code-block:: bash

    git clone https://github.com/BASTAcode/BASTA.git

Now, assuming you have downloaded the code, you can run the following to setup a virtual environment in the same folder (feel free to do it any other way you prefer; we strongly recommend to install the code in a fresh virtual environment):

.. code-block:: bash

    cd BASTA
    python3 -m venv bastaenv
    source bastaenv/bin/activate

Then you can install the code and dependencies into the virtual environment:

.. code-block:: bash

    pip install -e .

Using `-e` will let you modify the source code and it will take effect at next run without reinstalling the code.


.. _ref_dust:

Before first use
****************

To finalise the setup, you will need to download a example grid and obtain the dustmaps. BASTA is shipped with a tool to do so:

.. code-block:: bash

    BASTAdownload

If you cloned BASTA from GitHub, you most likely wish to use the default location and can just run `BASTAdownload 16CygA`. Otherwise, you can do something like:

.. code-block:: bash

    mkdir -p ~/BASTA/grids
    mkdir -p ~/BASTA/dust
    BASTAdownload --gridpath ~/BASTA/grids --dustpath ~/BASTA/dust 16CygA


If you installed BASTA from PyPI and wish to obtain the examples and template input file(s), take a look at:

.. code-block:: bash

    cd ~/BASTA
    BASTAexamples full

If you only need the input template, run `BASTAexamples simple` in the directory where you need the template.

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
