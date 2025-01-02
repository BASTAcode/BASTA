.. _trouble:

Troubleshoot
############

Here is an overview of some common/known problems and how to solve them.


.. _trouble_quick:

I don't want to read this whole thing — I just have a question!
***************************************************************

**Please do not file an issue to ask a question.** You can use the `discussion tab
<https://github.com/BASTAcode/BASTA/discussions>`_ on GitHub if you have a
question. You are also free to get in touch with us using some other channel.


.. _trouble_virt:

Installation of the virtual environment
***************************************

On certain Linux systems (e.g., specific versions of Debian and Ubuntu), the
creation of the virtualenv will fail with the output:

.. code-block:: bash

    Error: Command [...] returned non-zero exit status 1

This due to a problem with a specifc version of ``pyenv``. To fix the problem
do the following:

.. code-block:: bash

    rm -r venv
    python3 -m venv --without-pip venv
    source venv/bin/activate
    curl https://bootstrap.pypa.io/get-pip.py | python
    deactivate
    source venv/bin/activate
    pip install -r requirements.txt
    deactivate

and everything should be setup correctly. Now, continue following the :ref:`install` instructions.


.. _trouble_hdf5:

Unable to open HDF5 file
************************

HDF5 files are known to sometimes cause problems on NFS-mounted filesystems. This can cause errors of the
form:

.. code-block:: bash

    OSError: Unable to open file (unable to lock file, errno = 37, error message = 'No locks available')

To fix this, just run the following in the shell:

.. code-block:: bash

    export HDF5_USE_FILE_LOCKING=‘FALSE’

and try again. If the problem persists, just add the line to your ``~/.bashrc`` (or equivalent).
However, it might have unknown side-effects!
