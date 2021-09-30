.. _contrib:

Contributing to BASTA
=====================

First off all, thanks for taking the time to contribute!

Found a bug? Have a new feature to suggest? Want to contribute changes? Make sure to read this first.

.. _contrib_bugs:

How can I report bugs and errors?
---------------------------------

This section guides you through submitting a bug report. Following these guidelines helps maintainers understand your report,
reproduce the behavior, and find related reports.

Report bugs
^^^^^^^^^^^

Before creating bug reports, please check the following:

    * If you can reproduce the problem in the latest version of BASTA.
    * If the problem has already been reported. Perform a `cursory search <https://github.com/vaguirrebkoch/BASTA/issues>`_ and if the issue is still open add a comment to it instead of opening a new one.

How Do I Submit A (Good) Bug Report?
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Bugs are tracked as `Github issues <https://guides.github.com/features/issues/>`_. Explain the problem and include additional details to help
maintainers reproduce the problem:

    * Use a clear and descriptive title** for the issue to identify the problem.
    * Describe the exact steps which produce the problem in as many details as possible.
    * Describe the behavior you observed after following the steps and point out what exactly is the problem with that behavior.
    * Explain which behavior you expected to see instead and why.
    * If the problem was not triggered by a specific action, describe what you were doing before the problem happened and share more information using the guidelines below.
    * Did the problem start happening recently (e.g. after updating to a new version of BASTA) or was this always a problem?
    * If the problem started happening recently, can you reproduce the problem in an older version of BASTA? What is the most recent version in which the problem does not happen?

Include details about your configuration and environment:

    * Which version of BASTA are you using?
    * Are you running BASTA in the virtual environment?
    * What is the name and version of the OS you are using?

.. _contrib_add:

How can I add new features?
---------------------------

This section guides you through how you can begin contribution to BASTA.

Before making a branch, make sure you are working on a fully updated version of the devel-branch. Check:

.. code-block:: bash

    cd ${BASTADIR}
    git checkout devel
    git fetch
    git status

and if it is not up to date, then:

.. code-block:: bash

    git pull
    git status

Make a new branch for your feature
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Use the following commands to create a new branch and implement the feature:

.. code-block:: bash

    git branch new-feature
    git checkout new-feature

A few useful tips in the process:

    * Commit often, but read the section about :ref:`contrib_style` first.
    * When comitting, make sure it passes all ``pre-commit`` checks and that the commit is processed.

To push the branch in the online remote repository, do

.. code-block:: bash

    git push

It will probably fail, but then just

.. code-block:: bash

    git push --set-upstream origin new-feature

You are more than welcome to create a merge request as soon as your branch is made. If the new-feature is still a work-in-progress,
please state so in the title of the merge request (e.g. 'WIP: Resample corner plots'). You can change the merge request message every time you
complete a minor or major change, see the section about :ref:`contrib_style`.

When your change is ready to be merged, please check that

    * Your branch is up-to-date with the current ``devel`` branch. You can check this by running the following command and resolving any appearing conflicts:

    .. code-block:: bash

        git merge devel

    * You have removed all unnecessary print-statements/comments/tests from your debugging process in your branch.
    * All new packages/versions of packages are added to the appropriate requirements file. Why you want to add/update packages should be explained in the merge request message.
    * Your changes (to a great extend) `follow the PEP-8 standard for Python code <https://www.python.org/dev/peps/pep-0008/>`_. This can be checked automatically by most editors or small tools like `flake8 <http://flake8.pycqa.org/en/latest/>`_.
    * That your merge request follow the list in the section about :ref:`contrib_style`.

Your change will be then reviewed before it is merged first into the ``devel`` branch, and later in the next published version of the ``master`` branch.

.. _contrib_enhanc:

Suggesting enhancements to the core-devel team
----------------------------------------------

This section guides you through submitting an enhancement suggestion for BASTA, including completely new features and minor improvements to
existing functionality. Following these guidelines helps developers and the community understand your suggestion and find related suggestions.

Before Submitting An Enhancement Suggestion
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Perform a `cursory search <https://github.com/vaguirrebkoch/BASTA/issues>`_  to see if the enhancement has already been suggested.
If it has, add a comment to the existing issue instead of opening a new one.

How Do I Submit A (Good) Enhancement Suggestion?
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Enhancement suggestions are tracked as `Github issues <https://guides.github.com/features/issues/>`_. Please create an issue and provide the following information:

    * Use a clear and descriptive title for the issue to identify the suggestion.
    * Provide a step-by-step description of the suggested enhancement in as many details as possible.
    * Describe the current behavior and explain which behavior you expected to see instead and why.

.. _contrib_style:

Styleguides
-----------

Git Commit Messages
^^^^^^^^^^^^^^^^^^^

* Use the imperative mood ("Move cursor to..." not "Moves cursor to..." or "Moved cursor to...")
* Limit the first line to 72 characters or less

Merge requests
^^^^^^^^^^^^^^

* When you make the merge request, it is very important to set the target branch to ``devel``. Be sure to do this as the first thing, since your description text will disappear then you change the target.
* If the new-feature is still a work-in-progress, please state so in the title of the merge request (e.g. 'WIP: Resample corner plots'). When your branch is ready to be merged, please remove the 'WIP' in the title of your merge request.
* If applicable, refer to the issue(s) your merge request will fix.

Code style
^^^^^^^^^^

BASTA uses the code style defined by the `Black formatter <https://github.com/psf/black>`_.

Disclaimer
^^^^^^^^^^

This contribution guide was inspired by the amazing contribution guide to the `Atom project <https://github.com/atom>`_.
