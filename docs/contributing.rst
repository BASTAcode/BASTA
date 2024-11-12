.. _contrib:

Contributing to BASTA
#####################

First off all, thanks for taking the time to contribute!

Found a bug? Have a new feature to suggest? Want to contribute changes? Make sure to read this first.

This is still work-in-progress, as we have only just made the code public and changed workflow as a result of a migration from GitLab to GitHub.


.. _contrib_bugs:

How can I report bugs and errors?
*********************************

This section guides you through submitting a bug report. Following these guidelines helps maintainers understand your report, reproduce the behavior, and find related reports.

Report bugs
===========

Before creating bug reports, please check the following:

    * If you can reproduce the problem in the latest version of BASTA.
    * If the problem has already been reported. Perform a `cursory search <https://github.com/BASTAcode/BASTA/issues>`_ and if the issue is still open add a comment to it instead of opening a new one.

How Do I Submit A (Good) Bug Report?
====================================

Bugs are tracked as `Github issues <https://guides.github.com/features/issues/>`_. Explain the problem and include additional details to help
maintainers reproduce the problem:

    * Use a clear and descriptive title for the issue to identify the problem.
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
***************************

This section guides you through how you can begin contribution to BASTA. Affected by the migration and still work in progress...

Please make a fork of the repository. In your fork, please make a new branch for the feature you want to add (or bug you want to fix or...).

If you did not activate the git hooks during installation, now is the time to do so:

.. code-block:: bash

    source venv/bin/activate
    pre-commit install


It might take a minute or two to complete. Now, to ensure everything is
correctly setup, run the command:

.. code-block:: bash

    pre-commit run --all-files


It should pass all checks.

To share your improvements with us, please make a pull request. Before doing that, have a look at :ref:`contrib_style`.


.. _contrib_enhanc:

Suggesting enhancements to the core-devel team
**********************************************

This section guides you through submitting an enhancement suggestion for BASTA, including completely new features and minor improvements to existing functionality. Following these guidelines helps developers and the community understand your suggestion and find related suggestions.

Before Submitting An Enhancement Suggestion
===========================================

Perform a `cursory search <https://github.com/BASTAcode/BASTA/issues>`_  to see if the enhancement has already been suggested. If it has, add a comment to the existing issue instead of opening a new one.

How Do I Submit A (Good) Enhancement Suggestion?
================================================

Enhancement suggestions are tracked as `Github issues <https://guides.github.com/features/issues/>`_. Please create an issue and provide the following information:

    * Use a clear and descriptive title for the issue to identify the suggestion.
    * Provide a step-by-step description of the suggested enhancement in as many details as possible.
    * Describe the current behavior and explain which behavior you expected to see instead and why.


.. _contrib_style:

Styleguides
***********

Git Commit Messages
===================

* Use the imperative mood ("Move cursor to..." not "Moves cursor to..." or "Moved cursor to...")
* Limit the first line to 72 characters or less

Pull requests
=============

* When you make the pull request, it is important to set the target branch to ``devel``. Be sure to do this as the first thing, since your description text will disappear then you change the target.
* If the new-feature is still a work-in-progress, please state so in the title of the merge request (e.g. 'WIP: Resample corner plots'). When your branch is ready to be merged, please remove the 'WIP' in the title of your merge request.
* If applicable, refer to the issue(s) your merge request will fix.

Code style
==========

BASTA uses the code style defined by the `Black formatter <https://github.com/psf/black>`_.


Disclaimer
**********

This contribution guide was inspired by the amazing contribution guide to the `Atom project <https://github.com/atom>`_.
