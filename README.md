# BASTA: The BAyesian STellar Algorithm

[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![License: MIT](https://img.shields.io/badge/license-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Documentation Status](https://readthedocs.org/projects/basta/badge/?version=latest)](https://basta.readthedocs.io/en/latest/?badge=latest)
[![arXiv](https://img.shields.io/badge/arXiv-2109.14622-b31b1b.svg)](https://arxiv.org/abs/2109.14622)
[![ADS](https://img.shields.io/badge/ads-2022MNRAS.509.4344A-blue.svg)](https://ui.adsabs.harvard.edu/abs/2022MNRAS.509.4344A/abstract)
[![DOI](https://img.shields.io/badge/doi-10.1093/mnras/stab2911-orange.svg)](https://doi.org/10.1093/mnras/stab2911)

Current stable version: v1.5.3

**Important note:** BASTA is currently developed for Python 3.12, but Python >= 3.10 should suffice.


## Before you begin

*Please follow the repository on GitHub to get notifications on new releases: Click "Watch" then "Custom" and tick "Releases.*

Please have a look at [our documentation](https://basta.readthedocs.io/en/latest/index.html#).

There we have written a guide [guide to installing BASTA](https://basta.readthedocs.io/en/latest/install.html).

On there, you will also find an [introduction to running BASTA](https://basta.readthedocs.io/en/latest/running.html).

If you are curious on what BASTA can do, we have created several [fitting examples](https://basta.readthedocs.io/en/latest/examples.html) and the exact code to run them available.

If you have any questions, or encounter any issues, feel free to write to us through the [discussions page](https://github.com/orgs/BASTAcode/discussions).


## Quick start guide

BASTA can be obtained from GitHub or from the Python Package Index (PyPI); the full details are given in the documentation.

*We strongly recommend to create a fresh virtual environment to install BASTA in!*

With the virtual environment activated:

```
pip install basta
```
or
```
pip install https://github.com/BASTAcode/BASTA/archive/refs/heads/main.zip
```

To make the code ready to run you need to download some additional assets: a grid of stellar models and the dustmaps. To complete the setup of the code and download the grid used in most of our examples run the following (feel free to change the paths as you like):

```
mkdir -p ~/BASTA/grids
mkdir -p ~/BASTA/dust
BASTAdownload --gridpath ~/BASTA/grids --dustpath ~/BASTA/dust 16CygA
```

Finally, to obtain the full suite of examples run:

```
cd ~/BASTA
BASTAexamples full
```

Congratulations! You now have a fully functional installation of BASTA! We strongly recommend looking at the documentation for information on how to use the code.





## References and acknowledgments

There are two papers containing the rationale, main features, and capabilities of the code:

* [The BASTA paper I](https://ui.adsabs.harvard.edu/abs/2015MNRAS.452.2127S/abstract).
* [The BASTA paper II](https://ui.adsabs.harvard.edu/abs/2022MNRAS.509.4344A/abstract).

Please cite these references if you use BASTA in your work, and include the link to the code's repository `https://github.com/BASTAcode/BASTA`.

Due to its versatility, BASTA is used in a large variety of studies requiring robust determination of fundamental stellar properties. We have compiled a (non-exhaustive) [list of papers using BASTA results](https://ui.adsabs.harvard.edu/public-libraries/x2tCt52HR_yqG-oaUabo_A) that showcases these applications. If your paper using BASTA results is missing from the list please contact us.


## Authors

The current core developing team members are:

* Jakob Lysgaard Rørsted (maintainer)
* Mark Lykke Winther (co-maintainer)
* Amalie Stokholm (co-maintainer)
* Kuldeep Verma


The original author of the code is:

* Víctor Aguirre Børsen-Koch


Throughout the years, many people have contributed to the addition and development of various parts and modules of BASTA. We welcome further contributions from the community as well as issues reporting. Please look at the [contribution section](https://basta.readthedocs.io/en/latest/contributing.html) in the documentation for further details.


## Listings

In addition to the shields/icons in the top, BASTA can be found in the following software listings/catalogues:

* Astrophysics Source Code Library / [ASCL](https://ascl.net/2110.010)
* Exoplanet Modeling and Analysis Center / [EMAC](https://emac.gsfc.nasa.gov#bbcded4b-27d8-49f5-be4d-76e1fec748eb)
* The Python Package Index / [PyPI](https://pypi.org/project/basta/)
