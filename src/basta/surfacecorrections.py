"""
This module contains the possible corrections for the surface effect that can be specified and used in BASTA.
"""

import h5py  # type: ignore[import]
import numpy as np
from scipy.interpolate import CubicSpline  # type: ignore[import]
from scipy.optimize import minimize  # type: ignore[import]
from sklearn import linear_model  # type: ignore[import]

from basta import core
from basta import utils_seismic as su


SURFACECORRECTIONS = {}


def register_surfacecorrection(fn):
    SURFACECORRECTIONS[fn.__name__] = fn
    return fn


@register_surfacecorrection
def KBC08(joinkeys, join, nuref, bcor):
    pass


@register_surfacecorrection
def cubicterm_BG14():
    pass


@register_surfacecorrection
def twoterm_BG14(joinkeys, join, scalnu, method="l1", onlyl0=False):
    pass


# TODO(Amalie) Consider writing one function that applies it
def apply_KBC08(modkey, mod, coeffs, scalnu):
    pass


def apply_twoterm_BG14(modkey, mod, coeffs, scalnu):
    pass
