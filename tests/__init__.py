"""Tests for the Maldi Pipeline.

Disable tqdm for tests
"""
from functools import partialmethod

import matplotlib
from tqdm import tqdm

tqdm.__init__ = partialmethod(tqdm.__init__, disable=True)


# Tk is a pain, don't use it.
matplotlib.use("Agg")
