"""Tests for the Maldi Pipeline.

Disable tqdm for tests
"""
from functools import partialmethod

from tqdm import tqdm

tqdm.__init__ = partialmethod(tqdm.__init__, disable=True)
