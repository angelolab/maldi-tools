"""Tests the `extraction.py` module.

- TODO: signal_extraction
- TODO: get_peak_widths
- TODO: peak_spectra
- TODO: coordinate_integration
- TODO: _matching_vec
- TODO: library_matching
"""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from pyimzml.ImzMLParser import ImzMLParser
from pyimzml.ImzMLWriter import ImzMLWriter
from pytest import TempPathFactory

from maldi_tools import extraction


@pytest.fixture(scope="module")
def rng() -> np.random.Generator:
    rng = np.random.default_rng(12345)
    yield rng


@pytest.fixture(scope="module")
def imz_data(tmp_path_factory: TempPathFactory, rng: np.random.Generator) -> ImzMLParser:
    # Working with a 10 x 10 image.
    img_dim: int = 10

    # Generate random integers n for each coordinate (10 x 10). These will be used for creating
    # random m/z and intensity values of length n. Lengths n are distributed along the standard gamma.
    ns: np.ndarray = np.rint(rng.standard_gamma(shape=2.5, size=(img_dim**2)) * 100).astype(int)

    # Generate random masses and sample different amounts of them, so we get duplicates
    total_mzs: np.ndarray = (10000 - 100) * rng.random(size=img_dim**2 * 2) + 100

    coords = [(x, y, 1) for x in range(1, img_dim + 1) for y in range(1, img_dim + 1)]

    output_file_name: Path = tmp_path_factory.mktemp("data") / "test_data.imzML"

    with ImzMLWriter(output_filename=output_file_name, mode="processed") as imzml:
        for coord, n in zip(coords, ns):
            # Masses: 100 <= mz < 10000, of length n, sampled randomly
            mzs = rng.choice(a=total_mzs, size=n)

            # Intensities: 0 <= int < 1e8, of length n
            ints: np.ndarray = rng.exponential(size=n)

            imzml.addSpectrum(mzs=mzs, intensities=ints, coords=coord)

    yield ImzMLParser(filename=output_file_name)


@pytest.fixture(scope="module")
def total_mass_df(rng: np.random.Generator) -> pd.DataFrame:
    df = pd.DataFrame(
        data={"m/z": np.linspace(start=1, stop=101, num=100), "intensity": rng.random(size=100)}
    )
    yield df


def test_extract_spectra(imz_data: ImzMLParser) -> None:
    intensity_percentile: int = 99
    scan_setting: str = "scanSettings1"

    total_mass_df, thresholds = extraction.extract_spectra(
        imz_data=imz_data, intensity_percentile=intensity_percentile, scan_setting=scan_setting
    )

    assert thresholds.shape == (10, 10)

    assert total_mass_df["m/z"].max() <= 1e4
    assert total_mass_df["intensity"].max() <= 1e8


def test_rolling_window(total_mass_df: pd.DataFrame) -> None:
    intensity_percentile: int = 99
    window_setting: int = 10
    log_intensities, log_int_percentile = extraction.rolling_window(
        total_mass_df=total_mass_df, intensity_percentile=intensity_percentile, window_size=window_setting
    )
    assert log_intensities.shape == (100,)
    assert log_int_percentile.shape == (100,)
