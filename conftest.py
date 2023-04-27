"""Shared Fixtures for tests."""

from pathlib import Path
from typing import Generator

import numpy as np
import pandas as pd
import pytest
import xarray as xr
from pyimzml.ImzMLParser import ImzMLParser
from pyimzml.ImzMLWriter import ImzMLWriter
from pytest import TempPathFactory

from maldi_tools import extraction


@pytest.fixture(scope="session")
def rng() -> np.random.Generator:
    rng = np.random.default_rng(12345)
    yield rng


@pytest.fixture(scope="session")
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


@pytest.fixture(scope="session")
def total_mass_df(rng: np.random.Generator) -> pd.DataFrame:
    mz_count: int = 10000
    df = pd.DataFrame(
        data={"m/z": np.linspace(start=1, stop=101, num=mz_count), "intensity": rng.random(size=mz_count)}
    )
    yield df


@pytest.fixture(scope="session")
def percentile_intensities(
    total_mass_df: pd.DataFrame,
) -> Generator[tuple[np.ndarray, np.ndarray], None, None]:
    log_intensities, log_int_percentile = extraction.rolling_window(
        total_mass_df=total_mass_df, intensity_percentile=99, window_size=10
    )

    yield (log_intensities, log_int_percentile)


@pytest.fixture(scope="session")
def peak_idx_candidates(
    total_mass_df: pd.DataFrame, percentile_intensities: tuple[np.ndarray, np.ndarray]
) -> Generator[tuple[np.ndarray, np.ndarray], None, None]:
    _, log_int_percentile = percentile_intensities

    peak_candidate_indexes, peak_candidates = extraction.signal_extraction(
        total_mass_df=total_mass_df, log_int_percentile=log_int_percentile
    )
    yield (peak_candidate_indexes, peak_candidates)


@pytest.fixture(scope="session")
def peak_widths(
    total_mass_df, peak_idx_candidates
) -> Generator[tuple[pd.DataFrame, np.ndarray, np.ndarray, np.ndarray], None, None]:
    peak_candidate_idxs, peak_candidates = peak_idx_candidates
    peak_df, l_ips_r, r_ips_r, peak_widths_height = extraction.get_peak_widths(
        total_mass_df=total_mass_df,
        peak_candidate_idxs=peak_candidate_idxs,
        peak_candidates=peak_candidates,
        thresholds=np.zeros(shape=(10, 10)),
    )

    yield (peak_df, l_ips_r, r_ips_r, peak_widths_height)


@pytest.fixture(scope="session")
def library() -> Generator[pd.DataFrame, None, None]:
    lib = pd.DataFrame(data={"mz": [30, 45], "composition": ["A", "B"]})
    yield lib


@pytest.fixture(scope="session")
def image_xr(rng: np.random.Generator, library: pd.DataFrame) -> Generator[xr.DataArray, None, None]:
    img_xr = xr.DataArray(
        data=rng.random(size=(6, 10, 10)),
        coords={
            "peak": np.hstack([rng.integers(low=10, high=100, size=(4,)), library.mz.values]),
            "x": range(10),
            "y": range(10),
        },
        dims=["peak", "x", "y"],
    )
    yield img_xr
