"""Shared Fixtures for tests."""

import json
from pathlib import Path
from typing import Generator, List

import numpy as np
import pandas as pd
import pytest
import skimage.io as io
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
    # random m/z and intensity values of length n.
    # Lengths n are distributed along the standard gamma.
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


@pytest.fixture(scope="session")
def glycan_img_path(tmp_path_factory: TempPathFactory, imz_data: ImzMLParser, rng: np.random.Generator):
    coords: np.ndarray = np.array([coord[:2] for coord in imz_data.coordinates])

    glycan_img: np.ndarray = np.zeros((10, 10))
    glycan_img[coords[:, 1] - 1, coords[:, 0] - 1] = rng.random(coords.shape[0])

    glycan_img_file: Path = tmp_path_factory.mktemp("glycan_imgs") / "glycan_img.tiff"
    io.imsave(glycan_img_file, glycan_img)

    yield glycan_img_file


@pytest.fixture(scope="session")
def poslog_dir(tmp_path_factory: TempPathFactory, imz_data: ImzMLParser, rng: np.random.Generator):
    columns_write: List[str] = ["Date", "Time", "Region", "PosX", "PosY", "X", "Y", "Z"]
    poslog_base_dir: Path = tmp_path_factory.mktemp("poslogs")

    for i in np.arange(2):
        poslog_data: pd.DataFrame = pd.DataFrame(
            rng.random(size=(int(len(imz_data.coordinates) / 2) + 2, len(columns_write))),
            columns=columns_write,
        )

        poslog_regions: List[str] = []
        for j in np.arange(2):
            poslog_regions.append("__")
            poslog_regions.extend([f"R{j}XY"] * 25)
        poslog_data["Region"] = poslog_regions

        poslog_file: Path = poslog_base_dir / f"poslog{i}.txt"
        poslog_data.to_csv(poslog_file, header=None, index=False, sep=" ", mode="w", columns=columns_write)

    yield poslog_base_dir


@pytest.fixture(scope="session")
def centroid_path(tmp_path_factory: TempPathFactory, imz_data: ImzMLParser):
    coords: np.ndarray = np.array([coord[:2] for coord in imz_data.coordinates])
    center_coord_indices: np.ndarray = np.arange(10, coords.shape[0], 25)

    centroid_data: dict = {}
    centroid_data["exportDateTime"] = None
    centroid_data["fovs"] = []
    for i, cci in enumerate(center_coord_indices):
        center_coord = coords[cci, :]
        center_point_data = {
            "name": f"Region{i}",
            "centerPointPixels": {"x": center_coord[0].item(), "y": center_coord[1].item()},
        }
        centroid_data["fovs"].append(center_point_data)

    centroid_file: Path = tmp_path_factory.mktemp("centroids") / "centroids.json"
    with open(centroid_file, "w") as outfile:
        outfile.write(json.dumps(centroid_data))

    yield centroid_file


@pytest.fixture(scope="session")
def bad_centroid_path(tmp_path_factory: TempPathFactory, imz_data: ImzMLParser):
    coords: np.ndarray = np.array([coord[:2] for coord in imz_data.coordinates])
    center_coord_indices: np.ndarray = np.arange(10, coords.shape[0], 25)

    centroid_data: dict = {}
    centroid_data["exportDateTime"] = None
    centroid_data["fovs"] = []
    for i, cci in enumerate(center_coord_indices):
        center_coord = coords[cci, :]
        center_point_data = {
            "name": f"Region{i}",
            "centerPointPixels": {"x": center_coord[0].item() + 10000, "y": center_coord[1].item() + 10000},
        }
        centroid_data["fovs"].append(center_point_data)

    centroid_file: Path = tmp_path_factory.mktemp("centroids") / "centroids.json"
    with open(centroid_file, "w") as outfile:
        outfile.write(json.dumps(centroid_data))

    yield centroid_file
