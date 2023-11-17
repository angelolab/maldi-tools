"""Tests the `extraction.py` module."""

import os
import pathlib

import numpy as np
import pandas as pd
import pytest
import xarray as xr
from pyimzml.ImzMLParser import ImzMLParser

from maldi_tools import extraction


def test_extract_spectra(imz_data: ImzMLParser) -> None:
    intensity_percentile: int = 99

    total_mass_df, thresholds = extraction.extract_spectra(
        imz_data=imz_data, intensity_percentile=intensity_percentile
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
    assert log_intensities.shape == (10000,)
    assert log_int_percentile.shape == (10000,)


def test_signal_extraction(
    total_mass_df: pd.DataFrame, percentile_intensities: tuple[np.ndarray, np.ndarray]
) -> None:
    _, log_int_percentile = percentile_intensities
    peak_candidate_indexes, peak_candidates = extraction.signal_extraction(
        total_mass_df=total_mass_df, log_int_percentile=log_int_percentile
    )

    # Assert shapes
    assert peak_candidate_indexes.shape == peak_candidates.shape

    # Assert monotomically increasing peaks and their indexes.
    assert np.all(peak_candidate_indexes[1:] >= peak_candidate_indexes[:-1])
    assert np.all(peak_candidates[1:] >= peak_candidates[:-1])


def test_get_peak_widths(total_mass_df: pd.DataFrame, peak_idx_candidates: tuple[np.ndarray, np.ndarray]):
    peak_candidate_idxs, peak_candidates = peak_idx_candidates
    peak_df, l_ips_r, r_ips_r, peak_widths_height = extraction.get_peak_widths(
        total_mass_df=total_mass_df,
        peak_candidate_idxs=peak_candidate_idxs,
        peak_candidates=peak_candidates,
        thresholds=np.zeros(shape=(10, 10)),
    )
    # Check column names
    assert set(peak_df.columns) == set(["m/z", "intensity", "peak", "peak_height"])

    # Check lower and upper bound indicies and peak widths shapes are the same
    assert l_ips_r.shape == r_ips_r.shape == peak_widths_height.shape


def test_peak_spectra(
    total_mass_df: pd.DataFrame,
    peak_idx_candidates: tuple[np.ndarray, np.ndarray],
    peak_widths: tuple[pd.DataFrame, np.ndarray, np.ndarray, np.ndarray],
    tmp_path: pathlib.Path,
):
    debug_dir = tmp_path / "debug_dir"
    debug_dir.mkdir(parents=True, exist_ok=True)

    peak_candidate_idxs, peak_candidates = peak_idx_candidates
    peak_df, l_ips_r, r_ips_r, peak_widths_height = peak_widths

    panel_df = extraction.peak_spectra(
        total_mass_df=total_mass_df,
        peak_df=peak_df[:10],
        peak_candidate_idxs=peak_candidate_idxs,
        peak_candidates=peak_candidates,
        peak_widths_height=peak_widths_height,
        l_ips_r=l_ips_r,
        r_ips_r=r_ips_r,
        save_peak_spectra_debug=True,
        debug_dir=debug_dir,
    )

    # Assert that the debug images exist
    for peak in panel_df.itertuples():
        assert os.path.exists(debug_dir / f"{peak.peak:.4f}.png".replace(".", "_", 1))


def test_coordinate_integration(imz_data, peak_widths):
    peak_df, *_ = peak_widths
    img_data = extraction.coordinate_integration(peak_df=peak_df, imz_data=imz_data)

    # Make sure the shape of any given image is correct.
    assert img_data.shape[1:] == (10, 10)


@pytest.mark.parametrize(
    argnames="obs_mz, true_values",
    argvalues=[
        (30.00001, pd.Series(data=[30, True, "A", (1.0 / 3)])),
        (44.99999, pd.Series(data=[45, True, "B", (2.0 / 9)])),
        (70, pd.Series(data=[np.nan, False, np.nan, np.nan])),
    ],
)
@pytest.mark.parametrize(argnames="_ppm", argvalues=[99])
def test__matching_vec(library: pd.DataFrame, obs_mz: int, true_values: pd.Series, _ppm):
    result: pd.Series = extraction._matching_vec(obs_mz=obs_mz, library_peak_df=library, ppm=_ppm)

    pd.testing.assert_series_equal(left=result, right=true_values)


@pytest.mark.parametrize(argnames="_ppm", argvalues=[99])
def test_library_matching(image_xr: xr.DataArray, library: pd.DataFrame, _ppm: int, tmp_path: pathlib.Path):
    extraction_dir = tmp_path / "extraction_dir"
    extraction_dir.mkdir(parents=True, exist_ok=True)

    peak_df: pd.DataFrame = extraction.library_matching(
        image_xr=image_xr, library_peak_df=library, ppm=_ppm, extraction_dir=extraction_dir
    )

    for idx, row in enumerate(peak_df.itertuples()):
        if idx < 4:
            assert row.matched is False
            assert np.isnan(row.composition)
            assert np.isnan(row.mass_error)
            assert np.isnan(row.lib_mz)
        else:
            assert row.mass_error == 0
            assert row.composition in {"A", "B"}
            assert row.peak in {30, 45}


def test_generate_glycan_mask(imz_data: ImzMLParser, glycan_img_path: pathlib.Path):
    glycan_mask: np.ndarray = extraction.generate_glycan_mask(imz_data, glycan_img_path)
    coords: np.ndarray = np.array([coord[:2] for coord in imz_data.coordinates])
    assert np.all(glycan_mask[coords[:, 1] - 1, coords[:, 0] - 1] == 255)


def test_map_coordinates_to_core_name(
    imz_data: ImzMLParser, centroid_path: pathlib.Path, poslog_path: pathlib.Path
):
    region_core_info: pd.DataFrame = extraction.map_coordinates_to_core_name(
        imz_data, centroid_path, poslog_path
    )

    region_core_mapping = region_core_info[["Region", "Core"]].drop_duplicates()
    region_core_dict = region_core_mapping.set_index("Region")["Core"].to_dict()

    assert region_core_dict["R0"] == "Region0"
    assert region_core_dict["R1"] == "Region1"
    assert len(region_core_dict) == 2
