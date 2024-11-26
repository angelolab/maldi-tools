"""Tests the `extraction.py` module."""

import os
import pathlib
from typing import List, Tuple

import numpy as np
import pandas as pd
import pytest
import xarray as xr
from alpineer.io_utils import list_files, remove_file_extensions
from alpineer.misc_utils import verify_same_elements
from pyimzml.ImzMLParser import ImzMLParser
from pytest import TempPathFactory
from skimage.io import imread

from maldi_tools import extraction, plotting


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
    total_mass_df: pd.DataFrame, percentile_intensities: Tuple[np.ndarray, np.ndarray]
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


def test_get_peak_widths(total_mass_df: pd.DataFrame, peak_idx_candidates: Tuple[np.ndarray, np.ndarray]):
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
    peak_idx_candidates: Tuple[np.ndarray, np.ndarray],
    peak_widths: Tuple[pd.DataFrame, np.ndarray, np.ndarray, np.ndarray],
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


def test_coordinate_integration(
    imz_data_coord_int: ImzMLParser,
    peak_widths_coord_int: pd.DataFrame,
    image_xr: xr.DataArray,
    tmp_path: pathlib.Path,
):
    # peak_df, *_ = peak_widths
    extraction_dir = tmp_path / "extraction_dir"

    extraction.coordinate_integration(
        peak_df=peak_widths_coord_int, imz_data=imz_data_coord_int, extraction_dir=extraction_dir
    )

    # Make sure the shape of any given image is correct for both float and int
    test_float_peak_img = list_files(extraction_dir / "float")[0]
    float_img_data = imread(extraction_dir / "float" / test_float_peak_img)
    assert float_img_data.shape == (1, 1)

    test_int_peak_img = list_files(extraction_dir / "int")[0]
    int_img_data = imread(extraction_dir / "int" / test_int_peak_img)
    assert int_img_data.shape == (1, 1)


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
def test_library_matching(library: pd.DataFrame, image_xr: xr.DataArray, _ppm: int, tmp_path: pathlib.Path):
    extraction_dir = tmp_path / "extraction_dir"
    extraction_dir.mkdir(parents=True, exist_ok=True)
    plotting.save_peak_images(image_xr, extraction_dir)

    peak_df: pd.DataFrame = extraction.library_matching(
        library_peak_df=library, ppm=_ppm, extraction_dir=extraction_dir
    )

    for idx, row in enumerate(peak_df.itertuples()):
        if row.peak not in {30.0, 45.0}:
            assert row.matched is False
            assert np.isnan(row.composition)
            assert np.isnan(row.mass_error)
            assert np.isnan(row.lib_mz)
        else:
            assert row.matched is True
            assert row.mass_error == 0
            assert row.composition in {"A", "B"}
            assert row.lib_mz in {30.0, 45.0}


def test_generate_glycan_mask(
    tmp_path_factory: TempPathFactory, imz_data: ImzMLParser, glycan_img_path: pathlib.Path
):
    glycan_mask_path: pathlib.Path = tmp_path_factory.mktemp("glycan_mask") / "glycan_mask.tiff"
    extraction.generate_glycan_mask(imz_data, glycan_img_path, glycan_mask_path)
    assert os.path.exists(glycan_mask_path)

    glycan_mask: np.ndarray = imread(glycan_mask_path)
    coords: np.ndarray = np.array([coord[:2] for coord in imz_data.coordinates])
    assert np.all(glycan_mask[coords[:, 1] - 1, coords[:, 0] - 1] == 255)

    all_coords_X, all_coords_Y = np.meshgrid(np.arange(1, 11), np.arange(1, 11))
    all_coords: np.ndarray = np.vstack((all_coords_X.ravel(), all_coords_Y.ravel())).T
    coords_set: set = set(map(tuple, coords))
    non_hit_indices: np.ndarray = np.array([tuple(coord) not in coords_set for coord in all_coords])
    non_hit_coords: np.ndarray = all_coords[non_hit_indices]

    assert np.all(glycan_mask[non_hit_coords[:, 1] - 1, non_hit_coords[:, 0] - 1] == 0)


def test_map_coordinates_to_core_name(
    imz_data: ImzMLParser, centroid_path: pathlib.Path, poslog_dir: pathlib.Path
):
    poslog_paths: List[pathlib.Path] = [poslog_dir / pf for pf in os.listdir(poslog_dir)]
    region_core_info: pd.DataFrame = extraction.map_coordinates_to_core_name(
        imz_data, centroid_path, poslog_paths
    )

    region_core_mapping: pd.DataFrame = region_core_info[["Region", "Core"]].drop_duplicates()
    region_core_dict: dict = region_core_mapping.set_index("Region")["Core"].to_dict()

    assert len(region_core_dict) == 4
    assert region_core_dict[0] == "Region0"
    assert region_core_dict[1] == "Region1"
    assert region_core_dict[2] == "Region2"
    assert region_core_dict[3] == "Region3"


def test_map_coordinates_to_core_name_malformed(
    imz_data: ImzMLParser, bad_centroid_path: pathlib.Path, poslog_dir: pathlib.Path
):
    poslog_paths: List[pathlib.Path] = [poslog_dir / pf for pf in os.listdir(poslog_dir)]
    with pytest.warns(match="Could not find mapping of core Region0"):
        extraction.map_coordinates_to_core_name(imz_data, bad_centroid_path, poslog_paths)


def test_generate_glycan_crop_masks(
    tmp_path_factory: TempPathFactory, glycan_img_path: pathlib.Path, region_core_info: pd.DataFrame
):
    glycan_crop_save_dir: pathlib.Path = tmp_path_factory.mktemp("glycan_crops")
    extraction.generate_glycan_crop_masks(glycan_img_path, region_core_info, glycan_crop_save_dir)

    core_names: List[str] = remove_file_extensions(list_files(glycan_crop_save_dir, substrs=".tiff"))
    verify_same_elements(generated_core_masks=core_names, all_core_masks=region_core_info["Core"].unique())

    all_coords_X, all_coords_Y = np.meshgrid(np.arange(1, 11), np.arange(1, 11))
    all_coords: np.ndarray = np.vstack((all_coords_X.ravel(), all_coords_Y.ravel())).T
    for core in core_names:
        core_mask: np.ndarray = imread(glycan_crop_save_dir / f"{core}.tiff")
        core_coords: np.ndarray = region_core_info.loc[region_core_info["Core"] == core, ["X", "Y"]].values
        assert np.all(core_mask[core_coords[:, 1] - 1, core_coords[:, 0] - 1] == 255)

        coords_set: set = set(map(tuple, core_coords))
        non_hit_indices: np.ndarray = np.array([tuple(coord) not in coords_set for coord in all_coords])
        non_hit_coords: np.ndarray = all_coords[non_hit_indices]
        assert np.all(core_mask[non_hit_coords[:, 1] - 1, non_hit_coords[:, 0] - 1] == 0)


def test_load_glycan_crop_masks(glycan_crop_save_dir: pathlib.Path, region_core_info: pd.DataFrame):
    core_names: List[str] = remove_file_extensions(list_files(glycan_crop_save_dir))

    all_coords_X, all_coords_Y = np.meshgrid(np.arange(1, 11), np.arange(1, 11))
    all_coords: np.ndarray = np.vstack((all_coords_X.ravel(), all_coords_Y.ravel())).T

    # test for a subset of FOVs
    core_cropped_mask: np.ndarray = extraction.load_glycan_crop_masks(glycan_crop_save_dir, [core_names[0]])
    coords: np.ndarray = region_core_info.loc[region_core_info["Core"] == core_names[0], ["X", "Y"]].values
    assert np.all(core_cropped_mask[coords[:, 1] - 1, coords[:, 0] - 1] == 255)

    coords_set: set = set(map(tuple, coords))
    non_hit_indices: np.ndarray = np.array([tuple(coord) not in coords_set for coord in all_coords])
    non_hit_coords: np.ndarray = all_coords[non_hit_indices]
    assert np.all(core_cropped_mask[non_hit_coords[:, 1] - 1, non_hit_coords[:, 0] - 1] == 0)

    # test for all FOVs
    core_cropped_mask = extraction.load_glycan_crop_masks(glycan_crop_save_dir)
    coords = region_core_info.loc[:, ["X", "Y"]].values
    assert np.all(core_cropped_mask[coords[:, 1] - 1, coords[:, 0] - 1] == 255)

    coords_set = set(map(tuple, coords))
    non_hit_indices = np.array([tuple(coord) not in coords_set for coord in all_coords])
    non_hit_coords = all_coords[non_hit_indices]
    assert np.all(core_cropped_mask[non_hit_coords[:, 1] - 1, non_hit_coords[:, 0] - 1] == 0)
