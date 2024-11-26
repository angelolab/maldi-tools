"""Tests the `plotting.py` module."""

import os
import pathlib

import numpy as np
import pandas as pd
import pytest
import xarray as xr
from skimage.io import imread

from maldi_tools import plotting


def test_plot_intensities(total_mass_df: pd.DataFrame, percentile_intensities: tuple[np.ndarray, np.ndarray]):
    log_intensities, log_int_percentile = percentile_intensities
    _global_intensity_threshold = 7
    plotting.plot_intensities(
        total_mass_df=total_mass_df,
        log_intensities=log_intensities,
        log_int_percentile=log_int_percentile,
        global_intensity_threshold=_global_intensity_threshold,
    )


def test_plot_discovered_peaks(
    total_mass_df: pd.DataFrame, peak_idx_candidates: tuple[np.ndarray, np.ndarray]
):
    peak_candidate_indexes, peak_candidates = peak_idx_candidates
    _global_intensity_threshold = 7
    plotting.plot_discovered_peaks(
        total_mass_df=total_mass_df,
        peak_candidate_idxs=peak_candidate_indexes,
        peak_candidates=peak_candidates,
        global_intensity_threshold=_global_intensity_threshold,
    )


def test_save_peak_spectra(
    total_mass_df: pd.DataFrame, peak_idx_candidates: tuple[np.ndarray, np.ndarray], tmp_path: pathlib.Path
):
    debug_dir = tmp_path / "debug_dir"
    debug_dir.mkdir(parents=True, exist_ok=True)
    peak_candidate_indexes, peak_candidates = peak_idx_candidates
    _peak = 30.1
    _idx = 8
    _line_height = 2
    _lower_bound = 3
    _upper_bound = 4
    _df_subset = np.array([1, 2, 3])
    plotting.save_peak_spectra(
        peak=_peak,
        idx=_idx,
        line_height=_line_height,
        lower_bound=_lower_bound,
        upper_bound=_upper_bound,
        total_mass_df=total_mass_df,
        df_subset=_df_subset,
        peak_candidate_idxs=peak_candidate_indexes,
        debug_dir=debug_dir,
    )

    assert os.path.exists(debug_dir / "30_1000.png")


def test_save_peak_images(image_xr: xr.DataArray, tmp_path: pathlib.Path):
    extraction_dir = tmp_path / "extraction_dir"
    extraction_dir.mkdir(parents=True, exist_ok=True)

    plotting.save_peak_images(image_xr=image_xr, extraction_dir=extraction_dir)

    for v in image_xr:
        peak = v.peak.values
        float_img = v.values.T
        fname = extraction_dir / "float" / f"{peak:.4f}.tiff".replace(".", "_", 1)
        assert os.path.exists(fname)

        # Load the image and verify that the values are correct.
        float_imgdata = imread(fname=fname)
        np.testing.assert_allclose(float_img, float_imgdata)

        iname = extraction_dir / "int" / f"{peak:.4f}.tiff".replace(".", "_", 1)
        assert os.path.exists(iname)


def test_plot_peak_hist(image_xr: xr.DataArray, tmp_path: pathlib.Path):
    extraction_dir = tmp_path / "extraction_dir"
    extraction_dir.mkdir(parents=True, exist_ok=True)

    # ensure the test actually truncates to 4 digits correctly
    image_xr = image_xr.assign_coords(peak=np.random.rand(6) * 100)

    plotting.save_peak_images(image_xr=image_xr, extraction_dir=extraction_dir)

    # this test should run to completion, since the peak can be loaded
    plotting.plot_peak_hist(peak=image_xr.peak.values[0], bin_count=30, extraction_dir=extraction_dir)

    # this test should fail since the peak does not exist
    with pytest.raises(FileNotFoundError):
        plotting.plot_peak_hist(peak=50.0123, bin_count=30, extraction_dir=extraction_dir)


def test_save_matched_peak_images(rng: np.random.Generator, image_xr: xr.DataArray, tmp_path: pathlib.Path):
    extraction_dir = tmp_path / "extraction_dir"
    extraction_dir.mkdir(parents=True, exist_ok=True)
    plotting.save_peak_images(image_xr, extraction_dir)

    peaks = image_xr.peak.values
    img_shape = (image_xr.shape[1], image_xr.shape[2])
    matched = [False] * len(peaks)
    matched[-1] = True
    composition = [np.nan] * len(peaks)
    composition[-1] = rng.random(size=(1,))
    mass_error = [np.nan] * len(peaks)
    mass_error[-1] = rng.random(size=(1,))
    matched_peaks_df = pd.DataFrame(
        data={
            "lib_mz": peaks,
            "matched": matched,
            "composition": composition,
            "mass_error": mass_error,
        }
    )

    plotting.save_matched_peak_images(matched_peaks_df=matched_peaks_df, extraction_dir=extraction_dir)

    for peak in matched_peaks_df.itertuples():
        float_peak_path = extraction_dir / "library_matched" / "float" / f"{peak.composition}.tiff"
        int_peak_path = extraction_dir / "library_matched" / "int" / f"{peak.composition}.tiff"

        # Assert that the float and integer images are saved for all matched peaks.
        # Check that the peak images match the desired shape
        if peak.matched is True:
            assert os.path.exists(float_peak_path)
            assert os.path.exists(int_peak_path)

            matched_float_img = imread(float_peak_path)
            matched_int_img = imread(int_peak_path)

            assert matched_float_img.shape == img_shape
            assert matched_int_img.shape == img_shape
        # Otherwise, ensure peaks are not saved
        else:
            assert not os.path.exists(float_peak_path)
            assert not os.path.exists(int_peak_path)
