"""Used for generating plots, TIFF images for extracted and post-filtered masses. In addition creates
plots for debug peaks.

- TODO: Add axis labels to plots

"""

import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import skimage.io as io
import xarray as xr
from alpineer import image_utils
from tqdm.notebook import tqdm


def plot_intensities(
    total_mass_df: pd.DataFrame,
    log_intensities: np.ndarray,
    log_int_percentile: np.ndarray,
    global_intensity_threshold: float,
) -> None:
    """Plots the log intensites and the log intensity percentile with the global intensity threhsold.

    Args:
    ----
        total_mass_df (pd.DataFrame): A dataframe containing all the masses and their relative intensities.
        log_intensities (np.ndarray): The log of the intensities
        log_int_percentile (np.ndarray): The percentile of the log intensities.
        global_intensity_threshold (float): The intensity percentile of the total mass' intensities.
    """
    plt.plot(total_mass_df["m/z"], log_intensities, color="w")
    plt.plot(total_mass_df["m/z"], log_int_percentile, color="g")
    plt.hlines(
        np.log(global_intensity_threshold),
        total_mass_df.iloc[0, 0],
        total_mass_df.iloc[-1, 0],
        color="r",
    )
    plt.show()


def plot_discovered_peaks(
    total_mass_df: pd.DataFrame,
    peak_candidate_idxs: np.ndarray,
    peak_candidates: np.ndarray,
    global_intensity_threshold: float,
) -> None:
    """Plots the discovered peaks' maximum value.

    Args:
    ----
        total_mass_df (pd.DataFrame): A dataframe containing all the masses and their relative intensities.
        peak_candidate_idxs (np.ndarray): A list containing the indexes of the discovered peaks.
        peak_candidates (np.ndarray): A list containing the discovered peaks.
        global_intensity_threshold (float): The intensity percentile of the total mass' intensities.
    """
    plt_range_min_ind: int = 0
    plt_range_max_ind: int = len(total_mass_df["intensity"]) - 1

    plt.plot(total_mass_df["m/z"], total_mass_df["intensity"], color="w")
    plt.hlines(
        global_intensity_threshold,
        np.min(total_mass_df["m/z"]),
        np.max(total_mass_df["m/z"]),
        color="r",
    )

    plt.scatter(
        peak_candidates[
            (peak_candidate_idxs < plt_range_max_ind) * (peak_candidate_idxs > plt_range_min_ind)
        ],
        total_mass_df.loc[
            peak_candidate_idxs[
                (peak_candidate_idxs < plt_range_max_ind) * (peak_candidate_idxs > plt_range_min_ind)
            ],
            "intensity",
        ],
        color="m",
    )
    plt.show()


def save_peak_spectra(
    peak: np.float64,
    idx: int,
    line_height: np.float64,
    lower_bound: np.float64,
    upper_bound: np.float64,
    total_mass_df: pd.DataFrame,
    df_subset: np.ndarray,
    peak_candidate_idxs: np.ndarray,
    debug_dir: Path,
) -> None:
    """Saves the debug peak spectra plot.

    Args:
    ----
        peak (np.float64): The peak to save.
        idx (int): The index of the unique peak.
        line_height (np.float64): The height of the line to plot.
        lower_bound (np.float64): The lower bound mass.
        upper_bound (np.float64): The upper bound mass.
        total_mass_df (pd.DataFrame): A dataframe containing all the masses and their relative intensities.
        df_subset (np.ndarray): A subset of values within the a peak width range.
        peak_candidate_idxs (np.ndarray): A list containing the indexes of the discovered peaks.
        debug_dir (Path): The directory where the debug information is saved in.
    """
    fig, ax = plt.subplots()
    ax.hlines(line_height, lower_bound, upper_bound, color="g")
    ax.plot(
        total_mass_df.loc[df_subset, "m/z"],
        total_mass_df.loc[df_subset, "intensity"],
        color="w",
    )
    ax.scatter(peak, total_mass_df.loc[peak_candidate_idxs[idx], "intensity"], color="m")

    save_path = debug_dir / f"{peak:.4f}".replace(".", "_")

    fig.savefig(save_path)
    plt.close(fig=fig)


def save_peak_images(image_xr: xr.DataArray, extraction_dir: Path) -> None:
    """Saves the peak images discovered as floating point and integer images.

    Args:
    ----
        image_xr (xr.DataArray): A data structure which holds all the images for each peak.
        extraction_dir (Path): The directory to save extracted data in.
    """
    # Create image directories if they do not exist
    float_dir: Path = extraction_dir / "float"
    int_dir: Path = extraction_dir / "int"
    for img_dir in [float_dir, int_dir]:
        if not os.path.exists(img_dir):
            img_dir.mkdir(parents=True, exist_ok=True)

    for v in tqdm(image_xr, total=image_xr.shape[0]):
        peak: np.float64 = v.peak.values
        float_img: np.ndarray = v.values.T
        integer_img: np.ndarray = (float_img * (2**32 - 1) / np.max(float_img)).astype(np.uint32)

        img_name: str = f"{peak:.4f}".replace(".", "_")

        # save floating point image
        image_utils.save_image(fname=float_dir / f"{img_name}.tiff", data=float_img)

        # save integer image
        image_utils.save_image(fname=int_dir / f"{img_name}.tiff", data=integer_img)


def plot_peak_hist(peak: float, bin_count: int, extraction_dir: Path) -> None:
    """Plot a histogram of the intensities of a provided peak image.

    Args:
    ----
        peak (float): The desired peak to visualize
        bin_count (int): The bin size to use for the histogram
        extraction_dir (Path): The directory the peak images are saved in
    """
    # verify that the peak provided exists
    peak_path = extraction_dir / f"{str(peak).replace('.', '_')}.tiff"
    if not os.path.exists(peak_path):
        raise FileNotFoundError(f"Peak {peak} does not have a corresponding peak image in {extraction_dir}")

    # load the peak image in and display histogram
    peak_img: np.ndarray = io.imread(peak_path)
    plt.hist(peak_img.values, bins=bin_count)


def save_matched_peak_images(
    matched_peaks_df: pd.DataFrame,
    extraction_dir: Path,
) -> None:
    """Saves the images which were matched with the library.

    Args:
    ----
        matched_peaks_df (pd.DataFrame): A dataframe containing the peaks matched with the library.
        extraction_dir (Path): The directory to save extracted data in.
    """
    # Create image directories if they do not exist
    float_dir: Path = Path(extraction_dir) / "library_matched" / "float"
    int_dir: Path = Path(extraction_dir) / "library_matched" / "int"
    for img_dir in [float_dir, int_dir]:
        if not os.path.exists(img_dir):
            img_dir.mkdir(parents=True, exist_ok=True)

    matched_peaks_df_filtered: pd.DataFrame = matched_peaks_df.dropna()

    for row in tqdm(matched_peaks_df_filtered.itertuples(), total=len(matched_peaks_df_filtered)):
        if row.matched is True:
            peak_file_name: str = f"{row.lib_mz:.4f}".replace(".", "_") + ".tiff"
            # load in the corresponding float and integer images
            float_img: np.ndarray = io.imread(Path(extraction_dir) / "float" / peak_file_name)
            integer_img: np.ndarray = io.imread(Path(extraction_dir) / "int" / peak_file_name)

            img_name: str = row.composition

            # save floating point image
            image_utils.save_image(fname=float_dir / f"{img_name}.tiff", data=float_img)

            # save integer image
            image_utils.save_image(fname=int_dir / f"{img_name}.tiff", data=integer_img)
