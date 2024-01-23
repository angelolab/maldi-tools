"""Used for extracting spectra data, filtering intensities and masses, and matching the discovered m/z peaks
to the user supplied library.

- TODO: Parallel spectra extraction
- TODO: Adduct matching

"""

import json
import os
from functools import partial
from operator import itemgetter
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import xarray as xr
from alpineer.io_utils import list_files, remove_file_extensions, validate_paths
from alpineer.misc_utils import verify_in_list
from pyimzml.ImzMLParser import ImzMLParser
from scipy import signal
from skimage.io import imread, imsave
from tqdm.notebook import tqdm

from maldi_tools import plotting


def extract_spectra(imz_data: ImzMLParser, intensity_percentile: int) -> tuple[pd.DataFrame, np.ndarray]:
    """Iterates over all coordinates after opening the `imzML` data and extracts all masses,
    and sums the intensities for all masses. Creates an intensity image, thresholded on
    `intensity_percentile` with `np.percentile`. The masses are then sorted.

    Args:
    ----
        imz_data (ImzMLParser): The imzML object.
        intensity_percentile (int): Used to compute the q-th percentile of the intensities.

    Returns:
    -------
        tuple[pd.DataFrame, np.ndarray]: A tuple where the first element is the dataframe containing
        the total masses and their intensities, and the second element is the thresholds matrix
        of the image.
    """
    imz_coordinates: list = imz_data.coordinates

    x_size: int = max(imz_coordinates, key=itemgetter(0))[0]
    y_size: int = max(imz_coordinates, key=itemgetter(1))[1]

    image_shape: Tuple[int, int] = (x_size, y_size)

    thresholds: np.ndarray = np.zeros(image_shape)
    total_spectra: Dict[float, float] = {}

    for idx, (x, y, _) in tqdm(enumerate(imz_data.coordinates), total=len(imz_coordinates)):
        mzs, intensities = imz_data.getspectrum(idx)
        for mass_idx, mz in enumerate(mzs):
            total_spectra[mz] = (0 if mz not in total_spectra else total_spectra[mz]) + intensities[mass_idx]

        thresholds[x - 1, y - 1] = np.percentile(intensities, intensity_percentile)

    total_mass_df = pd.DataFrame(total_spectra.items(), columns=["m/z", "intensity"])

    total_mass_df.sort_values(by="m/z", inplace=True)
    total_mass_df.reset_index(drop=True, inplace=True)

    return (total_mass_df, thresholds)


def rolling_window(
    total_mass_df: pd.DataFrame, intensity_percentile: int, window_size: int = 5000
) -> tuple[np.ndarray, np.ndarray]:
    """Computes the rolling window log intensities and the rolling window log intensity percentiles.

    Args:
    ----
        total_mass_df (pd.DataFrame): A dataframe containing all the masses and their
            relative intensities.
        intensity_percentile (int): The intensity for the quantile calculation.
        window_size (int, optional): The sizve of the window for the rolling window method.
            Defaults to 5000.

    Returns:
    -------
        tuple[np.ndarray, np.ndarray]: A tuple where the first element is the log intensities,
            and the second element is the log intensity percentiles.
    """
    plt_range_min_ind: int = 0
    plt_range_max_ind: int = len(total_mass_df["intensity"]) - 1

    log_intensities: np.ndarray = np.log(total_mass_df.loc[plt_range_min_ind:plt_range_max_ind, "intensity"])

    log_int_percentile: np.ndarray = (
        pd.Series(log_intensities)
        .rolling(window=window_size, center=True, min_periods=1)
        .quantile(intensity_percentile / 100)
    ).to_numpy()

    return (log_intensities, log_int_percentile)


def signal_extraction(
    total_mass_df: pd.DataFrame,
    log_int_percentile: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Extracts all found peaks and their indexes.

    Args:
    ----
        total_mass_df (pd.DataFrame): A dataframe containing all the masses and their
            relative intensities.
        log_int_percentile (np.ndarray): An array for the log intensity percentiles.

    Returns:
    -------
        tuple[np.ndarray, np.ndarray]: A tuple where the first element is the peak candidate
            indexes, and the second is the candidate peaks.
    """
    peak_candidate_indexes, _peak_properties = signal.find_peaks(
        total_mass_df["intensity"].values, prominence=np.exp(log_int_percentile)
    )

    peak_candidates: np.ndarray = total_mass_df.loc[peak_candidate_indexes, "m/z"].to_numpy()

    return (peak_candidate_indexes, peak_candidates)


def get_peak_widths(
    total_mass_df: pd.DataFrame,
    peak_candidate_idxs: np.ndarray,
    peak_candidates: np.ndarray,
    thresholds: np.ndarray,
) -> tuple[pd.DataFrame, np.ndarray, np.ndarray, np.ndarray]:
    """Computes the widths of the peaks discovered from the data.

    Args:
    ----
        total_mass_df (pd.DataFrame): A dataframe containing all the masses and their
            relative intensities.
        peak_candidate_idxs (np.ndarray): A list containing the indexes of the discovered peaks.
        peak_candidates (np.ndarray): A list containing the discovered peaks.
        thresholds (np.ndarray): The threshold matrix of the image.

    Returns:
    -------
        tuple[pd.DataFrame, np.ndarray, np.ndarray, np.ndarray]: A tuple where the first element is
            a DataFrame of the unique peaks discovered from the signal extraction process.
            The second and third elements are the lower and upper bounds, and the fourth element is
            the height of the contour lines at which the peak widths were calculated from.
    """
    plt_range_min_ind: int = 0
    plt_range_max_ind: int = len(total_mass_df["intensity"]) - 1

    peak_widths, peak_widths_height, l_ips, r_ips = signal.peak_widths(
        total_mass_df["intensity"],
        peak_candidate_idxs[
            ((peak_candidate_idxs < plt_range_max_ind) * (peak_candidate_idxs > plt_range_min_ind))
        ]
        - plt_range_min_ind,
        rel_height=0.90,
    )

    l_ips_r = np.round(l_ips + plt_range_min_ind)
    r_ips_r = np.round(r_ips + plt_range_min_ind)

    for idx, (lower_bound, upper_bound, pwh) in enumerate(zip(l_ips_r, r_ips_r, peak_widths_height)):
        total_mass_df.loc[lower_bound:upper_bound, "peak"] = peak_candidates[idx]
        total_mass_df.loc[lower_bound:upper_bound, "peak_height"] = pwh / thresholds.size

    peak_df: pd.DataFrame = total_mass_df.dropna().reset_index(drop=True)
    return (peak_df, l_ips_r.astype(int), r_ips_r.astype(int), peak_widths_height)


def peak_spectra(
    total_mass_df: pd.DataFrame,
    peak_df: pd.DataFrame,
    peak_candidate_idxs: np.ndarray,
    peak_candidates: np.ndarray,
    peak_widths_height: np.ndarray,
    l_ips_r: np.ndarray,
    r_ips_r: np.ndarray,
    save_peak_spectra_debug: bool,
    debug_dir: Path,
) -> pd.DataFrame:
    """Creates a panel file for debugging with the peaks, and their lower and upper bounds.
    Can also output plots of these peaks for investigative purposes.

    Args:
    ----
        total_mass_df (pd.DataFrame): A DataFrame containing all the masses and their relative intensities.
        peak_df (pd.DataFrame): A DataFrame of the unique peaks.
        peak_candidate_idxs (np.ndarray): A list containing the indexes of the discovered peaks.
        peak_candidates (np.ndarray): A list containing the discovered peaks.
        peak_widths_height (np.ndarray): The height of the contour lines at which the peak widths were
        calculated from.
        l_ips_r (np.ndarray): The rounded left (lower) bound.
        r_ips_r (np.ndarray): The rounded right (upper) bound.
        save_peak_spectra_debug (bool): A debug parameter for saving each peak found.
        debug_dir (Path): The directory where the debug information is saved in.

    Returns:
    -------
        pd.DataFrame: The panel which contains the peaks along with their lower and upper bounds
        for the discovered peaks.
    """
    panel_dict: dict[str, list] = {"peak": [], "start": [], "stop": []}

    for peak in tqdm(peak_df["peak"].unique()):
        idx: int = list(peak_candidates).index(peak)

        lower_bound: np.float64 = total_mass_df.loc[l_ips_r[idx], "m/z"]
        upper_bound: np.float64 = total_mass_df.loc[r_ips_r[idx], "m/z"]

        panel_dict["peak"].append(peak)
        panel_dict["start"].append(lower_bound)
        panel_dict["stop"].append(upper_bound)

        width: np.float64 = upper_bound - lower_bound

        line_height: np.float64 = peak_widths_height[idx]

        df_subset = np.abs(total_mass_df["m/z"] - peak) < width

        if save_peak_spectra_debug:
            plotting.save_peak_spectra(
                peak=peak,
                idx=idx,
                line_height=line_height,
                lower_bound=lower_bound,
                upper_bound=upper_bound,
                total_mass_df=total_mass_df,
                df_subset=df_subset,
                peak_candidate_idxs=peak_candidate_idxs,
                debug_dir=debug_dir,
            )

        panel_df = pd.DataFrame(panel_dict)
        panel_df.to_csv(debug_dir / "panel.csv")

    return panel_df


def coordinate_integration(peak_df: pd.DataFrame, imz_data: ImzMLParser) -> xr.DataArray:
    """Integrates the coordinates with the discovered, post-processed peaks and generates an image for
    each of the peaks using the imzML coordinate data.

    Args:
    ----
        peak_df (pd.DataFrame): The unique peaks from the data.
        imz_data (ImzMLParser): The imzML object.

    Returns:
    -------
        xr.DataArray: A data structure which holds all the images for each peak.
    """
    unique_peaks = peak_df["peak"].unique()
    peak_dict = dict(zip(unique_peaks, np.arange((len(unique_peaks)))))

    imz_coordinates: list = imz_data.coordinates

    x_size: int = max(imz_coordinates, key=itemgetter(0))[0]
    y_size: int = max(imz_coordinates, key=itemgetter(1))[1]

    image_shape: Tuple[int, int] = (x_size, y_size)

    imgs = np.zeros((len(unique_peaks), *image_shape))

    for idx, (x, y, _) in tqdm(enumerate(imz_data.coordinates), total=len(imz_data.coordinates)):
        mzs, intensities = imz_data.getspectrum(idx)

        intensity: np.ndarray = intensities[np.isin(mzs, peak_df["m/z"])]

        for i_idx, peak in peak_df.loc[peak_df["m/z"].isin(mzs), "peak"].reset_index(drop=True).items():
            imgs[peak_dict[peak], x - 1, y - 1] += intensity[i_idx]

    img_data = xr.DataArray(
        data=imgs,
        coords={"peak": unique_peaks, "x": range(x_size), "y": range(y_size)},
        dims=["peak", "x", "y"],
    )

    return img_data


def _matching_vec(obs_mz: pd.Series, library_peak_df: pd.DataFrame, ppm: int) -> pd.Series:
    """Finds the first matching mass in the target library for each observed mass if it exists within a
    tolerance determined by ppm.

    Args:
    ----
        obs_mz (pd.Series): The observed mass.
        library_peak_df (pd.DataFrame): The library of interest to match the observed peaks with.
        ppm (int): The ppm for an acceptable mass error range between the observed mass and any target
        mass in the library.

    Returns:
    -------
        pd.Series: A series containing the library mass, a boolean if a match is discovered, the composition,
        and the error mass error value.
    """
    for row in library_peak_df.itertuples():
        lib_mz = row.mz
        mass_error = np.absolute((1 - lib_mz / obs_mz) * 1e6)
        if mass_error <= ppm:
            return pd.Series([lib_mz, True, row.composition, mass_error])
        else:
            continue
    return pd.Series([np.nan, False, np.nan, np.nan])


def library_matching(
    image_xr: xr.DataArray,
    library_peak_df: pd.DataFrame,
    ppm: int,
    extraction_dir: Path,
    adducts: bool = False,
) -> pd.DataFrame:
    """Matches the image peaks to the library, and creates a csv which contains the library target masses,
    and their associated peaks found in the data file, if they match within a tolerance.

    Args:
    ----
        image_xr (xr.DataArray): A data structure which holds all the images for each peak.
        library_peak_df (pd.DataFrame): The library of interest to match the observed peaks with.
        ppm (int): The ppm for an acceptable mass error range between the observed mass and any target
        mass in the library.
        extraction_dir (Path): The directory to save extracted data in.
        adducts (bool, optional): Add adducts together. Defaults to False. (Not implemented feature)

    Returns:
    -------
        pd.DataFrame: Contains the peak, the library target mass, a boolean stating if a match was found
        or not, the composition name and the mass error if a match was found or not.
    """
    peak_df = pd.DataFrame({"peak": image_xr.peak.to_numpy()})
    match_fun = partial(_matching_vec, library_peak_df=library_peak_df, ppm=ppm)

    peak_df[["lib_mz", "matched", "composition", "mass_error"]] = peak_df["peak"].apply(
        lambda row: match_fun(row)
    )
    lib_matched_dir: Path = extraction_dir / "library_matched"
    if not os.path.exists(lib_matched_dir):
        lib_matched_dir.mkdir(parents=True, exist_ok=True)

    peak_df.to_csv(lib_matched_dir / "matched_peaks.csv")

    return peak_df


def generate_glycan_mask(
    imz_data: ImzMLParser,
    glycan_img_path: Path,
    glycan_mask_path: Path,
):
    """Given a glycan image, generates an equivalent mask.

    Args:
    ---
        imz_data (ImzMLParser): The imzML object, needed for coordinate identification.
        glycan_img_path (Path): Location of the .png file containing the glycan scan
        glycan_mask_path (Path): Location where the mask will be saved
    """
    validate_paths([glycan_img_path])

    glycan_img: np.ndarray = imread(glycan_img_path)
    glycan_mask: np.ndarray = np.zeros(glycan_img.shape, dtype=np.uint8)

    coords: np.ndarray = np.array([coord[:2] for coord in imz_data.coordinates])
    glycan_mask[coords[:, 1] - 1, coords[:, 0] - 1] = 255
    imsave(glycan_mask_path, glycan_mask)


def map_coordinates_to_core_name(
    imz_data: ImzMLParser,
    centroid_path: Path,
    poslog_paths: List[Path],
):
    """Maps each scanned coordinate on a slide to their respective core name (created by TSAI tiler).

    Args:
    ---
        imz_data (ImzMLParser): The imzML object, needed for coordinate identification.
        centroid_path (Path): A JSON file mapping each core name to their respective centroid.
            Generated by the TSAI tiler.
        poslog_paths (List[Path]): A list of .txt files listing all the coordinates scanned,
            needed to map coordinates to their respective core. They must be specified
            in the order of extraction.

    Returns:
    -------
        pd.DataFrame:
            Maps each coordinate to their core
    """
    validate_paths([centroid_path] + poslog_paths)

    coords: np.ndarray = np.array([coord[:2] for coord in imz_data.coordinates])
    region_core_info: pd.DataFrame = pd.DataFrame(columns=["Region", "X", "Y"])
    coord_index: int = 0
    num_regions: int = 0

    for poslog_path in poslog_paths:
        region_core_sub: pd.DataFrame = pd.read_csv(
            poslog_path,
            delimiter=" ",
            names=["Date", "Time", "Region", "PosX", "PosY", "X", "Y", "Z"],
            usecols=["Region", "X", "Y"],
            index_col=False,
            skiprows=1,
        )
        region_core_sub = region_core_sub[region_core_sub["Region"] != "__"].copy()
        extracted: pd.Series = (
            region_core_sub["Region"].str.extract(r"R(\d+)X", expand=False).astype(int) + num_regions
        )
        region_core_sub["Region"] = extracted

        coords_subset: np.ndarray = coords[coord_index : (coord_index + region_core_sub.shape[0]), :]
        region_core_sub[["X", "Y"]] = coords_subset
        region_core_info = pd.concat([region_core_info, region_core_sub])

        coord_index += region_core_sub.shape[0]
        num_regions += len(region_core_sub["Region"].unique())

    with open(centroid_path, "r") as infile:
        centroid_data: dict = json.load(infile)

    core_region_mapping: dict = {}
    for core in centroid_data["fovs"]:
        center_point: dict = core["centerPointPixels"]
        region_match: pd.Series = region_core_info.loc[
            (region_core_info["X"] == center_point["x"]) & (region_core_info["Y"] == center_point["y"]),
            "Region",
        ]
        if region_match.shape[0] == 0:
            raise ValueError(
                f"Could not find mapping of core {core['name']} to any location on the slide, "
                "please verify that you positioned the central point of the core correctly "
                "using the TSAI tiler, or that you've set the right poslog file."
            )

        core_region_mapping[region_match.values[0]] = core["name"]

    region_core_info[["Region", "X", "Y"]] = region_core_info[["Region", "X", "Y"]].astype(int)
    region_core_info["Core"] = region_core_info["Region"].map(core_region_mapping)
    return region_core_info


def generate_glycan_crop_masks(
    glycan_mask_path: Path,
    region_core_info: pd.DataFrame,
    glycan_crop_save_dir: Path,
):
    """Generates and saves masks for each core in `region_core_info` for cropping purposes.

    Args:
    ---
        glycan_mask_path (Path): The path to the glycan mask .tiff, needed to create the cropped mask.
        region_core_info (pd.DataFrame): Defines the coordinates associated with each FOV.
        glycan_crop_save_dir (Path): The directory to save the glycan crop masks.
    """
    validate_paths([glycan_mask_path])
    glycan_mask: np.ndarray = imread(glycan_mask_path)

    for core in region_core_info["Core"].unique():
        core_cropped_mask: np.ndarray = np.zeros(glycan_mask.shape, dtype=np.uint8)
        coords: np.ndarray = region_core_info.loc[region_core_info["Core"] == core, ["X", "Y"]].values
        core_cropped_mask[coords[:, 1] - 1, coords[:, 0] - 1] = 255
        imsave(Path(glycan_crop_save_dir) / f"{core}.tiff", core_cropped_mask)


def load_glycan_crop_masks(glycan_crop_save_dir: Path, cores_to_crop: Optional[List[str]] = None):
    """Generate a mask for cropping out the specified cores.

    Args:
    ---
        glycan_crop_save_dir (Path): The directory containing the glycan crop mask for each individual core.
        cores_to_crop (Optional[List[str]]): Which cores to segment out. If None, use all.

    Returns:
    -------
        np.ndarray:
            The binary segmentation mask of the glycan image
    """
    validate_paths([glycan_crop_save_dir])

    all_core_masks = remove_file_extensions(list_files(glycan_crop_save_dir, substrs=".tiff"))
    cores = cores_to_crop if cores_to_crop else all_core_masks
    verify_in_list(specified_cores=cores_to_crop, all_cores=all_core_masks)

    test_mask: np.ndarray = imread(Path(glycan_crop_save_dir) / f"{cores[0]}.tiff")
    glycan_mask: np.ndarray = np.zeros(test_mask.shape, dtype=np.uint8)

    for core in cores:
        core_mask: np.ndarray = imread(Path(glycan_crop_save_dir) / f"{core}.tiff")
        glycan_mask += core_mask

    return glycan_mask
