"""Used for loading and binning. spectra data across multiple TMA runs into a single DataFrame.

This is expected to serve as a drop-in replacement for SCiLS. Because we to combine, normalize,
and bin across multiple, we need to directly interact with pyTDFSDK over using an existing tool
like timsconvert.

TODO: need to access TIC normalization, or else add it in manually.
"""

import os
from bisect import bisect_left
from concurrent.futures import ProcessPoolExecutor, as_completed
from ctypes import CDLL
from pathlib import Path
from typing import Dict, List, Tuple, Union

import numpy as np
import pandas as pd
from pyTDFSDK.classes import TsfData
from pyTDFSDK.init_tdf_sdk import init_tdf_sdk_api
from pyTDFSDK.tsf import tsf_index_to_mz, tsf_read_line_spectrum_v2

# Path to the TDFSDK binary file
BASE_PATH = Path(os.path.dirname(os.path.realpath(__file__)))


def generate_mz_bins(
    min_mz: float = 800,
    max_mz: float = 4000
) -> np.ndarray:
    """Given a range of mz values, generate the bins as would be calculated by Bruker's SCiLS.

    To convert from an mz value to its corresponding bin size, use mz_val / 200 / 1000, since the
    bin sizes are measured in mDa.

    Note that the computed values may be slightly off the SCiLS values, as the latter has floating
    point errors.

    Args:
    ----
        min_mz (float): The minimum mz extracted to start the binning at
        max_mz (float): The maximum mz extracted to start the binning at
    Returns:
    -------
        np.ndarray: The list of mz values to use for binning the observed mz values.
    """
    mz_bins: List[float] = [min_mz]
    while True:
        mz_right: float = mz_bins[-1] + mz_bins[-1] / 200000

        if mz_right >= max_mz:
            if mz_bins[-1] != max_mz:
                mz_bins.append(max_mz)
            break
        mz_bins.append(mz_right)

    return np.array(mz_bins)


def init_tsf_load_object(
    maldi_data_path: Union[str, Path],
    tdf_sdk_binary: CDLL
) -> TsfData:
    """Initialize the cursor (TsfData object).

    Args:
    ----
        maldi_data_path (Union[str, pathlib.Path]):
            The path to the raw MALDI data, must end with a `.d` extension
        tdf_sdk_binary (CDLL):
            The dynamically loaded library created for the `TsfData` object

    Returns:
    -------
        pyTDFSDK.classes.TsfData:
            The pyTDFSDK cursor for the `maldi_data_path` provided
    """
    return TsfData(maldi_data_path, tdf_sdk=tdf_sdk_binary)


def extract_maldi_tsf_data(
    maldi_data_path: Union[str, Path],
    min_mz: float = 800,
    max_mz: float = 4000
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Extract the spectra data for a particular MALDI run.

    Args:
    ====
        maldi_data_path (Union[str, Path]):
            The path to the raw MALDI data, must end with a `.d` extension
        min_mz (float):
            The minimum m/z value observed during the run
        max_mz (float):
            The maximum m/z value observed during the run

    Returns:
    -------
        Tuple[pandas.DataFrame, pandas.DataFrame]:
            Two DataFrames containing the spectra and poslog info across the run respectively
    """
    tdf_sdk_binary: CDLL = init_tdf_sdk_api(os.path.join(BASE_PATH, "timsdata.dll"))
    tsf_cursor: TsfData = init_tsf_load_object(maldi_data_path, tdf_sdk_binary)

    mz_bins: np.ndarray = generate_mz_bins(min_mz, max_mz)
    spectra_dict: Dict[float, float] = {}
    tsf_spot_info: pd.DataFrame = tsf_cursor.analysis["Frames"]
    for sid in tsf_spot_info["Id"].values:
        index_arr, intensity_arr = tsf_read_line_spectrum_v2(
            tdf_sdk=tdf_sdk_binary, handle=tsf_cursor.handle, frame_id=sid
        )
        mz_arr: np.ndarray = tsf_index_to_mz(
            tdf_sdk=tdf_sdk_binary, handle=tsf_cursor.handle, frame_id=sid, indices=index_arr
        )

        for mz, intensity in zip(mz_arr, intensity_arr):
            binned_mz = mz_bins[bisect_left(mz_bins, mz)]
            spectra_dict[binned_mz] = (
                0 if binned_mz not in spectra_dict else spectra_dict[binned_mz]
            ) + intensity

    run_name = os.path.basename(os.path.splitext(maldi_data_path)[0])
    tsf_spectra: pd.DataFrame = pd.DataFrame(spectra_dict.items(), columns=["m/z", "intensity"])
    tsf_spectra["run_name"] = run_name
    tsf_spectra.sort_values(by="m/z", inplace=True)

    tsf_poslog: pd.DataFrame = tsf_cursor.analysis["MaldiFrameInfo"]
    tsf_poslog["run_name"] = run_name

    return tsf_spectra, tsf_poslog


def extract_maldi_run_spectra(
    maldi_paths: List[Union[str, Path]],
    min_mz: float = 800,
    max_mz: float = 4000,
    num_workers: int = 16
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Extract the full spectra and corresponding poslog information from the MALDI files.

    Args:
    ----
        maldi_paths (List[Union[str, pathlib.Path]]):
            The list of MALDI runs to use, must be `.d` folders
        min_mz (float):
            The minimum m/z value observed during the run
        max_mz (float):
            The maximum m/z value observed during the run
        num_workers (int):
            The number of workers to use for the process, default to all

    Returns:
    -------
        Tuple[pandas.DataFrame, pandas.DataFrame:
            Two DataFrames containing the spectra and poslog info across all runs respectively
    """
    if num_workers <= 0:
        raise ValueError("num_workers specified must be positive")

    poslog_df: pd.DataFrame = pd.DataFrame()
    spectra_df: pd.DataFrame = pd.DataFrame()

    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        future_maldi_data = {
            executor.submit(extract_maldi_tsf_data, mp, min_mz, max_mz): mp for mp in maldi_paths
        }

        for future in as_completed(future_maldi_data):
            mp = future_maldi_data[future]
            try:
                poslog_mp, spectra_mp = future.result()
                poslog_df = pd.concat([poslog_df, poslog_mp])
                spectra_df = pd.concat([spectra_df, spectra_mp])
            except Exception as e:
                print(f"Exception raised while processing {mp}")

    poslog_df = poslog_df.reset_index(drop=True)
    spectra_df = spectra_df.reset_index(drop=True)

    return poslog_df, spectra_df
