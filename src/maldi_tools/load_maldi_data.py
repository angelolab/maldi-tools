"""Used for loading and binning. spectra data across multiple TMA runs into a single DataFrame.

This is expected to serve as a drop-in replacement for SCiLS. Because we to combine, normalize,
and bin across multiple, we need to directly interact with pyTDFSDK over using an existing tool
like timsconvert.

TODO: need to access TIC normalization, or else add it in manually.
"""

import os
from bisect import bisect_left
from ctypes import CDLL
from pathlib import Path
from typing import Dict, List, Tuple, Union

import dask.dataframe as dd
import numpy as np
import pandas as pd
from dask.diagnostics import ProgressBar
from pyTDFSDK.classes import TsfData
from pyTDFSDK.init_tdf_sdk import init_tdf_sdk_api
from pyTDFSDK.tsf import tsf_index_to_mz, tsf_read_line_spectrum_v2

BASE_PATH = Path(os.path.dirname(os.path.realpath(__file__)))


def generate_mz_bins(min_mz: float = 800, max_mz: float = 4000) -> np.ndarray:
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
    data_path: Union[str, Path], binary_path: Union[str, Path] = os.path.join(BASE_PATH, "timsdata.dll")
) -> Tuple[CDLL, TsfData]:
    """Initialize both the connection (.dll load) and the cursor (TsfData object).

    Args:
        data_path (Union[str, pathlib.Path]):
            The path to the raw MALDI data, must end with a ".d" extension
        binary_path (Union[str, pathlib.Path]):
            The path to the TDFSDK Windows binary

    Returns:
        Tuple[ctypes.CDLL, pyTDFSDK.classes.TsfData]:
            The pyTDFSDK connection and associated cursor for the MALDI data_path provided
    """
    tdf_sdk_binary: CDLL = init_tdf_sdk_api(binary_path)
    return tdf_sdk_binary, TsfData(data_path, tdf_sdk=tdf_sdk_binary)


def extract_spectra(frame_partition_data, tsf_conn, tsf_cursor, run_name):
    """Helper function to extract data in parallel.

    Args:
    ----
        frame_data (pandas.DataFrame):
            The partition of frames to process in this cohort
        tsf_conn (ctypes.CDLL):
            The pyTDFSDK connection
        tsf_cursor (pyTDFSDK.classes.TsfData):
            The pyTDFSDK cursor to interact with the MALDI data
        run_name (str):
            The name of the run

    Returns:
    -------
        pandas.DataFrame:
            A processed "mz" and "intensity" spectra listing
    """
    total_spectra: Dict[float, float] = {}
    for index, row in frame_partition_data.iterrows():
        index_array, intensity_array = tsf_read_line_spectrum_v2(
            tdf_sdk=tsf_conn, handle=tsf_cursor.handle, frame_id=int(row["Id"])
        )
        mz_array: np.ndarray = tsf_index_to_mz(
            tdf_sdk=tsf_conn, handle=tsf_cursor.handle, frame_id=int(row["Id"]), indices=index_array
        )
        for mass_idx, mz in enumerate(mz_array):
            total_spectra[mz] = (0 if mz not in total_spectra else total_spectra[mz]) + intensity_array[
                mass_idx
            ]

    binned_spectra_df: pd.DataFrame = pd.DataFrame(
        {"mz": list(total_spectra.keys()), "intensity": list(total_spectra.values())}
    )
    binned_spectra_df["run_name"] = run_name

    return binned_spectra_df


def load_coordinate_and_spectra_data(
    run_name: str, tsf_conn: CDLL, tsf_cursor: TsfData, min_mz: float, max_mz: float
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Loads the MALDI spectra and coordinate info for a single run.

    Args:
    ----
        run_name (str):
            The name of the run
        tsf_conn (ctypes.CDLL):
            The pyTDFSDK connection
        tsf_cursor (pyTDFSDK.classes.TsfData):
            The pyTDFSDK cursor to interact with the MALDI data
        min_mz (float):
            The minimum mz range that was extracted
            NOTE: assumed to be the same across all TMAs
        max_mz (float):
            The maximum mz range that was extracted
            NOTE: assumed to be the same across all TMAs
    Returns:
    -------
        Tuple[dask.dataframe, dask.dataframe]:
            The coordinate DataFrame identifying each spot and the DataFrame mapping m/z to intensity
    """
    print(f"Extracting data from {run_name}")

    # this is going to be the new poslog
    frame_df: dd = dd.from_pandas(tsf_cursor.analysis["MaldiFrameInfo"], npartitions=16)
    frame_df["run_name"] = run_name
    mz_bins: np.ndarray = generate_mz_bins(min_mz, max_mz)

    # extract the m/z data
    tsf_data: dd = dd.from_pandas(tsf_cursor.analysis["Frames"], npartitions=16)

    # process the data in parallel
    print("Extracting raw spectra")
    with ProgressBar():
        bin_spectra_df: dd = tsf_data.map_partitions(
            extract_spectra, tsf_conn, tsf_cursor, run_name
        ).persist()

    # bin the data by m/z
    print("Binning spectra")
    with ProgressBar():
        bin_spectra_df["mz_bin"] = bin_spectra_df.map_partitions(
            lambda row: bisect_left(mz_bins, row["mz"])
        ).persist()

    # sum up the intensity per bin
    print("Summing intensity by binned spectra values")
    with ProgressBar():
        bin_spectra_df = bin_spectra_df.groupby("mz_bin")["intensity"].sum().reset_index().persist()

    return frame_df, bin_spectra_df


def run_coordinate_spectra_extraction(
    data_paths: List[Union[str, Path]], min_mz: float = 800, max_mz: float = 4000
):
    """For each run specified, extract the coordinate and spectra info.

    Args:
    ----
        data_paths (List[Union[str, pathlib.Path]]):
            The list of paths to each TMA MALDI run to include, each must have a .d extension
        min_mz (float):
            The minimum mz range that was extracted
            NOTE: assumed to be the same across all TMAs
        max_mz (float):
            The maximum mz range that was extracted
            NOTE: assumed to be the same across all TMAs
    Returns:
    -------
        Tuple[dd.dataframe, dd.dataframe]:
            Two DataFrames, one identifying the coordinate info, one identifying the spectra info.
    """
    # data path validation
    for dp in data_paths:
        if os.path.splitext(dp)[1] != ".d":
            raise ValueError(f"Invalid data_path {dp}: folder specified must end with '.d'")
        if not os.path.exists(dp):
            raise FileNotFoundError(f"Data path {dp} does not exist")

    frame_data: dd = dd.from_pandas(pd.DataFrame(), npartitions=32)
    spectra_data: dd = dd.from_pandas(pd.DataFrame(), npartiitons=32)

    # # build the full fram (coordinate) and spectra info from each data path specifieds
    for i, dp in enumerate(data_paths):
        run_name = os.path.basename(os.path.splitext(dp)[0])
        tsf_conn, tsf_cursor = init_tsf_load_object(data_path=dp)
        frame_run, spectra_run = load_coordinate_and_spectra_data(
            run_name=run_name, tsf_conn=tsf_conn, tsf_cursor=tsf_cursor, min_mz=min_mz, max_mz=max_mz
        )
        frame_data = dd.concat([frame_data, frame_run])
        spectra_data = dd.concat([spectra_data, spectra_run])

    return frame_data, spectra_data
