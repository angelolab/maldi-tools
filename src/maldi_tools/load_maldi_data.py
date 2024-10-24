# """Used for loading and binning. spectra data across multiple TMA runs into a single DataFrame.

# This is expected to serve as a drop-in replacement for SCiLS. Because we to combine, normalize,
# and bin across multiple, we need to directly interact with pyTDFSDK over using an existing tool
# like timsconvert.

# TODO: need to access TIC normalization, or else add it in manually.
# """


import joblib
import os
import threading
from bisect import bisect_left
import copy
from ctypes import CDLL
from dask import delayed
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import dask
import dask.dataframe as dd
import mapply
import numpy as np
import pandas as pd
from dask.diagnostics import ProgressBar
from dask.distributed import Client
from pyTDFSDK.classes import TsfData
from pyTDFSDK.init_tdf_sdk import init_tdf_sdk_api
from pyTDFSDK.tsf import tsf_index_to_mz, tsf_read_line_spectrum_v2

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


# def init_tsf_load_object(data_path: Union[str, Path]) -> TsfData:
#     """Initialize both the connection (.dll load) and the cursor (TsfData object).

#     Args:
#     ----
#         data_path (Union[str, pathlib.Path]):
#             The path to the raw MALDI data, must end with a ".d" extension
#         binary_path (Union[str, pathlib.Path]):
#             The path to the TDFSDK Windows binary

#     Returns:
#     -------
#         pyTDFSDK.classes.TsfData:
#             The pyTDFSDK cursor for the MALDI data_path provided
#     """
#     global TSF_CURSOR
#     TSF_CURSOR = TsfData(data_path, tdf_sdk=TDF_SDK_BINARY)


import dask.dataframe as dd
import pandas as pd
from pyTDFSDK.classes import TsfData
from pyTDFSDK import tsf_read_line_spectrum
from functools import lru_cache

_worker_sdk_object = None

BASE_PATH = Path(os.path.dirname(os.path.realpath(__file__)))
# mapply.init(n_workers=-1)
# _thread_local_storage = threading.local()

# TDF_SDK_BINARY: CDLL = init_tdf_sdk_api(os.path.join(BASE_PATH, "timsdata.dll"))
# TSF_CURSOR: Optional[TsfData] = None

# def initialize_sdk():
#     """Initialize the SDK object once per worker"""
#     global _worker_sdk_object
#     if _worker_sdk_object is None:
#         _worker_sdk_object = pyTDFSDK.init_df_sdk.init_tdf_sdk_api()
#     return _worker_sdk_object

# client.run(initialize_sdk)

@lru_cache(maxsize=None)
def get_sdk_object():
    """Return the SDK object, initializing if necessary."""
    global _worker_sdk_object
    if _worker_sdk_object is None:
        _worker_sdk_object = pyTDFSDK.init_df_sdk.init_tdf_sdk_api(os.path.join(BASE_PATH, "timsdata.dll"))
    return _worker_sdk_object

def process_spot_range(path_to_d_folder, spot_df_partition, mz_bins):
    """Processes a partition of spot_df by creating a local TsfData object and extracting spectra"""
    sdk_object = get_sdk_object()

    # Initialize TsfData inside the partition to avoid passing it between workers
    tsf_data = TsfData(sdk_object, path_to_d_folder)

    results = []
    for spot_id in spot_df_partition['Id']:  # Assuming 'Id' is the column name
        mz_list, intensity_list = tsf_read_line_spectrum(tsf_data, spot_id)
        binned_spectra = bin_mz_values(mz_list, intensity_list, mz_bins)
        results.append(binned_spectra)

    return results

def bin_mz_values(mz_list, intensity_list, mz_bins):
    """Optimized m/z binning function"""
    binned_intensities = {bin_value: 0 for bin_value in mz_bins}
    
    for mz, intensity in zip(mz_list, intensity_list):
        bin_idx = bisect_left(mz_bins, mz)
        bin_value = mz_bins[bin_idx]
        binned_intensities[bin_value] += intensity
    
    return binned_intensities

def process_partition(path_to_d_folder, mz_bins):
    """Process each .d file: extract DataFrames and partition spot_df"""
    sdk_object = get_sdk_object()
    tsf_data = TsfData(sdk_object, path_to_d_folder)
    
    # Extract both DataFrames
    frame_df = tsf_data.analysis["MaldiFrameInfo"]
    spot_df = tsf_data.analysis["Frames"]
    
    # Partition spot_df over Dask
    spot_ddf = dd.from_pandas(spot_df, npartitions=10)  # Adjust the number of partitions
    
    # Map the spot partition processing to Dask
    spot_results = spot_ddf.map_partitions(
        lambda df: process_spot_range(path_to_d_folder, df, mz_bins)
    ).compute()

    return frame_df, spot_results


def process_paths(paths, min_mz=800, max_mz=4000):
    # Example m/z bin list (predefined)
    # mz_bins = [100, 150, 200, 250, 300]  # Adjust based on your binning requirements
    mz_bins = generate_mz_bins(min_mz, max_mz)

    # Processing each .d file with partitioning
    final_results = []
    for path in paths:
        frame_df, spot_results = process_partition(path, mz_bins)
        final_results.append((frame_df, spot_results))

    frame_df = pd.concat([x[0] for x in final_results])
    spot_df = pd.concat([x[1] for x in final_results])

    return frame_df, spot_df

# import joblib
# import os
# import threading
# from bisect import bisect_left
# import copy
# from ctypes import CDLL
# from dask import delayed
# from pathlib import Path
# from typing import Dict, List, Optional, Tuple, Union

# import dask
# import dask.dataframe as dd
# import mapply
# import numpy as np
# import pandas as pd
# from dask.diagnostics import ProgressBar
# from pyTDFSDK.classes import TsfData
# from pyTDFSDK.init_tdf_sdk import init_tdf_sdk_api
# from pyTDFSDK.tsf import tsf_index_to_mz, tsf_read_line_spectrum_v2

# BASE_PATH = Path(os.path.dirname(os.path.realpath(__file__)))
# mapply.init(n_workers=-1)
# _thread_local_storage = threading.local()

# TDF_SDK_BINARY: CDLL = init_tdf_sdk_api(os.path.join(BASE_PATH, "timsdata.dll"))
# TSF_CURSOR: Optional[TsfData] = None


# def generate_mz_bins(min_mz: float = 800, max_mz: float = 4000) -> np.ndarray:
#     """Given a range of mz values, generate the bins as would be calculated by Bruker's SCiLS.

#     To convert from an mz value to its corresponding bin size, use mz_val / 200 / 1000, since the
#     bin sizes are measured in mDa.

#     Note that the computed values may be slightly off the SCiLS values, as the latter has floating
#     point errors.

#     Args:
#     ----
#         min_mz (float): The minimum mz extracted to start the binning at
#         max_mz (float): The maximum mz extracted to start the binning at
#     Returns:
#     -------
#         np.ndarray: The list of mz values to use for binning the observed mz values.
#     """
#     mz_bins: List[float] = [min_mz]
#     while True:
#         mz_right: float = mz_bins[-1] + mz_bins[-1] / 200000

#         if mz_right >= max_mz:
#             if mz_bins[-1] != max_mz:
#                 mz_bins.append(max_mz)
#             break
#         mz_bins.append(mz_right)

#     return np.array(mz_bins)


# def init_tsf_load_object(data_path: Union[str, Path]) -> TsfData:
#     """Initialize both the connection (.dll load) and the cursor (TsfData object).

#     Args:
#     ----
#         data_path (Union[str, pathlib.Path]):
#             The path to the raw MALDI data, must end with a ".d" extension
#         binary_path (Union[str, pathlib.Path]):
#             The path to the TDFSDK Windows binary

#     Returns:
#     -------
#         pyTDFSDK.classes.TsfData:
#             The pyTDFSDK cursor for the MALDI data_path provided
#     """
#     global TSF_CURSOR
#     TSF_CURSOR = TsfData(data_path, tdf_sdk=TDF_SDK_BINARY)


# # def get_tsf_connection(
# #     data_path: Union[str, Path], binary_path: Union[str, Path] = os.path.join(BASE_PATH, "timsdata.dll")
# # ) -> Tuple[CDLL, TsfData]:
# #     """Initialize the TSF connection in a thread-safe, reusable environment

# #     Args:
# #         data_path (Union[str, pathlib.Path]):
# #             The path to the raw MALDI data, must end with a ".d" extension
# #         binary_path (Union[str, pathlib.Path]):
# #             The path to the TDFSDK Windows binary

# #     Returns:
# #         Tuple[ctypes.CDLL, pyTDFSDK.classes.TsfData]:
# #             The pyTDFSDK connection and associated cursor for the MALDI data_path provided
# #     """
# #     if not hasattr(_thread_local_storage, "tsf_conn"):
# #         _thread_local_storage.tsf_conn, _thread_local_storage.tsf_cursor = init_tsf_load_object(
# #             data_path, binary_path
# #         )
    
# #     return _thread_local_storage.tsf_conn, _thread_local_storage.tsf_cursor


# def extract_spot_data(spot_id: int, total_spectra: Dict[float, float]):
#     """Extract the spectra for one spot

#     Args:
#     ----
#         tsf_cursor (pyTDFSDK.classes.TsfData):
#             The pyTDFSDK cursor to interact with the MALDI data
#         spot_id (int):
#             The id of the spot to extract
#         total_spectra (Dict[float, float]):
#             The mapping of m/z values to spectra, will be written to asynchronously
#     """
#     # print(f"Running on spot {spot_id}")
#     # tsf_cursor = joblib.load("tsf_object.joblib")
#     index_array, intensity_array = tsf_read_line_spectrum_v2(
#         tdf_sdk=TDF_SDK_BINARY, handle=TSF_CURSOR.handle, frame_id=int(spot_id), profile_buffer_size=1024 ** 1024
#     )
#     mz_array: np.ndarray = tsf_index_to_mz(
#         tdf_sdk=TDF_SDK_BINARY, handle=TSF_CURSOR.handle, frame_id=int(spot_id), indices=index_array
#     )
#     for mass_idx, mz in enumerate(mz_array):
#         total_spectra[mz] = (0 if mz not in total_spectra else total_spectra[mz]) + intensity_array[
#             mass_idx
#         ]


# def extract_spectra(
#     frame_partition_data: pd.DataFrame, data_path: Union[str, Path], run_name: str
# ):
#     """Helper function to extract data in parallel.

#     Args:
#     ----
#         frame_partition_data (pandas.DataFrame):
#             The partition of frames to process in this cohort
#         data_path (Union[str, Path]):
#             The path to the MALDI TMA run, must have a .d extension
#         run_name (str):
#             The name of the run (the file name of the data_path with ".d" removed

#     Returns:
#     -------
#         pandas.DataFrame:
#             A processed "mz" and "intensity" spectra listing
#     """
#     tdf_sdk_binary = copy.copy(TDF_SDK_BINARY)
#     tsf_cursor = copy.copy(TSF_CURSOR)
#     print("The partition to look at is:")
#     print(frame_partition_data)
#     total_spectra: Dict[float, float] = {}
#     # tsf_cursor = init_tsf_load_object(data_path=data_path)

#     # frame_partition_data.mapply(
#     #     lambda row: extract_spot_data(row["Id"], total_spectra),
#     #     axis=1
#     # )
#     for index, row in frame_partition_data.iterrows():
#         index_array, intensity_array = tsf_read_line_spectrum_v2(
#             tdf_sdk=tdf_sdk_binary, handle=tsf_cursor.handle, frame_id=int(row["Id"])
#         )
#         # print(f"Running on spot {row['Id']}")
#         # index_array = None
#         # intensity_array = None
#         # while index_array is None and intensity_array is None:
#         #     index_array, intensity_array = tsf_read_line_spectrum_v2(
#         #             tdf_sdk=TDF_SDK_BINARY, handle=TSF_CURSOR.handle, frame_id=int(row["Id"])
#         #     )
#             # try:
#             #     index_array, intensity_array = tsf_read_line_spectrum_v2(
#             #         tdf_sdk=TDF_SDK_BINARY, handle=TSF_CURSOR.handle, frame_id=int(row["Id"])
#             #     )
#             # except (OSError, RuntimeError) as e:
#             #     print(f"Bad read on {row['Id']}, trying again")
#             # except Exception as e:
#             #     print(f"Unknown exception on {row['Id']}, trying again")
#         mz_array: np.ndarray = tsf_index_to_mz(
#             tdf_sdk=tdf_sdk_binary, handle=tsf_cursor.handle, frame_id=int(row["Id"]), indices=index_array
#         )
#         for mass_idx, mz in enumerate(mz_array):
#             total_spectra[mz] = (0 if mz not in total_spectra else total_spectra[mz]) + intensity_array[
#                 mass_idx
#             ]

#     binned_spectra_df: pd.DataFrame = pd.DataFrame(
#         {"mz": list(total_spectra.keys()), "intensity": list(total_spectra.values())}
#     )
#     binned_spectra_df["run_name"] = run_name

#     return binned_spectra_df


# def load_coordinate_and_spectra_data_old(
#     data_path: str, min_mz: float, max_mz: float
# ) -> Tuple[pd.DataFrame, pd.DataFrame]:
#     """Loads the MALDI spectra and coordinate info for a single run.

#     Args:
#     ----
#         data_path (str):
#             The path to the MALDI TMA run, must have a .d extension
#         min_mz (float):
#             The minimum mz range that was extracted
#             NOTE: assumed to be the same across all TMAs
#         max_mz (float):
#             The maximum mz range that was extracted
#             NOTE: assumed to be the same across all TMAs
#     Returns:
#     -------
#         Tuple[dask.dataframe, dask.dataframe]:
#             The coordinate DataFrame identifying each spot and the DataFrame mapping m/z to intensity
#     """
#     run_name: str = os.path.basename(os.path.splitext(data_path)[0])
#     print(f"Extracting data from {run_name}")

#     init_tsf_load_object(data_path=data_path)

#     # pickle TsfData to joblib to ensure proper loading in multithread/dask
#     # joblib.dump(tsf_cursor, "tsf_object.joblib")

#     # frame_df = tsf_cursor.analysis["MaldiFrameInfo"]
#     # tsf_id_data = tsf_cursor.analysis["Frames"]

#     # total_spectra: Dict[float, float] = {}
#     # tsf_id_data.mapply(
#     #     lambda row: extract_spot_data(tsf_conn, tsf_cursor, row["Id"], total_spectra),
#     #     axis=1
#     # )

#     # binned_spectra_df: pd.DataFrame = pd.DataFrame(
#     #     {"mz": list(total_spectra.keys()), "intensity": list(total_spectra.values())}
#     # )
#     # binned_spectra_df["run_name"] = run_name

#     # this is going to be the new poslog
#     frame_df: dd = dd.from_pandas(TSF_CURSOR.analysis["MaldiFrameInfo"], npartitions=16)
#     frame_df["run_name"] = run_name
#     mz_bins: np.ndarray = generate_mz_bins(min_mz, max_mz)

#     # extract the m/z data
#     tsf_id_data: dd = dd.from_pandas(TSF_CURSOR.analysis["Frames"].iloc[:, 0:2], npartitions=16)
#     print(tsf_id_data)

#     # process the data in parallel
#     print("Extracting raw spectra")
#     with ProgressBar():
#         bin_spectra_df: dd = tsf_id_data.map_partitions(
#             extract_spectra, data_path, run_name
#         ).persist()

#     # bin the data by m/z
#     print("Binning spectra")
#     with ProgressBar():
#         bin_spectra_df["mz_bin"] = bin_spectra_df.map_partitions(
#             lambda row: bisect_left(mz_bins, row["mz"])
#         ).persist()

#     # sum up the intensity per bin
#     print("Summing intensity by binned spectra values")
#     with ProgressBar():
#         bin_spectra_df = bin_spectra_df.groupby("mz_bin")["intensity"].sum().reset_index().persist()

#     os.remove("tsf_object.joblib")

#     return frame_df, bin_spectra_df


# def load_coordinate_and_spectra_data(
#     data_path: str, min_mz: float, max_mz: float
# ):
#     run_name: str = os.path.basename(os.path.splitext(data_path)[0])
#     print(f"Extracting data from {run_name}")

#     init_tsf_load_object(data_path=data_path)
#     frame_df = TSF_CURSOR.analysis["MaldiFrameInfo"]
#     tsf_id_data = TSF_CURSOR.analysis["Frames"]

#     chunks = []
#     n_chunks = 16
#     chunk_size = len(tsf_id_data) // n_chunks

#     for i in range(n_chunks):
#         start_idx = i * chunk_size
#         if i == n_chunks - 1:  # Handle the last chunk (may have more rows)
#             chunk = tsf_id_data.iloc[start_idx:]
#         else:
#             chunk = tsf_id_data.iloc[start_idx:start_idx + chunk_size]

#         spectra_comp = delayed(extract_spectra)(chunk, data_path, run_name)
#         chunks.append(spectra_comp)

#     print("Computing chunks")
#     full_spectra = dask.persist(*chunks)

#     return frame_df, full_spectra


# def run_coordinate_spectra_extraction(
#     data_paths: List[Union[str, Path]], min_mz: float = 800, max_mz: float = 4000
# ):
#     """For each run specified, extract the coordinate and spectra info.

#     Args:
#     ----
#         data_paths (List[Union[str, pathlib.Path]]):
#             The list of paths to each TMA MALDI run to include, each must have a .d extension
#         min_mz (float):
#             The minimum mz range that was extracted
#             NOTE: assumed to be the same across all TMAs
#         max_mz (float):
#             The maximum mz range that was extracted
#             NOTE: assumed to be the same across all TMAs
#     Returns:
#     -------
#         Tuple[dd.dataframe, dd.dataframe]:
#             Two DataFrames, one identifying the coordinate info, one identifying the spectra info.
#     """
#     # data path validation
#     for dp in data_paths:
#         if os.path.splitext(dp)[1] != ".d":
#             raise ValueError(f"Invalid data_path {dp}: folder specified must end with '.d'")
#         if not os.path.exists(dp):
#             raise FileNotFoundError(f"Data path {dp} does not exist")

#     frame_data: dd = dd.from_pandas(pd.DataFrame(), npartitions=32)
#     spectra_data: dd = dd.from_pandas(pd.DataFrame(), npartitions=32)

#     # # build the full fram (coordinate) and spectra info from each data path specifieds
#     for i, dp in enumerate(data_paths):
#         # run_name = os.path.basename(os.path.splitext(dp)[0])
#         # tsf_conn, tsf_cursor = init_tsf_load_object(data_path=dp)
#         frame_run, spectra_run = load_coordinate_and_spectra_data(
#             data_path=dp,
#             min_mz=min_mz,
#             max_mz=max_mz
#         )
#         frame_data = dd.concat([frame_data, frame_run])
#         spectra_data = dd.concat([spectra_data, spectra_run])

#     return frame_data, spectra_data





# import joblib
# import os
# import threading
# from bisect import bisect_left
# from concurrent.futures import ProcessPoolExecutor
# import copy
# from ctypes import CDLL
# from dask import delayed
# from dataclasses import dataclass
# from pathlib import Path
# from typing import Dict, List, Optional, Tuple, Union

# import dask
# import dask.dataframe as dd
# import logging
# import mapply
# import numpy as np
# import pandas as pd
# from dask.diagnostics import ProgressBar
# from pyTDFSDK.classes import TsfData
# from pyTDFSDK.init_tdf_sdk import init_tdf_sdk_api
# from pyTDFSDK.tsf import tsf_index_to_mz, tsf_read_line_spectrum_v2

# logging.basicConfig(
#     level=logging.INFO,
#     format='%(processName)s - %(message)s',
#     filename="maldi_log.txt",
#     filemode="a"
# )

# BASE_PATH = Path(os.path.dirname(os.path.realpath(__file__)))
# TDF_SDK_API = init_tdf_sdk_api(os.path.join(BASE_PATH, "timsdata.dll"))


# def generate_mz_bins(min_mz: float = 800, max_mz: float = 4000) -> np.ndarray:
#     """Given a range of mz values, generate the bins as would be calculated by Bruker's SCiLS.

#     To convert from an mz value to its corresponding bin size, use mz_val / 200 / 1000, since the
#     bin sizes are measured in mDa.

#     Note that the computed values may be slightly off the SCiLS values, as the latter has floating
#     point errors.

#     Args:
#     ----
#         min_mz (float): The minimum mz extracted to start the binning at
#         max_mz (float): The maximum mz extracted to start the binning at
#     Returns:
#     -------
#         np.ndarray: The list of mz values to use for binning the observed mz values.
#     """
#     mz_bins: List[float] = [min_mz]
#     while True:
#         mz_right: float = mz_bins[-1] + mz_bins[-1] / 200000

#         if mz_right >= max_mz:
#             if mz_bins[-1] != max_mz:
#                 mz_bins.append(max_mz)
#             break
#         mz_bins.append(mz_right)

#     return np.array(mz_bins)


# import dask.dataframe as dd
# import pandas as pd
# from multiprocessing import Pool
# from ctypes import CDLL
# import numpy as np
# from concurrent.futures import ProcessPoolExecutor

# # Define the external functions
# def init_tdf_sdk_api():
#     """Loads the .dll binary and returns the CDLL object."""
#     return CDLL('path/to/your.dll')

# class TsfData:
#     """Represents the object required by tsf_read_line_spectrum_v2."""
#     def __init__(self, api):
#         self.api = api  # The result of init_tdf_sdk_api

# def tsf_read_line_spectrum_v2(tsf_data, pixel_index):
#     """Extracts the spectra for a given pixel (index)."""
#     # Call the external function and return mass/intensity list.
#     # This is a placeholder for the actual function.
#     # Returns a list of (mass, intensity) tuples
#     return [(m, np.random.random()) for m in range(10000)]  # Mock example

# def process_pixel_spectra(tsf_data, pixel_index):
#     """Processes a single pixel, extracts its spectra, and bins the mass ranges."""
#     spectra = tsf_read_line_spectrum_v2(tsf_data, pixel_index)
#     df = pd.DataFrame(spectra, columns=['mass', 'intensity'])
    
#     # Define your mass bins
#     mass_bins = np.arange(0, 10000, 100)  # Example bins, adjust accordingly
#     df['mass_bin'] = pd.cut(df['mass'], bins=mass_bins)
    
#     # Group by 'mass_bin' and sum the intensities
#     df_grouped = df.groupby('mass_bin').agg({'intensity': 'sum'}).reset_index()
    
#     return df_grouped

# def process_spot_chunk(spot_chunk_df, maldi_path):
#     """Processes a chunk of spots (pixels)."""
#     tsf_data = TsfData(maldi_path, TDF_SDK_API)  # Create a new instance for this worker
#     results = []
#     for spot_id in spot_chunk_df["Id"].values:
#         pixel_index = spot_chunk_df.loc[idx, 'pixel_index']
#         pixel_spectra_df = process_pixel_spectra(tsf_data, pixel_index)
#         results.append(pixel_spectra_df)
    
#     return pd.concat(results)

# def process_spot_dataframe(maldi_path, spot_df, n_workers=4):
#     """Processes the entire spot DataFrame using Dask and parallel processing."""
#     # Convert pandas DataFrame to Dask DataFrame
#     spot_df_sub = spot_df.iloc[:, 0:2]
#     ddf = dd.from_pandas(spot_df_sub, npartitions=n_workers)
    
#     # Parallel processing of each chunk
#     results = ddf.map_partitions(process_spot_chunk, maldi_path).compute()
    
#     return pd.concat(results)

# def process_multiple_spot_dfs(spot_dfs, n_workers=4):
#     # """Processes multiple spot DataFrames in parallel."""
#     # api = init_tdf_sdk_api()  # Load the API once and pass it around
#     # with ProcessPoolExecutor(max_workers=n_workers) as executor:
#     #     futures = [executor.submit(process_spot_dataframe, df, api, n_workers)
#     #                for df in spot_dfs]
#     #     results = [f.result() for f in futures]
    
#     # return results
#     for maldi_path, spot_df in spot_dfs.items():
#         spectra_df = process_spot_dataframe(maldi_path, spot_df)


# def load_df_data(maldi_paths):
#     frame_df = {}
#     spot_df = {}
#     for mp in maldi_paths:
#         tsf_cursor = TsfData(mp, TDF_SDK_API)
#         frame_df[mp] = tsf_cursor.analysis["MaldiFrameInfo"]
#         spot_df[mp] = tsf_cursor.analysis["Frames"]

#     return frame_df, spot_df


# # Example usage
# if __name__ == "__main__":
#     # Example spot DataFrame
#     spot_df = pd.DataFrame({'pixel_index': range(5000)})  # Simplified example
    
#     # Process the single DataFrame
#     processed_df = process_spot_dataframe(spot_df, init_tdf_sdk_api(), n_workers=8)
    
#     # If you have multiple spot DataFrames
#     spot_dfs = [spot_df, spot_df.copy(), spot_df.copy()]  # Mock multiple DataFrames
#     processed_results = process_multiple_spot_dfs(spot_dfs, n_workers=4)


# @dataclass
# class TsfDataFrame:
#     tsf_cursor: TsfData
#     frame_df: pd.DataFrame
#     spot_df: pd.DataFrame
#     spectra_df: pd.DataFrame


# TDF_SDK_BINARY = init_tdf_sdk_api(os.path.join(BASE_PATH, "timsdata.dll"))


# class TsfDataLoader:
#     def __init__(self, maldi_run_paths, min_mz=800, max_mz=4000):
#         # self.tdf_sdk_binary = init_tdf_sdk_api(os.path.join(BASE_PATH, "timsdata.dll"))
#         self.maldi_run_paths = maldi_run_paths
#         self.full_tsf_data = {}
#         self.mz_bins = generate_mz_bins(min_mz, max_mz)

#         # for mrp in maldi_run_paths:
#         #     print(f"Initializing path {mrp}")
#         #     run_name = os.path.basename(os.path.splitext(mrp)[0])
#         #     tsf_cursor = TsfData(mrp, self.tdf_sdk_binary)
#         #     self.tsf_dfs[run_name] = TsfDataFrame(
#         #         tsf_cursor, tsf_cursor.analysis["MaldiFrameInfo"], tsf_cursor.analysis["Frames"], None
#         #     )

#     def process_pixel_spectra(self, tsf_cursor, spot_index):
#         total_spectra = {}

#         index_array, intensity_array = tsf_read_line_spectrum_v2(
#             tdf_sdk=self.tdf_sdk_binary, handle=tsf_cursor.handle, frame_id=int(spot_index)
#         )
#         mz_array: np.ndarray = tsf_index_to_mz(
#             tdf_sdk=TDF_SDK_BINARY, handle=TSF_CURSOR.handle, frame_id=int(spot_index), indices=index_array
#         )
#         for mass_idx, mz in enumerate(mz_array):
#             mz_bin_index: int = bisect.bisect_left(self.mz_bins, mz)
#             if mz_bin_index > 0 and mz_bins[mz_bin_index - 1] >= mz:
#                 mz_bin_index -= 1
#             mz_bin = self.mz_bins[mz_bin_index]
#             total_spectra[mz_bin] = (0 if mz_bin not in total_spectra else total_spectra[mz_bin]) + intensity_array[mass_idx]

#         spectra_df = pd.DataFrame(total_spectra, columns=["m/z", "intensity"])
#         return spectra_df

#     def process_spot_chunk(self, spot_chunk_df, run_name):
#         """Processes a chunk of spots (pixels)."""
#         tsf_cursor = TsfData(maldi_run_path, self.tdf_sdk_binary)
#         tsf_cursor = self.tsf_dfs[run_name].tsf_cursor
#         results = []
#         for spot_index in spot_chunk_df["Id"].values:
#             print(f"Processing spot_index {spot_index} on run {run_name}")
#             pixel_spectra_df = self.process_pixel_spectra(tsf_cursor, spot_index)
#             results.append(pixel_spectra_df)

#         results = pd.concat(results)
#         results["run_name"] = run_name
        
#         return run_name, results

#     def process_spot_dataframe(self, maldi_run_path, n_workers=4):
#         run_name = os.path.basename(os.path.splitext(maldi_run_path)[0])
#         # print(f"Initializing TsfData object for {run_name}")
#         # print(f"Initializing TsfDataFrame object for {run_name}")
#         # tsf_df = TsfDataFrame(
#         #     tsf_cursor, tsf_cursor.analysis["MaldiFrameInfo"], tsf_cursor.analysis["Frames"], None
#         # )
#         print(f"Initializing Dask DataFrame object for {run_name}")
#         print(tsf_df)
#         ddf = dd.from_pandas(tsf_df.spot_df.iloc[:, 0:2], npartitions=n_workers)
        
#         # Parallel processing of each chunk
#         print(f"Running spectra processing in parallel")
#         spectra_dfs = ddf.map_partitions(self.process_spot_chunk, run_name).compute()
#         tsf_df.spectra_df = pd.concat(tsf_df.spectra_dfs)

#         self.full_tsf_data[run_name] = tsf_df
        
#         # # Return the final concatenated DataFrames
#         # return run_name, tsf_df.frame_df, tsf_df.spectra_df

#     def process_multiple_spot_dfs(self, n_workers=4):
#         """Processes multiple spot DataFrames in parallel."""
#         for mrp in self.maldi_run_paths:
#             self.process_spot_dataframe(mrp)
#         # with ProcessPoolExecutor(max_workers=n_workers) as executor:
#         #     _ = [
#         #         executor.submit(self.process_spot_dataframe, maldi_path)
#         #         for maldi_path in self.maldi_run_paths
#         #     ]
#             # spectra_futures = [
#             #     executor.submit(self.process_spot_dataframe, maldi_path)
#             #     for maldi_path in self.maldi_run_paths
#             # ]
#             # spectra_dfs = [sf.result() for sf in spectra_futures]

#         # print(spectra_dfs)

#         # for sd in spectra_dfs:
#         #     self.tsf_dfs[sd[0]].spectra_df = df[1]
        
#         # return results
