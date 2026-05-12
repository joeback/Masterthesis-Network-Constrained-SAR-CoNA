import pandas as pd
import numpy as np
import re
from pathlib import Path

def normalize_stop_id(x):
    """
    The purpose of this function is to standardize the stop IDs to a common format across the input GTFS files.
    It turns the input into a string, and extracts the ID sequence (7-9 digits))
    """

    if pd.isna(x):
        return None
    x = str(x)
    # Extract the numeric ID with regex
    m = re.search(r"(\d{7,9})", x)
    return m.group(1) if m else x

def aggregate_realtime_by_station(data_dir, stops_file, by_time_bin=True):
    """
    Load and aggregate the real-time delay data by station and category.
    The 'by_time_bin' option allows aggregation by time bins.
    """

    data_dir = Path(data_dir)
    # Collect all csv files matching the data pattern.
    files = sorted(data_dir.glob("vbb_realtime_delays_buses_*.csv"))
    # Ensure that the files are found.
    if len(files) == 0:
        raise FileNotFoundError(f"No files matching vbb_realtime_delays_buses_*.csv in {data_dir}")

    df = pd.concat((pd.read_csv(f) for f in files), ignore_index=True)

    # Build the delay categories.
    def categorize_delay(sec):
        if pd.isna(sec):
            return None
        if sec < -60:
            return "early"
        if sec <= 180:
            return "on_time"
        return "delayed"

    # Build the delay labels and normalize the stop ids using the predefined function.
    df["category"] = df["delay_seconds"].apply(categorize_delay)
    df = df.dropna(subset=["category"])
    df["stop_id"] = df["stop_id"].apply(normalize_stop_id)

    # Load stop metadata.
    stops = pd.read_csv(stops_file, dtype=str, usecols=["stop_id", "parent_station"])
    stops["stop_id"] = stops["stop_id"].apply(normalize_stop_id)
    stops["parent_station"] = stops["parent_station"].apply(normalize_stop_id)

    # Aggregate the data by station from stop level.
    df = df.merge(stops, on="stop_id", how="left")
    df["station_id"] = df["parent_station"].fillna(df["stop_id"])

    # Create columns to group by optionally adding time bins.
    group_cols = ["station_id", "category"]
    if by_time_bin:
        if "time_bin" not in df.columns:
            raise ValueError("by_time_bin=True but no 'time_bin' column found")
        group_cols.insert(1, "time_bin")

    # Compute the counts for each station and category.
    counts = (
        df.groupby(group_cols).size().unstack(fill_value=0).reset_index()
    )

    # Ensure zero counts are present.
    for c in ["early", "on_time", "delayed"]:
        if c not in counts:
            counts[c] = 0
    counts = counts.rename_axis(None, axis=1)

    # Compute total counts and proportions.
    counts["total_n"] = counts[["early", "on_time", "delayed"]].sum(axis=1)
    # Create the composition from the proportions of categories.
    for c in ["early", "on_time", "delayed"]:
        counts[f"{c}_n"] = counts[c]
        counts[c] = counts[c] / counts["total_n"]

    return counts

def alpha_ilr_transform(Y, alpha=0.2):
    """
    Perform the alpha transformation introduced by Tsagris et al. (2011).
    This transformation is robust to zero shares and uses a predefined alpha level.
    """
    # Ensure format and normalizatuion.
    Y = np.asarray(Y, float)
    Y = Y / Y.sum(axis=1, keepdims=True)

    # Perform the alpha transformation.
    U = (Y ** alpha - 1) / alpha
    U_centered = U - U.mean(axis=1, keepdims=True)

    # Define the log-contrast matrix for the 3-part composition.
    H = np.array([
        [ 1/np.sqrt(2), -1/np.sqrt(2), 0],
        [ 1/np.sqrt(6),  1/np.sqrt(6), -2/np.sqrt(6)]
    ])
    
    # Create the transformed coordinates.
    Z = U_centered @ H.T
    return Z

def ilr_transform(Y):
    """
    Standard ILR transform for 3-part compositions.
    Assumes strictly positive components.
    """
    # Ensure format and normalizatuion.
    Y = np.asarray(Y, float)
    Y = Y / Y.sum(axis=1, keepdims=True)
    
    # Extract the ILR coordinates using the log-ratio of components.
    z1 = np.sqrt(1/2) * np.log(Y[:, 0] / Y[:, 1])
    z2 = np.sqrt(2/3) * np.log(np.sqrt(Y[:, 0] * Y[:, 1]) / Y[:, 2])
    return np.column_stack([z1, z2])


def ilr_inverse(Z, alpha=0.2):
    """
    Set up the backtransform for the alpha-ilr-transformation.
    """
    # Ensure format and create log-contrast matrix.
    Z = np.asarray(Z, float)
    H = np.array([
        [ 1/np.sqrt(2), -1/np.sqrt(2), 0],
        [ 1/np.sqrt(6),  1/np.sqrt(6), -2/np.sqrt(6)]
    ])

    # Undo the alpha transformation step by step.
    U_centered = Z @ H 
    T = 1.0 + alpha * U_centered
    eps = 1e-12
    T = np.maximum(T, eps)
    Y = T ** (1.0 / alpha)
    # Ensure 1-sum constraint.
    Y = Y / Y.sum(axis=1, keepdims=True)
    return Y