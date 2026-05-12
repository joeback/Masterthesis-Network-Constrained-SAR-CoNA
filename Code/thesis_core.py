from pathlib import Path
import pandas as pd
import numpy as np
from scipy.sparse import coo_matrix
from libpysal.weights import WSP
from spreg import ML_Lag
from scipy.sparse import identity
from libpysal.weights.util import w_subset
import os
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = Path(
    os.getenv(
        "THESIS_DATA_DIR",
        PROJECT_ROOT / "Data" / "data_raw"
    )
)

GTFS_DIR = Path(
    os.getenv(
        "THESIS_GTFS_DIR",
        PROJECT_ROOT / "Data" / "GTFS_Structure_Data" / "GTFS_160126"
    )
)

PROCESSED_DIR = Path(
    os.getenv(
        "THESIS_PROCESSED_DIR",
        PROJECT_ROOT / "Data" / "processed"
    )
)

WORKFILES_DIR = PROJECT_ROOT / "Code" / "Data_Processing" / "Workfiles"

if str(WORKFILES_DIR) not in sys.path:
    sys.path.append(str(WORKFILES_DIR))

from preprocessing_busdata import (
    normalize_stop_id,
    aggregate_realtime_by_station,
    ilr_inverse,
)

def load_static_gtfs():
    """
    Load the static GTFS metadata for stops, trips, stop times, and routes.
    """
    stops = pd.read_csv(GTFS_DIR / "stops.txt", dtype=str)
    trips = pd.read_csv(GTFS_DIR / "trips.txt", dtype=str)
    stop_times = pd.read_csv(GTFS_DIR / "stop_times.txt", dtype=str)
    routes = pd.read_csv(GTFS_DIR / "routes.txt")
    return stops, trips, stop_times, routes

def canonical_stations(stops):
    """
    Yield the normalized station IDs.
    """
    stops = stops.copy()
    stops["stop_id_norm"] = stops["stop_id"].apply(normalize_stop_id)
    stops["parent_station_norm"] = stops["parent_station"].apply(normalize_stop_id)
    stops["station_id"] = stops["parent_station_norm"].fillna(stops["stop_id_norm"])
    return stops

def build_station_meta(stops):
    """
    Build the locational metadata for the stations.
    """
    stops = stops.copy()
    # Yield the location of the stops as numeric values.
    stops["stop_lat"] = pd.to_numeric(stops["stop_lat"], errors="coerce")
    stops["stop_lon"] = pd.to_numeric(stops["stop_lon"], errors="coerce")

    # Build the station metadata by location and name.
    return (
        stops
        .dropna(subset=["stop_lat", "stop_lon"])
        .groupby("station_id", as_index=False)
        .agg({
            "stop_name": "first",
            "stop_lat": "mean",
            "stop_lon": "mean",
        })
    )

def build_bus_station_edges(
    stops: pd.DataFrame,
    stop_times: pd.DataFrame,
    trips: pd.DataFrame,
    routes: pd.DataFrame,
    bus_route_types=(3, 700),
) -> pd.DataFrame:
    """
    Build directed station-to-station edges from static GTFS.
    """

    # Get the required metadata.
    stops = stops.copy()
    stop_times = stop_times.copy()
    stops["stop_id"] = stops["stop_id"].apply(normalize_stop_id)
    stops["parent_station"] = stops["parent_station"].apply(normalize_stop_id)
    stop_times["stop_id"] = stop_times["stop_id"].apply(normalize_stop_id)

    # Filter for bus routes and relevant stop times.
    bus_routes = routes.loc[routes["route_type"].isin(bus_route_types), "route_id"]

    # Filter for bus trips and relevant stop times.
    bus_trips = trips[trips["route_id"].isin(bus_routes)]
    bus_stop_times = stop_times[stop_times["trip_id"].isin(bus_trips["trip_id"])].copy()

    # Ensure stop times are sorted by trip and stop sequence.
    bus_stop_times["stop_sequence"] = bus_stop_times["stop_sequence"].astype(int)
    bus_stop_times = bus_stop_times.sort_values(["trip_id", "stop_sequence"])

    # Keep only relevant information.
    stops_min = stops[["stop_id", "parent_station"]]
    # Merge the stop information to get the parent station IDs, which will be used for edge construction.
    bus_stop_times = bus_stop_times.merge(stops_min, on="stop_id", how="left")
    # Assign the station_id as parent_station if available.
    bus_stop_times["station_id"] = (bus_stop_times["parent_station"]
        .fillna(bus_stop_times["stop_id"])
        .astype(str)
    )

    # Create a sequence of station IDs for each trip.
    station_sequences = (bus_stop_times.groupby("trip_id")["station_id"].apply(list))

    # Build a list of edges through the station sequences.
    edge_list = []
    for seq in station_sequences:
        for i in range(len(seq) - 1):
            if seq[i] != seq[i + 1]:
                edge_list.append((seq[i], seq[i + 1]))

    # Build a DataFrame of edges and drop duplicates.
    # This results in the overall edgeset.
    edges_df = (
        pd.DataFrame(edge_list, columns=["from_station", "to_station"])
        .drop_duplicates()
        .reset_index(drop=True)
    )

    return edges_df

def build_sar_weights(
    edges_df: pd.DataFrame,
    stations: list[str],
):
    """
    Build row-standardized SAR weight matrix from station edges.
    Returns A (CSR), W_ps (PySAL), keep_idx (non-isolates).
    """
    # Create an index mapping for the stations.
    idx = {s: i for i, s in enumerate(stations)}
    n = len(stations)

    # Build an undirected adjacency matrix from the edges.
    E = edges_df[["from_station", "to_station"]].astype(str)
    E_rev = E.rename(
        columns={"from_station": "to_station", "to_station": "from_station"}
    )
    E_undirected = pd.concat([E, E_rev]).drop_duplicates()
    rows, cols = [], []
    for a, b in E_undirected.itertuples(index=False):
        if a in idx and b in idx and a != b:
            rows.append(idx[a])
            cols.append(idx[b])
    A = coo_matrix(
        (np.ones(len(rows)), (rows, cols)),
        shape=(n, n)
    ).tocsr()

    # Build the PySAL W object and identify isolates. Row-standardization is done in the W object.
    W_ps = WSP(A).to_W()
    W_ps.transform = "r"

    # Filter for non-isolte stations.
    isolates = {i for i, c in W_ps.cardinalities.items() if c == 0}
    keep_idx = [i for i in range(n) if i not in isolates]

    return A, W_ps, keep_idx

def fit_sar_ilr(
    Z: np.ndarray,
    W_ps,
    keep_idx: list[int],
    X: np.ndarray | None = None,
):
    """
    Fit SAR models separately for each ILR dimension.
    """
    # Filter viable stations and create designated design-matrix.
    Z_cent = Z[keep_idx, :]
    if X is None:
        X_cent = np.ones((len(keep_idx), 1))
    else:
        X_cent = X[keep_idx, :]

    # Create the weights matrix.
    W_cent = w_subset(W_ps, keep_idx)
    W_cent.transform = "r"

    # Fit the models and store the fitted values in ilr-space.
    models = []
    fitted_Z = np.zeros_like(Z_cent)
    for k in range(Z_cent.shape[1]):
        yk = Z_cent[:, k]
        m = ML_Lag(yk, X_cent, w=W_cent)
        models.append(m)
        fitted_Z[:, k] = m.predy.flatten()

    return models, fitted_Z, W_cent

def reconstruct_compositions(
    fitted_Z: np.ndarray,
    bus_comp_grouped: pd.DataFrame,
    bus_comp_sar: pd.DataFrame,
    keep_idx: list[int],
):
    """
    Map SAR predictions back to full station set and invert ILR.
    """
    # Create a matrix placeholder for the predictions.
    n_full = bus_comp_grouped.shape[0]
    q = fitted_Z.shape[1]
    fitted_Z_full = np.full((n_full, q), np.nan)

    # Extract the viable indices and subset the predictions.
    sar_idx = bus_comp_grouped.index[bus_comp_grouped["station_id"].isin(bus_comp_sar["station_id"])]
    sar_keep_idx = sar_idx[keep_idx]
    fitted_Z_full[sar_keep_idx, :] = fitted_Z

    # Create the backtransformed compositions.
    Y_hat = ilr_inverse(fitted_Z_full)
    return Y_hat

def simulate_unstructured_dirichlet(n, mu=(0.1,0.8,0.1), c=40, seed=None):
    """
    Create unstructured Dirichlet compositions for simulation purposes.
    The parameter c controls the concentration around the mean mu.
    """
    if seed is not None:
        np.random.seed(seed)
    alpha = np.array(mu) * c
    return np.random.dirichlet(alpha, size=n)

def simulate_sar_ilr(W, rho, sigma=0.3, seed=None):
    """
    Simulate SAR ILR coordinates with given noise and spatial dependence rho.
    """
    if seed is not None:
        np.random.seed(seed)
    n = W.shape[0]
    eps = np.random.normal(0, sigma, size=(n,2))
    I = identity(n, format="csr")
    # Create coordinates with the specified DGP.
    z1 = np.linalg.solve((I - rho * W).toarray(), eps[:,0])
    z2 = np.linalg.solve((I - rho * W).toarray(), eps[:,1])

    return np.column_stack([z1, z2])

def simulate_sar_ilr_with_covariates(
    W, X, beta1, beta2, rho, sigma=0.3, seed=None
):
    """
    Simulate SAR ILR coordinates with given noise and spatial dependence rho.
    Additionally, include the centrality covarites that are prespecified.
    """
    if seed is not None:
        np.random.seed(seed)
    n = W.shape[0]
    eps = np.random.normal(0, sigma, size=(n,2))
    I = identity(n, format="csr")
    # Create coordinates with the specified DGP.
    mu1 = X @ beta1
    mu2 = X @ beta2
    z1 = np.linalg.solve((I - rho * W).toarray(), mu1 + eps[:,0])
    z2 = np.linalg.solve((I - rho * W).toarray(), mu2 + eps[:,1])

    return np.column_stack([z1, z2])