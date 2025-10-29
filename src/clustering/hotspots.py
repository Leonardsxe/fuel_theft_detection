"""
Hotspot clustering using DBSCAN with haversine distance.
Identifies recurring locations where theft events occur.
"""

import pandas as pd
import numpy as np
from typing import Dict, Tuple, Optional
from sklearn.cluster import DBSCAN
import logging

from src.config.settings import ClusteringConfig

logger = logging.getLogger(__name__)


def haversine_dbscan(
    lat: np.ndarray,
    lon: np.ndarray,
    eps_m: float,
    min_samples: int
) -> np.ndarray:
    """
    DBSCAN clustering using haversine distance metric.
    
    Args:
        lat: Latitude values (degrees)
        lon: Longitude values (degrees)
        eps_m: Maximum distance in meters for cluster membership
        min_samples: Minimum samples to form a cluster
    
    Returns:
        Cluster labels (-1 for noise points)
    """
    # Convert to radians for haversine
    coords_rad = np.deg2rad(np.c_[lat, lon])
    
    # Convert eps from meters to radians
    eps_rad = eps_m / 6371000.0  # Earth radius in meters
    
    # Apply DBSCAN
    clustering = DBSCAN(
        eps=eps_rad,
        min_samples=min_samples,
        metric='haversine'
    )
    
    labels = clustering.fit_predict(coords_rad)
    
    return labels


class HotspotClusterer:
    """
    Cluster stationary locations to identify hotspots.
    """
    
    def __init__(self, config: ClusteringConfig):
        """
        Initialize clusterer with configuration.
        
        Args:
            config: Clustering configuration
        """
        self.config = config
        self.cluster_centroids = {}
    
    def fit(
        self,
        stationary_pts: pd.DataFrame,
        train_mask: Optional[np.ndarray] = None
    ) -> pd.DataFrame:
        """
        Fit clusters on stationary points (training data only if mask provided).
        
        Args:
            stationary_pts: DataFrame with stationary GPS coordinates
            train_mask: Optional mask for training data
        
        Returns:
            DataFrame with cluster_id column added
        """

        if train_mask is not None:
            # Validate mask size matches stationary_pts
            if len(train_mask) != len(stationary_pts):
                raise ValueError(
                    f"Train mask size {len(train_mask)} doesn't match "
                    f"stationary points size {len(stationary_pts)}"
                )
            pts_for_fitting = stationary_pts.loc[train_mask]
        else:
            pts_for_fitting = stationary_pts
        
        logger.info("Clustering stationary locations...")
        
        stationary_pts = stationary_pts.copy()
        stationary_pts["cluster_id"] = -1
        
        # Use training data only if mask provided
        if train_mask is not None:
            pts_for_fitting = stationary_pts.loc[stationary_pts.index.isin(stationary_pts.index[train_mask])]
            logger.info("Fitting clusters on training data only")
        else:
            pts_for_fitting = stationary_pts
        
        for vid, vehicle_pts in pts_for_fitting.groupby("vehicle_id"):
            if len(vehicle_pts) < self.config.min_samples:
                logger.warning(f"Vehicle {vid} has only {len(vehicle_pts)} points, skipping clustering")
                continue
            
            try:
                # Cluster on training points
                labels_train = haversine_dbscan(
                    vehicle_pts["latitude"].values,
                    vehicle_pts["longitude"].values,
                    self.config.eps_meters,
                    self.config.min_samples
                )
                
                # Store cluster centroids
                centroids = {}
                for cluster_id in set(labels_train):
                    if cluster_id == -1:
                        continue
                    
                    mask = labels_train == cluster_id
                    centroids[cluster_id] = (
                        vehicle_pts["latitude"].values[mask].mean(),
                        vehicle_pts["longitude"].values[mask].mean()
                    )
                
                self.cluster_centroids[vid] = centroids
                
                # Predict for ALL points (train + test)
                all_vehicle_pts = stationary_pts[stationary_pts["vehicle_id"] == vid]
                all_labels = self._predict_clusters(
                    all_vehicle_pts["latitude"].values,
                    all_vehicle_pts["longitude"].values,
                    centroids
                )
                
                stationary_pts.loc[all_vehicle_pts.index, "cluster_id"] = all_labels
                
                n_clusters = len(centroids)
                n_noise = int((all_labels == -1).sum())
                logger.info(f"Vehicle {vid}: {n_clusters} clusters, {n_noise} noise points")
            
            except Exception as e:
                logger.error(f"Clustering failed for vehicle {vid}: {e}")
                continue
        
        logger.info("✓ Hotspot clustering complete")
        
        return stationary_pts
    
    def _predict_clusters(
        self,
        lat: np.ndarray,
        lon: np.ndarray,
        centroids: Dict[int, Tuple[float, float]]
    ) -> np.ndarray:
        """
        Assign points to nearest cluster centroid if within eps.
        
        Args:
            lat: Latitude values
            lon: Longitude values
            centroids: Dictionary of cluster centroids
        
        Returns:
            Cluster labels
        """
        if not centroids:
            return np.full(len(lat), -1, dtype=int)
        
        from sklearn.metrics.pairwise import haversine_distances
        
        # Convert to radians
        coords_rad = np.deg2rad(np.c_[lat, lon])
        centroid_coords = np.array(list(centroids.values()))
        centroid_rad = np.deg2rad(centroid_coords)
        centroid_ids = list(centroids.keys())
        
        # Calculate distances
        distances_rad = haversine_distances(coords_rad, centroid_rad)
        distances_m = distances_rad * 6371000  # Convert to meters
        
        # Assign to nearest centroid if within eps
        min_distances = distances_m.min(axis=1)
        nearest_idx = distances_m.argmin(axis=1)
        
        labels = np.full(len(lat), -1, dtype=int)
        within_eps = min_distances <= self.config.eps_meters
        labels[within_eps] = [centroid_ids[i] for i in nearest_idx[within_eps]]
        
        return labels
    
    def transform(self, stationary_pts: pd.DataFrame) -> pd.DataFrame:
        """
        Assign cluster labels to new stationary points.
        
        Args:
            stationary_pts: DataFrame with stationary GPS coordinates
        
        Returns:
            DataFrame with cluster_id column added
        """
        stationary_pts = stationary_pts.copy()
        stationary_pts["cluster_id"] = -1
        
        for vid, vehicle_pts in stationary_pts.groupby("vehicle_id"):
            if vid not in self.cluster_centroids:
                continue
            
            labels = self._predict_clusters(
                vehicle_pts["latitude"].values,
                vehicle_pts["longitude"].values,
                self.cluster_centroids[vid]
            )
            
            stationary_pts.loc[vehicle_pts.index, "cluster_id"] = labels
        
        return stationary_pts


def cluster_stationary_points(
    df: pd.DataFrame,
    config: ClusteringConfig,
    train_mask: Optional[np.ndarray] = None
) -> pd.DataFrame:
    """
    Extract and cluster stationary points.
    
    Args:
        df: Raw DataFrame with stationary flags
        config: Clustering configuration
        train_mask: Optional training data mask
    
    Returns:
        DataFrame with stationary points and cluster IDs
    """
    # Extract stationary points
    stationary_pts = df.loc[
        df["stationary"],
        ["vehicle_id", "timestamp", "latitude", "longitude", "stationary_on", "ign_off", "dt_s"]
    ].dropna(subset=["latitude", "longitude"]).copy()
    
    if stationary_pts.empty:
        logger.warning("No valid stationary points found")
        stationary_pts["cluster_id"] = []
        return stationary_pts
    
    # Cluster
    clusterer = HotspotClusterer(config)
    stationary_pts = clusterer.fit(stationary_pts, train_mask)
    
    return stationary_pts