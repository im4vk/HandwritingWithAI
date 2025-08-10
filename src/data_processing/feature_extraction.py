"""
Feature Extraction for Handwriting Data
=======================================

Extract meaningful features from handwriting trajectories for analysis,
classification, and machine learning tasks.
"""

import numpy as np
from typing import Dict, Any, List, Tuple, Optional, Union
from dataclasses import dataclass
import logging
from scipy import signal
from scipy.spatial.distance import pdist, squareform
from scipy.stats import entropy
import cv2

from .dataset_loader import HandwritingSample

logger = logging.getLogger(__name__)


@dataclass
class HandwritingFeatures:
    """
    Container for handwriting features.
    
    Attributes:
        kinematic_features: Movement-related features
        geometric_features: Shape and spatial features
        temporal_features: Time-related features
        pressure_features: Pressure-related features (if available)
        statistical_features: Statistical measures
        frequency_features: Frequency domain features
        complexity_features: Complexity and entropy measures
    """
    kinematic_features: Dict[str, float]
    geometric_features: Dict[str, float]
    temporal_features: Dict[str, float]
    pressure_features: Optional[Dict[str, float]] = None
    statistical_features: Optional[Dict[str, float]] = None
    frequency_features: Optional[Dict[str, float]] = None
    complexity_features: Optional[Dict[str, float]] = None
    
    def to_vector(self, include_optional: bool = True) -> np.ndarray:
        """Convert features to vector format."""
        features = []
        
        # Add kinematic features
        features.extend(list(self.kinematic_features.values()))
        
        # Add geometric features
        features.extend(list(self.geometric_features.values()))
        
        # Add temporal features
        features.extend(list(self.temporal_features.values()))
        
        # Add optional features if available and requested
        if include_optional:
            if self.pressure_features:
                features.extend(list(self.pressure_features.values()))
            if self.statistical_features:
                features.extend(list(self.statistical_features.values()))
            if self.frequency_features:
                features.extend(list(self.frequency_features.values()))
            if self.complexity_features:
                features.extend(list(self.complexity_features.values()))
        
        return np.array(features)
    
    def get_feature_names(self, include_optional: bool = True) -> List[str]:
        """Get names of features in vector order."""
        names = []
        
        # Add kinematic feature names
        names.extend([f"kinematic_{k}" for k in self.kinematic_features.keys()])
        
        # Add geometric feature names
        names.extend([f"geometric_{k}" for k in self.geometric_features.keys()])
        
        # Add temporal feature names
        names.extend([f"temporal_{k}" for k in self.temporal_features.keys()])
        
        # Add optional feature names
        if include_optional:
            if self.pressure_features:
                names.extend([f"pressure_{k}" for k in self.pressure_features.keys()])
            if self.statistical_features:
                names.extend([f"statistical_{k}" for k in self.statistical_features.keys()])
            if self.frequency_features:
                names.extend([f"frequency_{k}" for k in self.frequency_features.keys()])
            if self.complexity_features:
                names.extend([f"complexity_{k}" for k in self.complexity_features.keys()])
        
        return names


@dataclass
class TemporalFeatures:
    """
    Time-series specific features for handwriting analysis.
    
    Attributes:
        duration: Total writing duration
        writing_time: Time with pen down
        pause_time: Time with pen up
        num_pauses: Number of pen lifts
        avg_pause_duration: Average pause duration
        writing_speed: Average writing speed
        speed_variation: Variation in writing speed
        acceleration_patterns: Acceleration pattern features
    """
    duration: float
    writing_time: float
    pause_time: float
    num_pauses: int
    avg_pause_duration: float
    writing_speed: float
    speed_variation: float
    acceleration_patterns: Dict[str, float]


class FeatureExtractor:
    """
    Comprehensive feature extractor for handwriting data.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize feature extractor.
        
        Args:
            config: Extractor configuration
        """
        self.config = config
        
        # Feature extraction settings
        self.extract_kinematic = config.get('extract_kinematic', True)
        self.extract_geometric = config.get('extract_geometric', True)
        self.extract_temporal = config.get('extract_temporal', True)
        self.extract_pressure = config.get('extract_pressure', True)
        self.extract_statistical = config.get('extract_statistical', True)
        self.extract_frequency = config.get('extract_frequency', True)
        self.extract_complexity = config.get('extract_complexity', True)
        
        # Feature computation parameters
        self.velocity_window = config.get('velocity_window', 3)
        self.acceleration_window = config.get('acceleration_window', 5)
        self.frequency_bands = config.get('frequency_bands', [(0, 2), (2, 5), (5, 10), (10, 25)])
        
        logger.info("Initialized FeatureExtractor")
    
    def extract_features(self, sample: HandwritingSample) -> HandwritingFeatures:
        """
        Extract comprehensive features from handwriting sample.
        
        Args:
            sample: Handwriting sample
            
        Returns:
            features: Extracted handwriting features
        """
        trajectory = sample.trajectory
        timestamps = sample.timestamps
        pressure = sample.pressure
        pen_states = sample.pen_states
        
        # Extract different feature groups
        kinematic_features = self._extract_kinematic_features(trajectory, timestamps) if self.extract_kinematic else {}
        geometric_features = self._extract_geometric_features(trajectory) if self.extract_geometric else {}
        temporal_features = self._extract_temporal_features(trajectory, timestamps, pen_states) if self.extract_temporal else {}
        
        pressure_features = None
        if self.extract_pressure and pressure is not None:
            pressure_features = self._extract_pressure_features(pressure, pen_states)
        
        statistical_features = None
        if self.extract_statistical:
            statistical_features = self._extract_statistical_features(trajectory, timestamps)
        
        frequency_features = None
        if self.extract_frequency and timestamps is not None:
            frequency_features = self._extract_frequency_features(trajectory, timestamps)
        
        complexity_features = None
        if self.extract_complexity:
            complexity_features = self._extract_complexity_features(trajectory)
        
        return HandwritingFeatures(
            kinematic_features=kinematic_features,
            geometric_features=geometric_features,
            temporal_features=temporal_features,
            pressure_features=pressure_features,
            statistical_features=statistical_features,
            frequency_features=frequency_features,
            complexity_features=complexity_features
        )
    
    def _extract_kinematic_features(self, trajectory: np.ndarray, 
                                  timestamps: Optional[np.ndarray] = None) -> Dict[str, float]:
        """Extract movement-related features."""
        features = {}
        
        # Compute velocities
        if timestamps is not None:
            dt = np.diff(timestamps)
            velocities = np.diff(trajectory, axis=0) / dt.reshape(-1, 1)
        else:
            velocities = np.diff(trajectory, axis=0)
        
        speeds = np.linalg.norm(velocities, axis=1)
        
        # Speed statistics
        features['mean_speed'] = np.mean(speeds)
        features['std_speed'] = np.std(speeds)
        features['max_speed'] = np.max(speeds)
        features['min_speed'] = np.min(speeds)
        features['speed_range'] = features['max_speed'] - features['min_speed']
        
        # Compute accelerations
        if len(velocities) > 1:
            if timestamps is not None:
                dt_acc = dt[1:]
                accelerations = np.diff(velocities, axis=0) / dt_acc.reshape(-1, 1)
            else:
                accelerations = np.diff(velocities, axis=0)
            
            acc_magnitudes = np.linalg.norm(accelerations, axis=1)
            
            features['mean_acceleration'] = np.mean(acc_magnitudes)
            features['std_acceleration'] = np.std(acc_magnitudes)
            features['max_acceleration'] = np.max(acc_magnitudes)
        else:
            features['mean_acceleration'] = 0.0
            features['std_acceleration'] = 0.0
            features['max_acceleration'] = 0.0
        
        # Jerk (rate of change of acceleration)
        if len(velocities) > 2:
            jerks = np.diff(accelerations, axis=0)
            jerk_magnitudes = np.linalg.norm(jerks, axis=1)
            
            features['mean_jerk'] = np.mean(jerk_magnitudes)
            features['std_jerk'] = np.std(jerk_magnitudes)
            features['smoothness'] = 1.0 / (1.0 + features['mean_jerk'])  # Inverse of jerk
        else:
            features['mean_jerk'] = 0.0
            features['std_jerk'] = 0.0
            features['smoothness'] = 1.0
        
        # Velocity direction changes
        if len(velocities) > 1:
            velocity_angles = np.arctan2(velocities[:, 1], velocities[:, 0])
            angle_changes = np.abs(np.diff(velocity_angles))
            
            # Handle angle wrapping
            angle_changes = np.minimum(angle_changes, 2*np.pi - angle_changes)
            
            features['mean_direction_change'] = np.mean(angle_changes)
            features['direction_variability'] = np.std(angle_changes)
        else:
            features['mean_direction_change'] = 0.0
            features['direction_variability'] = 0.0
        
        return features
    
    def _extract_geometric_features(self, trajectory: np.ndarray) -> Dict[str, float]:
        """Extract shape and spatial features."""
        features = {}
        
        # Bounding box features
        min_coords = np.min(trajectory, axis=0)
        max_coords = np.max(trajectory, axis=0)
        bbox_size = max_coords - min_coords
        
        features['bbox_width'] = bbox_size[0]
        features['bbox_height'] = bbox_size[1]
        features['bbox_area'] = bbox_size[0] * bbox_size[1]
        features['bbox_aspect_ratio'] = bbox_size[0] / (bbox_size[1] + 1e-8)
        
        # Trajectory length
        path_segments = np.diff(trajectory, axis=0)
        segment_lengths = np.linalg.norm(path_segments, axis=1)
        features['total_length'] = np.sum(segment_lengths)
        
        # Displacement features
        displacement = trajectory[-1] - trajectory[0]
        features['net_displacement'] = np.linalg.norm(displacement)
        features['displacement_x'] = displacement[0]
        features['displacement_y'] = displacement[1]
        
        # Efficiency (ratio of net displacement to total path length)
        features['path_efficiency'] = features['net_displacement'] / (features['total_length'] + 1e-8)
        
        # Curvature features
        curvatures = self._compute_curvature(trajectory)
        if len(curvatures) > 0:
            features['mean_curvature'] = np.mean(curvatures)
            features['std_curvature'] = np.std(curvatures)
            features['max_curvature'] = np.max(curvatures)
        else:
            features['mean_curvature'] = 0.0
            features['std_curvature'] = 0.0
            features['max_curvature'] = 0.0
        
        # Turning angles
        turning_angles = self._compute_turning_angles(trajectory)
        if len(turning_angles) > 0:
            features['mean_turning_angle'] = np.mean(np.abs(turning_angles))
            features['std_turning_angle'] = np.std(turning_angles)
            features['total_absolute_turning'] = np.sum(np.abs(turning_angles))
        else:
            features['mean_turning_angle'] = 0.0
            features['std_turning_angle'] = 0.0
            features['total_absolute_turning'] = 0.0
        
        # Centroid and spread
        centroid = np.mean(trajectory, axis=0)
        distances_to_centroid = np.linalg.norm(trajectory - centroid, axis=1)
        features['centroid_x'] = centroid[0]
        features['centroid_y'] = centroid[1]
        features['mean_distance_to_centroid'] = np.mean(distances_to_centroid)
        features['std_distance_to_centroid'] = np.std(distances_to_centroid)
        
        # Convex hull area
        try:
            from scipy.spatial import ConvexHull
            if len(trajectory) >= 3:
                hull = ConvexHull(trajectory)
                features['convex_hull_area'] = hull.volume  # In 2D, volume is area
                features['solidity'] = features['bbox_area'] / (hull.volume + 1e-8)
            else:
                features['convex_hull_area'] = 0.0
                features['solidity'] = 0.0
        except:
            features['convex_hull_area'] = 0.0
            features['solidity'] = 0.0
        
        return features
    
    def _extract_temporal_features(self, trajectory: np.ndarray,
                                 timestamps: Optional[np.ndarray] = None,
                                 pen_states: Optional[np.ndarray] = None) -> Dict[str, float]:
        """Extract time-related features."""
        features = {}
        
        if timestamps is not None:
            # Duration features
            total_duration = timestamps[-1] - timestamps[0]
            features['total_duration'] = total_duration
            
            # Writing vs pause time analysis
            if pen_states is not None:
                pen_down_mask = pen_states > 0.5
                writing_intervals = np.diff(timestamps)[pen_down_mask[:-1]]
                pause_intervals = np.diff(timestamps)[~pen_down_mask[:-1]]
                
                features['writing_time'] = np.sum(writing_intervals)
                features['pause_time'] = np.sum(pause_intervals)
                features['writing_ratio'] = features['writing_time'] / (total_duration + 1e-8)
                
                # Pause analysis
                pen_lifts = np.diff(pen_down_mask.astype(int))
                num_pauses = np.sum(pen_lifts == -1)  # Transitions from down to up
                
                features['num_pauses'] = num_pauses
                features['avg_pause_duration'] = np.mean(pause_intervals) if len(pause_intervals) > 0 else 0.0
                features['pause_frequency'] = num_pauses / (total_duration + 1e-8)
            else:
                features['writing_time'] = total_duration
                features['pause_time'] = 0.0
                features['writing_ratio'] = 1.0
                features['num_pauses'] = 0
                features['avg_pause_duration'] = 0.0
                features['pause_frequency'] = 0.0
            
            # Velocity timing
            velocities = np.diff(trajectory, axis=0) / np.diff(timestamps).reshape(-1, 1)
            speeds = np.linalg.norm(velocities, axis=1)
            
            # Time spent at different speeds
            speed_percentiles = np.percentile(speeds, [25, 50, 75])
            low_speed_time = np.sum(np.diff(timestamps)[speeds < speed_percentiles[0]])
            high_speed_time = np.sum(np.diff(timestamps)[speeds > speed_percentiles[2]])
            
            features['low_speed_time_ratio'] = low_speed_time / total_duration
            features['high_speed_time_ratio'] = high_speed_time / total_duration
            
            # Acceleration timing
            if len(velocities) > 1:
                accelerations = np.diff(velocities, axis=0) / np.diff(timestamps)[1:].reshape(-1, 1)
                acc_magnitudes = np.linalg.norm(accelerations, axis=1)
                
                # Time spent accelerating vs decelerating
                acc_threshold = np.percentile(acc_magnitudes, 75)
                high_acc_time = np.sum(np.diff(timestamps)[1:][acc_magnitudes > acc_threshold])
                features['high_acceleration_time_ratio'] = high_acc_time / total_duration
            else:
                features['high_acceleration_time_ratio'] = 0.0
        
        else:
            # Use point indices as proxy for time
            features['total_duration'] = len(trajectory)
            features['writing_time'] = len(trajectory)
            features['pause_time'] = 0.0
            features['writing_ratio'] = 1.0
            features['num_pauses'] = 0
            features['avg_pause_duration'] = 0.0
            features['pause_frequency'] = 0.0
            features['low_speed_time_ratio'] = 0.0
            features['high_speed_time_ratio'] = 0.0
            features['high_acceleration_time_ratio'] = 0.0
        
        return features
    
    def _extract_pressure_features(self, pressure: np.ndarray,
                                 pen_states: Optional[np.ndarray] = None) -> Dict[str, float]:
        """Extract pressure-related features."""
        features = {}
        
        # Basic pressure statistics
        features['mean_pressure'] = np.mean(pressure)
        features['std_pressure'] = np.std(pressure)
        features['max_pressure'] = np.max(pressure)
        features['min_pressure'] = np.min(pressure)
        features['pressure_range'] = features['max_pressure'] - features['min_pressure']
        
        # Pressure variation
        pressure_changes = np.abs(np.diff(pressure))
        features['mean_pressure_change'] = np.mean(pressure_changes)
        features['pressure_variability'] = np.std(pressure_changes)
        
        # Pressure percentiles
        percentiles = np.percentile(pressure, [10, 25, 50, 75, 90])
        features['pressure_p10'] = percentiles[0]
        features['pressure_p25'] = percentiles[1]
        features['pressure_median'] = percentiles[2]
        features['pressure_p75'] = percentiles[3]
        features['pressure_p90'] = percentiles[4]
        
        # Pressure patterns
        if pen_states is not None:
            pen_down_pressure = pressure[pen_states > 0.5]
            if len(pen_down_pressure) > 0:
                features['mean_writing_pressure'] = np.mean(pen_down_pressure)
                features['std_writing_pressure'] = np.std(pen_down_pressure)
            else:
                features['mean_writing_pressure'] = 0.0
                features['std_writing_pressure'] = 0.0
        else:
            features['mean_writing_pressure'] = features['mean_pressure']
            features['std_writing_pressure'] = features['std_pressure']
        
        # Pressure impulses (rapid changes)
        pressure_impulses = pressure_changes > (np.mean(pressure_changes) + 2*np.std(pressure_changes))
        features['pressure_impulse_count'] = np.sum(pressure_impulses)
        features['pressure_impulse_rate'] = features['pressure_impulse_count'] / len(pressure)
        
        return features
    
    def _extract_statistical_features(self, trajectory: np.ndarray,
                                    timestamps: Optional[np.ndarray] = None) -> Dict[str, float]:
        """Extract statistical measures."""
        features = {}
        
        # Coordinate statistics
        for i, coord in enumerate(['x', 'y']):
            coord_data = trajectory[:, i]
            
            features[f'{coord}_mean'] = np.mean(coord_data)
            features[f'{coord}_std'] = np.std(coord_data)
            features[f'{coord}_skewness'] = self._compute_skewness(coord_data)
            features[f'{coord}_kurtosis'] = self._compute_kurtosis(coord_data)
            features[f'{coord}_range'] = np.max(coord_data) - np.min(coord_data)
        
        # Trajectory correlation
        if trajectory.shape[1] >= 2:
            correlation = np.corrcoef(trajectory[:, 0], trajectory[:, 1])[0, 1]
            features['xy_correlation'] = correlation if not np.isnan(correlation) else 0.0
        
        # Distribution of distances between consecutive points
        distances = np.linalg.norm(np.diff(trajectory, axis=0), axis=1)
        features['mean_step_size'] = np.mean(distances)
        features['std_step_size'] = np.std(distances)
        features['step_size_skewness'] = self._compute_skewness(distances)
        
        # Auto-correlation of coordinates
        for i, coord in enumerate(['x', 'y']):
            coord_data = trajectory[:, i]
            if len(coord_data) > 10:
                autocorr = self._compute_autocorrelation(coord_data, max_lag=min(10, len(coord_data)//3))
                features[f'{coord}_autocorr_lag1'] = autocorr[1] if len(autocorr) > 1 else 0.0
                features[f'{coord}_autocorr_lag5'] = autocorr[5] if len(autocorr) > 5 else 0.0
            else:
                features[f'{coord}_autocorr_lag1'] = 0.0
                features[f'{coord}_autocorr_lag5'] = 0.0
        
        return features
    
    def _extract_frequency_features(self, trajectory: np.ndarray,
                                  timestamps: np.ndarray) -> Dict[str, float]:
        """Extract frequency domain features."""
        features = {}
        
        # Estimate sampling rate
        dt = np.mean(np.diff(timestamps))
        fs = 1.0 / dt
        
        # Compute power spectral density for each coordinate
        for i, coord in enumerate(['x', 'y']):
            coord_data = trajectory[:, i]
            
            # Remove DC component
            coord_data = coord_data - np.mean(coord_data)
            
            # Compute PSD
            freqs, psd = signal.welch(coord_data, fs=fs, nperseg=min(256, len(coord_data)//4))
            
            # Total power
            features[f'{coord}_total_power'] = np.sum(psd)
            
            # Power in different frequency bands
            for j, (f_low, f_high) in enumerate(self.frequency_bands):
                band_mask = (freqs >= f_low) & (freqs <= f_high)
                band_power = np.sum(psd[band_mask])
                features[f'{coord}_power_band_{j}'] = band_power
                features[f'{coord}_power_ratio_band_{j}'] = band_power / (features[f'{coord}_total_power'] + 1e-8)
            
            # Dominant frequency
            dominant_freq_idx = np.argmax(psd)
            features[f'{coord}_dominant_frequency'] = freqs[dominant_freq_idx]
            
            # Spectral centroid
            spectral_centroid = np.sum(freqs * psd) / (np.sum(psd) + 1e-8)
            features[f'{coord}_spectral_centroid'] = spectral_centroid
            
            # Spectral bandwidth
            spectral_bandwidth = np.sqrt(np.sum(((freqs - spectral_centroid) ** 2) * psd) / (np.sum(psd) + 1e-8))
            features[f'{coord}_spectral_bandwidth'] = spectral_bandwidth
        
        return features
    
    def _extract_complexity_features(self, trajectory: np.ndarray) -> Dict[str, float]:
        """Extract complexity and entropy measures."""
        features = {}
        
        # Fractal dimension (box counting method)
        features['fractal_dimension'] = self._compute_fractal_dimension(trajectory)
        
        # Sample entropy for coordinates
        for i, coord in enumerate(['x', 'y']):
            coord_data = trajectory[:, i]
            features[f'{coord}_sample_entropy'] = self._compute_sample_entropy(coord_data)
        
        # Approximate entropy
        for i, coord in enumerate(['x', 'y']):
            coord_data = trajectory[:, i]
            features[f'{coord}_approximate_entropy'] = self._compute_approximate_entropy(coord_data)
        
        # Lempel-Ziv complexity
        features['lz_complexity'] = self._compute_lz_complexity(trajectory)
        
        # Recurrence quantification analysis
        rqa_features = self._compute_rqa_features(trajectory)
        features.update(rqa_features)
        
        # Multi-scale entropy
        mse_features = self._compute_multiscale_entropy(trajectory)
        features.update(mse_features)
        
        return features
    
    def _compute_curvature(self, trajectory: np.ndarray) -> np.ndarray:
        """Compute curvature along trajectory."""
        if len(trajectory) < 3:
            return np.array([])
        
        # First and second derivatives
        first_deriv = np.diff(trajectory, axis=0)
        second_deriv = np.diff(first_deriv, axis=0)
        
        # Curvature formula: |x'y'' - y'x''| / (x'^2 + y'^2)^(3/2)
        first_deriv_aligned = first_deriv[:-1]  # Align with second derivative
        
        cross_product = (first_deriv_aligned[:, 0] * second_deriv[:, 1] - 
                        first_deriv_aligned[:, 1] * second_deriv[:, 0])
        
        speed_cubed = (np.sum(first_deriv_aligned**2, axis=1) + 1e-8) ** 1.5
        
        curvature = np.abs(cross_product) / speed_cubed
        
        return curvature
    
    def _compute_turning_angles(self, trajectory: np.ndarray) -> np.ndarray:
        """Compute turning angles between consecutive line segments."""
        if len(trajectory) < 3:
            return np.array([])
        
        # Vectors between consecutive points
        vectors = np.diff(trajectory, axis=0)
        
        # Angles between consecutive vectors
        angles = []
        for i in range(len(vectors) - 1):
            v1 = vectors[i]
            v2 = vectors[i + 1]
            
            # Compute angle using dot product
            dot_product = np.dot(v1, v2)
            norms = np.linalg.norm(v1) * np.linalg.norm(v2)
            
            if norms > 1e-8:
                cos_angle = np.clip(dot_product / norms, -1.0, 1.0)
                angle = np.arccos(cos_angle)
                
                # Determine sign using cross product
                cross_product = v1[0] * v2[1] - v1[1] * v2[0]
                if cross_product < 0:
                    angle = -angle
                
                angles.append(angle)
            else:
                angles.append(0.0)
        
        return np.array(angles)
    
    def _compute_skewness(self, data: np.ndarray) -> float:
        """Compute skewness of data."""
        if len(data) < 3:
            return 0.0
        
        mean = np.mean(data)
        std = np.std(data)
        
        if std == 0:
            return 0.0
        
        skewness = np.mean(((data - mean) / std) ** 3)
        return skewness
    
    def _compute_kurtosis(self, data: np.ndarray) -> float:
        """Compute kurtosis of data."""
        if len(data) < 4:
            return 0.0
        
        mean = np.mean(data)
        std = np.std(data)
        
        if std == 0:
            return 0.0
        
        kurtosis = np.mean(((data - mean) / std) ** 4) - 3  # Excess kurtosis
        return kurtosis
    
    def _compute_autocorrelation(self, data: np.ndarray, max_lag: int) -> np.ndarray:
        """Compute autocorrelation function."""
        if len(data) < max_lag + 1:
            return np.zeros(max_lag + 1)
        
        # Normalize data
        data_norm = data - np.mean(data)
        
        autocorr = np.correlate(data_norm, data_norm, mode='full')
        autocorr = autocorr[len(autocorr)//2:]  # Take positive lags
        
        # Normalize by variance
        autocorr = autocorr / autocorr[0]
        
        return autocorr[:max_lag + 1]
    
    def _compute_fractal_dimension(self, trajectory: np.ndarray) -> float:
        """Compute fractal dimension using box counting method."""
        # Simple implementation - can be improved
        if len(trajectory) < 10:
            return 1.0
        
        # Normalize trajectory to unit square
        min_coords = np.min(trajectory, axis=0)
        max_coords = np.max(trajectory, axis=0)
        range_coords = max_coords - min_coords
        
        if np.any(range_coords == 0):
            return 1.0
        
        normalized_traj = (trajectory - min_coords) / range_coords
        
        # Box counting at different scales
        scales = [2**i for i in range(2, 7)]  # Box sizes: 4, 8, 16, 32, 64
        counts = []
        
        for scale in scales:
            box_size = 1.0 / scale
            boxes = set()
            
            for point in normalized_traj:
                box_x = int(point[0] / box_size)
                box_y = int(point[1] / box_size)
                boxes.add((box_x, box_y))
            
            counts.append(len(boxes))
        
        # Fit log-log relationship
        if len(counts) > 1 and all(c > 0 for c in counts):
            log_scales = np.log(scales)
            log_counts = np.log(counts)
            
            # Linear regression
            slope = np.polyfit(log_scales, log_counts, 1)[0]
            fractal_dim = -slope
            
            return max(1.0, min(2.0, fractal_dim))  # Clamp to reasonable range
        
        return 1.5  # Default value
    
    def _compute_sample_entropy(self, data: np.ndarray, m: int = 2, r: float = 0.2) -> float:
        """Compute sample entropy."""
        if len(data) < m + 1:
            return 0.0
        
        # Simple implementation
        N = len(data)
        
        def _maxdist(xi, xj, m):
            return max([abs(ua - va) for ua, va in zip(xi, xj)])
        
        def _phi(m):
            patterns = np.array([data[i:i + m] for i in range(N - m + 1)])
            C = np.zeros(N - m + 1)
            
            for i in range(N - m + 1):
                template = patterns[i]
                for j in range(N - m + 1):
                    if _maxdist(template, patterns[j], m) <= r * np.std(data):
                        C[i] += 1
            
            phi = (N - m + 1.0) / np.sum(C)
            return phi
        
        try:
            return np.log(_phi(m) / _phi(m + 1))
        except:
            return 0.0
    
    def _compute_approximate_entropy(self, data: np.ndarray, m: int = 2, r: float = 0.2) -> float:
        """Compute approximate entropy."""
        # Simplified implementation
        if len(data) < m + 1:
            return 0.0
        
        try:
            std_data = np.std(data)
            if std_data == 0:
                return 0.0
            
            # This is a simplified version - full implementation would be more complex
            differences = np.abs(np.diff(data))
            entropy_estimate = -np.sum(differences * np.log(differences + 1e-8))
            return entropy_estimate / len(differences)
        except:
            return 0.0
    
    def _compute_lz_complexity(self, trajectory: np.ndarray) -> float:
        """Compute Lempel-Ziv complexity."""
        # Convert trajectory to symbolic sequence
        # Simple quantization approach
        if len(trajectory) < 5:
            return 1.0
        
        # Quantize each coordinate
        n_bins = 8
        quantized = []
        
        for i in range(trajectory.shape[1]):
            coord_data = trajectory[:, i]
            bins = np.linspace(np.min(coord_data), np.max(coord_data) + 1e-8, n_bins + 1)
            quantized_coord = np.digitize(coord_data, bins) - 1
            quantized.extend(quantized_coord)
        
        # Simple LZ complexity estimation
        sequence = ''.join(map(str, quantized))
        
        complexity = 1
        substring = sequence[0]
        
        for i in range(1, len(sequence)):
            if sequence[i] in substring:
                substring += sequence[i]
            else:
                complexity += 1
                substring = sequence[i]
        
        # Normalize by sequence length
        return complexity / len(sequence)
    
    def _compute_rqa_features(self, trajectory: np.ndarray) -> Dict[str, float]:
        """Compute Recurrence Quantification Analysis features."""
        # Simplified RQA implementation
        features = {}
        
        if len(trajectory) < 10:
            return {'rqa_recurrence_rate': 0.0, 'rqa_determinism': 0.0}
        
        # Use first coordinate for simplicity
        data = trajectory[:, 0]
        
        # Compute recurrence matrix (simplified)
        threshold = 0.1 * np.std(data)
        n = len(data)
        recurrence_matrix = np.zeros((n, n))
        
        for i in range(n):
            for j in range(n):
                if abs(data[i] - data[j]) < threshold:
                    recurrence_matrix[i, j] = 1
        
        # Recurrence rate
        recurrence_rate = np.sum(recurrence_matrix) / (n * n)
        features['rqa_recurrence_rate'] = recurrence_rate
        
        # Determinism (simplified)
        diagonal_lines = 0
        for diag in range(1, n):
            diagonal = np.diagonal(recurrence_matrix, offset=diag)
            if np.sum(diagonal) > 2:  # Line length > 2
                diagonal_lines += 1
        
        determinism = diagonal_lines / max(1, n - 1)
        features['rqa_determinism'] = determinism
        
        return features
    
    def _compute_multiscale_entropy(self, trajectory: np.ndarray) -> Dict[str, float]:
        """Compute multiscale entropy features."""
        features = {}
        
        # Use first coordinate
        data = trajectory[:, 0]
        
        # Compute entropy at different scales
        scales = [1, 2, 3, 5]
        
        for scale in scales:
            if len(data) >= scale * 10:  # Need sufficient data points
                # Coarse-grain the data
                coarse_grained = []
                for i in range(0, len(data) - scale + 1, scale):
                    coarse_grained.append(np.mean(data[i:i + scale]))
                
                # Compute sample entropy
                if len(coarse_grained) > 10:
                    mse = self._compute_sample_entropy(np.array(coarse_grained))
                    features[f'mse_scale_{scale}'] = mse
                else:
                    features[f'mse_scale_{scale}'] = 0.0
            else:
                features[f'mse_scale_{scale}'] = 0.0
        
        return features
    
    def extract_features_batch(self, samples: List[HandwritingSample]) -> List[HandwritingFeatures]:
        """
        Extract features from a batch of samples.
        
        Args:
            samples: List of handwriting samples
            
        Returns:
            features_list: List of extracted features
        """
        features_list = []
        
        for sample in samples:
            try:
                features = self.extract_features(sample)
                features_list.append(features)
            except Exception as e:
                logger.warning(f"Error extracting features from sample {sample.writer_id}: {e}")
                # Create empty features as fallback
                empty_features = HandwritingFeatures(
                    kinematic_features={},
                    geometric_features={},
                    temporal_features={}
                )
                features_list.append(empty_features)
        
        logger.info(f"Extracted features from {len(features_list)} samples")
        return features_list
    
    def get_feature_importance(self, features_list: List[HandwritingFeatures], 
                             labels: Optional[List[Any]] = None) -> Dict[str, float]:
        """
        Compute feature importance based on variance or mutual information.
        
        Args:
            features_list: List of extracted features
            labels: Optional labels for supervised importance
            
        Returns:
            importance_scores: Feature importance scores
        """
        if not features_list:
            return {}
        
        # Convert features to matrix
        feature_vectors = []
        for features in features_list:
            vector = features.to_vector(include_optional=True)
            feature_vectors.append(vector)
        
        feature_matrix = np.array(feature_vectors)
        feature_names = features_list[0].get_feature_names(include_optional=True)
        
        # Compute variance-based importance
        variances = np.var(feature_matrix, axis=0)
        
        # Normalize to [0, 1]
        if np.max(variances) > 0:
            normalized_variances = variances / np.max(variances)
        else:
            normalized_variances = variances
        
        importance_scores = dict(zip(feature_names, normalized_variances))
        
        return importance_scores