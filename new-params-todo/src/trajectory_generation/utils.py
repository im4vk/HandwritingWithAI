"""
Trajectory Generation Utilities
==============================

Utility functions for trajectory processing, analysis, and manipulation.
"""

import numpy as np
from typing import Dict, Any, List, Tuple, Optional, Union, Callable
from dataclasses import dataclass
import logging
from scipy import interpolate, signal, optimize
from scipy.spatial.distance import cdist
import matplotlib.pyplot as plt

logger = logging.getLogger(__name__)


@dataclass
class VelocityProfile:
    """
    Velocity profile data structure.
    
    Attributes:
        velocities: Velocity vectors [n_points, 2]
        speeds: Speed magnitudes [n_points]
        directions: Unit direction vectors [n_points, 2]
        timestamps: Time stamps [n_points]
        peak_speed: Maximum speed
        mean_speed: Average speed
        acceleration_phases: Indices of acceleration phases
        deceleration_phases: Indices of deceleration phases
    """
    velocities: np.ndarray
    speeds: np.ndarray
    directions: np.ndarray
    timestamps: np.ndarray
    peak_speed: float
    mean_speed: float
    acceleration_phases: Optional[List[Tuple[int, int]]] = None
    deceleration_phases: Optional[List[Tuple[int, int]]] = None


@dataclass
class AccelerationProfile:
    """
    Acceleration profile data structure.
    
    Attributes:
        accelerations: Acceleration vectors [n_points, 2]
        magnitudes: Acceleration magnitudes [n_points]
        timestamps: Time stamps [n_points]
        peak_acceleration: Maximum acceleration
        mean_acceleration: Average acceleration magnitude
        jerk: Jerk (rate of change of acceleration) [n_points, 2]
        smoothness_index: Trajectory smoothness measure
    """
    accelerations: np.ndarray
    magnitudes: np.ndarray
    timestamps: np.ndarray
    peak_acceleration: float
    mean_acceleration: float
    jerk: Optional[np.ndarray] = None
    smoothness_index: Optional[float] = None


class TrajectoryUtils:
    """
    Utility functions for trajectory analysis and manipulation.
    """
    
    @staticmethod
    def resample_trajectory(trajectory: np.ndarray,
                          timestamps: np.ndarray,
                          new_sampling_rate: float,
                          method: str = 'linear') -> Tuple[np.ndarray, np.ndarray]:
        """
        Resample trajectory to new sampling rate.
        
        Args:
            trajectory: Original trajectory [n_points, 2]
            timestamps: Original timestamps [n_points]
            new_sampling_rate: New sampling rate in Hz
            method: Interpolation method ('linear', 'cubic')
            
        Returns:
            resampled_trajectory: Resampled trajectory
            new_timestamps: New timestamps
        """
        if len(trajectory) < 2:
            return trajectory.copy(), timestamps.copy()
        
        # Calculate new time points
        duration = timestamps[-1] - timestamps[0]
        new_dt = 1.0 / new_sampling_rate
        new_timestamps = np.arange(timestamps[0], timestamps[-1] + new_dt/2, new_dt)
        
        # Interpolate trajectory
        resampled_trajectory = np.zeros((len(new_timestamps), trajectory.shape[1]))
        
        for i in range(trajectory.shape[1]):
            if method == 'linear':
                resampled_trajectory[:, i] = np.interp(new_timestamps, timestamps, trajectory[:, i])
            elif method == 'cubic':
                # Use cubic spline interpolation
                try:
                    cs = interpolate.CubicSpline(timestamps, trajectory[:, i])
                    resampled_trajectory[:, i] = cs(new_timestamps)
                except Exception:
                    # Fallback to linear
                    resampled_trajectory[:, i] = np.interp(new_timestamps, timestamps, trajectory[:, i])
            else:
                raise ValueError(f"Unknown interpolation method: {method}")
        
        return resampled_trajectory, new_timestamps
    
    @staticmethod
    def compute_velocity_profile(trajectory: np.ndarray,
                               timestamps: np.ndarray,
                               smoothing: bool = True) -> VelocityProfile:
        """
        Compute comprehensive velocity profile from trajectory.
        
        Args:
            trajectory: Trajectory points [n_points, 2]
            timestamps: Time stamps [n_points]
            smoothing: Whether to apply smoothing
            
        Returns:
            velocity_profile: Velocity profile data
        """
        if len(trajectory) < 2:
            return VelocityProfile(
                velocities=np.zeros((len(trajectory), 2)),
                speeds=np.zeros(len(trajectory)),
                directions=np.zeros((len(trajectory), 2)),
                timestamps=timestamps,
                peak_speed=0.0,
                mean_speed=0.0
            )
        
        # Compute velocities using finite differences
        dt = np.diff(timestamps)
        dt = np.append(dt, dt[-1])  # Extend for same length
        
        velocities = np.zeros_like(trajectory)
        velocities[1:] = np.diff(trajectory, axis=0) / dt[1:].reshape(-1, 1)
        velocities[0] = velocities[1]  # Copy first velocity
        
        # Apply smoothing if requested
        if smoothing and len(velocities) >= 5:
            window_length = min(5, len(velocities))
            if window_length % 2 == 0:
                window_length -= 1
            
            try:
                for i in range(velocities.shape[1]):
                    velocities[:, i] = signal.savgol_filter(velocities[:, i], window_length, 3)
            except Exception:
                pass  # Use unsmoothed velocities
        
        # Compute derived quantities
        speeds = np.linalg.norm(velocities, axis=1)
        directions = np.zeros_like(velocities)
        
        nonzero_mask = speeds > 1e-8
        directions[nonzero_mask] = velocities[nonzero_mask] / speeds[nonzero_mask].reshape(-1, 1)
        
        peak_speed = np.max(speeds) if len(speeds) > 0 else 0.0
        mean_speed = np.mean(speeds) if len(speeds) > 0 else 0.0
        
        # Detect acceleration/deceleration phases
        acceleration_phases, deceleration_phases = TrajectoryUtils._detect_movement_phases(speeds)
        
        return VelocityProfile(
            velocities=velocities,
            speeds=speeds,
            directions=directions,
            timestamps=timestamps,
            peak_speed=peak_speed,
            mean_speed=mean_speed,
            acceleration_phases=acceleration_phases,
            deceleration_phases=deceleration_phases
        )
    
    @staticmethod
    def compute_acceleration_profile(velocities: np.ndarray,
                                   timestamps: np.ndarray,
                                   compute_jerk: bool = True) -> AccelerationProfile:
        """
        Compute acceleration profile from velocities.
        
        Args:
            velocities: Velocity vectors [n_points, 2]
            timestamps: Time stamps [n_points]
            compute_jerk: Whether to compute jerk
            
        Returns:
            acceleration_profile: Acceleration profile data
        """
        if len(velocities) < 2:
            return AccelerationProfile(
                accelerations=np.zeros_like(velocities),
                magnitudes=np.zeros(len(velocities)),
                timestamps=timestamps,
                peak_acceleration=0.0,
                mean_acceleration=0.0
            )
        
        # Compute accelerations
        dt = np.diff(timestamps)
        dt = np.append(dt, dt[-1])
        
        accelerations = np.zeros_like(velocities)
        accelerations[1:] = np.diff(velocities, axis=0) / dt[1:].reshape(-1, 1)
        accelerations[0] = accelerations[1]
        
        # Compute magnitudes
        magnitudes = np.linalg.norm(accelerations, axis=1)
        peak_acceleration = np.max(magnitudes) if len(magnitudes) > 0 else 0.0
        mean_acceleration = np.mean(magnitudes) if len(magnitudes) > 0 else 0.0
        
        # Compute jerk if requested
        jerk = None
        smoothness_index = None
        
        if compute_jerk and len(accelerations) >= 2:
            jerk = np.zeros_like(accelerations)
            jerk[1:] = np.diff(accelerations, axis=0) / dt[1:].reshape(-1, 1)
            jerk[0] = jerk[1]
            
            # Compute smoothness index (inverse of mean jerk magnitude)
            jerk_magnitudes = np.linalg.norm(jerk, axis=1)
            mean_jerk = np.mean(jerk_magnitudes)
            smoothness_index = 1.0 / (1.0 + mean_jerk) if mean_jerk > 0 else 1.0
        
        return AccelerationProfile(
            accelerations=accelerations,
            magnitudes=magnitudes,
            timestamps=timestamps,
            peak_acceleration=peak_acceleration,
            mean_acceleration=mean_acceleration,
            jerk=jerk,
            smoothness_index=smoothness_index
        )
    
    @staticmethod
    def _detect_movement_phases(speeds: np.ndarray,
                              threshold: float = 0.1) -> Tuple[List[Tuple[int, int]], List[Tuple[int, int]]]:
        """Detect acceleration and deceleration phases in speed profile."""
        if len(speeds) < 3:
            return [], []
        
        # Compute speed changes
        speed_diff = np.diff(speeds)
        
        # Find acceleration and deceleration regions
        acceleration_mask = speed_diff > threshold * np.max(speed_diff)
        deceleration_mask = speed_diff < -threshold * np.max(speed_diff)
        
        # Find contiguous regions
        def find_contiguous_regions(mask):
            regions = []
            in_region = False
            start_idx = 0
            
            for i, val in enumerate(mask):
                if val and not in_region:
                    start_idx = i
                    in_region = True
                elif not val and in_region:
                    regions.append((start_idx, i))
                    in_region = False
            
            if in_region:
                regions.append((start_idx, len(mask)))
            
            return regions
        
        acceleration_phases = find_contiguous_regions(acceleration_mask)
        deceleration_phases = find_contiguous_regions(deceleration_mask)
        
        return acceleration_phases, deceleration_phases
    
    @staticmethod
    def filter_trajectory(trajectory: np.ndarray,
                         timestamps: np.ndarray,
                         filter_type: str = 'butterworth',
                         **filter_params) -> np.ndarray:
        """
        Apply filtering to trajectory data.
        
        Args:
            trajectory: Trajectory points [n_points, 2]
            timestamps: Time stamps [n_points]
            filter_type: Type of filter ('butterworth', 'gaussian', 'median')
            **filter_params: Filter-specific parameters
            
        Returns:
            filtered_trajectory: Filtered trajectory
        """
        if len(trajectory) < 3:
            return trajectory.copy()
        
        filtered_trajectory = trajectory.copy()
        
        if filter_type == 'butterworth':
            # Butterworth low-pass filter
            cutoff_freq = filter_params.get('cutoff_freq', 10.0)  # Hz
            order = filter_params.get('order', 4)
            
            # Estimate sampling rate
            sampling_rate = 1.0 / np.mean(np.diff(timestamps))
            nyquist_freq = sampling_rate / 2
            normalized_cutoff = cutoff_freq / nyquist_freq
            
            if normalized_cutoff < 1.0:
                try:
                    b, a = signal.butter(order, normalized_cutoff, btype='low')
                    
                    for i in range(trajectory.shape[1]):
                        filtered_trajectory[:, i] = signal.filtfilt(b, a, trajectory[:, i])
                except Exception as e:
                    logger.warning(f"Butterworth filtering failed: {e}")
        
        elif filter_type == 'gaussian':
            # Gaussian smoothing
            sigma = filter_params.get('sigma', 1.0)
            
            for i in range(trajectory.shape[1]):
                filtered_trajectory[:, i] = signal.gaussian_filter1d(trajectory[:, i], sigma)
        
        elif filter_type == 'median':
            # Median filter
            kernel_size = filter_params.get('kernel_size', 3)
            
            for i in range(trajectory.shape[1]):
                filtered_trajectory[:, i] = signal.medfilt(trajectory[:, i], kernel_size)
        
        elif filter_type == 'savgol':
            # Savitzky-Golay filter
            window_length = filter_params.get('window_length', 5)
            polyorder = filter_params.get('polyorder', 3)
            
            if window_length <= len(trajectory) and window_length % 2 == 1:
                try:
                    for i in range(trajectory.shape[1]):
                        filtered_trajectory[:, i] = signal.savgol_filter(
                            trajectory[:, i], window_length, polyorder
                        )
                except Exception as e:
                    logger.warning(f"Savgol filtering failed: {e}")
        
        else:
            logger.warning(f"Unknown filter type: {filter_type}")
        
        return filtered_trajectory
    
    @staticmethod
    def compute_curvature(trajectory: np.ndarray,
                         timestamps: np.ndarray,
                         method: str = 'finite_difference') -> np.ndarray:
        """
        Compute curvature along trajectory.
        
        Args:
            trajectory: Trajectory points [n_points, 2]
            timestamps: Time stamps [n_points]
            method: Computation method ('finite_difference', 'parametric')
            
        Returns:
            curvature: Curvature values [n_points]
        """
        if len(trajectory) < 3:
            return np.zeros(len(trajectory))
        
        if method == 'finite_difference':
            # Compute first and second derivatives
            velocity_profile = TrajectoryUtils.compute_velocity_profile(trajectory, timestamps, smoothing=True)
            acceleration_profile = TrajectoryUtils.compute_acceleration_profile(
                velocity_profile.velocities, timestamps, compute_jerk=False
            )
            
            velocities = velocity_profile.velocities
            accelerations = acceleration_profile.accelerations
            
            # Curvature formula: |v x a| / |v|^3
            cross_product = velocities[:, 0] * accelerations[:, 1] - velocities[:, 1] * accelerations[:, 0]
            speed_cubed = (velocity_profile.speeds + 1e-8) ** 3
            
            curvature = np.abs(cross_product) / speed_cubed
        
        elif method == 'parametric':
            # Parametric curvature computation
            # Use arc length parameterization
            distances = np.cumsum(np.concatenate([[0], np.linalg.norm(np.diff(trajectory, axis=0), axis=1)]))
            
            curvature = np.zeros(len(trajectory))
            
            for i in range(1, len(trajectory) - 1):
                # Three consecutive points
                p1, p2, p3 = trajectory[i-1:i+2]
                
                # Compute curvature using three-point formula
                try:
                    # Triangle area method
                    a = np.linalg.norm(p2 - p1)
                    b = np.linalg.norm(p3 - p2)
                    c = np.linalg.norm(p3 - p1)
                    
                    # Area of triangle
                    s = (a + b + c) / 2  # Semi-perimeter
                    area = np.sqrt(max(0, s * (s - a) * (s - b) * (s - c)))
                    
                    # Curvature = 4 * Area / (a * b * c)
                    if a * b * c > 1e-12:
                        curvature[i] = 4 * area / (a * b * c)
                    else:
                        curvature[i] = 0
                
                except Exception:
                    curvature[i] = 0
        
        else:
            raise ValueError(f"Unknown curvature computation method: {method}")
        
        return curvature
    
    @staticmethod
    def detect_turning_points(trajectory: np.ndarray,
                            timestamps: np.ndarray,
                            threshold: float = 0.1) -> List[int]:
        """
        Detect turning points (local extrema) in trajectory.
        
        Args:
            trajectory: Trajectory points [n_points, 2]
            timestamps: Time stamps [n_points]
            threshold: Minimum curvature threshold for turning points
            
        Returns:
            turning_points: Indices of turning points
        """
        if len(trajectory) < 5:
            return []
        
        # Compute curvature
        curvature = TrajectoryUtils.compute_curvature(trajectory, timestamps)
        
        # Find local maxima in curvature
        peaks, _ = signal.find_peaks(curvature, height=threshold, distance=3)
        
        return peaks.tolist()
    
    @staticmethod
    def segment_trajectory_by_velocity(trajectory: np.ndarray,
                                     timestamps: np.ndarray,
                                     velocity_threshold: float = 0.001) -> List[Tuple[int, int]]:
        """
        Segment trajectory based on velocity minima (pen lifts, pauses).
        
        Args:
            trajectory: Trajectory points [n_points, 2]
            timestamps: Time stamps [n_points]
            velocity_threshold: Velocity threshold for segmentation
            
        Returns:
            segments: List of (start_idx, end_idx) tuples
        """
        if len(trajectory) < 2:
            return [(0, len(trajectory))]
        
        # Compute velocity profile
        velocity_profile = TrajectoryUtils.compute_velocity_profile(trajectory, timestamps)
        speeds = velocity_profile.speeds
        
        # Find low-velocity regions
        low_velocity_mask = speeds < velocity_threshold
        
        # Find segment boundaries
        segments = []
        in_segment = not low_velocity_mask[0]
        segment_start = 0
        
        for i in range(1, len(speeds)):
            if not low_velocity_mask[i] and not in_segment:
                # Start of new segment
                segment_start = i
                in_segment = True
            elif low_velocity_mask[i] and in_segment:
                # End of current segment
                segments.append((segment_start, i))
                in_segment = False
        
        # Handle final segment
        if in_segment:
            segments.append((segment_start, len(trajectory)))
        
        # Filter out very short segments
        min_segment_length = 3
        segments = [(start, end) for start, end in segments if end - start >= min_segment_length]
        
        return segments
    
    @staticmethod
    def compute_trajectory_similarity(traj1: np.ndarray,
                                    traj2: np.ndarray,
                                    method: str = 'dtw') -> float:
        """
        Compute similarity between two trajectories.
        
        Args:
            traj1: First trajectory [n_points1, 2]
            traj2: Second trajectory [n_points2, 2]
            method: Similarity method ('dtw', 'hausdorff', 'frechet')
            
        Returns:
            similarity: Similarity score (higher = more similar)
        """
        if len(traj1) == 0 or len(traj2) == 0:
            return 0.0
        
        if method == 'dtw':
            return TrajectoryUtils._compute_dtw_similarity(traj1, traj2)
        elif method == 'hausdorff':
            return TrajectoryUtils._compute_hausdorff_similarity(traj1, traj2)
        elif method == 'frechet':
            return TrajectoryUtils._compute_frechet_similarity(traj1, traj2)
        else:
            raise ValueError(f"Unknown similarity method: {method}")
    
    @staticmethod
    def _compute_dtw_similarity(traj1: np.ndarray, traj2: np.ndarray) -> float:
        """Compute Dynamic Time Warping similarity."""
        # Compute distance matrix
        dist_matrix = cdist(traj1, traj2, metric='euclidean')
        
        n, m = dist_matrix.shape
        
        # DTW dynamic programming
        dtw_matrix = np.full((n + 1, m + 1), np.inf)
        dtw_matrix[0, 0] = 0
        
        for i in range(1, n + 1):
            for j in range(1, m + 1):
                cost = dist_matrix[i-1, j-1]
                dtw_matrix[i, j] = cost + min(
                    dtw_matrix[i-1, j],      # insertion
                    dtw_matrix[i, j-1],      # deletion
                    dtw_matrix[i-1, j-1]     # match
                )
        
        # Normalize by path length
        dtw_distance = dtw_matrix[n, m] / (n + m)
        
        # Convert to similarity (0-1 scale)
        max_distance = np.max([np.max(traj1), np.max(traj2)]) * np.sqrt(2)
        similarity = max(0, 1 - dtw_distance / max_distance)
        
        return similarity
    
    @staticmethod
    def _compute_hausdorff_similarity(traj1: np.ndarray, traj2: np.ndarray) -> float:
        """Compute Hausdorff distance-based similarity."""
        # Compute distance matrices
        dist_12 = cdist(traj1, traj2, metric='euclidean')
        dist_21 = cdist(traj2, traj1, metric='euclidean')
        
        # Hausdorff distance
        hausdorff_12 = np.max(np.min(dist_12, axis=1))
        hausdorff_21 = np.max(np.min(dist_21, axis=1))
        hausdorff_distance = max(hausdorff_12, hausdorff_21)
        
        # Convert to similarity
        max_distance = np.max([np.max(traj1), np.max(traj2)]) * np.sqrt(2)
        similarity = max(0, 1 - hausdorff_distance / max_distance)
        
        return similarity
    
    @staticmethod
    def _compute_frechet_similarity(traj1: np.ndarray, traj2: np.ndarray) -> float:
        """Compute discrete Frechet distance-based similarity."""
        n, m = len(traj1), len(traj2)
        
        # Compute distance matrix
        dist_matrix = cdist(traj1, traj2, metric='euclidean')
        
        # Dynamic programming for discrete Frechet distance
        frechet_matrix = np.full((n, m), np.inf)
        frechet_matrix[0, 0] = dist_matrix[0, 0]
        
        # Fill first row and column
        for i in range(1, n):
            frechet_matrix[i, 0] = max(frechet_matrix[i-1, 0], dist_matrix[i, 0])
        
        for j in range(1, m):
            frechet_matrix[0, j] = max(frechet_matrix[0, j-1], dist_matrix[0, j])
        
        # Fill remaining matrix
        for i in range(1, n):
            for j in range(1, m):
                frechet_matrix[i, j] = max(
                    dist_matrix[i, j],
                    min(frechet_matrix[i-1, j], frechet_matrix[i, j-1], frechet_matrix[i-1, j-1])
                )
        
        frechet_distance = frechet_matrix[n-1, m-1]
        
        # Convert to similarity
        max_distance = np.max([np.max(traj1), np.max(traj2)]) * np.sqrt(2)
        similarity = max(0, 1 - frechet_distance / max_distance)
        
        return similarity
    
    @staticmethod
    def visualize_trajectory(trajectory: np.ndarray,
                           timestamps: Optional[np.ndarray] = None,
                           velocities: Optional[np.ndarray] = None,
                           title: str = "Trajectory",
                           save_path: Optional[str] = None) -> None:
        """
        Visualize trajectory with optional velocity information.
        
        Args:
            trajectory: Trajectory points [n_points, 2]
            timestamps: Time stamps [n_points]
            velocities: Velocity vectors [n_points, 2]
            title: Plot title
            save_path: Path to save plot
        """
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        # Trajectory plot
        ax1 = axes[0, 0]
        ax1.plot(trajectory[:, 0], trajectory[:, 1], 'b-', linewidth=2, alpha=0.7)
        ax1.scatter(trajectory[0, 0], trajectory[0, 1], c='green', s=100, marker='o', label='Start')
        ax1.scatter(trajectory[-1, 0], trajectory[-1, 1], c='red', s=100, marker='s', label='End')
        ax1.set_xlabel('X Position')
        ax1.set_ylabel('Y Position')
        ax1.set_title(f'{title} - Spatial Path')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        ax1.axis('equal')
        
        if timestamps is not None:
            # Time-based plots
            ax2 = axes[0, 1]
            ax2.plot(timestamps, trajectory[:, 0], 'r-', label='X')
            ax2.plot(timestamps, trajectory[:, 1], 'b-', label='Y')
            ax2.set_xlabel('Time (s)')
            ax2.set_ylabel('Position')
            ax2.set_title('Position vs Time')
            ax2.legend()
            ax2.grid(True, alpha=0.3)
            
            if velocities is not None:
                # Velocity plots
                ax3 = axes[1, 0]
                speeds = np.linalg.norm(velocities, axis=1)
                ax3.plot(timestamps, speeds, 'g-', linewidth=2)
                ax3.set_xlabel('Time (s)')
                ax3.set_ylabel('Speed')
                ax3.set_title('Speed Profile')
                ax3.grid(True, alpha=0.3)
                
                # Velocity vectors (subsampled for clarity)
                ax4 = axes[1, 1]
                step = max(1, len(trajectory) // 20)  # Show ~20 arrows
                ax4.plot(trajectory[:, 0], trajectory[:, 1], 'b-', alpha=0.5)
                ax4.quiver(trajectory[::step, 0], trajectory[::step, 1],
                          velocities[::step, 0], velocities[::step, 1],
                          angles='xy', scale_units='xy', scale=1, alpha=0.7)
                ax4.set_xlabel('X Position')
                ax4.set_ylabel('Y Position')
                ax4.set_title('Velocity Vectors')
                ax4.grid(True, alpha=0.3)
                ax4.axis('equal')
            else:
                # Hide unused subplots
                axes[1, 0].set_visible(False)
                axes[1, 1].set_visible(False)
        else:
            # Hide time-dependent plots
            axes[0, 1].set_visible(False)
            axes[1, 0].set_visible(False)
            axes[1, 1].set_visible(False)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Saved trajectory plot to {save_path}")
        else:
            plt.show()
        
        plt.close()
    
    @staticmethod
    def export_trajectory(trajectory: np.ndarray,
                         timestamps: Optional[np.ndarray] = None,
                         velocities: Optional[np.ndarray] = None,
                         accelerations: Optional[np.ndarray] = None,
                         metadata: Optional[Dict[str, Any]] = None,
                         file_path: str = "trajectory.npz") -> None:
        """
        Export trajectory data to file.
        
        Args:
            trajectory: Trajectory points [n_points, 2]
            timestamps: Time stamps [n_points]
            velocities: Velocity vectors [n_points, 2]
            accelerations: Acceleration vectors [n_points, 2]
            metadata: Additional metadata
            file_path: Output file path
        """
        # Prepare data dictionary
        data = {'trajectory': trajectory}
        
        if timestamps is not None:
            data['timestamps'] = timestamps
        
        if velocities is not None:
            data['velocities'] = velocities
        
        if accelerations is not None:
            data['accelerations'] = accelerations
        
        if metadata is not None:
            data['metadata'] = metadata
        
        # Save to file
        if file_path.endswith('.npz'):
            np.savez_compressed(file_path, **data)
        elif file_path.endswith('.npy'):
            np.save(file_path, data)
        else:
            # Default to npz
            np.savez_compressed(file_path + '.npz', **data)
        
        logger.info(f"Exported trajectory data to {file_path}")
    
    @staticmethod
    def load_trajectory(file_path: str) -> Dict[str, Any]:
        """
        Load trajectory data from file.
        
        Args:
            file_path: Input file path
            
        Returns:
            trajectory_data: Loaded trajectory data
        """
        if file_path.endswith('.npz'):
            data = np.load(file_path, allow_pickle=True)
            return {key: data[key] for key in data.files}
        elif file_path.endswith('.npy'):
            return np.load(file_path, allow_pickle=True).item()
        else:
            raise ValueError(f"Unsupported file format: {file_path}")
    
    @staticmethod
    def validate_trajectory(trajectory: np.ndarray,
                          timestamps: Optional[np.ndarray] = None,
                          max_velocity: float = 10.0,
                          max_acceleration: float = 100.0) -> Dict[str, Any]:
        """
        Validate trajectory for physical plausibility.
        
        Args:
            trajectory: Trajectory points [n_points, 2]
            timestamps: Time stamps [n_points]
            max_velocity: Maximum allowed velocity (m/s)
            max_acceleration: Maximum allowed acceleration (m/s²)
            
        Returns:
            validation_results: Validation results and issues
        """
        issues = []
        warnings = []
        
        if len(trajectory) == 0:
            issues.append("Empty trajectory")
            return {'valid': False, 'issues': issues, 'warnings': warnings}
        
        if timestamps is not None and len(timestamps) != len(trajectory):
            issues.append("Timestamp and trajectory length mismatch")
        
        if len(trajectory) >= 2:
            # Check for NaN or infinite values
            if np.any(~np.isfinite(trajectory)):
                issues.append("Non-finite values in trajectory")
            
            # Compute velocity and acceleration
            if timestamps is not None:
                velocity_profile = TrajectoryUtils.compute_velocity_profile(trajectory, timestamps)
                
                # Check velocity limits
                max_speed = velocity_profile.peak_speed
                if max_speed > max_velocity:
                    issues.append(f"Maximum velocity ({max_speed:.3f}) exceeds limit ({max_velocity})")
                
                # Check acceleration limits
                acceleration_profile = TrajectoryUtils.compute_acceleration_profile(
                    velocity_profile.velocities, timestamps
                )
                
                max_acc = acceleration_profile.peak_acceleration
                if max_acc > max_acceleration:
                    issues.append(f"Maximum acceleration ({max_acc:.3f}) exceeds limit ({max_acceleration})")
                
                # Check for excessive jerk
                if acceleration_profile.smoothness_index is not None and acceleration_profile.smoothness_index < 0.1:
                    warnings.append("Trajectory has high jerk (low smoothness)")
            
            # Check for stationary points
            distances = np.linalg.norm(np.diff(trajectory, axis=0), axis=1)
            stationary_ratio = np.sum(distances < 1e-6) / len(distances)
            if stationary_ratio > 0.5:
                warnings.append("High proportion of stationary points")
        
        valid = len(issues) == 0
        
        return {
            'valid': valid,
            'issues': issues,
            'warnings': warnings,
            'num_points': len(trajectory),
            'duration': timestamps[-1] - timestamps[0] if timestamps is not None and len(timestamps) > 1 else None
        }