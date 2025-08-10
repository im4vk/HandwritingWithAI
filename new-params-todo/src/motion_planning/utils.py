"""
Motion Planning Utilities
=========================

Utility functions and data structures for motion planning operations.
"""

import numpy as np
from typing import Dict, Any, List, Tuple, Optional, Union
from dataclasses import dataclass
from enum import Enum
import logging
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.interpolate import interp1d, CubicSpline
from scipy.spatial.distance import cdist

logger = logging.getLogger(__name__)


@dataclass
class JointTrajectory:
    """
    Joint space trajectory representation.
    
    Attributes:
        positions: Joint positions [n_points, n_joints]
        velocities: Joint velocities [n_points, n_joints]
        accelerations: Joint accelerations [n_points, n_joints]
        timestamps: Time stamps [n_points]
        joint_names: Names of joints
        metadata: Additional trajectory metadata
    """
    positions: np.ndarray
    velocities: Optional[np.ndarray] = None
    accelerations: Optional[np.ndarray] = None
    timestamps: Optional[np.ndarray] = None
    joint_names: Optional[List[str]] = None
    metadata: Optional[Dict[str, Any]] = None
    
    def __post_init__(self):
        """Initialize default values and validate."""
        n_points, n_joints = self.positions.shape
        
        if self.velocities is None:
            self.velocities = np.zeros((n_points, n_joints))
        
        if self.accelerations is None:
            self.accelerations = np.zeros((n_points, n_joints))
        
        if self.timestamps is None:
            self.timestamps = np.linspace(0, 1, n_points)
        
        if self.joint_names is None:
            self.joint_names = [f'joint_{i+1}' for i in range(n_joints)]
        
        if self.metadata is None:
            self.metadata = {}
    
    def get_duration(self) -> float:
        """Get trajectory duration."""
        return self.timestamps[-1] - self.timestamps[0]
    
    def get_joint_index(self, joint_name: str) -> int:
        """Get index of joint by name."""
        try:
            return self.joint_names.index(joint_name)
        except ValueError:
            raise ValueError(f"Joint '{joint_name}' not found in trajectory")
    
    def get_joint_trajectory(self, joint_name: str) -> Dict[str, np.ndarray]:
        """Get trajectory for specific joint."""
        idx = self.get_joint_index(joint_name)
        
        return {
            'positions': self.positions[:, idx],
            'velocities': self.velocities[:, idx],
            'accelerations': self.accelerations[:, idx],
            'timestamps': self.timestamps
        }
    
    def interpolate(self, new_timestamps: np.ndarray, method: str = 'cubic') -> 'JointTrajectory':
        """Interpolate trajectory to new timestamps."""
        n_joints = self.positions.shape[1]
        new_positions = np.zeros((len(new_timestamps), n_joints))
        new_velocities = np.zeros((len(new_timestamps), n_joints))
        
        for j in range(n_joints):
            # Interpolate positions
            if method == 'linear':
                f_pos = interp1d(self.timestamps, self.positions[:, j], kind='linear')
                f_vel = interp1d(self.timestamps, self.velocities[:, j], kind='linear')
            elif method == 'cubic':
                f_pos = CubicSpline(self.timestamps, self.positions[:, j])
                f_vel = CubicSpline(self.timestamps, self.velocities[:, j])
            else:
                raise ValueError(f"Unknown interpolation method: {method}")
            
            new_positions[:, j] = f_pos(new_timestamps)
            new_velocities[:, j] = f_vel(new_timestamps)
        
        # Compute new accelerations
        new_accelerations = np.zeros_like(new_velocities)
        if len(new_timestamps) > 1:
            dt = np.diff(new_timestamps)
            new_accelerations[1:] = np.diff(new_velocities, axis=0) / dt.reshape(-1, 1)
        
        return JointTrajectory(
            positions=new_positions,
            velocities=new_velocities,
            accelerations=new_accelerations,
            timestamps=new_timestamps,
            joint_names=self.joint_names.copy(),
            metadata=self.metadata.copy()
        )
    
    def crop(self, start_time: float, end_time: float) -> 'JointTrajectory':
        """Crop trajectory to time interval."""
        # Find indices
        start_idx = np.searchsorted(self.timestamps, start_time)
        end_idx = np.searchsorted(self.timestamps, end_time)
        
        if start_idx >= len(self.timestamps) or end_idx <= 0:
            raise ValueError("Time interval is outside trajectory range")
        
        start_idx = max(0, start_idx)
        end_idx = min(len(self.timestamps), end_idx)
        
        return JointTrajectory(
            positions=self.positions[start_idx:end_idx],
            velocities=self.velocities[start_idx:end_idx],
            accelerations=self.accelerations[start_idx:end_idx],
            timestamps=self.timestamps[start_idx:end_idx],
            joint_names=self.joint_names.copy(),
            metadata=self.metadata.copy()
        )


@dataclass
class CartesianTrajectory:
    """
    Cartesian space trajectory representation.
    
    Attributes:
        positions: Cartesian positions [n_points, 3]
        orientations: Orientations [n_points, 3] (Euler angles)
        velocities: Cartesian velocities [n_points, 6] (linear + angular)
        accelerations: Cartesian accelerations [n_points, 6]
        timestamps: Time stamps [n_points]
        frame: Reference frame name
        metadata: Additional trajectory metadata
    """
    positions: np.ndarray
    orientations: Optional[np.ndarray] = None
    velocities: Optional[np.ndarray] = None
    accelerations: Optional[np.ndarray] = None
    timestamps: Optional[np.ndarray] = None
    frame: str = 'world'
    metadata: Optional[Dict[str, Any]] = None
    
    def __post_init__(self):
        """Initialize default values."""
        n_points = len(self.positions)
        
        if self.orientations is None:
            self.orientations = np.zeros((n_points, 3))
        
        if self.velocities is None:
            self.velocities = np.zeros((n_points, 6))
        
        if self.accelerations is None:
            self.accelerations = np.zeros((n_points, 6))
        
        if self.timestamps is None:
            self.timestamps = np.linspace(0, 1, n_points)
        
        if self.metadata is None:
            self.metadata = {}
    
    def get_pose_matrices(self) -> np.ndarray:
        """Get pose as 4x4 transformation matrices."""
        n_points = len(self.positions)
        poses = np.zeros((n_points, 4, 4))
        
        for i in range(n_points):
            # Create transformation matrix
            poses[i] = np.eye(4)
            poses[i][:3, 3] = self.positions[i]
            
            # Add rotation if available
            if self.orientations is not None:
                R = self._euler_to_rotation(self.orientations[i])
                poses[i][:3, :3] = R
        
        return poses
    
    def _euler_to_rotation(self, euler_angles: np.ndarray) -> np.ndarray:
        """Convert Euler angles to rotation matrix."""
        roll, pitch, yaw = euler_angles
        
        # Rotation matrices
        Rx = np.array([
            [1, 0, 0],
            [0, np.cos(roll), -np.sin(roll)],
            [0, np.sin(roll), np.cos(roll)]
        ])
        
        Ry = np.array([
            [np.cos(pitch), 0, np.sin(pitch)],
            [0, 1, 0],
            [-np.sin(pitch), 0, np.cos(pitch)]
        ])
        
        Rz = np.array([
            [np.cos(yaw), -np.sin(yaw), 0],
            [np.sin(yaw), np.cos(yaw), 0],
            [0, 0, 1]
        ])
        
        return Rz @ Ry @ Rx


class MotionPlanningUtils:
    """
    Utility functions for motion planning operations.
    """
    
    @staticmethod
    def compute_trajectory_length(trajectory: np.ndarray) -> float:
        """
        Compute total length of trajectory.
        
        Args:
            trajectory: Trajectory points [n_points, n_dims]
            
        Returns:
            length: Total trajectory length
        """
        if len(trajectory) < 2:
            return 0.0
        
        distances = np.linalg.norm(np.diff(trajectory, axis=0), axis=1)
        return np.sum(distances)
    
    @staticmethod
    def compute_trajectory_smoothness(trajectory: np.ndarray, timestamps: np.ndarray) -> float:
        """
        Compute trajectory smoothness metric.
        
        Args:
            trajectory: Trajectory points [n_points, n_dims]
            timestamps: Time stamps [n_points]
            
        Returns:
            smoothness: Smoothness metric (lower = smoother)
        """
        if len(trajectory) < 3:
            return 0.0
        
        # Compute velocities and accelerations
        dt = np.diff(timestamps)
        velocities = np.diff(trajectory, axis=0) / dt.reshape(-1, 1)
        
        if len(velocities) < 2:
            return 0.0
        
        dt2 = dt[1:]
        accelerations = np.diff(velocities, axis=0) / dt2.reshape(-1, 1)
        
        # Smoothness as variance in acceleration
        smoothness = np.mean(np.var(accelerations, axis=0))
        return smoothness
    
    @staticmethod
    def resample_trajectory(trajectory: np.ndarray, 
                          original_timestamps: np.ndarray,
                          target_sample_rate: float) -> Tuple[np.ndarray, np.ndarray]:
        """
        Resample trajectory to target sample rate.
        
        Args:
            trajectory: Original trajectory [n_points, n_dims]
            original_timestamps: Original timestamps [n_points]
            target_sample_rate: Target sample rate in Hz
            
        Returns:
            resampled_trajectory: Resampled trajectory
            new_timestamps: New timestamps
        """
        # Create new timestamp vector
        duration = original_timestamps[-1] - original_timestamps[0]
        dt = 1.0 / target_sample_rate
        new_timestamps = np.arange(original_timestamps[0], original_timestamps[-1] + dt/2, dt)
        
        # Interpolate each dimension
        n_dims = trajectory.shape[1]
        resampled_trajectory = np.zeros((len(new_timestamps), n_dims))
        
        for dim in range(n_dims):
            f = interp1d(original_timestamps, trajectory[:, dim], 
                        kind='cubic', bounds_error=False, fill_value='extrapolate')
            resampled_trajectory[:, dim] = f(new_timestamps)
        
        return resampled_trajectory, new_timestamps
    
    @staticmethod
    def smooth_trajectory(trajectory: np.ndarray, 
                         window_size: int = 5,
                         method: str = 'gaussian') -> np.ndarray:
        """
        Smooth trajectory using various methods.
        
        Args:
            trajectory: Input trajectory [n_points, n_dims]
            window_size: Smoothing window size
            method: Smoothing method ('gaussian', 'moving_average', 'savgol')
            
        Returns:
            smoothed_trajectory: Smoothed trajectory
        """
        if len(trajectory) < window_size:
            return trajectory.copy()
        
        smoothed = trajectory.copy()
        
        if method == 'moving_average':
            # Simple moving average
            for i in range(window_size//2, len(trajectory) - window_size//2):
                start_idx = i - window_size//2
                end_idx = i + window_size//2 + 1
                smoothed[i] = np.mean(trajectory[start_idx:end_idx], axis=0)
        
        elif method == 'gaussian':
            # Gaussian smoothing
            from scipy.ndimage import gaussian_filter1d
            sigma = window_size / 3.0
            
            for dim in range(trajectory.shape[1]):
                smoothed[:, dim] = gaussian_filter1d(trajectory[:, dim], sigma)
        
        elif method == 'savgol':
            # Savitzky-Golay filter
            from scipy.signal import savgol_filter
            
            if window_size % 2 == 0:
                window_size += 1  # Must be odd
            
            polyorder = min(3, window_size - 1)
            
            for dim in range(trajectory.shape[1]):
                smoothed[:, dim] = savgol_filter(trajectory[:, dim], window_size, polyorder)
        
        else:
            raise ValueError(f"Unknown smoothing method: {method}")
        
        return smoothed
    
    @staticmethod
    def compute_trajectory_similarity(traj1: np.ndarray, traj2: np.ndarray,
                                    method: str = 'dtw') -> float:
        """
        Compute similarity between two trajectories.
        
        Args:
            traj1: First trajectory [n_points1, n_dims]
            traj2: Second trajectory [n_points2, n_dims]
            method: Similarity method ('dtw', 'hausdorff', 'frechet')
            
        Returns:
            similarity: Similarity score (0-1, higher = more similar)
        """
        if method == 'dtw':
            return MotionPlanningUtils._compute_dtw_similarity(traj1, traj2)
        elif method == 'hausdorff':
            return MotionPlanningUtils._compute_hausdorff_similarity(traj1, traj2)
        elif method == 'frechet':
            return MotionPlanningUtils._compute_frechet_similarity(traj1, traj2)
        else:
            raise ValueError(f"Unknown similarity method: {method}")
    
    @staticmethod
    def _compute_dtw_similarity(traj1: np.ndarray, traj2: np.ndarray) -> float:
        """Compute Dynamic Time Warping similarity."""
        # Distance matrix
        dist_matrix = cdist(traj1, traj2, metric='euclidean')
        
        n, m = dist_matrix.shape
        dtw_matrix = np.full((n + 1, m + 1), np.inf)
        dtw_matrix[0, 0] = 0
        
        # Dynamic programming
        for i in range(1, n + 1):
            for j in range(1, m + 1):
                cost = dist_matrix[i-1, j-1]
                dtw_matrix[i, j] = cost + min(
                    dtw_matrix[i-1, j],      # insertion
                    dtw_matrix[i, j-1],      # deletion
                    dtw_matrix[i-1, j-1]     # match
                )
        
        # Normalize and convert to similarity
        dtw_distance = dtw_matrix[n, m] / (n + m)
        max_distance = np.max(dist_matrix)
        similarity = max(0, 1 - dtw_distance / max_distance)
        
        return similarity
    
    @staticmethod
    def _compute_hausdorff_similarity(traj1: np.ndarray, traj2: np.ndarray) -> float:
        """Compute Hausdorff distance-based similarity."""
        dist_12 = cdist(traj1, traj2, metric='euclidean')
        dist_21 = cdist(traj2, traj1, metric='euclidean')
        
        hausdorff_12 = np.max(np.min(dist_12, axis=1))
        hausdorff_21 = np.max(np.min(dist_21, axis=1))
        hausdorff_distance = max(hausdorff_12, hausdorff_21)
        
        max_distance = np.max([np.max(traj1), np.max(traj2)]) * np.sqrt(traj1.shape[1])
        similarity = max(0, 1 - hausdorff_distance / max_distance)
        
        return similarity
    
    @staticmethod
    def _compute_frechet_similarity(traj1: np.ndarray, traj2: np.ndarray) -> float:
        """Compute discrete Frechet distance-based similarity."""
        n, m = len(traj1), len(traj2)
        dist_matrix = cdist(traj1, traj2, metric='euclidean')
        
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
        max_distance = np.max(dist_matrix)
        similarity = max(0, 1 - frechet_distance / max_distance)
        
        return similarity
    
    @staticmethod
    def visualize_joint_trajectory(joint_trajectory: JointTrajectory,
                                 joint_indices: Optional[List[int]] = None,
                                 save_path: Optional[str] = None) -> None:
        """
        Visualize joint trajectory.
        
        Args:
            joint_trajectory: Joint trajectory to visualize
            joint_indices: Specific joints to plot (None for all)
            save_path: Path to save plot
        """
        if joint_indices is None:
            joint_indices = list(range(joint_trajectory.positions.shape[1]))
        
        n_joints = len(joint_indices)
        fig, axes = plt.subplots(3, 1, figsize=(12, 10))
        
        # Plot positions
        for i, joint_idx in enumerate(joint_indices):
            joint_name = joint_trajectory.joint_names[joint_idx]
            axes[0].plot(joint_trajectory.timestamps, joint_trajectory.positions[:, joint_idx], 
                        label=joint_name, alpha=0.8)
        
        axes[0].set_ylabel('Position (rad)')
        axes[0].set_title('Joint Positions')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        # Plot velocities
        for i, joint_idx in enumerate(joint_indices):
            joint_name = joint_trajectory.joint_names[joint_idx]
            axes[1].plot(joint_trajectory.timestamps, joint_trajectory.velocities[:, joint_idx], 
                        label=joint_name, alpha=0.8)
        
        axes[1].set_ylabel('Velocity (rad/s)')
        axes[1].set_title('Joint Velocities')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
        
        # Plot accelerations
        for i, joint_idx in enumerate(joint_indices):
            joint_name = joint_trajectory.joint_names[joint_idx]
            axes[2].plot(joint_trajectory.timestamps, joint_trajectory.accelerations[:, joint_idx], 
                        label=joint_name, alpha=0.8)
        
        axes[2].set_xlabel('Time (s)')
        axes[2].set_ylabel('Acceleration (rad/s²)')
        axes[2].set_title('Joint Accelerations')
        axes[2].legend()
        axes[2].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Saved joint trajectory plot to {save_path}")
        else:
            plt.show()
        
        plt.close()
    
    @staticmethod
    def visualize_cartesian_trajectory(cartesian_trajectory: CartesianTrajectory,
                                     show_orientation: bool = False,
                                     save_path: Optional[str] = None) -> None:
        """
        Visualize Cartesian trajectory.
        
        Args:
            cartesian_trajectory: Cartesian trajectory to visualize
            show_orientation: Whether to show orientation arrows
            save_path: Path to save plot
        """
        fig = plt.figure(figsize=(15, 5))
        
        # 3D trajectory plot
        ax1 = fig.add_subplot(131, projection='3d')
        positions = cartesian_trajectory.positions
        
        ax1.plot(positions[:, 0], positions[:, 1], positions[:, 2], 
                'b-', linewidth=2, alpha=0.8, label='Trajectory')
        ax1.scatter(positions[0, 0], positions[0, 1], positions[0, 2], 
                   c='green', s=100, marker='o', label='Start')
        ax1.scatter(positions[-1, 0], positions[-1, 1], positions[-1, 2], 
                   c='red', s=100, marker='s', label='End')
        
        # Show orientation arrows
        if show_orientation and cartesian_trajectory.orientations is not None:
            step = max(1, len(positions) // 10)  # Show every 10th orientation
            for i in range(0, len(positions), step):
                pos = positions[i]
                orient = cartesian_trajectory.orientations[i]
                
                # Simple orientation visualization (just yaw)
                arrow_length = 0.05
                dx = arrow_length * np.cos(orient[2])  # yaw
                dy = arrow_length * np.sin(orient[2])
                
                ax1.quiver(pos[0], pos[1], pos[2], dx, dy, 0, 
                          color='red', alpha=0.6, arrow_length_ratio=0.1)
        
        ax1.set_xlabel('X (m)')
        ax1.set_ylabel('Y (m)')
        ax1.set_zlabel('Z (m)')
        ax1.set_title('3D Trajectory')
        ax1.legend()
        
        # Position vs time
        ax2 = fig.add_subplot(132)
        ax2.plot(cartesian_trajectory.timestamps, positions[:, 0], 'r-', label='X')
        ax2.plot(cartesian_trajectory.timestamps, positions[:, 1], 'g-', label='Y')
        ax2.plot(cartesian_trajectory.timestamps, positions[:, 2], 'b-', label='Z')
        ax2.set_xlabel('Time (s)')
        ax2.set_ylabel('Position (m)')
        ax2.set_title('Position vs Time')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # Velocity profile
        ax3 = fig.add_subplot(133)
        if cartesian_trajectory.velocities is not None:
            linear_vel = cartesian_trajectory.velocities[:, :3]
            speeds = np.linalg.norm(linear_vel, axis=1)
            ax3.plot(cartesian_trajectory.timestamps, speeds, 'k-', linewidth=2)
            ax3.set_xlabel('Time (s)')
            ax3.set_ylabel('Speed (m/s)')
            ax3.set_title('Speed Profile')
            ax3.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Saved Cartesian trajectory plot to {save_path}")
        else:
            plt.show()
        
        plt.close()
    
    @staticmethod
    def export_trajectory(trajectory: Union[JointTrajectory, CartesianTrajectory],
                         file_path: str, format: str = 'npz') -> None:
        """
        Export trajectory to file.
        
        Args:
            trajectory: Trajectory to export
            file_path: Output file path
            format: Export format ('npz', 'csv', 'json')
        """
        if format == 'npz':
            # NumPy compressed format
            if isinstance(trajectory, JointTrajectory):
                np.savez_compressed(file_path,
                                  positions=trajectory.positions,
                                  velocities=trajectory.velocities,
                                  accelerations=trajectory.accelerations,
                                  timestamps=trajectory.timestamps,
                                  joint_names=trajectory.joint_names,
                                  metadata=trajectory.metadata)
            else:  # CartesianTrajectory
                np.savez_compressed(file_path,
                                  positions=trajectory.positions,
                                  orientations=trajectory.orientations,
                                  velocities=trajectory.velocities,
                                  accelerations=trajectory.accelerations,
                                  timestamps=trajectory.timestamps,
                                  frame=trajectory.frame,
                                  metadata=trajectory.metadata)
        
        elif format == 'csv':
            # CSV format (positions and timestamps only)
            import pandas as pd
            
            if isinstance(trajectory, JointTrajectory):
                data = {'time': trajectory.timestamps}
                for i, joint_name in enumerate(trajectory.joint_names):
                    data[f'{joint_name}_pos'] = trajectory.positions[:, i]
                    data[f'{joint_name}_vel'] = trajectory.velocities[:, i]
            else:  # CartesianTrajectory
                data = {
                    'time': trajectory.timestamps,
                    'x': trajectory.positions[:, 0],
                    'y': trajectory.positions[:, 1],
                    'z': trajectory.positions[:, 2]
                }
                
                if trajectory.orientations is not None:
                    data.update({
                        'roll': trajectory.orientations[:, 0],
                        'pitch': trajectory.orientations[:, 1],
                        'yaw': trajectory.orientations[:, 2]
                    })
            
            df = pd.DataFrame(data)
            df.to_csv(file_path, index=False)
        
        else:
            raise ValueError(f"Unsupported export format: {format}")
        
        logger.info(f"Exported trajectory to {file_path}")
    
    @staticmethod
    def load_trajectory(file_path: str, trajectory_type: str = 'joint') -> Union[JointTrajectory, CartesianTrajectory]:
        """
        Load trajectory from file.
        
        Args:
            file_path: Input file path
            trajectory_type: Type of trajectory ('joint' or 'cartesian')
            
        Returns:
            trajectory: Loaded trajectory
        """
        if file_path.endswith('.npz'):
            data = np.load(file_path, allow_pickle=True)
            
            if trajectory_type == 'joint':
                return JointTrajectory(
                    positions=data['positions'],
                    velocities=data.get('velocities'),
                    accelerations=data.get('accelerations'),
                    timestamps=data.get('timestamps'),
                    joint_names=data.get('joint_names', []).tolist() if 'joint_names' in data else None,
                    metadata=data.get('metadata', {}).item() if 'metadata' in data else None
                )
            else:  # cartesian
                return CartesianTrajectory(
                    positions=data['positions'],
                    orientations=data.get('orientations'),
                    velocities=data.get('velocities'),
                    accelerations=data.get('accelerations'),
                    timestamps=data.get('timestamps'),
                    frame=data.get('frame', 'world').item() if 'frame' in data else 'world',
                    metadata=data.get('metadata', {}).item() if 'metadata' in data else None
                )
        
        else:
            raise ValueError(f"Unsupported file format: {file_path}")
    
    @staticmethod
    def compute_trajectory_metrics(trajectory: Union[JointTrajectory, CartesianTrajectory]) -> Dict[str, float]:
        """
        Compute comprehensive trajectory metrics.
        
        Args:
            trajectory: Trajectory to analyze
            
        Returns:
            metrics: Dictionary of trajectory metrics
        """
        if isinstance(trajectory, JointTrajectory):
            positions = trajectory.positions
            velocities = trajectory.velocities
            accelerations = trajectory.accelerations
        else:  # CartesianTrajectory
            positions = trajectory.positions
            velocities = trajectory.velocities[:, :3] if trajectory.velocities is not None else None
            accelerations = trajectory.accelerations[:, :3] if trajectory.accelerations is not None else None
        
        timestamps = trajectory.timestamps
        
        metrics = {}
        
        # Duration
        metrics['duration'] = timestamps[-1] - timestamps[0]
        
        # Length
        metrics['path_length'] = MotionPlanningUtils.compute_trajectory_length(positions)
        
        # Smoothness
        metrics['smoothness'] = MotionPlanningUtils.compute_trajectory_smoothness(positions, timestamps)
        
        # Velocity metrics
        if velocities is not None:
            speeds = np.linalg.norm(velocities, axis=1)
            metrics['max_speed'] = np.max(speeds)
            metrics['avg_speed'] = np.mean(speeds)
            metrics['speed_variation'] = np.std(speeds)
        
        # Acceleration metrics
        if accelerations is not None:
            acc_magnitudes = np.linalg.norm(accelerations, axis=1)
            metrics['max_acceleration'] = np.max(acc_magnitudes)
            metrics['avg_acceleration'] = np.mean(acc_magnitudes)
        
        # Efficiency (path length / straight line distance)
        if len(positions) > 1:
            straight_line_distance = np.linalg.norm(positions[-1] - positions[0])
            metrics['efficiency'] = straight_line_distance / metrics['path_length'] if metrics['path_length'] > 0 else 0
        
        return metrics