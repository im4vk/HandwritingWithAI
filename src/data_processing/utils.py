"""
Data Processing Utilities
=========================

General utilities for data processing operations.
"""

import numpy as np
from typing import Dict, Any, List, Tuple, Optional
from dataclasses import dataclass
import logging
import matplotlib.pyplot as plt

from .dataset_loader import HandwritingSample

logger = logging.getLogger(__name__)


@dataclass
class TrajectoryStats:
    """Statistics for trajectory analysis."""
    num_points: int
    duration: float
    total_length: float
    mean_velocity: float
    max_velocity: float
    mean_acceleration: float
    bounding_box: Tuple[float, float, float, float]  # x_min, y_min, x_max, y_max
    centroid: Tuple[float, float]


@dataclass
class DataValidation:
    """Data validation results."""
    is_valid: bool
    errors: List[str]
    warnings: List[str]
    num_samples: int
    num_valid_samples: int


class DataUtils:
    """Utility functions for data processing."""
    
    @staticmethod
    def compute_trajectory_stats(sample: HandwritingSample) -> TrajectoryStats:
        """Compute comprehensive statistics for a trajectory."""
        trajectory = sample.trajectory
        timestamps = sample.timestamps
        
        # Basic stats
        num_points = len(trajectory)
        duration = timestamps[-1] - timestamps[0] if timestamps is not None else num_points
        
        # Path length
        if num_points > 1:
            distances = np.linalg.norm(np.diff(trajectory, axis=0), axis=1)
            total_length = np.sum(distances)
        else:
            total_length = 0.0
        
        # Velocities
        if timestamps is not None and num_points > 1:
            dt = np.diff(timestamps)
            velocities = np.diff(trajectory, axis=0) / dt.reshape(-1, 1)
            speeds = np.linalg.norm(velocities, axis=1)
            mean_velocity = np.mean(speeds)
            max_velocity = np.max(speeds)
            
            # Accelerations
            if num_points > 2:
                accelerations = np.diff(velocities, axis=0) / dt[1:].reshape(-1, 1)
                acc_magnitudes = np.linalg.norm(accelerations, axis=1)
                mean_acceleration = np.mean(acc_magnitudes)
            else:
                mean_acceleration = 0.0
        else:
            mean_velocity = 0.0
            max_velocity = 0.0
            mean_acceleration = 0.0
        
        # Bounding box
        if num_points > 0:
            x_min, y_min = np.min(trajectory, axis=0)
            x_max, y_max = np.max(trajectory, axis=0)
            bounding_box = (x_min, y_min, x_max, y_max)
            
            # Centroid
            centroid = tuple(np.mean(trajectory, axis=0))
        else:
            bounding_box = (0, 0, 0, 0)
            centroid = (0, 0)
        
        return TrajectoryStats(
            num_points=num_points,
            duration=duration,
            total_length=total_length,
            mean_velocity=mean_velocity,
            max_velocity=max_velocity,
            mean_acceleration=mean_acceleration,
            bounding_box=bounding_box,
            centroid=centroid
        )
    
    @staticmethod
    def validate_dataset(samples: List[HandwritingSample]) -> DataValidation:
        """Validate dataset for common issues."""
        errors = []
        warnings = []
        num_valid_samples = 0
        
        for i, sample in enumerate(samples):
            sample_errors = []
            sample_warnings = []
            
            # Check trajectory
            if len(sample.trajectory) == 0:
                sample_errors.append(f"Sample {i}: Empty trajectory")
            elif len(sample.trajectory) < 3:
                sample_warnings.append(f"Sample {i}: Very short trajectory ({len(sample.trajectory)} points)")
            
            # Check for NaN/inf values
            if not np.all(np.isfinite(sample.trajectory)):
                sample_errors.append(f"Sample {i}: Non-finite values in trajectory")
            
            # Check text
            if not sample.text or len(sample.text.strip()) == 0:
                sample_warnings.append(f"Sample {i}: Empty or missing text")
            
            # Check timestamps consistency
            if sample.timestamps is not None:
                if len(sample.timestamps) != len(sample.trajectory):
                    sample_errors.append(f"Sample {i}: Timestamp length mismatch")
                elif not np.all(np.diff(sample.timestamps) >= 0):
                    sample_errors.append(f"Sample {i}: Non-monotonic timestamps")
            
            # Check pressure values
            if sample.pressure is not None:
                if len(sample.pressure) != len(sample.trajectory):
                    sample_errors.append(f"Sample {i}: Pressure length mismatch")
                elif np.any(sample.pressure < 0):
                    sample_warnings.append(f"Sample {i}: Negative pressure values")
            
            # Check pen states
            if sample.pen_states is not None:
                if len(sample.pen_states) != len(sample.trajectory):
                    sample_errors.append(f"Sample {i}: Pen states length mismatch")
                elif not np.all(np.isin(sample.pen_states, [0, 1])):
                    sample_warnings.append(f"Sample {i}: Pen states not binary")
            
            # Aggregate errors and warnings
            errors.extend(sample_errors)
            warnings.extend(sample_warnings)
            
            if not sample_errors:
                num_valid_samples += 1
        
        return DataValidation(
            is_valid=len(errors) == 0,
            errors=errors,
            warnings=warnings,
            num_samples=len(samples),
            num_valid_samples=num_valid_samples
        )
    
    @staticmethod
    def analyze_dataset(samples: List[HandwritingSample]) -> Dict[str, Any]:
        """Analyze dataset and provide comprehensive statistics."""
        if not samples:
            return {'error': 'Empty dataset'}
        
        # Collect statistics
        trajectory_lengths = []
        durations = []
        path_lengths = []
        writers = set()
        characters = set()
        
        for sample in samples:
            stats = DataUtils.compute_trajectory_stats(sample)
            trajectory_lengths.append(stats.num_points)
            durations.append(stats.duration)
            path_lengths.append(stats.total_length)
            writers.add(sample.writer_id)
            
            if sample.text:
                characters.update(sample.text)
        
        # Compute statistics
        analysis = {
            'dataset_size': len(samples),
            'num_writers': len(writers),
            'num_unique_characters': len(characters),
            'trajectory_stats': {
                'mean_length': np.mean(trajectory_lengths),
                'std_length': np.std(trajectory_lengths),
                'min_length': np.min(trajectory_lengths),
                'max_length': np.max(trajectory_lengths)
            },
            'duration_stats': {
                'mean_duration': np.mean(durations),
                'std_duration': np.std(durations),
                'min_duration': np.min(durations),
                'max_duration': np.max(durations),
                'total_duration': np.sum(durations)
            },
            'path_length_stats': {
                'mean_path_length': np.mean(path_lengths),
                'std_path_length': np.std(path_lengths),
                'min_path_length': np.min(path_lengths),
                'max_path_length': np.max(path_lengths)
            },
            'data_quality': DataUtils.validate_dataset(samples)
        }
        
        return analysis
    
    @staticmethod
    def visualize_sample(sample: HandwritingSample, save_path: Optional[str] = None) -> None:
        """Visualize a handwriting sample."""
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        # Trajectory plot
        ax1 = axes[0, 0]
        trajectory = sample.trajectory
        ax1.plot(trajectory[:, 0], trajectory[:, 1], 'b-', linewidth=2, alpha=0.7)
        ax1.scatter(trajectory[0, 0], trajectory[0, 1], c='green', s=100, marker='o', label='Start')
        ax1.scatter(trajectory[-1, 0], trajectory[-1, 1], c='red', s=100, marker='s', label='End')
        ax1.set_xlabel('X')
        ax1.set_ylabel('Y')
        ax1.set_title(f'Trajectory: "{sample.text}"')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        ax1.axis('equal')
        
        # Velocity profile
        ax2 = axes[0, 1]
        if sample.timestamps is not None:
            velocities = np.diff(trajectory, axis=0) / np.diff(sample.timestamps).reshape(-1, 1)
            speeds = np.linalg.norm(velocities, axis=1)
            ax2.plot(sample.timestamps[1:], speeds, 'g-', linewidth=2)
            ax2.set_xlabel('Time (s)')
            ax2.set_ylabel('Speed')
            ax2.set_title('Speed Profile')
        else:
            ax2.text(0.5, 0.5, 'No timestamp data', ha='center', va='center', transform=ax2.transAxes)
            ax2.set_title('Speed Profile (No Data)')
        ax2.grid(True, alpha=0.3)
        
        # Pressure profile
        ax3 = axes[1, 0]
        if sample.pressure is not None:
            if sample.timestamps is not None:
                ax3.plot(sample.timestamps, sample.pressure, 'r-', linewidth=2)
                ax3.set_xlabel('Time (s)')
            else:
                ax3.plot(sample.pressure, 'r-', linewidth=2)
                ax3.set_xlabel('Point Index')
            ax3.set_ylabel('Pressure')
            ax3.set_title('Pressure Profile')
        else:
            ax3.text(0.5, 0.5, 'No pressure data', ha='center', va='center', transform=ax3.transAxes)
            ax3.set_title('Pressure Profile (No Data)')
        ax3.grid(True, alpha=0.3)
        
        # Pen states
        ax4 = axes[1, 1]
        if sample.pen_states is not None:
            if sample.timestamps is not None:
                ax4.plot(sample.timestamps, sample.pen_states, 'k-', linewidth=2)
                ax4.set_xlabel('Time (s)')
            else:
                ax4.plot(sample.pen_states, 'k-', linewidth=2)
                ax4.set_xlabel('Point Index')
            ax4.set_ylabel('Pen State')
            ax4.set_title('Pen Up/Down')
            ax4.set_ylim(-0.1, 1.1)
        else:
            ax4.text(0.5, 0.5, 'No pen state data', ha='center', va='center', transform=ax4.transAxes)
            ax4.set_title('Pen States (No Data)')
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Saved sample visualization to {save_path}")
        else:
            plt.show()
        
        plt.close()
    
    @staticmethod
    def compare_samples(sample1: HandwritingSample, sample2: HandwritingSample) -> Dict[str, float]:
        """Compare two handwriting samples."""
        # Basic trajectory comparison
        traj1, traj2 = sample1.trajectory, sample2.trajectory
        
        # DTW distance (simplified)
        from scipy.spatial.distance import cdist
        dist_matrix = cdist(traj1, traj2)
        
        # Simple path comparison
        if len(traj1) == len(traj2):
            position_similarity = 1.0 / (1.0 + np.mean(np.linalg.norm(traj1 - traj2, axis=1)))
        else:
            position_similarity = 0.0
        
        # Text similarity
        text_similarity = 1.0 if sample1.text == sample2.text else 0.0
        
        # Writer similarity
        writer_similarity = 1.0 if sample1.writer_id == sample2.writer_id else 0.0
        
        # Duration comparison
        stats1 = DataUtils.compute_trajectory_stats(sample1)
        stats2 = DataUtils.compute_trajectory_stats(sample2)
        
        duration_similarity = 1.0 / (1.0 + abs(stats1.duration - stats2.duration))
        length_similarity = 1.0 / (1.0 + abs(stats1.total_length - stats2.total_length))
        
        return {
            'position_similarity': position_similarity,
            'text_similarity': text_similarity,
            'writer_similarity': writer_similarity,
            'duration_similarity': duration_similarity,
            'length_similarity': length_similarity,
            'overall_similarity': np.mean([
                position_similarity, text_similarity, duration_similarity, length_similarity
            ])
        }
    
    @staticmethod
    def filter_samples_by_quality(samples: List[HandwritingSample],
                                min_length: int = 5,
                                max_length: int = 1000,
                                min_duration: float = 0.1,
                                max_velocity: float = 10.0) -> List[HandwritingSample]:
        """Filter samples based on quality criteria."""
        filtered_samples = []
        
        for sample in samples:
            stats = DataUtils.compute_trajectory_stats(sample)
            
            # Quality checks
            if (min_length <= stats.num_points <= max_length and
                stats.duration >= min_duration and
                stats.max_velocity <= max_velocity and
                np.all(np.isfinite(sample.trajectory))):
                
                filtered_samples.append(sample)
        
        logger.info(f"Filtered {len(samples)} samples to {len(filtered_samples)} based on quality")
        return filtered_samples