"""
Handwriting Data Preprocessing Module
===================================

Data preprocessing utilities for handwriting datasets.
"""

import numpy as np
from typing import List, Dict, Any, Tuple, Optional
import json
from pathlib import Path

class HandwritingPreprocessor:
    """
    Preprocessor for handwriting data with filtering, normalization,
    and feature extraction capabilities.
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        """
        Initialize preprocessor.
        
        Args:
            config: Preprocessing configuration
        """
        self.config = config or {}
        self.sampling_rate = self.config.get('sampling_rate', 100.0)
        self.filter_cutoff = self.config.get('filter_cutoff', 10.0)
        self.normalize_position = self.config.get('normalize_position', True)
        self.normalize_velocity = self.config.get('normalize_velocity', True)
    
    def preprocess_trajectory(self, trajectory: List[List[float]]) -> Dict[str, np.ndarray]:
        """
        Preprocess a single trajectory.
        
        Args:
            trajectory: Raw trajectory points [[x, y, z], ...]
            
        Returns:
            Preprocessed data with positions, velocities, accelerations
        """
        trajectory = np.array(trajectory)
        
        # Apply smoothing filter
        smoothed = self._apply_smoothing_filter(trajectory)
        
        # Calculate velocities
        velocities = self._calculate_velocities(smoothed)
        
        # Calculate accelerations
        accelerations = self._calculate_accelerations(velocities)
        
        # Normalize if requested
        if self.normalize_position:
            smoothed = self._normalize_positions(smoothed)
        
        if self.normalize_velocity:
            velocities = self._normalize_velocities(velocities)
        
        return {
            'positions': smoothed,
            'velocities': velocities,
            'accelerations': accelerations,
            'timestamps': np.arange(len(smoothed)) / self.sampling_rate
        }
    
    def _apply_smoothing_filter(self, trajectory: np.ndarray) -> np.ndarray:
        """Apply smoothing filter to trajectory."""
        # Simple moving average filter
        window_size = max(1, int(self.sampling_rate / self.filter_cutoff))
        
        if window_size <= 1:
            return trajectory
        
        smoothed = trajectory.copy()
        for i in range(len(trajectory)):
            start_idx = max(0, i - window_size // 2)
            end_idx = min(len(trajectory), i + window_size // 2 + 1)
            smoothed[i] = np.mean(trajectory[start_idx:end_idx], axis=0)
        
        return smoothed
    
    def _calculate_velocities(self, positions: np.ndarray) -> np.ndarray:
        """Calculate velocities from positions."""
        velocities = np.zeros_like(positions)
        dt = 1.0 / self.sampling_rate
        
        for i in range(1, len(positions)):
            velocities[i] = (positions[i] - positions[i-1]) / dt
        
        return velocities
    
    def _calculate_accelerations(self, velocities: np.ndarray) -> np.ndarray:
        """Calculate accelerations from velocities."""
        accelerations = np.zeros_like(velocities)
        dt = 1.0 / self.sampling_rate
        
        for i in range(1, len(velocities)):
            accelerations[i] = (velocities[i] - velocities[i-1]) / dt
        
        return accelerations
    
    def _normalize_positions(self, positions: np.ndarray) -> np.ndarray:
        """Normalize positions to [0, 1] range."""
        normalized = positions.copy()
        
        for dim in range(positions.shape[1]):
            min_val = np.min(positions[:, dim])
            max_val = np.max(positions[:, dim])
            
            if max_val > min_val:
                normalized[:, dim] = (positions[:, dim] - min_val) / (max_val - min_val)
        
        return normalized
    
    def _normalize_velocities(self, velocities: np.ndarray) -> np.ndarray:
        """Normalize velocities by maximum magnitude."""
        magnitudes = np.linalg.norm(velocities, axis=1)
        max_magnitude = np.max(magnitudes)
        
        if max_magnitude > 0:
            return velocities / max_magnitude
        else:
            return velocities
    
    def load_and_preprocess_dataset(self, dataset_path: str) -> List[Dict[str, Any]]:
        """
        Load and preprocess an entire dataset.
        
        Args:
            dataset_path: Path to dataset JSON file
            
        Returns:
            List of preprocessed samples
        """
        with open(dataset_path, 'r') as f:
            samples = json.load(f)
        
        preprocessed_samples = []
        
        for sample in samples:
            trajectory = sample['trajectory']
            preprocessed_data = self.preprocess_trajectory(trajectory)
            
            # Add preprocessed data to sample
            sample['preprocessed'] = preprocessed_data
            preprocessed_samples.append(sample)
        
        return preprocessed_samples
    
    def extract_features(self, trajectory: np.ndarray) -> Dict[str, float]:
        """
        Extract features from trajectory.
        
        Args:
            trajectory: Trajectory points
            
        Returns:
            Dictionary of extracted features
        """
        if len(trajectory) < 2:
            return {}
        
        # Basic features
        total_distance = np.sum(np.linalg.norm(np.diff(trajectory, axis=0), axis=1))
        total_time = len(trajectory) / self.sampling_rate
        avg_velocity = total_distance / total_time if total_time > 0 else 0
        
        # Velocity features
        velocities = self._calculate_velocities(trajectory)
        velocity_magnitudes = np.linalg.norm(velocities, axis=1)
        max_velocity = np.max(velocity_magnitudes)
        velocity_variance = np.var(velocity_magnitudes)
        
        # Acceleration features
        accelerations = self._calculate_accelerations(velocities)
        acceleration_magnitudes = np.linalg.norm(accelerations, axis=1)
        max_acceleration = np.max(acceleration_magnitudes)
        
        # Jerk features (rate of change of acceleration)
        jerks = self._calculate_accelerations(accelerations)
        jerk_magnitudes = np.linalg.norm(jerks, axis=1)
        avg_jerk = np.mean(jerk_magnitudes)
        
        return {
            'total_distance': total_distance,
            'total_time': total_time,
            'avg_velocity': avg_velocity,
            'max_velocity': max_velocity,
            'velocity_variance': velocity_variance,
            'max_acceleration': max_acceleration,
            'avg_jerk': avg_jerk,
            'num_points': len(trajectory),
            'smoothness': 1.0 / (1.0 + avg_jerk) if avg_jerk > 0 else 1.0
        }
