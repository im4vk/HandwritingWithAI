"""
Data Augmentation for Handwriting
=================================

Data augmentation techniques to increase dataset diversity
and improve model generalization.
"""

import numpy as np
from typing import Dict, Any, List, Tuple, Optional
from dataclasses import dataclass
from enum import Enum
import logging

from .dataset_loader import HandwritingSample

logger = logging.getLogger(__name__)


class AugmentationMethod(Enum):
    """Available augmentation methods."""
    ROTATION = "rotation"
    SCALING = "scaling"
    TRANSLATION = "translation"
    SHEARING = "shearing"
    NOISE_ADDITION = "noise_addition"
    TIME_WARPING = "time_warping"
    VELOCITY_SCALING = "velocity_scaling"
    ELASTIC_DEFORMATION = "elastic_deformation"


@dataclass
class AugmentationParams:
    """Parameters for data augmentation."""
    rotation_range: Tuple[float, float] = (-15.0, 15.0)  # degrees
    scaling_range: Tuple[float, float] = (0.8, 1.2)
    translation_range: Tuple[float, float] = (-0.1, 0.1)  # relative to trajectory size
    shear_range: Tuple[float, float] = (-0.1, 0.1)
    noise_std: float = 0.01
    time_warp_strength: float = 0.1
    velocity_scale_range: Tuple[float, float] = (0.8, 1.2)
    elastic_strength: float = 0.05


class DataAugmenter:
    """Data augmenter for handwriting trajectories."""
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize data augmenter."""
        self.config = config
        self.params = AugmentationParams(**config.get('augmentation_params', {}))
        self.enabled_methods = config.get('enabled_methods', [
            AugmentationMethod.ROTATION,
            AugmentationMethod.SCALING,
            AugmentationMethod.NOISE_ADDITION
        ])
        
        logger.info("Initialized DataAugmenter")
    
    def augment_sample(self, sample: HandwritingSample, 
                      methods: Optional[List[AugmentationMethod]] = None) -> HandwritingSample:
        """Augment a single sample."""
        if methods is None:
            methods = self.enabled_methods
        
        # Create copy
        augmented_sample = HandwritingSample(
            trajectory=sample.trajectory.copy(),
            text=sample.text,
            writer_id=sample.writer_id,
            character_labels=sample.character_labels,
            timestamps=sample.timestamps.copy() if sample.timestamps is not None else None,
            pressure=sample.pressure.copy() if sample.pressure is not None else None,
            pen_states=sample.pen_states.copy() if sample.pen_states is not None else None,
            metadata=sample.metadata.copy() if sample.metadata else {}
        )
        
        # Apply augmentations
        for method in methods:
            if method == AugmentationMethod.ROTATION:
                augmented_sample = self._apply_rotation(augmented_sample)
            elif method == AugmentationMethod.SCALING:
                augmented_sample = self._apply_scaling(augmented_sample)
            elif method == AugmentationMethod.TRANSLATION:
                augmented_sample = self._apply_translation(augmented_sample)
            elif method == AugmentationMethod.SHEARING:
                augmented_sample = self._apply_shearing(augmented_sample)
            elif method == AugmentationMethod.NOISE_ADDITION:
                augmented_sample = self._apply_noise(augmented_sample)
            elif method == AugmentationMethod.TIME_WARPING:
                augmented_sample = self._apply_time_warping(augmented_sample)
            elif method == AugmentationMethod.VELOCITY_SCALING:
                augmented_sample = self._apply_velocity_scaling(augmented_sample)
            elif method == AugmentationMethod.ELASTIC_DEFORMATION:
                augmented_sample = self._apply_elastic_deformation(augmented_sample)
        
        # Mark as augmented
        if augmented_sample.metadata is None:
            augmented_sample.metadata = {}
        augmented_sample.metadata['augmented'] = True
        augmented_sample.metadata['augmentation_methods'] = [m.value for m in methods]
        
        return augmented_sample
    
    def _apply_rotation(self, sample: HandwritingSample) -> HandwritingSample:
        """Apply rotation augmentation."""
        angle = np.random.uniform(*self.params.rotation_range)
        angle_rad = np.radians(angle)
        
        # Rotation matrix
        cos_a, sin_a = np.cos(angle_rad), np.sin(angle_rad)
        rotation_matrix = np.array([[cos_a, -sin_a], [sin_a, cos_a]])
        
        # Apply rotation around centroid
        centroid = np.mean(sample.trajectory, axis=0)
        centered_traj = sample.trajectory - centroid
        rotated_traj = centered_traj @ rotation_matrix.T
        sample.trajectory = rotated_traj + centroid
        
        return sample
    
    def _apply_scaling(self, sample: HandwritingSample) -> HandwritingSample:
        """Apply scaling augmentation."""
        scale = np.random.uniform(*self.params.scaling_range)
        
        # Scale around centroid
        centroid = np.mean(sample.trajectory, axis=0)
        centered_traj = sample.trajectory - centroid
        scaled_traj = centered_traj * scale
        sample.trajectory = scaled_traj + centroid
        
        # Scale velocities if timestamps available
        if sample.timestamps is not None:
            velocities = np.diff(sample.trajectory, axis=0) / np.diff(sample.timestamps).reshape(-1, 1)
            # Velocities scale with the spatial scaling
            
        return sample
    
    def _apply_translation(self, sample: HandwritingSample) -> HandwritingSample:
        """Apply translation augmentation."""
        # Compute trajectory bounding box
        bbox_size = np.max(sample.trajectory, axis=0) - np.min(sample.trajectory, axis=0)
        
        # Translation relative to trajectory size
        translation = np.random.uniform(*self.params.translation_range, size=2) * bbox_size
        sample.trajectory += translation
        
        return sample
    
    def _apply_shearing(self, sample: HandwritingSample) -> HandwritingSample:
        """Apply shearing augmentation."""
        shear_x = np.random.uniform(*self.params.shear_range)
        shear_y = np.random.uniform(*self.params.shear_range)
        
        shear_matrix = np.array([[1, shear_x], [shear_y, 1]])
        
        centroid = np.mean(sample.trajectory, axis=0)
        centered_traj = sample.trajectory - centroid
        sheared_traj = centered_traj @ shear_matrix.T
        sample.trajectory = sheared_traj + centroid
        
        return sample
    
    def _apply_noise(self, sample: HandwritingSample) -> HandwritingSample:
        """Apply noise augmentation."""
        noise = np.random.normal(0, self.params.noise_std, sample.trajectory.shape)
        sample.trajectory += noise
        
        # Add noise to pressure if available
        if sample.pressure is not None:
            pressure_noise = np.random.normal(0, self.params.noise_std * 0.1, sample.pressure.shape)
            sample.pressure = np.clip(sample.pressure + pressure_noise, 0, None)
        
        return sample
    
    def _apply_time_warping(self, sample: HandwritingSample) -> HandwritingSample:
        """Apply time warping augmentation."""
        if sample.timestamps is None:
            return sample
        
        n_points = len(sample.trajectory)
        if n_points < 5:
            return sample
        
        # Create warping function
        warp_points = np.sort(np.random.choice(n_points, size=max(2, n_points//10), replace=False))
        warp_factors = 1 + np.random.uniform(-self.params.time_warp_strength, 
                                           self.params.time_warp_strength, len(warp_points))
        
        # Apply warping to timestamps
        original_times = sample.timestamps.copy()
        duration = original_times[-1] - original_times[0]
        
        # Simple linear warping
        warped_times = original_times.copy()
        for i, (point_idx, factor) in enumerate(zip(warp_points, warp_factors)):
            # Warp local section
            if i < len(warp_points) - 1:
                next_idx = warp_points[i + 1]
            else:
                next_idx = n_points
            
            local_duration = original_times[next_idx - 1] - original_times[point_idx]
            warped_duration = local_duration * factor
            
            # Update timestamps in this section
            section_times = np.linspace(
                warped_times[point_idx],
                warped_times[point_idx] + warped_duration,
                next_idx - point_idx
            )
            warped_times[point_idx:next_idx] = section_times
        
        sample.timestamps = warped_times
        
        return sample
    
    def _apply_velocity_scaling(self, sample: HandwritingSample) -> HandwritingSample:
        """Apply velocity scaling augmentation."""
        if sample.timestamps is None:
            return sample
        
        scale_factor = np.random.uniform(*self.params.velocity_scale_range)
        
        # Scale timestamps to change velocity
        duration = sample.timestamps[-1] - sample.timestamps[0]
        new_duration = duration / scale_factor
        
        # Rescale timestamps
        normalized_times = (sample.timestamps - sample.timestamps[0]) / duration
        sample.timestamps = sample.timestamps[0] + normalized_times * new_duration
        
        return sample
    
    def _apply_elastic_deformation(self, sample: HandwritingSample) -> HandwritingSample:
        """Apply elastic deformation augmentation."""
        n_points = len(sample.trajectory)
        if n_points < 10:
            return sample
        
        # Create smooth random displacement field
        grid_size = max(5, n_points // 10)
        
        # Generate random displacement at grid points
        displacement_grid = np.random.normal(0, self.params.elastic_strength, (grid_size, 2))
        
        # Interpolate displacements to trajectory points
        grid_indices = np.linspace(0, n_points - 1, grid_size).astype(int)
        
        displacements = np.zeros((n_points, 2))
        for i in range(2):  # x, y components
            displacements[:, i] = np.interp(
                np.arange(n_points), 
                grid_indices, 
                displacement_grid[:, i]
            )
        
        # Apply elastic deformation
        sample.trajectory += displacements
        
        return sample
    
    def augment_dataset(self, samples: List[HandwritingSample], 
                       augmentation_factor: int = 2) -> List[HandwritingSample]:
        """
        Augment entire dataset.
        
        Args:
            samples: Original samples
            augmentation_factor: Number of augmented versions per sample
            
        Returns:
            augmented_samples: Original + augmented samples
        """
        augmented_samples = samples.copy()
        
        for sample in samples:
            for _ in range(augmentation_factor - 1):  # -1 because original is included
                # Randomly select augmentation methods
                num_methods = np.random.randint(1, min(4, len(self.enabled_methods) + 1))
                selected_methods = np.random.choice(
                    self.enabled_methods, size=num_methods, replace=False
                ).tolist()
                
                augmented_sample = self.augment_sample(sample, selected_methods)
                augmented_samples.append(augmented_sample)
        
        logger.info(f"Augmented dataset from {len(samples)} to {len(augmented_samples)} samples")
        return augmented_samples