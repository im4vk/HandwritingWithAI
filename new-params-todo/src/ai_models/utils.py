"""
AI Models Utilities
==================

Utility functions and classes for training, evaluation, and deployment
of AI models in the robotic handwriting system.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from typing import Dict, Any, List, Tuple, Optional, Callable, Union
import numpy as np
import logging
import time
import os
import json
import pickle
from collections import defaultdict, deque
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

logger = logging.getLogger(__name__)


class ModelUtils:
    """
    General utilities for model management, training, and evaluation.
    """
    
    @staticmethod
    def count_parameters(model: nn.Module, trainable_only: bool = True) -> int:
        """
        Count the number of parameters in a model.
        
        Args:
            model: PyTorch model
            trainable_only: If True, count only trainable parameters
            
        Returns:
            num_parameters: Number of parameters
        """
        if trainable_only:
            return sum(p.numel() for p in model.parameters() if p.requires_grad)
        else:
            return sum(p.numel() for p in model.parameters())
    
    @staticmethod
    def get_model_size_mb(model: nn.Module) -> float:
        """
        Get model size in megabytes.
        
        Args:
            model: PyTorch model
            
        Returns:
            size_mb: Model size in MB
        """
        param_size = 0
        buffer_size = 0
        
        for param in model.parameters():
            param_size += param.nelement() * param.element_size()
        
        for buffer in model.buffers():
            buffer_size += buffer.nelement() * buffer.element_size()
        
        size_bytes = param_size + buffer_size
        size_mb = size_bytes / (1024 ** 2)
        
        return size_mb
    
    @staticmethod
    def initialize_weights(model: nn.Module, method: str = 'xavier') -> None:
        """
        Initialize model weights.
        
        Args:
            model: PyTorch model
            method: Initialization method ('xavier', 'kaiming', 'normal')
        """
        for module in model.modules():
            if isinstance(module, (nn.Linear, nn.Conv2d)):
                if method == 'xavier':
                    nn.init.xavier_uniform_(module.weight)
                elif method == 'kaiming':
                    nn.init.kaiming_uniform_(module.weight, nonlinearity='relu')
                elif method == 'normal':
                    nn.init.normal_(module.weight, 0, 0.02)
                
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            
            elif isinstance(module, (nn.LSTM, nn.GRU)):
                for name, param in module.named_parameters():
                    if 'weight' in name:
                        nn.init.xavier_uniform_(param)
                    elif 'bias' in name:
                        nn.init.zeros_(param)
    
    @staticmethod
    def freeze_parameters(model: nn.Module, layer_names: Optional[List[str]] = None) -> None:
        """
        Freeze model parameters.
        
        Args:
            model: PyTorch model
            layer_names: Specific layer names to freeze (if None, freeze all)
        """
        if layer_names is None:
            # Freeze all parameters
            for param in model.parameters():
                param.requires_grad = False
        else:
            # Freeze specific layers
            for name, param in model.named_parameters():
                if any(layer_name in name for layer_name in layer_names):
                    param.requires_grad = False
        
        logger.info(f"Frozen parameters in model")
    
    @staticmethod
    def unfreeze_parameters(model: nn.Module, layer_names: Optional[List[str]] = None) -> None:
        """
        Unfreeze model parameters.
        
        Args:
            model: PyTorch model
            layer_names: Specific layer names to unfreeze (if None, unfreeze all)
        """
        if layer_names is None:
            # Unfreeze all parameters
            for param in model.parameters():
                param.requires_grad = True
        else:
            # Unfreeze specific layers
            for name, param in model.named_parameters():
                if any(layer_name in name for layer_name in layer_names):
                    param.requires_grad = True
        
        logger.info(f"Unfrozen parameters in model")
    
    @staticmethod
    def get_learning_rate(optimizer: optim.Optimizer) -> float:
        """Get current learning rate from optimizer."""
        return optimizer.param_groups[0]['lr']
    
    @staticmethod
    def set_learning_rate(optimizer: optim.Optimizer, lr: float) -> None:
        """Set learning rate for optimizer."""
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr


class TrainingUtils:
    """
    Utilities for model training, validation, and monitoring.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize training utilities.
        
        Args:
            config: Training configuration
        """
        self.config = config
        self.metrics_history = defaultdict(list)
        self.best_metrics = {}
        self.patience_counter = 0
        self.best_epoch = 0
        
        # Early stopping
        self.early_stopping_enabled = config.get('early_stopping', True)
        self.patience = config.get('patience', 50)
        self.min_delta = config.get('min_delta', 1e-6)
        
        # Gradient monitoring
        self.monitor_gradients = config.get('monitor_gradients', False)
        self.gradient_history = deque(maxlen=1000)
        
        logger.info("Initialized TrainingUtils")
    
    def update_metrics(self, metrics: Dict[str, float], epoch: int) -> None:
        """
        Update training metrics.
        
        Args:
            metrics: Dictionary of metric values
            epoch: Current epoch number
        """
        for metric_name, metric_value in metrics.items():
            self.metrics_history[metric_name].append(metric_value)
        
        # Update best metrics
        if 'val_loss' in metrics:
            if 'val_loss' not in self.best_metrics or metrics['val_loss'] < self.best_metrics['val_loss']:
                self.best_metrics.update(metrics)
                self.best_epoch = epoch
                self.patience_counter = 0
            else:
                self.patience_counter += 1
    
    def should_early_stop(self) -> bool:
        """Check if training should stop early."""
        if not self.early_stopping_enabled:
            return False
        
        return self.patience_counter >= self.patience
    
    def monitor_model_gradients(self, model: nn.Module) -> Dict[str, float]:
        """
        Monitor gradient statistics.
        
        Args:
            model: PyTorch model
            
        Returns:
            gradient_stats: Gradient statistics
        """
        total_norm = 0.0
        param_count = 0
        max_grad = 0.0
        min_grad = float('inf')
        
        for param in model.parameters():
            if param.grad is not None:
                param_norm = param.grad.data.norm(2)
                total_norm += param_norm.item() ** 2
                param_count += 1
                
                max_grad = max(max_grad, param.grad.data.abs().max().item())
                min_grad = min(min_grad, param.grad.data.abs().min().item())
        
        total_norm = total_norm ** (1.0 / 2)
        avg_grad = total_norm / max(param_count, 1)
        
        gradient_stats = {
            'grad_norm': total_norm,
            'avg_grad': avg_grad,
            'max_grad': max_grad,
            'min_grad': min_grad if min_grad != float('inf') else 0.0
        }
        
        if self.monitor_gradients:
            self.gradient_history.append(gradient_stats)
        
        return gradient_stats
    
    def get_training_summary(self) -> Dict[str, Any]:
        """Get comprehensive training summary."""
        summary = {
            'best_epoch': self.best_epoch,
            'best_metrics': self.best_metrics,
            'total_epochs': len(self.metrics_history.get('train_loss', [])),
            'patience_counter': self.patience_counter,
            'metrics_history': dict(self.metrics_history)
        }
        
        if self.monitor_gradients and self.gradient_history:
            summary['gradient_stats'] = {
                'final_grad_norm': self.gradient_history[-1]['grad_norm'],
                'avg_grad_norm': np.mean([g['grad_norm'] for g in self.gradient_history]),
                'max_grad_norm': np.max([g['grad_norm'] for g in self.gradient_history])
            }
        
        return summary
    
    def plot_training_curves(self, save_path: Optional[str] = None) -> None:
        """
        Plot training curves.
        
        Args:
            save_path: Path to save the plot
        """
        if not self.metrics_history:
            logger.warning("No metrics to plot")
            return
        
        # Create subplots
        n_metrics = len(self.metrics_history)
        n_cols = min(3, n_metrics)
        n_rows = (n_metrics + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 5 * n_rows))
        if n_metrics == 1:
            axes = [axes]
        elif n_rows == 1:
            axes = axes.reshape(1, -1)
        
        # Plot each metric
        for idx, (metric_name, values) in enumerate(self.metrics_history.items()):
            row = idx // n_cols
            col = idx % n_cols
            ax = axes[row, col] if n_rows > 1 else axes[col]
            
            ax.plot(values, label=metric_name)
            ax.set_title(f'{metric_name.title()}')
            ax.set_xlabel('Epoch')
            ax.set_ylabel('Value')
            ax.grid(True, alpha=0.3)
            ax.legend()
            
            # Mark best epoch for validation metrics
            if 'val' in metric_name and self.best_epoch < len(values):
                ax.axvline(x=self.best_epoch, color='red', linestyle='--', alpha=0.7, label='Best Epoch')
        
        # Remove empty subplots
        for idx in range(n_metrics, n_rows * n_cols):
            row = idx // n_cols
            col = idx % n_cols
            fig.delaxes(axes[row, col] if n_rows > 1 else axes[col])
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Saved training curves to {save_path}")
        else:
            plt.show()
        
        plt.close()


class DataUtils:
    """
    Utilities for data preprocessing, augmentation, and handling.
    """
    
    @staticmethod
    def normalize_trajectories(trajectories: np.ndarray, method: str = 'standard') -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Normalize trajectory data.
        
        Args:
            trajectories: Trajectory data [N, seq_len, features]
            method: Normalization method ('standard', 'minmax', 'robust')
            
        Returns:
            normalized_trajectories: Normalized data
            normalization_params: Parameters for denormalization
        """
        if method == 'standard':
            mean = np.mean(trajectories, axis=(0, 1), keepdims=True)
            std = np.std(trajectories, axis=(0, 1), keepdims=True)
            std = np.where(std == 0, 1, std)  # Avoid division by zero
            
            normalized = (trajectories - mean) / std
            params = {'mean': mean, 'std': std, 'method': 'standard'}
        
        elif method == 'minmax':
            min_vals = np.min(trajectories, axis=(0, 1), keepdims=True)
            max_vals = np.max(trajectories, axis=(0, 1), keepdims=True)
            range_vals = max_vals - min_vals
            range_vals = np.where(range_vals == 0, 1, range_vals)
            
            normalized = (trajectories - min_vals) / range_vals
            params = {'min': min_vals, 'max': max_vals, 'range': range_vals, 'method': 'minmax'}
        
        elif method == 'robust':
            median = np.median(trajectories, axis=(0, 1), keepdims=True)
            mad = np.median(np.abs(trajectories - median), axis=(0, 1), keepdims=True)
            mad = np.where(mad == 0, 1, mad)
            
            normalized = (trajectories - median) / (1.4826 * mad)  # 1.4826 makes MAD consistent with std
            params = {'median': median, 'mad': mad, 'method': 'robust'}
        
        else:
            raise ValueError(f"Unsupported normalization method: {method}")
        
        return normalized, params
    
    @staticmethod
    def denormalize_trajectories(normalized_trajectories: np.ndarray, params: Dict[str, Any]) -> np.ndarray:
        """
        Denormalize trajectory data.
        
        Args:
            normalized_trajectories: Normalized trajectory data
            params: Normalization parameters
            
        Returns:
            denormalized_trajectories: Original scale data
        """
        method = params['method']
        
        if method == 'standard':
            return normalized_trajectories * params['std'] + params['mean']
        elif method == 'minmax':
            return normalized_trajectories * params['range'] + params['min']
        elif method == 'robust':
            return normalized_trajectories * (1.4826 * params['mad']) + params['median']
        else:
            raise ValueError(f"Unsupported denormalization method: {method}")
    
    @staticmethod
    def augment_trajectory(trajectory: np.ndarray, 
                          rotation_range: float = 5.0,
                          scale_range: Tuple[float, float] = (0.9, 1.1),
                          noise_level: float = 0.01,
                          time_warp_strength: float = 0.1) -> np.ndarray:
        """
        Apply data augmentation to trajectory.
        
        Args:
            trajectory: Input trajectory [seq_len, features]
            rotation_range: Rotation range in degrees
            scale_range: Scale factor range
            noise_level: Gaussian noise standard deviation
            time_warp_strength: Time warping strength
            
        Returns:
            augmented_trajectory: Augmented trajectory
        """
        augmented = trajectory.copy()
        seq_len, n_features = trajectory.shape
        
        # Assume first 2 features are x, y coordinates
        if n_features >= 2:
            positions = augmented[:, :2]
            
            # Rotation
            angle = np.random.uniform(-rotation_range, rotation_range) * np.pi / 180
            cos_a, sin_a = np.cos(angle), np.sin(angle)
            rotation_matrix = np.array([[cos_a, -sin_a], [sin_a, cos_a]])
            
            rotated_positions = positions @ rotation_matrix.T
            augmented[:, :2] = rotated_positions
            
            # Scaling
            scale_factor = np.random.uniform(scale_range[0], scale_range[1])
            augmented[:, :2] *= scale_factor
            
            # If velocities are present (features 2, 3), scale them too
            if n_features >= 4:
                augmented[:, 2:4] *= scale_factor
        
        # Add noise
        noise = np.random.normal(0, noise_level, augmented.shape)
        augmented += noise
        
        # Time warping (if enabled)
        if time_warp_strength > 0:
            augmented = DataUtils._apply_time_warp(augmented, time_warp_strength)
        
        return augmented
    
    @staticmethod
    def _apply_time_warp(trajectory: np.ndarray, strength: float) -> np.ndarray:
        """Apply time warping to trajectory."""
        seq_len = trajectory.shape[0]
        
        # Create warping function
        original_times = np.linspace(0, 1, seq_len)
        
        # Random warping points
        n_warp_points = max(2, seq_len // 10)
        warp_indices = np.sort(np.random.choice(seq_len, n_warp_points, replace=False))
        warp_offsets = np.random.uniform(-strength, strength, n_warp_points)
        
        # Interpolate warping function
        warp_times = original_times[warp_indices] + warp_offsets
        warp_times = np.clip(warp_times, 0, 1)
        warp_times = np.sort(warp_times)  # Ensure monotonic
        
        # Interpolate trajectory to warped times
        warped_trajectory = np.zeros_like(trajectory)
        for i in range(trajectory.shape[1]):
            warped_trajectory[:, i] = np.interp(original_times, warp_times, trajectory[warp_indices, i])
        
        return warped_trajectory
    
    @staticmethod
    def create_sliding_windows(data: np.ndarray, 
                             window_size: int, 
                             stride: int = 1,
                             pad_mode: str = 'constant') -> np.ndarray:
        """
        Create sliding windows from sequential data.
        
        Args:
            data: Input data [seq_len, features]
            window_size: Size of each window
            stride: Stride between windows
            pad_mode: Padding mode for insufficient data
            
        Returns:
            windows: Sliding windows [num_windows, window_size, features]
        """
        seq_len, n_features = data.shape
        
        if seq_len < window_size:
            # Pad data if too short
            if pad_mode == 'constant':
                padding = np.zeros((window_size - seq_len, n_features))
            elif pad_mode == 'reflect':
                padding = np.flip(data, axis=0)[:window_size - seq_len]
            else:
                raise ValueError(f"Unsupported pad_mode: {pad_mode}")
            
            data = np.concatenate([data, padding], axis=0)
            seq_len = data.shape[0]
        
        # Create windows
        num_windows = (seq_len - window_size) // stride + 1
        windows = np.zeros((num_windows, window_size, n_features))
        
        for i in range(num_windows):
            start_idx = i * stride
            end_idx = start_idx + window_size
            windows[i] = data[start_idx:end_idx]
        
        return windows


class EvaluationUtils:
    """
    Utilities for model evaluation and performance metrics.
    """
    
    @staticmethod
    def compute_trajectory_similarity(pred_traj: np.ndarray, 
                                    target_traj: np.ndarray,
                                    metric: str = 'dtw') -> float:
        """
        Compute similarity between predicted and target trajectories.
        
        Args:
            pred_traj: Predicted trajectory [seq_len, features]
            target_traj: Target trajectory [seq_len, features]
            metric: Similarity metric ('dtw', 'mse', 'cosine')
            
        Returns:
            similarity: Similarity score
        """
        if metric == 'mse':
            # Mean squared error (lower is better, convert to similarity)
            mse = np.mean((pred_traj - target_traj) ** 2)
            similarity = 1.0 / (1.0 + mse)
        
        elif metric == 'cosine':
            # Cosine similarity (only for position features)
            pred_flat = pred_traj[:, :2].flatten()
            target_flat = target_traj[:, :2].flatten()
            
            norm_pred = np.linalg.norm(pred_flat)
            norm_target = np.linalg.norm(target_flat)
            
            if norm_pred == 0 or norm_target == 0:
                similarity = 0.0
            else:
                similarity = np.dot(pred_flat, target_flat) / (norm_pred * norm_target)
        
        elif metric == 'dtw':
            # Dynamic Time Warping (simplified implementation)
            similarity = EvaluationUtils._compute_dtw_similarity(pred_traj[:, :2], target_traj[:, :2])
        
        else:
            raise ValueError(f"Unsupported similarity metric: {metric}")
        
        return similarity
    
    @staticmethod
    def _compute_dtw_similarity(seq1: np.ndarray, seq2: np.ndarray) -> float:
        """Compute DTW similarity between two sequences."""
        n, m = len(seq1), len(seq2)
        
        # DTW distance matrix
        dtw_matrix = np.full((n + 1, m + 1), np.inf)
        dtw_matrix[0, 0] = 0
        
        for i in range(1, n + 1):
            for j in range(1, m + 1):
                cost = np.linalg.norm(seq1[i-1] - seq2[j-1])
                dtw_matrix[i, j] = cost + min(
                    dtw_matrix[i-1, j],      # insertion
                    dtw_matrix[i, j-1],      # deletion
                    dtw_matrix[i-1, j-1]     # match
                )
        
        # Convert distance to similarity
        max_distance = max(n, m) * np.sqrt(2)  # Maximum possible distance
        dtw_distance = dtw_matrix[n, m]
        similarity = 1.0 - (dtw_distance / max_distance)
        
        return max(0.0, similarity)
    
    @staticmethod
    def evaluate_handwriting_quality(trajectory: np.ndarray) -> Dict[str, float]:
        """
        Evaluate handwriting quality metrics.
        
        Args:
            trajectory: Handwriting trajectory [seq_len, features]
            
        Returns:
            quality_metrics: Dictionary of quality metrics
        """
        positions = trajectory[:, :2]
        
        # Smoothness (inverse of jerk)
        if len(positions) >= 3:
            velocities = np.diff(positions, axis=0)
            accelerations = np.diff(velocities, axis=0)
            jerk = np.diff(accelerations, axis=0)
            smoothness = 1.0 / (1.0 + np.mean(np.linalg.norm(jerk, axis=1)))
        else:
            smoothness = 0.0
        
        # Consistency (velocity variation)
        if len(positions) >= 2:
            velocities = np.diff(positions, axis=0)
            speeds = np.linalg.norm(velocities, axis=1)
            speed_consistency = 1.0 / (1.0 + np.std(speeds))
        else:
            speed_consistency = 0.0
        
        # Legibility (simplified as trajectory coverage)
        if len(positions) > 0:
            bbox_area = EvaluationUtils._compute_bounding_box_area(positions)
            path_length = np.sum(np.linalg.norm(np.diff(positions, axis=0), axis=1))
            legibility = min(1.0, bbox_area / (path_length + 1e-8))
        else:
            legibility = 0.0
        
        return {
            'smoothness': smoothness,
            'speed_consistency': speed_consistency,
            'legibility': legibility,
            'overall_quality': (smoothness + speed_consistency + legibility) / 3.0
        }
    
    @staticmethod
    def _compute_bounding_box_area(positions: np.ndarray) -> float:
        """Compute bounding box area of trajectory."""
        if len(positions) == 0:
            return 0.0
        
        min_pos = np.min(positions, axis=0)
        max_pos = np.max(positions, axis=0)
        
        width = max_pos[0] - min_pos[0]
        height = max_pos[1] - min_pos[1]
        
        return width * height