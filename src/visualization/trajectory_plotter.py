"""
Trajectory plotting and analysis for robotic handwriting.

This module provides comprehensive trajectory visualization including
path plots, velocity profiles, acceleration analysis, and quality metrics.
"""

import numpy as np
from typing import Dict, Any, Optional, List, Tuple, Union
import logging

from .base_visualizer import BaseVisualizer

logger = logging.getLogger(__name__)


class TrajectoryPlotter(BaseVisualizer):
    """
    Comprehensive trajectory plotting and analysis tool.
    
    Provides visualization of:
    - 2D and 3D trajectory paths
    - Velocity and acceleration profiles
    - Curvature analysis
    - Quality metrics visualization
    - Comparative trajectory analysis
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the trajectory plotter.
        
        Args:
            config: Configuration dictionary with plotting settings
        """
        super().__init__(config)
        
        # Required fields for validation
        self.required_fields = ['trajectory']
        
        # Plot configuration
        self.plot_types = config.get('plot_types', ['path', 'velocity', 'acceleration'])
        self.subplot_layout = config.get('subplot_layout', (2, 2))
        
        # Trajectory display settings
        self.trajectory_colors = config.get('trajectory_colors', ['blue', 'red', 'green', 'purple'])
        self.line_width = config.get('line_width', 2)
        self.marker_size = config.get('marker_size', 4)
        self.show_markers = config.get('show_markers', True)
        self.show_start_end = config.get('show_start_end', True)
        
        # Analysis settings
        self.compute_velocity = config.get('compute_velocity', True)
        self.compute_acceleration = config.get('compute_acceleration', True)
        self.compute_curvature = config.get('compute_curvature', True)
        self.compute_quality_metrics = config.get('compute_quality_metrics', True)
        
        # Time settings
        self.timestep = config.get('timestep', 0.001)
        self.time_units = config.get('time_units', 'seconds')
        
        # Grid and formatting
        self.show_grid = config.get('show_grid', True)
        self.show_legend = config.get('show_legend', True)
        
        # Stored trajectories for comparison
        self.trajectories = {}
        self.trajectory_analyses = {}
        
        # Matplotlib setup
        self.use_matplotlib = config.get('use_matplotlib', True)
        if self.use_matplotlib:
            self.setup_matplotlib()
    
    def setup_matplotlib(self):
        """Setup matplotlib for plotting."""
        try:
            import matplotlib.pyplot as plt
            import matplotlib.patches as patches
            from mpl_toolkits.mplot3d import Axes3D
            
            self.plt = plt
            self.patches = patches
            
            # Set style
            style = self.config.get('plot_style', 'default')
            if style != 'default':
                plt.style.use(style)
            
            # Set default figure parameters
            plt.rcParams['figure.figsize'] = (12, 8)
            plt.rcParams['font.size'] = 10
            plt.rcParams['lines.linewidth'] = self.line_width
            
        except ImportError:
            logger.error("Matplotlib not available. Install matplotlib for trajectory plotting.")
            self.use_matplotlib = False
    
    def initialize(self) -> bool:
        """
        Initialize the trajectory plotter.
        
        Returns:
            bool: True if initialization successful
        """
        try:
            self.is_initialized = True
            self.is_active = True
            
            if self.enable_logging:
                self.viz_logger.info("Trajectory plotter initialized")
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize trajectory plotter: {e}")
            return False
    
    def update(self, data: Dict[str, Any]) -> bool:
        """
        Update with new trajectory data.
        
        Args:
            data: Data dictionary containing trajectory information
            
        Returns:
            bool: True if update successful
        """
        if not self.validate_data(data):
            return False
        
        try:
            # Process the data
            processed_data = self.process_data(data)
            
            # Store trajectory if it has a name/id
            trajectory_id = processed_data.get('id', f'trajectory_{len(self.trajectories)}')
            self.add_trajectory(trajectory_id, processed_data['trajectory'])
            
            # Add to buffer
            self.add_data(processed_data)
            
            return True
            
        except Exception as e:
            if self.enable_logging:
                self.viz_logger.error(f"Update failed: {e}")
            return False
    
    def add_trajectory(self, trajectory_id: str, trajectory: np.ndarray, metadata: Optional[Dict] = None):
        """
        Add a trajectory for analysis and plotting.
        
        Args:
            trajectory_id: Unique identifier for the trajectory
            trajectory: Array of trajectory points [N, 3]
            metadata: Optional metadata about the trajectory
        """
        if isinstance(trajectory, list):
            trajectory = np.array(trajectory)
        
        self.trajectories[trajectory_id] = {
            'points': trajectory,
            'metadata': metadata or {},
            'timestamp': time.time()
        }
        
        # Compute analysis
        if len(trajectory) > 1:
            analysis = self.analyze_trajectory(trajectory)
            self.trajectory_analyses[trajectory_id] = analysis
        
        if self.enable_logging:
            self.viz_logger.info(f"Added trajectory: {trajectory_id} ({len(trajectory)} points)")
    
    def analyze_trajectory(self, trajectory: np.ndarray) -> Dict[str, Any]:
        """
        Compute comprehensive trajectory analysis.
        
        Args:
            trajectory: Array of trajectory points [N, 3]
            
        Returns:
            Dict containing analysis results
        """
        analysis = {}
        
        if len(trajectory) < 2:
            return analysis
        
        # Basic metrics
        analysis['num_points'] = len(trajectory)
        analysis['total_distance'] = self.compute_path_length(trajectory)
        analysis['total_time'] = len(trajectory) * self.timestep
        
        # Velocity analysis
        if self.compute_velocity:
            velocities = self.compute_velocities(trajectory)
            speeds = np.linalg.norm(velocities, axis=1)
            
            analysis['velocities'] = velocities
            analysis['speeds'] = speeds
            analysis['mean_speed'] = np.mean(speeds)
            analysis['max_speed'] = np.max(speeds)
            analysis['speed_std'] = np.std(speeds)
        
        # Acceleration analysis
        if self.compute_acceleration and 'velocities' in analysis:
            accelerations = self.compute_accelerations(analysis['velocities'])
            acceleration_magnitudes = np.linalg.norm(accelerations, axis=1)
            
            analysis['accelerations'] = accelerations
            analysis['acceleration_magnitudes'] = acceleration_magnitudes
            analysis['mean_acceleration'] = np.mean(acceleration_magnitudes)
            analysis['max_acceleration'] = np.max(acceleration_magnitudes)
        
        # Curvature analysis
        if self.compute_curvature:
            curvatures = self.compute_curvatures(trajectory)
            analysis['curvatures'] = curvatures
            analysis['mean_curvature'] = np.mean(curvatures) if len(curvatures) > 0 else 0
            analysis['max_curvature'] = np.max(curvatures) if len(curvatures) > 0 else 0
        
        # Quality metrics
        if self.compute_quality_metrics:
            quality = self.compute_quality_metrics_detailed(trajectory, analysis)
            analysis['quality_metrics'] = quality
        
        return analysis
    
    def compute_path_length(self, trajectory: np.ndarray) -> float:
        """Compute total path length."""
        if len(trajectory) < 2:
            return 0.0
        
        distances = np.linalg.norm(np.diff(trajectory, axis=0), axis=1)
        return np.sum(distances)
    
    def compute_velocities(self, trajectory: np.ndarray) -> np.ndarray:
        """Compute velocity vectors."""
        if len(trajectory) < 2:
            return np.array([])
        
        velocities = np.diff(trajectory, axis=0) / self.timestep
        return velocities
    
    def compute_accelerations(self, velocities: np.ndarray) -> np.ndarray:
        """Compute acceleration vectors."""
        if len(velocities) < 2:
            return np.array([])
        
        accelerations = np.diff(velocities, axis=0) / self.timestep
        return accelerations
    
    def compute_curvatures(self, trajectory: np.ndarray) -> np.ndarray:
        """Compute curvature at each point."""
        if len(trajectory) < 3:
            return np.array([])
        
        curvatures = []
        for i in range(1, len(trajectory) - 1):
            p1, p2, p3 = trajectory[i-1], trajectory[i], trajectory[i+1]
            
            # Compute vectors
            v1 = p2 - p1
            v2 = p3 - p2
            
            # Compute curvature using the formula: |v1 x v2| / |v1|^3
            if np.linalg.norm(v1) > 0 and np.linalg.norm(v2) > 0:
                cross_product = np.cross(v1, v2)
                if len(cross_product.shape) == 0:  # 2D case
                    cross_magnitude = abs(cross_product)
                else:  # 3D case
                    cross_magnitude = np.linalg.norm(cross_product)
                
                curvature = cross_magnitude / (np.linalg.norm(v1) ** 3)
                curvatures.append(curvature)
            else:
                curvatures.append(0.0)
        
        return np.array(curvatures)
    
    def compute_quality_metrics_detailed(self, trajectory: np.ndarray, analysis: Dict[str, Any]) -> Dict[str, float]:
        """Compute detailed quality metrics."""
        metrics = {}
        
        # Smoothness (inverse of acceleration variance)
        if 'acceleration_magnitudes' in analysis and len(analysis['acceleration_magnitudes']) > 0:
            acc_var = np.var(analysis['acceleration_magnitudes'])
            metrics['smoothness'] = 1.0 / (1.0 + acc_var)
        else:
            metrics['smoothness'] = 0.0
        
        # Speed consistency (inverse of speed variance)
        if 'speeds' in analysis and len(analysis['speeds']) > 0:
            speed_var = np.var(analysis['speeds'])
            metrics['speed_consistency'] = 1.0 / (1.0 + speed_var)
        else:
            metrics['speed_consistency'] = 0.0
        
        # Path efficiency (ratio of direct distance to path length)
        if len(trajectory) >= 2:
            direct_distance = np.linalg.norm(trajectory[-1] - trajectory[0])
            path_length = analysis.get('total_distance', 0)
            if path_length > 0:
                metrics['path_efficiency'] = direct_distance / path_length
            else:
                metrics['path_efficiency'] = 0.0
        else:
            metrics['path_efficiency'] = 0.0
        
        # Curvature consistency
        if 'curvatures' in analysis and len(analysis['curvatures']) > 0:
            curv_var = np.var(analysis['curvatures'])
            metrics['curvature_consistency'] = 1.0 / (1.0 + curv_var)
        else:
            metrics['curvature_consistency'] = 0.0
        
        # Overall quality score (weighted combination)
        weights = {'smoothness': 0.3, 'speed_consistency': 0.3, 'path_efficiency': 0.2, 'curvature_consistency': 0.2}
        quality_score = sum(weights[key] * metrics[key] for key in weights if key in metrics)
        metrics['overall_quality'] = quality_score
        
        return metrics
    
    def render(self) -> bool:
        """
        Render trajectory plots.
        
        Returns:
            bool: True if rendering successful
        """
        if not self.is_active or not self.use_matplotlib:
            return False
        
        try:
            # Create subplots based on plot types
            self.create_subplot_layout()
            
            # Plot each type
            subplot_idx = 0
            for plot_type in self.plot_types:
                if hasattr(self, f'plot_{plot_type}'):
                    getattr(self, f'plot_{plot_type}')(subplot_idx)
                    subplot_idx += 1
            
            # Add overall title and formatting
            self.plt.suptitle('Trajectory Analysis', fontsize=14, fontweight='bold')
            self.plt.tight_layout()
            
            # Show or save
            if self.config.get('show_plot', True):
                self.plt.show()
            
            self.frame_count += 1
            return True
            
        except Exception as e:
            if self.enable_logging:
                self.viz_logger.error(f"Render failed: {e}")
            return False
    
    def create_subplot_layout(self):
        """Create subplot layout based on plot types."""
        num_plots = len(self.plot_types)
        
        if num_plots == 1:
            rows, cols = 1, 1
        elif num_plots == 2:
            rows, cols = 1, 2
        elif num_plots <= 4:
            rows, cols = 2, 2
        else:
            rows, cols = 3, 3
        
        self.fig, self.axes = self.plt.subplots(rows, cols, figsize=(cols*5, rows*4))
        
        if num_plots == 1:
            self.axes = [self.axes]
        elif rows == 1 or cols == 1:
            self.axes = self.axes.flatten()
        else:
            self.axes = self.axes.flatten()
        
        # Hide unused subplots
        for i in range(num_plots, len(self.axes)):
            self.axes[i].set_visible(False)
    
    def plot_path(self, subplot_idx: int):
        """Plot 2D/3D trajectory paths."""
        ax = self.axes[subplot_idx]
        
        # Determine if 3D plot is needed
        use_3d = any(np.any(traj['points'][:, 2] != traj['points'][0, 2]) 
                    for traj in self.trajectories.values() if len(traj['points']) > 0)
        
        if use_3d:
            # Remove 2D axis and create 3D
            ax.remove()
            ax = self.fig.add_subplot(self.subplot_layout[0], self.subplot_layout[1], subplot_idx+1, projection='3d')
            self.axes[subplot_idx] = ax
        
        # Plot each trajectory
        for i, (traj_id, traj_data) in enumerate(self.trajectories.items()):
            trajectory = traj_data['points']
            if len(trajectory) == 0:
                continue
            
            color = self.trajectory_colors[i % len(self.trajectory_colors)]
            
            if use_3d:
                ax.plot(trajectory[:, 0], trajectory[:, 1], trajectory[:, 2], 
                       color=color, linewidth=self.line_width, label=traj_id)
                
                if self.show_markers:
                    ax.scatter(trajectory[::10, 0], trajectory[::10, 1], trajectory[::10, 2],
                             color=color, s=self.marker_size)
                
                if self.show_start_end:
                    ax.scatter([trajectory[0, 0]], [trajectory[0, 1]], [trajectory[0, 2]],
                             color='green', s=50, marker='o', label=f'{traj_id} start')
                    ax.scatter([trajectory[-1, 0]], [trajectory[-1, 1]], [trajectory[-1, 2]],
                             color='red', s=50, marker='s', label=f'{traj_id} end')
                
                ax.set_xlabel('X (m)')
                ax.set_ylabel('Y (m)')
                ax.set_zlabel('Z (m)')
                ax.set_title('3D Trajectory Path')
            
            else:
                ax.plot(trajectory[:, 0], trajectory[:, 1], 
                       color=color, linewidth=self.line_width, label=traj_id)
                
                if self.show_markers:
                    ax.scatter(trajectory[::10, 0], trajectory[::10, 1],
                             color=color, s=self.marker_size)
                
                if self.show_start_end:
                    ax.scatter([trajectory[0, 0]], [trajectory[0, 1]],
                             color='green', s=50, marker='o', label=f'{traj_id} start')
                    ax.scatter([trajectory[-1, 0]], [trajectory[-1, 1]],
                             color='red', s=50, marker='s', label=f'{traj_id} end')
                
                ax.set_xlabel('X (m)')
                ax.set_ylabel('Y (m)')
                ax.set_title('2D Trajectory Path')
                ax.set_aspect('equal')
        
        if self.show_grid:
            ax.grid(True, alpha=0.3)
        
        if self.show_legend and self.trajectories:
            ax.legend()
    
    def plot_velocity(self, subplot_idx: int):
        """Plot velocity profiles."""
        ax = self.axes[subplot_idx]
        
        for i, (traj_id, analysis) in enumerate(self.trajectory_analyses.items()):
            if 'speeds' not in analysis:
                continue
            
            speeds = analysis['speeds']
            time_axis = np.arange(len(speeds)) * self.timestep
            
            color = self.trajectory_colors[i % len(self.trajectory_colors)]
            ax.plot(time_axis, speeds, color=color, linewidth=self.line_width, label=traj_id)
        
        ax.set_xlabel(f'Time ({self.time_units})')
        ax.set_ylabel('Speed (m/s)')
        ax.set_title('Velocity Profile')
        
        if self.show_grid:
            ax.grid(True, alpha=0.3)
        
        if self.show_legend and self.trajectory_analyses:
            ax.legend()
    
    def plot_acceleration(self, subplot_idx: int):
        """Plot acceleration profiles."""
        ax = self.axes[subplot_idx]
        
        for i, (traj_id, analysis) in enumerate(self.trajectory_analyses.items()):
            if 'acceleration_magnitudes' not in analysis:
                continue
            
            accelerations = analysis['acceleration_magnitudes']
            time_axis = np.arange(len(accelerations)) * self.timestep
            
            color = self.trajectory_colors[i % len(self.trajectory_colors)]
            ax.plot(time_axis, accelerations, color=color, linewidth=self.line_width, label=traj_id)
        
        ax.set_xlabel(f'Time ({self.time_units})')
        ax.set_ylabel('Acceleration (m/s²)')
        ax.set_title('Acceleration Profile')
        
        if self.show_grid:
            ax.grid(True, alpha=0.3)
        
        if self.show_legend and self.trajectory_analyses:
            ax.legend()
    
    def plot_curvature(self, subplot_idx: int):
        """Plot curvature profiles."""
        ax = self.axes[subplot_idx]
        
        for i, (traj_id, analysis) in enumerate(self.trajectory_analyses.items()):
            if 'curvatures' not in analysis:
                continue
            
            curvatures = analysis['curvatures']
            time_axis = np.arange(len(curvatures)) * self.timestep
            
            color = self.trajectory_colors[i % len(self.trajectory_colors)]
            ax.plot(time_axis, curvatures, color=color, linewidth=self.line_width, label=traj_id)
        
        ax.set_xlabel(f'Time ({self.time_units})')
        ax.set_ylabel('Curvature (1/m)')
        ax.set_title('Curvature Profile')
        
        if self.show_grid:
            ax.grid(True, alpha=0.3)
        
        if self.show_legend and self.trajectory_analyses:
            ax.legend()
    
    def plot_quality_metrics(self, subplot_idx: int):
        """Plot quality metrics comparison."""
        ax = self.axes[subplot_idx]
        
        # Collect metrics for all trajectories
        metrics_data = {}
        for traj_id, analysis in self.trajectory_analyses.items():
            if 'quality_metrics' in analysis:
                for metric, value in analysis['quality_metrics'].items():
                    if metric not in metrics_data:
                        metrics_data[metric] = []
                    metrics_data[metric].append((traj_id, value))
        
        if not metrics_data:
            ax.text(0.5, 0.5, 'No quality metrics available', 
                   ha='center', va='center', transform=ax.transAxes)
            return
        
        # Create bar chart
        metric_names = list(metrics_data.keys())
        x_pos = np.arange(len(metric_names))
        
        # Plot bars for each trajectory
        bar_width = 0.8 / len(self.trajectory_analyses) if self.trajectory_analyses else 0.8
        
        for i, traj_id in enumerate(self.trajectory_analyses.keys()):
            values = [metrics_data[metric][i][1] if len(metrics_data[metric]) > i else 0 
                     for metric in metric_names]
            
            color = self.trajectory_colors[i % len(self.trajectory_colors)]
            ax.bar(x_pos + i * bar_width, values, bar_width, 
                  color=color, alpha=0.7, label=traj_id)
        
        ax.set_xlabel('Quality Metrics')
        ax.set_ylabel('Score')
        ax.set_title('Quality Metrics Comparison')
        ax.set_xticks(x_pos + bar_width * (len(self.trajectory_analyses) - 1) / 2)
        ax.set_xticklabels(metric_names, rotation=45, ha='right')
        
        if self.show_grid:
            ax.grid(True, alpha=0.3, axis='y')
        
        if self.show_legend and self.trajectory_analyses:
            ax.legend()
    
    def compare_trajectories(self, trajectory_ids: List[str]) -> Dict[str, Any]:
        """
        Compare multiple trajectories.
        
        Args:
            trajectory_ids: List of trajectory IDs to compare
            
        Returns:
            Dict containing comparison results
        """
        comparison = {}
        
        # Validate trajectory IDs
        valid_ids = [tid for tid in trajectory_ids if tid in self.trajectory_analyses]
        if len(valid_ids) < 2:
            return comparison
        
        # Compare basic metrics
        metrics_to_compare = ['total_distance', 'total_time', 'mean_speed', 'max_speed']
        
        for metric in metrics_to_compare:
            values = []
            for tid in valid_ids:
                analysis = self.trajectory_analyses[tid]
                if metric in analysis:
                    values.append(analysis[metric])
            
            if values:
                comparison[metric] = {
                    'values': dict(zip(valid_ids, values)),
                    'mean': np.mean(values),
                    'std': np.std(values),
                    'min': np.min(values),
                    'max': np.max(values)
                }
        
        # Compare quality metrics
        quality_comparison = {}
        for tid in valid_ids:
            analysis = self.trajectory_analyses[tid]
            if 'quality_metrics' in analysis:
                for metric, value in analysis['quality_metrics'].items():
                    if metric not in quality_comparison:
                        quality_comparison[metric] = {}
                    quality_comparison[metric][tid] = value
        
        comparison['quality_metrics'] = quality_comparison
        
        return comparison
    
    def save_plot(self, filename: str, **kwargs) -> bool:
        """
        Save the current plot to file.
        
        Args:
            filename: Output filename
            **kwargs: Additional arguments for savefig
            
        Returns:
            bool: True if save successful
        """
        if not hasattr(self, 'fig') or self.fig is None:
            return False
        
        try:
            self.fig.savefig(filename, dpi=self.dpi, bbox_inches='tight', **kwargs)
            if self.enable_logging:
                self.viz_logger.info(f"Plot saved to {filename}")
            return True
        except Exception as e:
            if self.enable_logging:
                self.viz_logger.error(f"Failed to save plot: {e}")
            return False
    
    def clear_trajectories(self):
        """Clear all stored trajectories."""
        self.trajectories.clear()
        self.trajectory_analyses.clear()
        if self.enable_logging:
            self.viz_logger.info("All trajectories cleared")
    
    def get_trajectory_summary(self) -> Dict[str, Any]:
        """
        Get summary of all trajectories.
        
        Returns:
            Dict containing trajectory summary
        """
        summary = {
            'num_trajectories': len(self.trajectories),
            'trajectory_ids': list(self.trajectories.keys()),
            'total_points': sum(len(traj['points']) for traj in self.trajectories.values()),
            'analyses_available': len(self.trajectory_analyses)
        }
        
        if self.trajectory_analyses:
            # Overall statistics
            all_distances = [analysis.get('total_distance', 0) for analysis in self.trajectory_analyses.values()]
            all_times = [analysis.get('total_time', 0) for analysis in self.trajectory_analyses.values()]
            
            if all_distances:
                summary['distance_stats'] = {
                    'mean': np.mean(all_distances),
                    'std': np.std(all_distances),
                    'min': np.min(all_distances),
                    'max': np.max(all_distances)
                }
            
            if all_times:
                summary['time_stats'] = {
                    'mean': np.mean(all_times),
                    'std': np.std(all_times),
                    'min': np.min(all_times),
                    'max': np.max(all_times)
                }
        
        return summary
    
    def close(self):
        """Close the trajectory plotter."""
        if self.use_matplotlib and hasattr(self, 'plt'):
            self.plt.close('all')
        
        self.is_active = False
        if self.enable_logging:
            self.viz_logger.info("Trajectory plotter closed")