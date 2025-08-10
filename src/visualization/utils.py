"""
Utility functions for robotic handwriting visualization.

This module provides helper functions for setting up plots, creating animations,
handling colors and styling, and managing visualization resources.
"""

import numpy as np
from typing import Dict, Any, Optional, List, Tuple, Union, Callable
import os
import logging

logger = logging.getLogger(__name__)


def setup_matplotlib(style: str = 'default', backend: str = 'TkAgg') -> bool:
    """
    Setup matplotlib with specified style and backend.
    
    Args:
        style: Matplotlib style to use
        backend: Backend to use for rendering
        
    Returns:
        bool: True if setup successful
    """
    try:
        import matplotlib
        matplotlib.use(backend)
        import matplotlib.pyplot as plt
        
        # Set style
        if style != 'default':
            plt.style.use(style)
        
        # Configure default parameters
        plt.rcParams['figure.figsize'] = (10, 6)
        plt.rcParams['font.size'] = 10
        plt.rcParams['lines.linewidth'] = 2
        plt.rcParams['grid.alpha'] = 0.3
        
        # Enable interactive mode
        plt.ion()
        
        logger.info(f"Matplotlib setup completed with style '{style}' and backend '{backend}'")
        return True
        
    except ImportError:
        logger.error("Matplotlib not available")
        return False
    except Exception as e:
        logger.error(f"Failed to setup matplotlib: {e}")
        return False


def create_figure(width: int = 800, height: int = 600, dpi: int = 100, 
                 title: str = "", style: str = None) -> Tuple[Any, Any]:
    """
    Create a matplotlib figure with specified parameters.
    
    Args:
        width: Figure width in pixels
        height: Figure height in pixels
        dpi: Dots per inch
        title: Figure title
        style: Optional style to apply
        
    Returns:
        Tuple of (figure, axes) objects
    """
    try:
        import matplotlib.pyplot as plt
        
        # Apply style if specified
        if style:
            plt.style.use(style)
        
        # Create figure
        fig, ax = plt.subplots(figsize=(width/dpi, height/dpi), dpi=dpi)
        
        if title:
            fig.suptitle(title, fontsize=14, fontweight='bold')
        
        return fig, ax
        
    except ImportError:
        logger.error("Matplotlib not available")
        return None, None
    except Exception as e:
        logger.error(f"Failed to create figure: {e}")
        return None, None


def save_plot(fig, filename: str, dpi: int = 300, format: str = 'png', 
             bbox_inches: str = 'tight', **kwargs) -> bool:
    """
    Save a matplotlib figure to file.
    
    Args:
        fig: Matplotlib figure object
        filename: Output filename
        dpi: Resolution for raster formats
        format: Output format
        bbox_inches: Bounding box setting
        **kwargs: Additional arguments for savefig
        
    Returns:
        bool: True if save successful
    """
    try:
        # Ensure directory exists
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        
        # Save figure
        fig.savefig(filename, dpi=dpi, format=format, bbox_inches=bbox_inches, **kwargs)
        
        logger.info(f"Plot saved to {filename}")
        return True
        
    except Exception as e:
        logger.error(f"Failed to save plot: {e}")
        return False


def animate_trajectory(trajectory: np.ndarray, 
                      pen_positions: Optional[np.ndarray] = None,
                      save_path: Optional[str] = None,
                      fps: int = 30,
                      interval: int = 50,
                      trail_length: int = 50) -> Any:
    """
    Create an animated visualization of a trajectory.
    
    Args:
        trajectory: Array of trajectory points [N, 2] or [N, 3]
        pen_positions: Optional pen positions for robot visualization
        save_path: Optional path to save animation
        fps: Frames per second for saved animation
        interval: Interval between frames in milliseconds
        trail_length: Number of points to show in trail
        
    Returns:
        Animation object
    """
    try:
        import matplotlib.pyplot as plt
        import matplotlib.animation as animation
        
        # Setup figure
        fig, ax = plt.subplots(figsize=(10, 8))
        
        # Determine if 3D
        is_3d = trajectory.shape[1] >= 3 and np.any(trajectory[:, 2] != trajectory[0, 2])
        
        if is_3d:
            ax.remove()
            ax = fig.add_subplot(111, projection='3d')
        
        # Set up plot limits
        margin = 0.1
        if is_3d:
            ax.set_xlim(np.min(trajectory[:, 0]) - margin, np.max(trajectory[:, 0]) + margin)
            ax.set_ylim(np.min(trajectory[:, 1]) - margin, np.max(trajectory[:, 1]) + margin)
            ax.set_zlim(np.min(trajectory[:, 2]) - margin, np.max(trajectory[:, 2]) + margin)
            ax.set_xlabel('X (m)')
            ax.set_ylabel('Y (m)')
            ax.set_zlabel('Z (m)')
        else:
            ax.set_xlim(np.min(trajectory[:, 0]) - margin, np.max(trajectory[:, 0]) + margin)
            ax.set_ylim(np.min(trajectory[:, 1]) - margin, np.max(trajectory[:, 1]) + margin)
            ax.set_xlabel('X (m)')
            ax.set_ylabel('Y (m)')
            ax.set_aspect('equal')
        
        ax.set_title('Trajectory Animation')
        ax.grid(True, alpha=0.3)
        
        # Initialize plot elements
        if is_3d:
            trail_line, = ax.plot([], [], [], 'b-', alpha=0.7, linewidth=2)
            current_point, = ax.plot([], [], [], 'ro', markersize=8)
            if pen_positions is not None:
                pen_point, = ax.plot([], [], [], 'go', markersize=6)
        else:
            trail_line, = ax.plot([], [], 'b-', alpha=0.7, linewidth=2)
            current_point, = ax.plot([], [], 'ro', markersize=8)
            if pen_positions is not None:
                pen_point, = ax.plot([], [], 'go', markersize=6)
        
        def animate_frame(frame):
            """Animation function for each frame."""
            # Determine trail start index
            start_idx = max(0, frame - trail_length)
            
            # Update trail
            if is_3d:
                trail_line.set_data_3d(trajectory[start_idx:frame+1, 0],
                                     trajectory[start_idx:frame+1, 1],
                                     trajectory[start_idx:frame+1, 2])
                current_point.set_data_3d([trajectory[frame, 0]], 
                                        [trajectory[frame, 1]], 
                                        [trajectory[frame, 2]])
                
                if pen_positions is not None and frame < len(pen_positions):
                    pen_point.set_data_3d([pen_positions[frame, 0]], 
                                        [pen_positions[frame, 1]], 
                                        [pen_positions[frame, 2]])
            else:
                trail_line.set_data(trajectory[start_idx:frame+1, 0],
                                  trajectory[start_idx:frame+1, 1])
                current_point.set_data([trajectory[frame, 0]], [trajectory[frame, 1]])
                
                if pen_positions is not None and frame < len(pen_positions):
                    pen_point.set_data([pen_positions[frame, 0]], [pen_positions[frame, 1]])
            
            # Update frame counter in title
            ax.set_title(f'Trajectory Animation - Frame {frame}/{len(trajectory)-1}')
            
            if pen_positions is not None:
                return trail_line, current_point, pen_point
            else:
                return trail_line, current_point
        
        # Create animation
        anim = animation.FuncAnimation(fig, animate_frame, frames=len(trajectory),
                                     interval=interval, blit=True, repeat=True)
        
        # Save animation if path provided
        if save_path:
            writer = animation.FFMpegWriter(fps=fps, metadata=dict(artist='Robotic Handwriting'))
            anim.save(save_path, writer=writer)
            logger.info(f"Animation saved to {save_path}")
        
        return anim
        
    except ImportError as e:
        logger.error(f"Required libraries not available for animation: {e}")
        return None
    except Exception as e:
        logger.error(f"Failed to create animation: {e}")
        return None


def create_video(image_sequence: List[str], output_path: str, fps: int = 30, 
                codec: str = 'libx264') -> bool:
    """
    Create video from sequence of images.
    
    Args:
        image_sequence: List of image file paths
        output_path: Output video file path
        fps: Frames per second
        codec: Video codec to use
        
    Returns:
        bool: True if video creation successful
    """
    try:
        import cv2
        
        if not image_sequence:
            logger.error("No images provided for video creation")
            return False
        
        # Read first image to get dimensions
        first_image = cv2.imread(image_sequence[0])
        if first_image is None:
            logger.error(f"Could not read first image: {image_sequence[0]}")
            return False
        
        height, width, layers = first_image.shape
        
        # Define video codec and create VideoWriter
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        video_writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        # Add images to video
        for image_path in image_sequence:
            image = cv2.imread(image_path)
            if image is not None:
                video_writer.write(image)
            else:
                logger.warning(f"Could not read image: {image_path}")
        
        # Release video writer
        video_writer.release()
        
        logger.info(f"Video created: {output_path}")
        return True
        
    except ImportError:
        logger.error("OpenCV not available for video creation")
        return False
    except Exception as e:
        logger.error(f"Failed to create video: {e}")
        return False


def setup_colors(color_scheme: str = 'default') -> Dict[str, str]:
    """
    Setup color palette for visualization.
    
    Args:
        color_scheme: Color scheme name ('default', 'dark', 'colorful', 'professional')
        
    Returns:
        Dict mapping color names to hex codes
    """
    color_schemes = {
        'default': {
            'background': '#FFFFFF',
            'text': '#000000',
            'grid': '#CCCCCC',
            'primary': '#1f77b4',
            'secondary': '#ff7f0e',
            'success': '#2ca02c',
            'warning': '#ffbb78',
            'error': '#d62728',
            'trajectory': '#1f77b4',
            'robot': '#888888',
            'pen': '#d62728',
            'paper': '#f7f7f7'
        },
        'dark': {
            'background': '#2E2E2E',
            'text': '#FFFFFF',
            'grid': '#555555',
            'primary': '#4CAF50',
            'secondary': '#FF9800',
            'success': '#8BC34A',
            'warning': '#FFC107',
            'error': '#F44336',
            'trajectory': '#00BCD4',
            'robot': '#CCCCCC',
            'pen': '#E91E63',
            'paper': '#424242'
        },
        'colorful': {
            'background': '#FFFFFF',
            'text': '#333333',
            'grid': '#E0E0E0',
            'primary': '#3F51B5',
            'secondary': '#FF5722',
            'success': '#4CAF50',
            'warning': '#FF9800',
            'error': '#F44336',
            'trajectory': '#9C27B0',
            'robot': '#607D8B',
            'pen': '#E91E63',
            'paper': '#FAFAFA'
        },
        'professional': {
            'background': '#FFFFFF',
            'text': '#2C3E50',
            'grid': '#BDC3C7',
            'primary': '#34495E',
            'secondary': '#3498DB',
            'success': '#27AE60',
            'warning': '#F39C12',
            'error': '#E74C3C',
            'trajectory': '#2980B9',
            'robot': '#7F8C8D',
            'pen': '#C0392B',
            'paper': '#ECF0F1'
        }
    }
    
    return color_schemes.get(color_scheme, color_schemes['default'])


def format_plot(ax, title: str = "", xlabel: str = "", ylabel: str = "", 
               grid: bool = True, legend: bool = False, 
               color_scheme: str = 'default') -> None:
    """
    Apply consistent formatting to a plot.
    
    Args:
        ax: Matplotlib axes object
        title: Plot title
        xlabel: X-axis label
        ylabel: Y-axis label
        grid: Whether to show grid
        legend: Whether to show legend
        color_scheme: Color scheme to use
    """
    try:
        colors = setup_colors(color_scheme)
        
        # Set title and labels
        if title:
            ax.set_title(title, fontsize=12, fontweight='bold', color=colors['text'])
        if xlabel:
            ax.set_xlabel(xlabel, fontsize=10, color=colors['text'])
        if ylabel:
            ax.set_ylabel(ylabel, fontsize=10, color=colors['text'])
        
        # Configure grid
        if grid:
            ax.grid(True, alpha=0.3, color=colors['grid'])
        
        # Configure legend
        if legend and ax.get_legend_handles_labels()[0]:
            ax.legend(loc='best', framealpha=0.9)
        
        # Set background color
        ax.set_facecolor(colors['background'])
        
        # Configure tick colors
        ax.tick_params(colors=colors['text'])
        
    except Exception as e:
        logger.error(f"Failed to format plot: {e}")


def create_subplot_grid(num_plots: int, max_cols: int = 3) -> Tuple[int, int]:
    """
    Calculate optimal subplot grid dimensions.
    
    Args:
        num_plots: Number of plots to arrange
        max_cols: Maximum number of columns
        
    Returns:
        Tuple of (rows, cols)
    """
    if num_plots <= 0:
        return 1, 1
    
    cols = min(num_plots, max_cols)
    rows = (num_plots + cols - 1) // cols  # Ceiling division
    
    return rows, cols


def add_colorbar(ax, mappable, label: str = "", orientation: str = 'vertical') -> Any:
    """
    Add colorbar to a plot.
    
    Args:
        ax: Matplotlib axes object
        mappable: Mappable object (e.g., from imshow, scatter)
        label: Colorbar label
        orientation: Colorbar orientation ('vertical' or 'horizontal')
        
    Returns:
        Colorbar object
    """
    try:
        import matplotlib.pyplot as plt
        
        cbar = plt.colorbar(mappable, ax=ax, orientation=orientation)
        
        if label:
            if orientation == 'vertical':
                cbar.set_label(label, rotation=270, labelpad=15)
            else:
                cbar.set_label(label)
        
        return cbar
        
    except Exception as e:
        logger.error(f"Failed to add colorbar: {e}")
        return None


def normalize_data(data: np.ndarray, method: str = 'minmax') -> np.ndarray:
    """
    Normalize data for visualization.
    
    Args:
        data: Input data array
        method: Normalization method ('minmax', 'zscore', 'robust')
        
    Returns:
        Normalized data array
    """
    try:
        if method == 'minmax':
            # Min-max normalization to [0, 1]
            data_min = np.min(data)
            data_max = np.max(data)
            if data_max > data_min:
                return (data - data_min) / (data_max - data_min)
            else:
                return np.zeros_like(data)
        
        elif method == 'zscore':
            # Z-score normalization
            mean = np.mean(data)
            std = np.std(data)
            if std > 0:
                return (data - mean) / std
            else:
                return np.zeros_like(data)
        
        elif method == 'robust':
            # Robust normalization using median and IQR
            median = np.median(data)
            q75, q25 = np.percentile(data, [75, 25])
            iqr = q75 - q25
            if iqr > 0:
                return (data - median) / iqr
            else:
                return np.zeros_like(data)
        
        else:
            logger.warning(f"Unknown normalization method: {method}")
            return data
        
    except Exception as e:
        logger.error(f"Failed to normalize data: {e}")
        return data


def create_trajectory_colormap(trajectory: np.ndarray, 
                             metric: Optional[np.ndarray] = None,
                             colormap: str = 'viridis') -> Tuple[np.ndarray, Any]:
    """
    Create color-mapped trajectory visualization.
    
    Args:
        trajectory: Trajectory points [N, 2] or [N, 3]
        metric: Optional metric values for coloring [N]
        colormap: Matplotlib colormap name
        
    Returns:
        Tuple of (colors, colormap object)
    """
    try:
        import matplotlib.pyplot as plt
        import matplotlib.cm as cm
        
        if metric is None:
            # Use time/sequence as metric
            metric = np.arange(len(trajectory))
        
        # Normalize metric values
        norm_metric = normalize_data(metric)
        
        # Create colormap
        cmap = cm.get_cmap(colormap)
        colors = cmap(norm_metric)
        
        return colors, cmap
        
    except ImportError:
        logger.error("Matplotlib not available for colormap creation")
        return None, None
    except Exception as e:
        logger.error(f"Failed to create trajectory colormap: {e}")
        return None, None


def calculate_plot_limits(data: np.ndarray, margin: float = 0.1) -> Tuple[List[float], List[float]]:
    """
    Calculate appropriate plot limits with margin.
    
    Args:
        data: Data array [N, 2] or [N, 3]
        margin: Margin as fraction of data range
        
    Returns:
        Tuple of (x_limits, y_limits) or (x_limits, y_limits, z_limits)
    """
    try:
        if len(data) == 0:
            return [0, 1], [0, 1]
        
        # Calculate ranges
        data_min = np.min(data, axis=0)
        data_max = np.max(data, axis=0)
        data_range = data_max - data_min
        
        # Add margin
        margin_size = data_range * margin
        limits_min = data_min - margin_size
        limits_max = data_max + margin_size
        
        # Ensure minimum range
        min_range = 0.001
        for i in range(len(data_range)):
            if data_range[i] < min_range:
                center = (limits_min[i] + limits_max[i]) / 2
                limits_min[i] = center - min_range / 2
                limits_max[i] = center + min_range / 2
        
        x_limits = [limits_min[0], limits_max[0]]
        y_limits = [limits_min[1], limits_max[1]]
        
        if data.shape[1] >= 3:
            z_limits = [limits_min[2], limits_max[2]]
            return x_limits, y_limits, z_limits
        else:
            return x_limits, y_limits
        
    except Exception as e:
        logger.error(f"Failed to calculate plot limits: {e}")
        return [0, 1], [0, 1]


def create_error_ellipse(mean: np.ndarray, cov: np.ndarray, 
                        confidence: float = 0.95, num_points: int = 100) -> np.ndarray:
    """
    Create error ellipse for 2D data visualization.
    
    Args:
        mean: Mean point [2]
        cov: Covariance matrix [2, 2]
        confidence: Confidence level (0-1)
        num_points: Number of points in ellipse
        
    Returns:
        Array of ellipse points [num_points, 2]
    """
    try:
        from scipy.stats import chi2
        
        # Calculate confidence interval
        chi2_val = chi2.ppf(confidence, df=2)
        
        # Eigenvalue decomposition
        eigenvals, eigenvecs = np.linalg.eigh(cov)
        
        # Calculate ellipse parameters
        angle = np.arctan2(eigenvecs[1, 0], eigenvecs[0, 0])
        width = 2 * np.sqrt(chi2_val * eigenvals[0])
        height = 2 * np.sqrt(chi2_val * eigenvals[1])
        
        # Generate ellipse points
        t = np.linspace(0, 2 * np.pi, num_points)
        ellipse = np.array([width * np.cos(t), height * np.sin(t)]).T
        
        # Rotate ellipse
        rotation_matrix = np.array([[np.cos(angle), -np.sin(angle)],
                                   [np.sin(angle), np.cos(angle)]])
        ellipse = ellipse @ rotation_matrix.T
        
        # Translate to mean
        ellipse += mean
        
        return ellipse
        
    except ImportError:
        logger.error("SciPy not available for error ellipse calculation")
        return np.array([])
    except Exception as e:
        logger.error(f"Failed to create error ellipse: {e}")
        return np.array([])


def cleanup_visualization_resources():
    """Clean up visualization resources and close plots."""
    try:
        import matplotlib.pyplot as plt
        plt.close('all')
        logger.info("Visualization resources cleaned up")
    except ImportError:
        pass
    except Exception as e:
        logger.error(f"Failed to cleanup visualization resources: {e}")


def validate_visualization_data(data: Dict[str, Any], required_fields: List[str]) -> bool:
    """
    Validate data for visualization.
    
    Args:
        data: Data dictionary to validate
        required_fields: List of required field names
        
    Returns:
        bool: True if data is valid
    """
    try:
        # Check if data is dictionary
        if not isinstance(data, dict):
            logger.error("Data must be a dictionary")
            return False
        
        # Check required fields
        for field in required_fields:
            if field not in data:
                logger.error(f"Missing required field: {field}")
                return False
        
        # Check for numpy arrays and convert if needed
        for key, value in data.items():
            if isinstance(value, list):
                try:
                    data[key] = np.array(value)
                except Exception:
                    pass  # Keep as list if conversion fails
        
        return True
        
    except Exception as e:
        logger.error(f"Data validation failed: {e}")
        return False