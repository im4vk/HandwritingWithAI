"""
Visualization package for robotic handwriting systems.

This package provides comprehensive visualization tools for real-time rendering,
trajectory analysis, performance metrics, and interactive displays.
"""

from .base_visualizer import BaseVisualizer
from .real_time_visualizer import RealTimeVisualizer
from .trajectory_plotter import TrajectoryPlotter
from .robot_renderer import RobotRenderer
from .metrics_dashboard import MetricsDashboard
from .utils import (
    setup_matplotlib,
    create_figure,
    save_plot,
    animate_trajectory,
    create_video,
    setup_colors,
    format_plot
)

__all__ = [
    'BaseVisualizer',
    'RealTimeVisualizer',
    'TrajectoryPlotter', 
    'RobotRenderer',
    'MetricsDashboard',
    'setup_matplotlib',
    'create_figure',
    'save_plot',
    'animate_trajectory',
    'create_video',
    'setup_colors',
    'format_plot'
]

__version__ = "1.0.0"