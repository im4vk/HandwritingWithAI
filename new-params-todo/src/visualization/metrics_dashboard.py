"""
Metrics dashboard for robotic handwriting visualization.

This module provides real-time and historical visualization of performance
metrics, quality indicators, and system diagnostics.
"""

import numpy as np
from typing import Dict, Any, Optional, List, Tuple, Union
import time
import logging

from .base_visualizer import BaseVisualizer

logger = logging.getLogger(__name__)


class MetricsDashboard(BaseVisualizer):
    """
    Comprehensive metrics dashboard for robotic handwriting analysis.
    
    Provides visualization of:
    - Real-time performance metrics
    - Quality assessment indicators
    - Historical trend analysis
    - System diagnostics and alerts
    - Comparative performance analysis
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the metrics dashboard.
        
        Args:
            config: Configuration dictionary with dashboard settings
        """
        super().__init__(config)
        
        # Required fields for validation
        self.required_fields = ['metrics']
        
        # Dashboard layout
        self.layout_type = config.get('layout_type', 'grid')  # 'grid', 'tabs', 'single'
        self.grid_size = config.get('grid_size', (3, 3))
        self.panel_spacing = config.get('panel_spacing', 0.1)
        
        # Metrics configuration
        self.tracked_metrics = config.get('tracked_metrics', [
            'position_error', 'velocity', 'acceleration', 'smoothness',
            'pressure', 'contact_force', 'quality_score'
        ])
        self.metric_units = config.get('metric_units', {
            'position_error': 'm',
            'velocity': 'm/s',
            'acceleration': 'm/s²',
            'smoothness': 'unitless',
            'pressure': 'N',
            'contact_force': 'N',
            'quality_score': 'unitless'
        })
        self.metric_ranges = config.get('metric_ranges', {
            'position_error': [0, 0.01],
            'velocity': [0, 0.5],
            'acceleration': [0, 5.0],
            'smoothness': [0, 1],
            'pressure': [0, 10],
            'contact_force': [0, 5],
            'quality_score': [0, 1]
        })
        
        # Display settings
        self.show_real_time = config.get('show_real_time', True)
        self.show_historical = config.get('show_historical', True)
        self.show_statistics = config.get('show_statistics', True)
        self.show_alerts = config.get('show_alerts', True)
        
        # Time window settings
        self.time_window = config.get('time_window', 30.0)  # seconds
        self.history_length = config.get('history_length', 1000)  # points
        
        # Alert thresholds
        self.alert_thresholds = config.get('alert_thresholds', {
            'position_error': {'warning': 0.005, 'critical': 0.01},
            'velocity': {'warning': 0.3, 'critical': 0.5},
            'acceleration': {'warning': 3.0, 'critical': 5.0},
            'quality_score': {'warning': 0.5, 'critical': 0.3}
        })
        
        # Colors and styling
        self.color_scheme = config.get('color_scheme', 'default')
        self.colors = config.get('colors', {
            'normal': 'green',
            'warning': 'orange',
            'critical': 'red',
            'background': 'white',
            'grid': 'lightgray',
            'text': 'black'
        })
        
        # Data storage
        self.metrics_history = {metric: [] for metric in self.tracked_metrics}
        self.timestamps = []
        self.current_metrics = {}
        self.alerts = []
        self.statistics = {}
        
        # Dashboard components
        self.panels = {}
        self.plots = {}
        
        # Matplotlib setup
        self.use_matplotlib = config.get('use_matplotlib', True)
        if self.use_matplotlib:
            self.setup_matplotlib()
    
    def setup_matplotlib(self):
        """Setup matplotlib for dashboard visualization."""
        try:
            import matplotlib.pyplot as plt
            import matplotlib.patches as patches
            from matplotlib.gridspec import GridSpec
            import matplotlib.animation as animation
            
            self.plt = plt
            self.patches = patches
            self.GridSpec = GridSpec
            self.animation = animation
            
            # Set style
            if self.color_scheme == 'dark':
                plt.style.use('dark_background')
            
            # Configure interactive mode
            plt.ion()
            
        except ImportError:
            logger.error("Matplotlib not available for dashboard visualization")
            self.use_matplotlib = False
    
    def initialize(self) -> bool:
        """
        Initialize the metrics dashboard.
        
        Returns:
            bool: True if initialization successful
        """
        try:
            if self.use_matplotlib:
                self.create_dashboard_layout()
            
            self.is_initialized = True
            self.is_active = True
            
            if self.enable_logging:
                self.viz_logger.info("Metrics dashboard initialized")
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize metrics dashboard: {e}")
            return False
    
    def create_dashboard_layout(self):
        """Create the dashboard layout with multiple panels."""
        if not self.use_matplotlib:
            return
        
        # Create main figure
        self.fig = self.plt.figure(figsize=(15, 10))
        self.fig.suptitle('Robotic Handwriting - Metrics Dashboard', fontsize=16)
        
        if self.layout_type == 'grid':
            self.create_grid_layout()
        elif self.layout_type == 'tabs':
            self.create_tab_layout()
        else:
            self.create_single_layout()
        
        # Show the dashboard
        self.plt.show(block=False)
    
    def create_grid_layout(self):
        """Create grid-based layout."""
        rows, cols = self.grid_size
        gs = self.GridSpec(rows, cols, figure=self.fig, hspace=0.3, wspace=0.3)
        
        # Define panel assignments
        panel_assignments = [
            ('real_time_metrics', 0, 0, 1, 2),      # Top left, spans 2 columns
            ('quality_trends', 0, 2, 1, 1),         # Top right
            ('velocity_profile', 1, 0, 1, 1),       # Middle left
            ('acceleration_profile', 1, 1, 1, 1),   # Middle center
            ('pressure_analysis', 1, 2, 1, 1),      # Middle right
            ('statistics_panel', 2, 0, 1, 1),       # Bottom left
            ('alerts_panel', 2, 1, 1, 1),           # Bottom center
            ('system_status', 2, 2, 1, 1)           # Bottom right
        ]
        
        # Create panels
        for panel_name, row, col, row_span, col_span in panel_assignments:
            if row < rows and col < cols:
                ax = self.fig.add_subplot(gs[row:row+row_span, col:col+col_span])
                self.panels[panel_name] = ax
                self.setup_panel(panel_name, ax)
    
    def create_tab_layout(self):
        """Create tab-based layout (simplified as subplots)."""
        # For matplotlib, simulate tabs with subplots
        self.create_grid_layout()
    
    def create_single_layout(self):
        """Create single panel layout."""
        ax = self.fig.add_subplot(111)
        self.panels['main'] = ax
        self.setup_panel('main', ax)
    
    def setup_panel(self, panel_name: str, ax):
        """Setup individual panel properties."""
        if panel_name == 'real_time_metrics':
            ax.set_title('Real-time Metrics')
            ax.set_ylabel('Value')
            ax.grid(True, alpha=0.3)
            
        elif panel_name == 'quality_trends':
            ax.set_title('Quality Trends')
            ax.set_ylabel('Quality Score')
            ax.set_ylim(0, 1)
            ax.grid(True, alpha=0.3)
            
        elif panel_name == 'velocity_profile':
            ax.set_title('Velocity Profile')
            ax.set_ylabel('Velocity (m/s)')
            ax.grid(True, alpha=0.3)
            
        elif panel_name == 'acceleration_profile':
            ax.set_title('Acceleration Profile')
            ax.set_ylabel('Acceleration (m/s²)')
            ax.grid(True, alpha=0.3)
            
        elif panel_name == 'pressure_analysis':
            ax.set_title('Pressure Analysis')
            ax.set_ylabel('Pressure (N)')
            ax.grid(True, alpha=0.3)
            
        elif panel_name == 'statistics_panel':
            ax.set_title('Statistics')
            ax.axis('off')  # Text-based panel
            
        elif panel_name == 'alerts_panel':
            ax.set_title('Alerts & Warnings')
            ax.axis('off')  # Text-based panel
            
        elif panel_name == 'system_status':
            ax.set_title('System Status')
            ax.axis('off')  # Text-based panel
        
        # Store reference for later updates
        self.plots[panel_name] = {'ax': ax, 'lines': [], 'texts': []}
    
    def update(self, data: Dict[str, Any]) -> bool:
        """
        Update dashboard with new metrics data.
        
        Args:
            data: Data dictionary containing metrics
            
        Returns:
            bool: True if update successful
        """
        if not self.validate_data(data):
            return False
        
        try:
            # Process the data
            processed_data = self.process_data(data)
            
            # Update metrics history
            self.update_metrics_history(processed_data)
            
            # Check for alerts
            self.check_alerts(processed_data)
            
            # Update statistics
            self.update_statistics()
            
            # Add to buffer
            self.add_data(processed_data)
            
            return True
            
        except Exception as e:
            if self.enable_logging:
                self.viz_logger.error(f"Update failed: {e}")
            return False
    
    def update_metrics_history(self, data: Dict[str, Any]):
        """Update metrics history with new data."""
        current_time = time.time()
        self.timestamps.append(current_time)
        
        # Extract metrics from data
        metrics = data.get('metrics', {})
        self.current_metrics = metrics
        
        # Update history for each tracked metric
        for metric in self.tracked_metrics:
            value = metrics.get(metric, 0.0)
            self.metrics_history[metric].append(value)
        
        # Maintain history length
        if len(self.timestamps) > self.history_length:
            self.timestamps.pop(0)
            for metric in self.tracked_metrics:
                if len(self.metrics_history[metric]) > self.history_length:
                    self.metrics_history[metric].pop(0)
    
    def check_alerts(self, data: Dict[str, Any]):
        """Check for alert conditions and update alerts list."""
        metrics = data.get('metrics', {})
        current_time = time.time()
        
        new_alerts = []
        
        for metric, value in metrics.items():
            if metric in self.alert_thresholds:
                thresholds = self.alert_thresholds[metric]
                
                alert_level = None
                if value > thresholds.get('critical', float('inf')):
                    alert_level = 'critical'
                elif value > thresholds.get('warning', float('inf')):
                    alert_level = 'warning'
                
                # For quality score, alerts are for low values
                if metric == 'quality_score':
                    if value < thresholds.get('critical', 0):
                        alert_level = 'critical'
                    elif value < thresholds.get('warning', 0):
                        alert_level = 'warning'
                
                if alert_level:
                    alert = {
                        'time': current_time,
                        'metric': metric,
                        'value': value,
                        'level': alert_level,
                        'message': f"{metric}: {value:.3f} ({alert_level})"
                    }
                    new_alerts.append(alert)
        
        # Add new alerts and maintain alert history
        self.alerts.extend(new_alerts)
        
        # Keep only recent alerts (last 50)
        if len(self.alerts) > 50:
            self.alerts = self.alerts[-50:]
    
    def update_statistics(self):
        """Update statistical summaries of metrics."""
        self.statistics = {}
        
        for metric in self.tracked_metrics:
            history = self.metrics_history[metric]
            if len(history) > 0:
                self.statistics[metric] = {
                    'current': history[-1],
                    'mean': np.mean(history),
                    'std': np.std(history),
                    'min': np.min(history),
                    'max': np.max(history),
                    'trend': self.calculate_trend(history)
                }
    
    def calculate_trend(self, data: List[float], window: int = 10) -> str:
        """Calculate trend direction for a metric."""
        if len(data) < window:
            return 'stable'
        
        recent = data[-window:]
        earlier = data[-2*window:-window] if len(data) >= 2*window else data[:-window]
        
        if len(earlier) == 0:
            return 'stable'
        
        recent_mean = np.mean(recent)
        earlier_mean = np.mean(earlier)
        
        diff = recent_mean - earlier_mean
        threshold = np.std(data) * 0.1  # 10% of standard deviation
        
        if diff > threshold:
            return 'increasing'
        elif diff < -threshold:
            return 'decreasing'
        else:
            return 'stable'
    
    def render(self) -> bool:
        """
        Render the metrics dashboard.
        
        Returns:
            bool: True if rendering successful
        """
        if not self.is_active or not self.use_matplotlib:
            return False
        
        try:
            # Update each panel
            for panel_name, panel_info in self.plots.items():
                self.update_panel(panel_name)
            
            # Refresh display
            self.plt.draw()
            self.plt.pause(0.001)
            
            self.frame_count += 1
            return True
            
        except Exception as e:
            if self.enable_logging:
                self.viz_logger.error(f"Render failed: {e}")
            return False
    
    def update_panel(self, panel_name: str):
        """Update individual panel content."""
        if panel_name not in self.plots:
            return
        
        ax = self.plots[panel_name]['ax']
        
        if panel_name == 'real_time_metrics':
            self.update_real_time_panel(ax)
        elif panel_name == 'quality_trends':
            self.update_quality_panel(ax)
        elif panel_name == 'velocity_profile':
            self.update_velocity_panel(ax)
        elif panel_name == 'acceleration_profile':
            self.update_acceleration_panel(ax)
        elif panel_name == 'pressure_analysis':
            self.update_pressure_panel(ax)
        elif panel_name == 'statistics_panel':
            self.update_statistics_panel(ax)
        elif panel_name == 'alerts_panel':
            self.update_alerts_panel(ax)
        elif panel_name == 'system_status':
            self.update_status_panel(ax)
    
    def update_real_time_panel(self, ax):
        """Update real-time metrics panel."""
        ax.clear()
        ax.set_title('Real-time Metrics')
        ax.set_ylabel('Value')
        
        if len(self.timestamps) == 0:
            return
        
        # Show last 100 points
        time_window = self.timestamps[-100:] if len(self.timestamps) > 100 else self.timestamps
        
        # Convert to relative time
        if len(time_window) > 0:
            rel_time = [(t - time_window[0]) for t in time_window]
            
            # Plot key metrics
            key_metrics = ['position_error', 'velocity', 'quality_score']
            colors = ['red', 'blue', 'green']
            
            for i, metric in enumerate(key_metrics):
                if metric in self.metrics_history:
                    history = self.metrics_history[metric]
                    metric_window = history[-len(time_window):] if len(history) >= len(time_window) else history
                    
                    if len(metric_window) > 0:
                        ax.plot(rel_time[:len(metric_window)], metric_window, 
                               color=colors[i], label=metric, linewidth=2)
        
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_xlabel('Time (s)')
    
    def update_quality_panel(self, ax):
        """Update quality trends panel."""
        ax.clear()
        ax.set_title('Quality Trends')
        ax.set_ylabel('Quality Score')
        ax.set_ylim(0, 1)
        
        if 'quality_score' in self.metrics_history and len(self.metrics_history['quality_score']) > 0:
            history = self.metrics_history['quality_score']
            rel_time = list(range(len(history)))
            
            ax.plot(rel_time, history, color='purple', linewidth=2)
            
            # Add horizontal lines for thresholds
            if 'quality_score' in self.alert_thresholds:
                thresholds = self.alert_thresholds['quality_score']
                if 'warning' in thresholds:
                    ax.axhline(y=thresholds['warning'], color='orange', linestyle='--', label='Warning')
                if 'critical' in thresholds:
                    ax.axhline(y=thresholds['critical'], color='red', linestyle='--', label='Critical')
                ax.legend()
        
        ax.grid(True, alpha=0.3)
        ax.set_xlabel('Time Steps')
    
    def update_velocity_panel(self, ax):
        """Update velocity profile panel."""
        ax.clear()
        ax.set_title('Velocity Profile')
        ax.set_ylabel('Velocity (m/s)')
        
        if 'velocity' in self.metrics_history and len(self.metrics_history['velocity']) > 0:
            history = self.metrics_history['velocity']
            rel_time = list(range(len(history)))
            
            ax.plot(rel_time, history, color='blue', linewidth=2)
            
            # Add mean line
            if len(history) > 1:
                mean_vel = np.mean(history)
                ax.axhline(y=mean_vel, color='blue', linestyle=':', alpha=0.7, label=f'Mean: {mean_vel:.3f}')
                ax.legend()
        
        ax.grid(True, alpha=0.3)
        ax.set_xlabel('Time Steps')
    
    def update_acceleration_panel(self, ax):
        """Update acceleration profile panel."""
        ax.clear()
        ax.set_title('Acceleration Profile')
        ax.set_ylabel('Acceleration (m/s²)')
        
        if 'acceleration' in self.metrics_history and len(self.metrics_history['acceleration']) > 0:
            history = self.metrics_history['acceleration']
            rel_time = list(range(len(history)))
            
            ax.plot(rel_time, history, color='orange', linewidth=2)
            
            # Add threshold lines
            if 'acceleration' in self.alert_thresholds:
                thresholds = self.alert_thresholds['acceleration']
                if 'warning' in thresholds:
                    ax.axhline(y=thresholds['warning'], color='orange', linestyle='--', alpha=0.7)
                if 'critical' in thresholds:
                    ax.axhline(y=thresholds['critical'], color='red', linestyle='--', alpha=0.7)
        
        ax.grid(True, alpha=0.3)
        ax.set_xlabel('Time Steps')
    
    def update_pressure_panel(self, ax):
        """Update pressure analysis panel."""
        ax.clear()
        ax.set_title('Pressure Analysis')
        ax.set_ylabel('Pressure (N)')
        
        if 'pressure' in self.metrics_history and len(self.metrics_history['pressure']) > 0:
            history = self.metrics_history['pressure']
            rel_time = list(range(len(history)))
            
            ax.plot(rel_time, history, color='green', linewidth=2)
            
            # Add statistics
            if len(history) > 1:
                mean_pressure = np.mean(history)
                std_pressure = np.std(history)
                ax.axhline(y=mean_pressure, color='green', linestyle=':', alpha=0.7)
                ax.fill_between(rel_time, mean_pressure - std_pressure, mean_pressure + std_pressure, 
                               alpha=0.2, color='green')
        
        ax.grid(True, alpha=0.3)
        ax.set_xlabel('Time Steps')
    
    def update_statistics_panel(self, ax):
        """Update statistics panel with text information."""
        ax.clear()
        ax.set_title('Statistics')
        ax.axis('off')
        
        if not self.statistics:
            ax.text(0.5, 0.5, 'No statistics available', ha='center', va='center', transform=ax.transAxes)
            return
        
        # Create statistics text
        stats_text = ""
        for metric, stats in self.statistics.items():
            stats_text += f"{metric}:\n"
            stats_text += f"  Current: {stats['current']:.3f}\n"
            stats_text += f"  Mean: {stats['mean']:.3f}\n"
            stats_text += f"  Std: {stats['std']:.3f}\n"
            stats_text += f"  Trend: {stats['trend']}\n\n"
        
        ax.text(0.05, 0.95, stats_text, ha='left', va='top', transform=ax.transAxes, 
               fontsize=9, fontfamily='monospace')
    
    def update_alerts_panel(self, ax):
        """Update alerts panel with recent alerts."""
        ax.clear()
        ax.set_title('Alerts & Warnings')
        ax.axis('off')
        
        if not self.alerts:
            ax.text(0.5, 0.5, 'No active alerts', ha='center', va='center', 
                   transform=ax.transAxes, color='green', fontweight='bold')
            return
        
        # Show recent alerts (last 5)
        recent_alerts = self.alerts[-5:]
        alerts_text = ""
        
        for alert in recent_alerts:
            color = 'red' if alert['level'] == 'critical' else 'orange'
            alerts_text += f"[{alert['level'].upper()}] {alert['message']}\n"
        
        ax.text(0.05, 0.95, alerts_text, ha='left', va='top', transform=ax.transAxes, 
               fontsize=9, fontfamily='monospace')
    
    def update_status_panel(self, ax):
        """Update system status panel."""
        ax.clear()
        ax.set_title('System Status')
        ax.axis('off')
        
        # System status information
        status_text = f"Dashboard Status: Active\n"
        status_text += f"Update Rate: {self.update_rate} FPS\n"
        status_text += f"Data Points: {len(self.timestamps)}\n"
        status_text += f"Active Alerts: {len([a for a in self.alerts if time.time() - a['time'] < 60])}\n"
        status_text += f"Frame Count: {self.frame_count}\n"
        
        # Current metric values
        if self.current_metrics:
            status_text += "\nCurrent Values:\n"
            for metric, value in self.current_metrics.items():
                unit = self.metric_units.get(metric, '')
                status_text += f"{metric}: {value:.3f} {unit}\n"
        
        ax.text(0.05, 0.95, status_text, ha='left', va='top', transform=ax.transAxes, 
               fontsize=9, fontfamily='monospace')
    
    def save_dashboard(self, filename: str, **kwargs) -> bool:
        """
        Save current dashboard to file.
        
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
                self.viz_logger.info(f"Dashboard saved to {filename}")
            return True
        except Exception as e:
            if self.enable_logging:
                self.viz_logger.error(f"Failed to save dashboard: {e}")
            return False
    
    def export_metrics_data(self, filename: str) -> bool:
        """
        Export metrics data to CSV file.
        
        Args:
            filename: Output CSV filename
            
        Returns:
            bool: True if export successful
        """
        try:
            import csv
            
            with open(filename, 'w', newline='') as csvfile:
                writer = csv.writer(csvfile)
                
                # Write header
                header = ['timestamp'] + self.tracked_metrics
                writer.writerow(header)
                
                # Write data
                for i, timestamp in enumerate(self.timestamps):
                    row = [timestamp]
                    for metric in self.tracked_metrics:
                        if i < len(self.metrics_history[metric]):
                            row.append(self.metrics_history[metric][i])
                        else:
                            row.append('')
                    writer.writerow(row)
            
            if self.enable_logging:
                self.viz_logger.info(f"Metrics data exported to {filename}")
            return True
            
        except Exception as e:
            if self.enable_logging:
                self.viz_logger.error(f"Failed to export metrics data: {e}")
            return False
    
    def get_dashboard_summary(self) -> Dict[str, Any]:
        """
        Get dashboard summary information.
        
        Returns:
            Dict containing dashboard summary
        """
        summary = {
            'tracked_metrics': self.tracked_metrics,
            'data_points': len(self.timestamps),
            'active_alerts': len([a for a in self.alerts if time.time() - a['time'] < 60]),
            'total_alerts': len(self.alerts),
            'frame_count': self.frame_count,
            'current_metrics': self.current_metrics,
            'statistics': self.statistics
        }
        
        return summary
    
    def reset_dashboard(self):
        """Reset dashboard data and displays."""
        # Clear data
        self.metrics_history = {metric: [] for metric in self.tracked_metrics}
        self.timestamps = []
        self.current_metrics = {}
        self.alerts = []
        self.statistics = {}
        
        # Reset frame counter
        self.frame_count = 0
        
        if self.enable_logging:
            self.viz_logger.info("Dashboard reset")
    
    def close(self):
        """Close the metrics dashboard."""
        if self.use_matplotlib and hasattr(self, 'plt'):
            self.plt.close('all')
        
        self.is_active = False
        if self.enable_logging:
            self.viz_logger.info("Metrics dashboard closed")