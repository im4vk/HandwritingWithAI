"""
Real-time visualizer for robotic handwriting simulation.

This module provides live visualization of the handwriting process including
robot movement, pen trajectory, and performance metrics in real-time.
"""

import numpy as np
from typing import Dict, Any, Optional, List, Tuple
import time
import threading
import logging

from .base_visualizer import BaseVisualizer

logger = logging.getLogger(__name__)


class RealTimeVisualizer(BaseVisualizer):
    """
    Real-time visualizer for robotic handwriting simulation.
    
    Provides live display of:
    - Robot arm position and movement
    - Pen trajectory and writing path
    - Paper surface and contact points
    - Real-time performance metrics
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the real-time visualizer.
        
        Args:
            config: Configuration dictionary with real-time visualization settings
        """
        super().__init__(config)
        
        # Required fields for validation
        self.required_fields = ['pen_position', 'timestamp']
        
        # Display components
        self.show_robot = config.get('show_robot', True)
        self.show_trajectory = config.get('show_trajectory', True)
        self.show_paper = config.get('show_paper', True)
        self.show_metrics = config.get('show_metrics', True)
        
        # Trajectory display settings
        self.trajectory_color = config.get('trajectory_color', 'blue')
        self.trajectory_width = config.get('trajectory_width', 2)
        self.max_trajectory_points = config.get('max_trajectory_points', 500)
        self.fade_trajectory = config.get('fade_trajectory', True)
        
        # Robot display settings
        self.robot_color = config.get('robot_color', 'gray')
        self.pen_color = config.get('pen_color', 'red')
        self.contact_color = config.get('contact_color', 'green')
        
        # Paper display settings
        self.paper_color = config.get('paper_color', 'white')
        self.paper_border_color = config.get('paper_border_color', 'black')
        self.paper_size = config.get('paper_size', [0.21, 0.297])  # A4
        self.paper_position = config.get('paper_position', [0.5, 0.0, 0.01])
        
        # View settings
        self.view_mode = config.get('view_mode', '3d')  # '3d', 'top', 'side'
        self.camera_follow = config.get('camera_follow', True)
        self.zoom_level = config.get('zoom_level', 1.0)
        
        # Threading for smooth updates
        self.use_threading = config.get('use_threading', True)
        self.update_thread = None
        self.stop_thread = False
        
        # Data storage
        self.trajectory_points = []
        self.contact_points = []
        self.current_pen_position = np.zeros(3)
        self.current_robot_state = {}
        self.current_metrics = {}
        
        # Performance tracking
        self.fps_counter = 0
        self.last_fps_time = time.time()
        self.current_fps = 0
        
        # Matplotlib/plotting backend
        self.use_matplotlib = config.get('use_matplotlib', True)
        if self.use_matplotlib:
            self.setup_matplotlib()
    
    def setup_matplotlib(self):
        """Setup matplotlib for visualization."""
        try:
            import matplotlib
            matplotlib.use('TkAgg')  # Use Tkinter backend
            import matplotlib.pyplot as plt
            from mpl_toolkits.mplot3d import Axes3D
            
            self.plt = plt
            self.fig = None
            self.ax = None
            
            # Enable interactive mode
            plt.ion()
            
        except ImportError:
            logger.error("Matplotlib not available. Install matplotlib for visualization.")
            self.use_matplotlib = False
    
    def initialize(self) -> bool:
        """
        Initialize the real-time visualizer.
        
        Returns:
            bool: True if initialization successful
        """
        try:
            if self.use_matplotlib:
                self.setup_matplotlib_display()
            
            # Start update thread if enabled
            if self.use_threading:
                self.start_update_thread()
            
            self.is_initialized = True
            self.is_active = True
            
            if self.enable_logging:
                self.viz_logger.info("Real-time visualizer initialized")
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize real-time visualizer: {e}")
            return False
    
    def setup_matplotlib_display(self):
        """Setup matplotlib figure and axes."""
        if not self.use_matplotlib:
            return
        
        # Create figure
        self.fig = self.plt.figure(figsize=(self.width/100, self.height/100), dpi=self.dpi)
        self.fig.patch.set_facecolor(self.background_color)
        
        if self.view_mode == '3d':
            self.ax = self.fig.add_subplot(111, projection='3d')
            self.setup_3d_view()
        else:
            self.ax = self.fig.add_subplot(111)
            self.setup_2d_view()
        
        # Setup paper surface
        if self.show_paper:
            self.draw_paper_surface()
        
        # Initial plot setup
        self.ax.set_title('Robotic Handwriting - Real-time View')
        
        # Show the figure
        self.plt.show(block=False)
    
    def setup_3d_view(self):
        """Setup 3D view parameters."""
        if not self.ax:
            return
        
        # Set equal aspect ratio
        self.ax.set_box_aspect([1,1,0.5])
        
        # Set axis labels
        self.ax.set_xlabel('X (m)')
        self.ax.set_ylabel('Y (m)')
        self.ax.set_zlabel('Z (m)')
        
        # Set initial view limits
        self.ax.set_xlim([0.3, 0.7])
        self.ax.set_ylim([-0.2, 0.2])
        self.ax.set_zlim([0, 0.3])
        
        # Set viewing angle
        self.ax.view_init(elev=30, azim=45)
    
    def setup_2d_view(self):
        """Setup 2D view parameters."""
        if not self.ax:
            return
        
        self.ax.set_aspect('equal')
        self.ax.set_xlabel('X (m)')
        self.ax.set_ylabel('Y (m)')
        
        # Set limits based on view mode
        if self.view_mode == 'top':
            self.ax.set_xlim([0.3, 0.7])
            self.ax.set_ylim([-0.2, 0.2])
        elif self.view_mode == 'side':
            self.ax.set_xlim([0.3, 0.7])
            self.ax.set_ylim([0, 0.3])
    
    def draw_paper_surface(self):
        """Draw the paper surface on the plot."""
        if not self.ax or not self.show_paper:
            return
        
        paper_x = self.paper_position[0]
        paper_y = self.paper_position[1]
        paper_z = self.paper_position[2]
        
        width = self.paper_size[0]
        height = self.paper_size[1]
        
        if self.view_mode == '3d':
            # Draw 3D paper surface
            x_corners = [paper_x - width/2, paper_x + width/2, paper_x + width/2, paper_x - width/2]
            y_corners = [paper_y - height/2, paper_y - height/2, paper_y + height/2, paper_y + height/2]
            z_corners = [paper_z, paper_z, paper_z, paper_z]
            
            # Paper surface
            self.ax.plot_trisurf(x_corners + [x_corners[0]], 
                               y_corners + [y_corners[0]], 
                               z_corners + [z_corners[0]], 
                               color=self.paper_color, alpha=0.7)
            
            # Paper border
            self.ax.plot(x_corners + [x_corners[0]], 
                        y_corners + [y_corners[0]], 
                        z_corners + [z_corners[0]], 
                        color=self.paper_border_color, linewidth=2)
        
        else:
            # Draw 2D paper rectangle
            from matplotlib.patches import Rectangle
            
            if self.view_mode == 'top':
                rect = Rectangle((paper_x - width/2, paper_y - height/2), 
                               width, height, 
                               facecolor=self.paper_color, 
                               edgecolor=self.paper_border_color, 
                               linewidth=2)
                self.ax.add_patch(rect)
    
    def update(self, data: Dict[str, Any]) -> bool:
        """
        Update visualization with new simulation data.
        
        Args:
            data: Data dictionary containing simulation state
            
        Returns:
            bool: True if update successful
        """
        if not self.validate_data(data):
            return False
        
        # Check update rate
        current_time = time.time()
        if not self.should_update(current_time):
            return True
        
        try:
            # Process the data
            processed_data = self.process_data(data)
            
            # Update internal state
            self.update_internal_state(processed_data)
            
            # Add to buffer
            self.add_data(processed_data)
            
            # Update FPS counter
            self.update_fps_counter()
            
            # If not using threading, render immediately
            if not self.use_threading:
                self.render()
            
            return True
            
        except Exception as e:
            if self.enable_logging:
                self.viz_logger.error(f"Update failed: {e}")
            return False
    
    def update_internal_state(self, data: Dict[str, Any]):
        """Update internal state with new data."""
        # Update pen position
        if 'pen_position' in data:
            self.current_pen_position = np.array(data['pen_position'])
            self.trajectory_points.append(self.current_pen_position.copy())
            
            # Limit trajectory length
            if len(self.trajectory_points) > self.max_trajectory_points:
                self.trajectory_points.pop(0)
        
        # Update contact points
        if 'is_in_contact' in data and data['is_in_contact']:
            self.contact_points.append(self.current_pen_position.copy())
        
        # Update robot state
        if 'robot_state' in data:
            self.current_robot_state = data['robot_state']
        
        # Update metrics
        if 'metrics' in data:
            self.current_metrics = data['metrics']
    
    def render(self) -> bool:
        """
        Render the current visualization.
        
        Returns:
            bool: True if rendering successful
        """
        if not self.is_active or not self.use_matplotlib or not self.ax:
            return False
        
        try:
            # Clear previous plots (except paper)
            self.clear_dynamic_elements()
            
            # Draw trajectory
            if self.show_trajectory and len(self.trajectory_points) > 1:
                self.draw_trajectory()
            
            # Draw pen position
            self.draw_pen()
            
            # Draw robot (if enabled)
            if self.show_robot:
                self.draw_robot()
            
            # Draw contact points
            self.draw_contact_points()
            
            # Update metrics display
            if self.show_metrics:
                self.update_metrics_display()
            
            # Update camera if following
            if self.camera_follow:
                self.update_camera()
            
            # Refresh display
            self.plt.draw()
            self.plt.pause(0.001)
            
            self.frame_count += 1
            return True
            
        except Exception as e:
            if self.enable_logging:
                self.viz_logger.error(f"Render failed: {e}")
            return False
    
    def clear_dynamic_elements(self):
        """Clear dynamic elements from the plot."""
        # Keep only the paper surface and static elements
        # This is a simplified approach - in practice, you'd track plot objects
        pass
    
    def draw_trajectory(self):
        """Draw the pen trajectory."""
        if len(self.trajectory_points) < 2:
            return
        
        trajectory = np.array(self.trajectory_points)
        
        if self.view_mode == '3d':
            # 3D trajectory
            if self.fade_trajectory:
                # Draw with fading colors
                for i in range(1, len(trajectory)):
                    alpha = i / len(trajectory)
                    self.ax.plot([trajectory[i-1, 0], trajectory[i, 0]],
                               [trajectory[i-1, 1], trajectory[i, 1]],
                               [trajectory[i-1, 2], trajectory[i, 2]],
                               color=self.trajectory_color, 
                               alpha=alpha,
                               linewidth=self.trajectory_width)
            else:
                self.ax.plot(trajectory[:, 0], trajectory[:, 1], trajectory[:, 2],
                           color=self.trajectory_color, 
                           linewidth=self.trajectory_width)
        else:
            # 2D trajectory
            if self.view_mode == 'top':
                self.ax.plot(trajectory[:, 0], trajectory[:, 1],
                           color=self.trajectory_color, 
                           linewidth=self.trajectory_width)
            elif self.view_mode == 'side':
                self.ax.plot(trajectory[:, 0], trajectory[:, 2],
                           color=self.trajectory_color, 
                           linewidth=self.trajectory_width)
    
    def draw_pen(self):
        """Draw the current pen position."""
        if len(self.current_pen_position) == 0:
            return
        
        pos = self.current_pen_position
        
        if self.view_mode == '3d':
            self.ax.scatter([pos[0]], [pos[1]], [pos[2]], 
                          color=self.pen_color, s=50, marker='o')
        else:
            if self.view_mode == 'top':
                self.ax.scatter([pos[0]], [pos[1]], 
                              color=self.pen_color, s=50, marker='o')
            elif self.view_mode == 'side':
                self.ax.scatter([pos[0]], [pos[2]], 
                              color=self.pen_color, s=50, marker='o')
    
    def draw_robot(self):
        """Draw a simplified robot representation."""
        # Simplified robot drawing - just show base and links
        # In practice, this would use the actual robot geometry
        
        if not self.current_robot_state:
            return
        
        # Draw robot base (simplified)
        base_pos = self.current_robot_state.get('base_position', [0, 0, 0.1])
        
        if self.view_mode == '3d':
            self.ax.scatter([base_pos[0]], [base_pos[1]], [base_pos[2]], 
                          color=self.robot_color, s=100, marker='s')
            
            # Draw line from base to pen (simplified arm)
            if len(self.current_pen_position) > 0:
                self.ax.plot([base_pos[0], self.current_pen_position[0]],
                           [base_pos[1], self.current_pen_position[1]],
                           [base_pos[2], self.current_pen_position[2]],
                           color=self.robot_color, linewidth=3)
    
    def draw_contact_points(self):
        """Draw points where pen contacted the paper."""
        if len(self.contact_points) == 0:
            return
        
        contact_array = np.array(self.contact_points)
        
        if self.view_mode == '3d':
            self.ax.scatter(contact_array[:, 0], contact_array[:, 1], contact_array[:, 2],
                          color=self.contact_color, s=20, marker='.')
        else:
            if self.view_mode == 'top':
                self.ax.scatter(contact_array[:, 0], contact_array[:, 1],
                              color=self.contact_color, s=20, marker='.')
            elif self.view_mode == 'side':
                self.ax.scatter(contact_array[:, 0], contact_array[:, 2],
                              color=self.contact_color, s=20, marker='.')
    
    def update_metrics_display(self):
        """Update the metrics display on the plot."""
        if not self.current_metrics:
            return
        
        # Create text display for metrics
        metrics_text = f"FPS: {self.current_fps:.1f}\n"
        
        for key, value in self.current_metrics.items():
            if isinstance(value, (int, float)):
                metrics_text += f"{key}: {value:.3f}\n"
        
        # Add text to plot
        self.ax.text(0.02, 0.98, metrics_text, 
                    transform=self.ax.transAxes, 
                    verticalalignment='top',
                    bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    def update_camera(self):
        """Update camera position to follow the pen."""
        if len(self.current_pen_position) == 0 or self.view_mode != '3d':
            return
        
        # Center view on pen position
        pos = self.current_pen_position
        
        # Adjust view limits to follow pen
        margin = 0.1
        self.ax.set_xlim([pos[0] - margin, pos[0] + margin])
        self.ax.set_ylim([pos[1] - margin, pos[1] + margin])
        self.ax.set_zlim([pos[2] - margin/2, pos[2] + margin])
    
    def update_fps_counter(self):
        """Update FPS counter."""
        current_time = time.time()
        self.fps_counter += 1
        
        if current_time - self.last_fps_time >= 1.0:  # Update every second
            self.current_fps = self.fps_counter / (current_time - self.last_fps_time)
            self.fps_counter = 0
            self.last_fps_time = current_time
    
    def start_update_thread(self):
        """Start the update thread for smooth rendering."""
        if self.update_thread is not None:
            return
        
        self.stop_thread = False
        self.update_thread = threading.Thread(target=self._update_loop, daemon=True)
        self.update_thread.start()
        
        if self.enable_logging:
            self.viz_logger.info("Update thread started")
    
    def stop_update_thread(self):
        """Stop the update thread."""
        if self.update_thread is None:
            return
        
        self.stop_thread = True
        self.update_thread.join(timeout=1.0)
        self.update_thread = None
        
        if self.enable_logging:
            self.viz_logger.info("Update thread stopped")
    
    def _update_loop(self):
        """Main update loop for the thread."""
        while not self.stop_thread and self.is_active:
            try:
                if len(self.data_buffer) > 0:
                    self.render()
                
                # Sleep to maintain update rate
                time.sleep(1.0 / self.update_rate)
                
            except Exception as e:
                if self.enable_logging:
                    self.viz_logger.error(f"Update loop error: {e}")
                break
    
    def save_frame(self, filename: str) -> bool:
        """
        Save current frame to file.
        
        Args:
            filename: Output filename
            
        Returns:
            bool: True if save successful
        """
        if not self.use_matplotlib or not self.fig:
            return False
        
        try:
            self.fig.savefig(filename, dpi=self.dpi, bbox_inches='tight')
            if self.enable_logging:
                self.viz_logger.info(f"Frame saved to {filename}")
            return True
        except Exception as e:
            if self.enable_logging:
                self.viz_logger.error(f"Failed to save frame: {e}")
            return False
    
    def set_view_mode(self, mode: str):
        """
        Set the view mode.
        
        Args:
            mode: View mode ('3d', 'top', 'side')
        """
        if mode in ['3d', 'top', 'side']:
            self.view_mode = mode
            if self.is_initialized:
                # Reinitialize display with new view mode
                self.setup_matplotlib_display()
    
    def toggle_component(self, component: str):
        """
        Toggle display of a component.
        
        Args:
            component: Component to toggle ('robot', 'trajectory', 'paper', 'metrics')
        """
        if component == 'robot':
            self.show_robot = not self.show_robot
        elif component == 'trajectory':
            self.show_trajectory = not self.show_trajectory
        elif component == 'paper':
            self.show_paper = not self.show_paper
        elif component == 'metrics':
            self.show_metrics = not self.show_metrics
    
    def clear_trajectory(self):
        """Clear the trajectory display."""
        self.trajectory_points.clear()
        self.contact_points.clear()
    
    def close(self):
        """Close the real-time visualizer."""
        self.is_active = False
        
        # Stop update thread
        if self.use_threading:
            self.stop_update_thread()
        
        # Close matplotlib
        if self.use_matplotlib and self.plt:
            self.plt.close('all')
        
        if self.enable_logging:
            self.viz_logger.info("Real-time visualizer closed")