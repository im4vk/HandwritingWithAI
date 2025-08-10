"""
3D robot and environment renderer for robotic handwriting.

This module provides 3D visualization of the robot arm, workspace,
paper surface, and handwriting environment using various rendering backends.
"""

import numpy as np
from typing import Dict, Any, Optional, List, Tuple, Union
import logging

from .base_visualizer import BaseVisualizer

logger = logging.getLogger(__name__)


class RobotRenderer(BaseVisualizer):
    """
    3D renderer for robotic handwriting environments.
    
    Provides visualization of:
    - Robot arm geometry and movement
    - Workspace boundaries
    - Paper surface and writing area
    - End-effector and pen representation
    - Environmental objects and constraints
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the robot renderer.
        
        Args:
            config: Configuration dictionary with rendering settings
        """
        super().__init__(config)
        
        # Required fields for validation
        self.required_fields = ['robot_state']
        
        # Rendering backend selection
        self.backend = config.get('backend', 'matplotlib')  # 'matplotlib', 'plotly', 'open3d'
        
        # Robot model settings
        self.robot_model_path = config.get('robot_model_path', None)
        self.link_colors = config.get('link_colors', ['gray', 'blue', 'red', 'green'])
        self.joint_colors = config.get('joint_colors', ['black'])
        self.show_joints = config.get('show_joints', True)
        self.show_links = config.get('show_links', True)
        self.show_frames = config.get('show_frames', False)
        
        # End-effector settings
        self.pen_length = config.get('pen_length', 0.15)
        self.pen_radius = config.get('pen_radius', 0.002)
        self.pen_color = config.get('pen_color', 'blue')
        self.show_pen_trajectory = config.get('show_pen_trajectory', True)
        
        # Environment settings
        self.paper_size = config.get('paper_size', [0.21, 0.297])  # A4
        self.paper_position = config.get('paper_position', [0.5, 0.0, 0.01])
        self.paper_color = config.get('paper_color', 'white')
        self.workspace_bounds = config.get('workspace_bounds', {
            'x': [0.2, 0.8], 'y': [-0.3, 0.3], 'z': [0.0, 0.3]
        })
        self.show_workspace = config.get('show_workspace', True)
        
        # Lighting and shading
        self.lighting_enabled = config.get('lighting_enabled', True)
        self.ambient_light = config.get('ambient_light', 0.3)
        self.directional_light = config.get('directional_light', 0.7)
        
        # Camera settings
        self.camera_position = config.get('camera_position', [1.0, 1.0, 0.5])
        self.camera_target = config.get('camera_target', [0.5, 0.0, 0.1])
        self.camera_up = config.get('camera_up', [0, 0, 1])
        self.field_of_view = config.get('field_of_view', 45)
        
        # Animation settings
        self.enable_animation = config.get('enable_animation', False)
        self.animation_speed = config.get('animation_speed', 1.0)
        
        # State tracking
        self.current_robot_state = {}
        self.robot_geometry = {}
        self.scene_objects = {}
        
        # Initialize backend
        self.setup_backend()
    
    def setup_backend(self):
        """Setup the selected rendering backend."""
        if self.backend == 'matplotlib':
            self.setup_matplotlib_backend()
        elif self.backend == 'plotly':
            self.setup_plotly_backend()
        elif self.backend == 'open3d':
            self.setup_open3d_backend()
        else:
            logger.warning(f"Unknown backend: {self.backend}, falling back to matplotlib")
            self.backend = 'matplotlib'
            self.setup_matplotlib_backend()
    
    def setup_matplotlib_backend(self):
        """Setup matplotlib 3D rendering."""
        try:
            import matplotlib.pyplot as plt
            from mpl_toolkits.mplot3d import Axes3D
            import matplotlib.patches as patches
            from mpl_toolkits.mplot3d.art3d import Poly3DCollection
            
            self.plt = plt
            self.patches = patches
            self.Poly3DCollection = Poly3DCollection
            
            # Set up interactive mode
            plt.ion()
            
        except ImportError:
            logger.error("Matplotlib not available for 3D rendering")
            self.backend = None
    
    def setup_plotly_backend(self):
        """Setup Plotly 3D rendering."""
        try:
            import plotly.graph_objects as go
            import plotly.express as px
            
            self.go = go
            self.px = px
            
        except ImportError:
            logger.error("Plotly not available. Install plotly for advanced 3D rendering.")
            self.backend = 'matplotlib'
            self.setup_matplotlib_backend()
    
    def setup_open3d_backend(self):
        """Setup Open3D rendering."""
        try:
            import open3d as o3d
            
            self.o3d = o3d
            
        except ImportError:
            logger.error("Open3D not available. Install open3d for high-quality 3D rendering.")
            self.backend = 'matplotlib'
            self.setup_matplotlib_backend()
    
    def initialize(self) -> bool:
        """
        Initialize the robot renderer.
        
        Returns:
            bool: True if initialization successful
        """
        try:
            if self.backend is None:
                return False
            
            # Initialize scene
            self.create_scene()
            
            # Load robot model
            if self.robot_model_path:
                self.load_robot_model(self.robot_model_path)
            else:
                self.create_default_robot()
            
            # Create environment objects
            self.create_environment()
            
            self.is_initialized = True
            self.is_active = True
            
            if self.enable_logging:
                self.viz_logger.info(f"Robot renderer initialized with {self.backend} backend")
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize robot renderer: {e}")
            return False
    
    def create_scene(self):
        """Create the 3D scene based on the backend."""
        if self.backend == 'matplotlib':
            self.fig = self.plt.figure(figsize=(self.width/100, self.height/100))
            self.ax = self.fig.add_subplot(111, projection='3d')
            self.setup_matplotlib_scene()
        
        elif self.backend == 'plotly':
            self.fig = self.go.Figure()
            self.setup_plotly_scene()
        
        elif self.backend == 'open3d':
            self.vis = self.o3d.visualization.Visualizer()
            self.vis.create_window(width=self.width, height=self.height)
            self.setup_open3d_scene()
    
    def setup_matplotlib_scene(self):
        """Setup matplotlib 3D scene."""
        # Set equal aspect ratio
        self.ax.set_box_aspect([1, 1, 0.5])
        
        # Set labels
        self.ax.set_xlabel('X (m)')
        self.ax.set_ylabel('Y (m)')
        self.ax.set_zlabel('Z (m)')
        
        # Set limits
        bounds = self.workspace_bounds
        self.ax.set_xlim(bounds['x'])
        self.ax.set_ylim(bounds['y'])
        self.ax.set_zlim(bounds['z'])
        
        # Set title
        self.ax.set_title('Robotic Handwriting System - 3D View')
        
        # Set viewing angle
        self.ax.view_init(elev=30, azim=45)
    
    def setup_plotly_scene(self):
        """Setup Plotly 3D scene."""
        bounds = self.workspace_bounds
        
        self.fig.update_layout(
            title='Robotic Handwriting System - 3D View',
            scene=dict(
                xaxis=dict(range=bounds['x'], title='X (m)'),
                yaxis=dict(range=bounds['y'], title='Y (m)'),
                zaxis=dict(range=bounds['z'], title='Z (m)'),
                aspectmode='cube',
                camera=dict(
                    eye=dict(x=1.5, y=1.5, z=1.0),
                    center=dict(x=0, y=0, z=0),
                    up=dict(x=0, y=0, z=1)
                )
            ),
            width=self.width,
            height=self.height
        )
    
    def setup_open3d_scene(self):
        """Setup Open3D scene."""
        # Create coordinate frame
        coord_frame = self.o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1)
        self.vis.add_geometry(coord_frame)
        
        # Setup camera
        ctr = self.vis.get_view_control()
        ctr.set_front(self.camera_position)
        ctr.set_lookat(self.camera_target)
        ctr.set_up(self.camera_up)
    
    def load_robot_model(self, model_path: str) -> bool:
        """
        Load robot model from file.
        
        Args:
            model_path: Path to robot model file
            
        Returns:
            bool: True if model loaded successfully
        """
        try:
            # This would load actual robot geometry from URDF/STL files
            # For now, create a simplified model
            self.create_default_robot()
            return True
        except Exception as e:
            logger.error(f"Failed to load robot model: {e}")
            return False
    
    def create_default_robot(self):
        """Create a default robot arm geometry."""
        # Define a simple 2-DOF robot arm
        self.robot_geometry = {
            'base': {
                'type': 'cylinder',
                'radius': 0.05,
                'height': 0.1,
                'position': [0, 0, 0.05],
                'color': self.link_colors[0]
            },
            'link1': {
                'type': 'cylinder',
                'radius': 0.02,
                'height': 0.3,
                'position': [0, 0, 0.15],  # Will be updated based on joint angles
                'color': self.link_colors[1]
            },
            'link2': {
                'type': 'cylinder',
                'radius': 0.015,
                'height': 0.25,
                'position': [0, 0, 0.4],  # Will be updated based on joint angles
                'color': self.link_colors[2]
            },
            'pen': {
                'type': 'cylinder',
                'radius': self.pen_radius,
                'height': self.pen_length,
                'position': [0, 0, 0.6],  # Will be updated based on kinematics
                'color': self.pen_color
            }
        }
        
        # Define joint positions
        self.joint_positions = {
            'base_joint': [0, 0, 0.1],
            'joint1': [0, 0, 0.3],
            'joint2': [0, 0, 0.55],
            'end_effector': [0, 0, 0.7]
        }
    
    def create_environment(self):
        """Create environment objects (paper, workspace, etc.)."""
        # Paper surface
        self.scene_objects['paper'] = {
            'type': 'box',
            'size': [self.paper_size[0], self.paper_size[1], 0.001],
            'position': self.paper_position,
            'color': self.paper_color
        }
        
        # Workspace boundaries (if enabled)
        if self.show_workspace:
            bounds = self.workspace_bounds
            self.scene_objects['workspace'] = {
                'type': 'wireframe_box',
                'bounds': bounds,
                'color': 'gray',
                'alpha': 0.3
            }
    
    def update(self, data: Dict[str, Any]) -> bool:
        """
        Update robot renderer with new state data.
        
        Args:
            data: Data dictionary containing robot state
            
        Returns:
            bool: True if update successful
        """
        if not self.validate_data(data):
            return False
        
        try:
            # Process the data
            processed_data = self.process_data(data)
            
            # Update robot state
            self.update_robot_state(processed_data)
            
            # Add to buffer
            self.add_data(processed_data)
            
            return True
            
        except Exception as e:
            if self.enable_logging:
                self.viz_logger.error(f"Update failed: {e}")
            return False
    
    def update_robot_state(self, data: Dict[str, Any]):
        """Update internal robot state."""
        if 'robot_state' in data:
            self.current_robot_state = data['robot_state']
            
            # Update robot geometry based on joint angles
            self.update_robot_geometry()
    
    def update_robot_geometry(self):
        """Update robot geometry based on current joint states."""
        # Simple forward kinematics for 2-DOF arm
        if 'joint_angles' in self.current_robot_state:
            angles = self.current_robot_state['joint_angles']
            
            # Update link positions based on joint angles
            # This is simplified - real implementation would use proper FK
            if len(angles) >= 2:
                # Base link (fixed)
                base_pos = [0, 0, 0.05]
                
                # Link 1
                l1_length = 0.3
                link1_end = [
                    base_pos[0] + l1_length * np.cos(angles[0]),
                    base_pos[1] + l1_length * np.sin(angles[0]),
                    base_pos[2] + 0.1
                ]
                
                # Link 2
                l2_length = 0.25
                link2_end = [
                    link1_end[0] + l2_length * np.cos(angles[0] + angles[1]),
                    link1_end[1] + l2_length * np.sin(angles[0] + angles[1]),
                    link1_end[2]
                ]
                
                # Update geometry
                self.robot_geometry['link1']['position'] = [
                    (base_pos[0] + link1_end[0]) / 2,
                    (base_pos[1] + link1_end[1]) / 2,
                    (base_pos[2] + link1_end[2]) / 2
                ]
                
                self.robot_geometry['link2']['position'] = [
                    (link1_end[0] + link2_end[0]) / 2,
                    (link1_end[1] + link2_end[1]) / 2,
                    (link1_end[2] + link2_end[2]) / 2
                ]
                
                # Pen position
                pen_pos = [
                    link2_end[0],
                    link2_end[1],
                    link2_end[2] - self.pen_length/2
                ]
                self.robot_geometry['pen']['position'] = pen_pos
    
    def render(self) -> bool:
        """
        Render the 3D robot scene.
        
        Returns:
            bool: True if rendering successful
        """
        if not self.is_active or self.backend is None:
            return False
        
        try:
            if self.backend == 'matplotlib':
                return self.render_matplotlib()
            elif self.backend == 'plotly':
                return self.render_plotly()
            elif self.backend == 'open3d':
                return self.render_open3d()
            
            return False
            
        except Exception as e:
            if self.enable_logging:
                self.viz_logger.error(f"Render failed: {e}")
            return False
    
    def render_matplotlib(self) -> bool:
        """Render using matplotlib."""
        # Clear previous frame
        self.ax.clear()
        self.setup_matplotlib_scene()
        
        # Render environment
        self.render_environment_matplotlib()
        
        # Render robot
        self.render_robot_matplotlib()
        
        # Update display
        self.plt.draw()
        self.plt.pause(0.001)
        
        self.frame_count += 1
        return True
    
    def render_environment_matplotlib(self):
        """Render environment objects using matplotlib."""
        # Render paper surface
        if 'paper' in self.scene_objects:
            paper = self.scene_objects['paper']
            pos = paper['position']
            size = paper['size']
            
            # Create paper rectangle
            x_corners = [pos[0] - size[0]/2, pos[0] + size[0]/2, pos[0] + size[0]/2, pos[0] - size[0]/2]
            y_corners = [pos[1] - size[1]/2, pos[1] - size[1]/2, pos[1] + size[1]/2, pos[1] + size[1]/2]
            z_corners = [pos[2], pos[2], pos[2], pos[2]]
            
            # Plot paper surface
            vertices = [list(zip(x_corners, y_corners, z_corners))]
            self.ax.add_collection3d(self.Poly3DCollection(vertices, alpha=0.7, facecolor=paper['color']))
        
        # Render workspace boundaries
        if 'workspace' in self.scene_objects:
            bounds = self.scene_objects['workspace']['bounds']
            
            # Draw wireframe box
            x_range = bounds['x']
            y_range = bounds['y']
            z_range = bounds['z']
            
            # Bottom face
            self.ax.plot([x_range[0], x_range[1], x_range[1], x_range[0], x_range[0]],
                        [y_range[0], y_range[0], y_range[1], y_range[1], y_range[0]],
                        [z_range[0], z_range[0], z_range[0], z_range[0], z_range[0]],
                        'k--', alpha=0.3)
            
            # Top face
            self.ax.plot([x_range[0], x_range[1], x_range[1], x_range[0], x_range[0]],
                        [y_range[0], y_range[0], y_range[1], y_range[1], y_range[0]],
                        [z_range[1], z_range[1], z_range[1], z_range[1], z_range[1]],
                        'k--', alpha=0.3)
            
            # Vertical edges
            for x in x_range:
                for y in y_range:
                    self.ax.plot([x, x], [y, y], z_range, 'k--', alpha=0.3)
    
    def render_robot_matplotlib(self):
        """Render robot using matplotlib."""
        # Render each robot component
        for component_name, component in self.robot_geometry.items():
            pos = component['position']
            color = component['color']
            
            if component['type'] == 'cylinder':
                # Simplified cylinder representation as a line for links
                if component_name in ['link1', 'link2']:
                    # Draw as thick line from base to end
                    height = component['height']
                    self.ax.plot([pos[0], pos[0]], [pos[1], pos[1]], 
                               [pos[2] - height/2, pos[2] + height/2],
                               color=color, linewidth=10, solid_capstyle='round')
                else:
                    # Draw as point for base and pen
                    self.ax.scatter([pos[0]], [pos[1]], [pos[2]], 
                                  color=color, s=100, marker='o')
        
        # Render joints if enabled
        if self.show_joints:
            for joint_name, joint_pos in self.joint_positions.items():
                self.ax.scatter([joint_pos[0]], [joint_pos[1]], [joint_pos[2]], 
                              color='black', s=50, marker='s')
    
    def render_plotly(self) -> bool:
        """Render using Plotly."""
        # Clear previous traces
        self.fig.data = []
        
        # Render environment
        self.render_environment_plotly()
        
        # Render robot
        self.render_robot_plotly()
        
        # Update display
        if self.config.get('show_plot', True):
            self.fig.show()
        
        self.frame_count += 1
        return True
    
    def render_environment_plotly(self):
        """Render environment using Plotly."""
        # Paper surface
        if 'paper' in self.scene_objects:
            paper = self.scene_objects['paper']
            pos = paper['position']
            size = paper['size']
            
            # Create paper mesh
            x = [pos[0] - size[0]/2, pos[0] + size[0]/2, pos[0] + size[0]/2, pos[0] - size[0]/2]
            y = [pos[1] - size[1]/2, pos[1] - size[1]/2, pos[1] + size[1]/2, pos[1] + size[1]/2]
            z = [pos[2], pos[2], pos[2], pos[2]]
            
            self.fig.add_trace(self.go.Mesh3d(
                x=x, y=y, z=z,
                color=paper['color'],
                opacity=0.7,
                name='Paper'
            ))
    
    def render_robot_plotly(self):
        """Render robot using Plotly."""
        # Render robot components
        for component_name, component in self.robot_geometry.items():
            pos = component['position']
            color = component['color']
            
            if component['type'] == 'cylinder':
                # Add cylinder representation
                self.fig.add_trace(self.go.Scatter3d(
                    x=[pos[0]], y=[pos[1]], z=[pos[2]],
                    mode='markers',
                    marker=dict(size=10, color=color),
                    name=component_name
                ))
    
    def render_open3d(self) -> bool:
        """Render using Open3D."""
        # Update geometries
        self.update_open3d_geometries()
        
        # Update visualizer
        self.vis.update_renderer()
        self.vis.poll_events()
        
        self.frame_count += 1
        return True
    
    def update_open3d_geometries(self):
        """Update Open3D geometries."""
        # This would update the 3D meshes in Open3D
        # Simplified implementation
        pass
    
    def save_frame(self, filename: str, **kwargs) -> bool:
        """
        Save current frame to file.
        
        Args:
            filename: Output filename
            **kwargs: Additional arguments for saving
            
        Returns:
            bool: True if save successful
        """
        try:
            if self.backend == 'matplotlib':
                self.fig.savefig(filename, dpi=self.dpi, bbox_inches='tight', **kwargs)
            elif self.backend == 'plotly':
                self.fig.write_image(filename, **kwargs)
            elif self.backend == 'open3d':
                self.vis.capture_screen_image(filename)
            
            if self.enable_logging:
                self.viz_logger.info(f"Frame saved to {filename}")
            return True
            
        except Exception as e:
            if self.enable_logging:
                self.viz_logger.error(f"Failed to save frame: {e}")
            return False
    
    def set_camera_position(self, position: List[float], target: List[float] = None):
        """
        Set camera position and target.
        
        Args:
            position: Camera position [x, y, z]
            target: Camera target [x, y, z] (optional)
        """
        self.camera_position = position
        if target:
            self.camera_target = target
        
        # Update camera based on backend
        if self.backend == 'matplotlib' and hasattr(self, 'ax'):
            self.ax.view_init(elev=position[2]*30, azim=position[0]*45)
        elif self.backend == 'open3d' and hasattr(self, 'vis'):
            ctr = self.vis.get_view_control()
            ctr.set_front(position)
            if target:
                ctr.set_lookat(target)
    
    def animate_trajectory(self, trajectory_data: List[Dict[str, Any]], save_path: Optional[str] = None):
        """
        Create animation of robot following a trajectory.
        
        Args:
            trajectory_data: List of robot states for animation
            save_path: Optional path to save animation
        """
        if not self.enable_animation:
            return
        
        # This would create an animation by updating robot state
        # and rendering each frame
        frames = []
        
        for state_data in trajectory_data:
            self.update(state_data)
            self.render()
            
            if save_path:
                # Save frame for video creation
                frame_filename = f"frame_{len(frames):06d}.png"
                self.save_frame(frame_filename)
                frames.append(frame_filename)
        
        if save_path and frames:
            # Create video from frames (would need additional video processing)
            if self.enable_logging:
                self.viz_logger.info(f"Animation frames saved: {len(frames)}")
    
    def toggle_component_visibility(self, component: str, visible: bool):
        """
        Toggle visibility of robot components.
        
        Args:
            component: Component name ('joints', 'links', 'frames', 'workspace')
            visible: Visibility state
        """
        if component == 'joints':
            self.show_joints = visible
        elif component == 'links':
            self.show_links = visible
        elif component == 'frames':
            self.show_frames = visible
        elif component == 'workspace':
            self.show_workspace = visible
    
    def get_render_statistics(self) -> Dict[str, Any]:
        """
        Get rendering performance statistics.
        
        Returns:
            Dict containing rendering stats
        """
        stats = self.get_statistics()
        stats.update({
            'backend': self.backend,
            'robot_components': len(self.robot_geometry),
            'scene_objects': len(self.scene_objects),
            'lighting_enabled': self.lighting_enabled,
            'animation_enabled': self.enable_animation
        })
        return stats
    
    def close(self):
        """Close the robot renderer."""
        if self.backend == 'matplotlib' and hasattr(self, 'plt'):
            self.plt.close('all')
        elif self.backend == 'open3d' and hasattr(self, 'vis'):
            self.vis.destroy_window()
        
        self.is_active = False
        if self.enable_logging:
            self.viz_logger.info("Robot renderer closed")