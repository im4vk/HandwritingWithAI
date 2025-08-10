#!/usr/bin/env python3
"""
Robotic Handwriting AI - Demo Script
===================================

Interactive demonstration of the robotic handwriting system.
Shows basic functionality including robot control, handwriting generation,
and visualization.

Usage:
    python scripts/demo.py --text "Hello World" --style casual
    python scripts/demo.py --interactive
    python scripts/demo.py --show-robot
"""

import argparse
import sys
import os
import logging
import numpy as np
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

try:
    import yaml
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation
    
    # Import project modules
    from src.robot_models.virtual_robot import VirtualRobotArm
    from src.robot_models.pen_gripper import PenGripper
    
except ImportError as e:
    logger.error(f"Missing dependencies: {e}")
    logger.error("Please install requirements: pip install -r requirements.txt")
    sys.exit(1)


class HandwritingDemo:
    """
    Interactive demonstration of robotic handwriting system.
    """
    
    def __init__(self, config_path: str = "config/config.yaml"):
        """Initialize demo with configuration"""
        self.config_path = config_path
        self.config = self._load_config()
        
        # Initialize robot
        self.robot = VirtualRobotArm(self.config['robot'])
        self.gripper = PenGripper(self.config['robot']['pen'])
        
        # Demo parameters
        self.writing_surface_z = 0.0
        self.current_trajectory = []
        
        logger.info("Handwriting demo initialized")
    
    def _load_config(self) -> dict:
        """Load configuration file"""
        try:
            with open(self.config_path, 'r') as f:
                config = yaml.safe_load(f)
            logger.info(f"Loaded configuration from {self.config_path}")
            return config
        except Exception as e:
            logger.error(f"Failed to load config: {e}")
            # Return default config
            return self._get_default_config()
    
    def _get_default_config(self) -> dict:
        """Get default configuration if file loading fails"""
        return {
            'robot': {
                'name': 'WriteBot',
                'kinematics': {
                    'dof': 7,
                    'joint_limits': [[-180, 180]] * 7,
                    'link_lengths': [0.15, 0.30, 0.25, 0.20, 0.10, 0.08, 0.05],
                    'max_velocity': [3.14] * 7,
                    'max_acceleration': [10.0] * 7
                },
                'hand': {
                    'type': 'pen_gripper',
                    'fingers': 3
                },
                'pen': {
                    'length': 0.15,
                    'weight': 0.02,
                    'max_grip_force': 20.0,
                    'max_writing_pressure': 10.0
                }
            },
            'writing_surface': {
                'position': [0.5, 0.0, 0.0],
                'dimensions': [0.21, 0.297]
            }
        }
    
    def demonstrate_robot_motion(self):
        """Demonstrate basic robot movement capabilities"""
        print("\n=== Robot Motion Demonstration ===")
        
        # Show initial state
        print(f"Robot: {self.robot}")
        print(f"Initial position: {self.robot.state.pen_position}")
        
        # Move to writing position
        writing_pos = np.array([0.5, 0.0, 0.1])  # 10cm above surface
        print(f"\nMoving to writing position: {writing_pos}")
        
        trajectory = self.robot.move_to_position(writing_pos, duration=2.0)
        if trajectory:
            print(f"Movement completed in {len(trajectory)} steps")
            print(f"Final position: {self.robot.state.pen_position}")
        else:
            print("Movement failed - position unreachable")
        
        # Demonstrate gripper
        print(f"\nGripper: {self.gripper}")
        success = self.gripper.grip_pen(self.robot.state.pen_position)
        print(f"Pen gripped: {success}")
        
        if success:
            self.gripper.set_writing_pressure(0.3)
            print(f"Writing pressure set to 30%")
            print(f"Pen tip position: {self.gripper.get_pen_tip_position()}")
    
    def demonstrate_simple_writing(self, text: str = "Hi"):
        """Demonstrate simple text writing"""
        print(f"\n=== Writing Demonstration: '{text}' ===")
        
        # Ensure pen is gripped
        if not self.gripper.state.is_gripped:
            writing_pos = np.array([0.5, 0.0, 0.1])
            self.robot.move_to_position(writing_pos)
            self.gripper.grip_pen(self.robot.state.pen_position)
        
        # Simple character-by-character writing simulation
        trajectory_points = []
        start_x, start_y = 0.45, -0.1
        
        for i, char in enumerate(text):
            char_trajectory = self._generate_character_trajectory(char, start_x + i * 0.02, start_y)
            trajectory_points.extend(char_trajectory)
        
        # Execute trajectory
        print(f"Executing trajectory with {len(trajectory_points)} points")
        self.current_trajectory = trajectory_points
        
        # Set writing pressure
        self.gripper.set_writing_pressure(0.4)
        
        # Move through trajectory
        for point in trajectory_points:
            success = self.robot.move_to_position(point, duration=0.1)
            if not success:
                print(f"Failed to reach point {point}")
                break
        
        print("Writing completed")
    
    def _generate_character_trajectory(self, char: str, start_x: float, start_y: float) -> list:
        """Generate simple trajectory for a character"""
        points = []
        char_height = 0.02  # 2cm
        char_width = 0.015  # 1.5cm
        
        # Very simple character shapes (could be replaced with proper font rendering)
        if char.upper() == 'H':
            # Draw H
            points.extend([
                [start_x, start_y, self.writing_surface_z],  # Bottom left
                [start_x, start_y + char_height, self.writing_surface_z],  # Top left
                [start_x, start_y + char_height/2, self.writing_surface_z],  # Middle left
                [start_x + char_width, start_y + char_height/2, self.writing_surface_z],  # Middle right
                [start_x + char_width, start_y + char_height, self.writing_surface_z],  # Top right
                [start_x + char_width, start_y, self.writing_surface_z],  # Bottom right
            ])
        elif char.upper() == 'I':
            # Draw I
            points.extend([
                [start_x, start_y, self.writing_surface_z],  # Bottom
                [start_x + char_width, start_y, self.writing_surface_z],  # Bottom line
                [start_x + char_width/2, start_y, self.writing_surface_z],  # Center bottom
                [start_x + char_width/2, start_y + char_height, self.writing_surface_z],  # Center top
                [start_x, start_y + char_height, self.writing_surface_z],  # Top left
                [start_x + char_width, start_y + char_height, self.writing_surface_z],  # Top right
            ])
        else:
            # Default: simple vertical line
            points.extend([
                [start_x, start_y, self.writing_surface_z],
                [start_x, start_y + char_height, self.writing_surface_z],
            ])
        
        return points
    
    def visualize_robot(self, show_trajectory: bool = True):
        """Create 3D visualization of robot and trajectory"""
        print("\n=== Robot Visualization ===")
        
        try:
            from mpl_toolkits.mplot3d import Axes3D
            
            fig = plt.figure(figsize=(12, 8))
            ax = fig.add_subplot(111, projection='3d')
            
            # Plot robot arm (simplified representation)
            self._plot_robot_arm(ax)
            
            # Plot writing surface
            self._plot_writing_surface(ax)
            
            # Plot trajectory if available
            if show_trajectory and self.current_trajectory:
                self._plot_trajectory(ax)
            
            # Set up plot
            ax.set_xlabel('X (m)')
            ax.set_ylabel('Y (m)')
            ax.set_zlabel('Z (m)')
            ax.set_title('Robotic Handwriting System')
            
            # Set equal aspect ratio
            max_range = 0.5
            ax.set_xlim([0, max_range])
            ax.set_ylim([-max_range/2, max_range/2])
            ax.set_zlim([0, max_range])
            
            plt.show()
            
        except ImportError:
            print("3D visualization requires matplotlib with 3D support")
            print("Install with: pip install matplotlib[3d]")
    
    def _plot_robot_arm(self, ax):
        """Plot simplified robot arm representation"""
        # Get current joint angles
        joint_angles = self.robot.state.joint_angles
        
        # Compute link positions using forward kinematics
        positions = []
        current_pos = np.array([0, 0, 0])  # Base position
        positions.append(current_pos.copy())
        
        # Simplified representation - just show links as straight segments
        link_lengths = self.robot.link_lengths
        for i, (length, angle) in enumerate(zip(link_lengths, joint_angles)):
            # Simple approximation: each joint rotates in a different plane
            if i % 2 == 0:  # X-Y plane rotation
                current_pos += length * np.array([np.cos(angle), np.sin(angle), 0])
            else:  # X-Z plane rotation
                current_pos += length * np.array([np.cos(angle), 0, np.sin(angle)])
            positions.append(current_pos.copy())
        
        # Plot arm links
        positions = np.array(positions)
        ax.plot(positions[:, 0], positions[:, 1], positions[:, 2], 
                'b-', linewidth=3, label='Robot Arm')
        
        # Plot joints
        ax.scatter(positions[:, 0], positions[:, 1], positions[:, 2], 
                  c='red', s=50, label='Joints')
        
        # Plot end-effector
        end_pos = self.robot.state.pen_position
        ax.scatter(end_pos[0], end_pos[1], end_pos[2], 
                  c='green', s=100, marker='^', label='End-Effector')
    
    def _plot_writing_surface(self, ax):
        """Plot writing surface"""
        surface_config = self.config.get('writing_surface', {})
        position = surface_config.get('position', [0.5, 0.0, 0.0])
        dimensions = surface_config.get('dimensions', [0.21, 0.297])
        
        # Create surface corners
        x_center, y_center, z = position
        width, height = dimensions
        
        x_corners = [x_center - width/2, x_center + width/2, 
                    x_center + width/2, x_center - width/2, x_center - width/2]
        y_corners = [y_center - height/2, y_center - height/2, 
                    y_center + height/2, y_center + height/2, y_center - height/2]
        z_corners = [z] * 5
        
        ax.plot(x_corners, y_corners, z_corners, 'k-', linewidth=2, label='Writing Surface')
    
    def _plot_trajectory(self, ax):
        """Plot writing trajectory"""
        if not self.current_trajectory:
            return
        
        trajectory = np.array(self.current_trajectory)
        ax.plot(trajectory[:, 0], trajectory[:, 1], trajectory[:, 2], 
                'r-', linewidth=2, label='Writing Trajectory')
        
        # Mark start and end points
        ax.scatter(trajectory[0, 0], trajectory[0, 1], trajectory[0, 2], 
                  c='green', s=100, marker='o', label='Start')
        ax.scatter(trajectory[-1, 0], trajectory[-1, 1], trajectory[-1, 2], 
                  c='red', s=100, marker='s', label='End')
    
    def interactive_mode(self):
        """Run interactive demonstration mode"""
        print("\n=== Interactive Mode ===")
        print("Available commands:")
        print("  'move' - Demonstrate robot movement")
        print("  'write <text>' - Write specified text")
        print("  'show' - Show 3D visualization")
        print("  'status' - Show robot status")
        print("  'reset' - Reset robot to home position")
        print("  'quit' - Exit")
        
        while True:
            try:
                command = input("\n> ").strip().lower()
                
                if command == 'quit':
                    break
                elif command == 'move':
                    self.demonstrate_robot_motion()
                elif command.startswith('write '):
                    text = command[6:]  # Remove 'write '
                    self.demonstrate_simple_writing(text)
                elif command == 'show':
                    self.visualize_robot()
                elif command == 'status':
                    self._show_status()
                elif command == 'reset':
                    self.robot.reset()
                    self.gripper.reset()
                    print("Robot reset to home position")
                else:
                    print("Unknown command. Type 'quit' to exit.")
                    
            except KeyboardInterrupt:
                print("\nExiting...")
                break
            except Exception as e:
                print(f"Error: {e}")
    
    def _show_status(self):
        """Show current robot and gripper status"""
        print("\n--- Robot Status ---")
        print(f"Joint angles (deg): {np.rad2deg(self.robot.state.joint_angles)}")
        print(f"End-effector position: {self.robot.state.end_effector_pose[:3]}")
        print(f"Pen position: {self.robot.state.pen_position}")
        print(f"Manipulability: {self.robot.get_manipulability():.3f}")
        
        print("\n--- Gripper Status ---")
        gripper_state = self.gripper.get_state_dict()
        print(f"Pen gripped: {gripper_state['is_gripped']}")
        print(f"Writing pressure: {gripper_state['writing_pressure']:.2f}")
        print(f"Grip stability: {gripper_state['grip_stability']:.2f}")
        if gripper_state['is_gripped']:
            print(f"Pen tip position: {gripper_state['tip_position']}")


def main():
    """Main function"""
    parser = argparse.ArgumentParser(description='Robotic Handwriting AI Demo')
    parser.add_argument('--text', type=str, default='Hello',
                       help='Text to write (default: Hello)')
    parser.add_argument('--style', type=str, default='casual',
                       help='Writing style (default: casual)')
    parser.add_argument('--interactive', action='store_true',
                       help='Run in interactive mode')
    parser.add_argument('--show-robot', action='store_true',
                       help='Show 3D robot visualization')
    parser.add_argument('--config', type=str, default='config/config.yaml',
                       help='Configuration file path')
    
    args = parser.parse_args()
    
    # Initialize demo
    demo = HandwritingDemo(args.config)
    
    try:
        if args.interactive:
            demo.interactive_mode()
        else:
            # Run automatic demo
            print("=== Robotic Handwriting AI Demo ===")
            
            # Demonstrate robot motion
            demo.demonstrate_robot_motion()
            
            # Demonstrate writing
            demo.demonstrate_simple_writing(args.text)
            
            # Show visualization if requested
            if args.show_robot:
                demo.visualize_robot()
            
            print("\nDemo completed successfully!")
            
    except Exception as e:
        logger.error(f"Demo failed: {e}")
        return 1
    
    return 0


if __name__ == '__main__':
    sys.exit(main())