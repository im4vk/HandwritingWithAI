#!/usr/bin/env python3
"""
Interactive Robotic Handwriting Interface
=========================================

Real-time interactive interface for controlling the robotic handwriting system.
Provides GUI controls for robot motion, writing parameters, and visualization.

Usage:
    python scripts/interactive.py
    python scripts/interactive.py --simulation-mode
    python scripts/interactive.py --config custom_config.yaml
"""

import argparse
import sys
import os
import logging
import threading
import time
from pathlib import Path
from typing import Dict, Any, Optional

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
    import numpy as np
    import tkinter as tk
    from tkinter import ttk, messagebox, filedialog
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
    from matplotlib.figure import Figure
    import matplotlib.animation as animation
    
    # Import project modules
    from src.robot_models.virtual_robot import VirtualRobotArm
    from src.robot_models.pen_gripper import PenGripper
    
except ImportError as e:
    logger.error(f"Missing dependencies: {e}")
    logger.error("Install GUI dependencies: pip install matplotlib tkinter")
    sys.exit(1)


class HandwritingGUI:
    """
    Interactive GUI for robotic handwriting system.
    """
    
    def __init__(self, config_path: str = "config/config.yaml"):
        """Initialize GUI application"""
        self.config = self._load_config(config_path)
        
        # Initialize robot system
        self.robot = VirtualRobotArm(self.config['robot'])
        self.gripper = PenGripper(self.config['robot']['pen'])
        
        # GUI state
        self.root = None
        self.is_running = False
        self.animation_active = False
        self.current_trajectory = []
        
        # Control variables
        self.control_vars = {}
        
        logger.info("Interactive GUI initialized")
    
    def _load_config(self, config_path: str) -> Dict[str, Any]:
        """Load configuration file"""
        try:
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)
            return config
        except Exception as e:
            logger.error(f"Failed to load config: {e}")
            return self._get_default_config()
    
    def _get_default_config(self) -> Dict[str, Any]:
        """Get default configuration"""
        return {
            'robot': {
                'name': 'WriteBot',
                'kinematics': {
                    'dof': 7,
                    'joint_limits': [[-180, 180]] * 7,
                    'link_lengths': [0.15, 0.30, 0.25, 0.20, 0.10, 0.08, 0.05]
                },
                'pen': {
                    'length': 0.15,
                    'max_grip_force': 20.0,
                    'max_writing_pressure': 10.0
                }
            }
        }
    
    def create_gui(self):
        """Create main GUI window"""
        self.root = tk.Tk()
        self.root.title("Robotic Handwriting AI - Interactive Control")
        self.root.geometry("1200x800")
        
        # Create main layout
        self._create_menu()
        self._create_main_layout()
        self._create_control_panels()
        self._create_visualization()
        self._create_status_bar()
        
        # Initialize control variables
        self._setup_control_variables()
        
        logger.info("GUI created successfully")
    
    def _create_menu(self):
        """Create menu bar"""
        menubar = tk.Menu(self.root)
        self.root.config(menu=menubar)
        
        # File menu
        file_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="File", menu=file_menu)
        file_menu.add_command(label="Load Config", command=self._load_config_dialog)
        file_menu.add_command(label="Save Config", command=self._save_config_dialog)
        file_menu.add_separator()
        file_menu.add_command(label="Exit", command=self._on_closing)
        
        # Robot menu
        robot_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="Robot", menu=robot_menu)
        robot_menu.add_command(label="Reset Position", command=self._reset_robot)
        robot_menu.add_command(label="Emergency Stop", command=self._emergency_stop)
        robot_menu.add_command(label="Calibrate", command=self._calibrate_robot)
        
        # Help menu
        help_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="Help", menu=help_menu)
        help_menu.add_command(label="About", command=self._show_about)
    
    def _create_main_layout(self):
        """Create main layout with panels"""
        # Main frame
        main_frame = ttk.Frame(self.root)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Left panel (controls)
        self.left_panel = ttk.Frame(main_frame, width=300)
        self.left_panel.pack(side=tk.LEFT, fill=tk.Y, padx=(0, 5))
        self.left_panel.pack_propagate(False)
        
        # Right panel (visualization)
        self.right_panel = ttk.Frame(main_frame)
        self.right_panel.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)
    
    def _create_control_panels(self):
        """Create control panels"""
        # Robot Control Panel
        robot_frame = ttk.LabelFrame(self.left_panel, text="Robot Control")
        robot_frame.pack(fill=tk.X, pady=(0, 10))
        
        self._create_joint_controls(robot_frame)
        self._create_position_controls(robot_frame)
        
        # Writing Control Panel
        writing_frame = ttk.LabelFrame(self.left_panel, text="Writing Control")
        writing_frame.pack(fill=tk.X, pady=(0, 10))
        
        self._create_writing_controls(writing_frame)
        
        # Gripper Control Panel
        gripper_frame = ttk.LabelFrame(self.left_panel, text="Gripper Control")
        gripper_frame.pack(fill=tk.X, pady=(0, 10))
        
        self._create_gripper_controls(gripper_frame)
        
        # Status Panel
        status_frame = ttk.LabelFrame(self.left_panel, text="Status")
        status_frame.pack(fill=tk.BOTH, expand=True)
        
        self._create_status_display(status_frame)
    
    def _create_joint_controls(self, parent):
        """Create individual joint controls"""
        ttk.Label(parent, text="Joint Angles (degrees):").pack(anchor=tk.W)
        
        self.joint_scales = []
        for i in range(7):
            frame = ttk.Frame(parent)
            frame.pack(fill=tk.X, pady=2)
            
            ttk.Label(frame, text=f"J{i+1}:", width=3).pack(side=tk.LEFT)
            
            scale = ttk.Scale(
                frame, 
                from_=-180, 
                to=180, 
                orient=tk.HORIZONTAL,
                command=lambda val, joint=i: self._on_joint_change(joint, val)
            )
            scale.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(5, 10))
            scale.set(0)
            
            value_label = ttk.Label(frame, text="0°", width=5)
            value_label.pack(side=tk.RIGHT)
            
            self.joint_scales.append((scale, value_label))
    
    def _create_position_controls(self, parent):
        """Create Cartesian position controls"""
        ttk.Separator(parent, orient=tk.HORIZONTAL).pack(fill=tk.X, pady=10)
        ttk.Label(parent, text="End-Effector Position:").pack(anchor=tk.W)
        
        # Position inputs
        pos_frame = ttk.Frame(parent)
        pos_frame.pack(fill=tk.X, pady=5)
        
        ttk.Label(pos_frame, text="X:").grid(row=0, column=0, sticky=tk.W)
        self.x_var = tk.DoubleVar(value=0.5)
        ttk.Entry(pos_frame, textvariable=self.x_var, width=8).grid(row=0, column=1, padx=5)
        
        ttk.Label(pos_frame, text="Y:").grid(row=0, column=2, sticky=tk.W, padx=(10, 0))
        self.y_var = tk.DoubleVar(value=0.0)
        ttk.Entry(pos_frame, textvariable=self.y_var, width=8).grid(row=0, column=3, padx=5)
        
        ttk.Label(pos_frame, text="Z:").grid(row=1, column=0, sticky=tk.W)
        self.z_var = tk.DoubleVar(value=0.1)
        ttk.Entry(pos_frame, textvariable=self.z_var, width=8).grid(row=1, column=1, padx=5)
        
        # Move button
        ttk.Button(
            parent, 
            text="Move to Position", 
            command=self._move_to_position
        ).pack(pady=5)
    
    def _create_writing_controls(self, parent):
        """Create writing-specific controls"""
        # Text input
        ttk.Label(parent, text="Text to Write:").pack(anchor=tk.W)
        self.text_var = tk.StringVar(value="Hello")
        ttk.Entry(parent, textvariable=self.text_var, width=30).pack(fill=tk.X, pady=2)
        
        # Writing parameters
        param_frame = ttk.Frame(parent)
        param_frame.pack(fill=tk.X, pady=5)
        
        ttk.Label(param_frame, text="Speed:").grid(row=0, column=0, sticky=tk.W)
        self.speed_var = tk.DoubleVar(value=1.0)
        ttk.Scale(
            param_frame, 
            from_=0.1, 
            to=2.0, 
            variable=self.speed_var,
            orient=tk.HORIZONTAL
        ).grid(row=0, column=1, sticky=tk.EW, padx=5)
        
        ttk.Label(param_frame, text="Size:").grid(row=1, column=0, sticky=tk.W)
        self.size_var = tk.DoubleVar(value=1.0)
        ttk.Scale(
            param_frame, 
            from_=0.5, 
            to=2.0, 
            variable=self.size_var,
            orient=tk.HORIZONTAL
        ).grid(row=1, column=1, sticky=tk.EW, padx=5)
        
        param_frame.columnconfigure(1, weight=1)
        
        # Writing buttons
        button_frame = ttk.Frame(parent)
        button_frame.pack(fill=tk.X, pady=5)
        
        ttk.Button(
            button_frame, 
            text="Start Writing", 
            command=self._start_writing
        ).pack(side=tk.LEFT, padx=(0, 5))
        
        ttk.Button(
            button_frame, 
            text="Stop", 
            command=self._stop_writing
        ).pack(side=tk.LEFT)
    
    def _create_gripper_controls(self, parent):
        """Create gripper controls"""
        # Grip force
        ttk.Label(parent, text="Grip Force:").pack(anchor=tk.W)
        self.grip_force_var = tk.DoubleVar(value=0.5)
        grip_scale = ttk.Scale(
            parent, 
            from_=0.0, 
            to=1.0, 
            variable=self.grip_force_var,
            orient=tk.HORIZONTAL,
            command=self._on_grip_force_change
        )
        grip_scale.pack(fill=tk.X, pady=2)
        
        # Writing pressure
        ttk.Label(parent, text="Writing Pressure:").pack(anchor=tk.W)
        self.pressure_var = tk.DoubleVar(value=0.3)
        pressure_scale = ttk.Scale(
            parent, 
            from_=0.0, 
            to=1.0, 
            variable=self.pressure_var,
            orient=tk.HORIZONTAL,
            command=self._on_pressure_change
        )
        pressure_scale.pack(fill=tk.X, pady=2)
        
        # Gripper buttons
        button_frame = ttk.Frame(parent)
        button_frame.pack(fill=tk.X, pady=5)
        
        ttk.Button(
            button_frame, 
            text="Grip Pen", 
            command=self._grip_pen
        ).pack(side=tk.LEFT, padx=(0, 5))
        
        ttk.Button(
            button_frame, 
            text="Release", 
            command=self._release_pen
        ).pack(side=tk.LEFT)
    
    def _create_status_display(self, parent):
        """Create status display area"""
        self.status_text = tk.Text(parent, height=10, wrap=tk.WORD)
        self.status_text.pack(fill=tk.BOTH, expand=True)
        
        scrollbar = ttk.Scrollbar(parent, orient=tk.VERTICAL, command=self.status_text.yview)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.status_text.config(yscrollcommand=scrollbar.set)
    
    def _create_visualization(self):
        """Create 3D visualization panel"""
        # Create matplotlib figure
        self.fig = Figure(figsize=(8, 6))
        self.ax = self.fig.add_subplot(111, projection='3d')
        
        # Create canvas
        self.canvas = FigureCanvasTkAgg(self.fig, self.right_panel)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
        # Setup plot
        self._setup_3d_plot()
        
        # Start animation
        self.animation = animation.FuncAnimation(
            self.fig, 
            self._update_plot, 
            interval=100,  # 10 FPS
            blit=False
        )
    
    def _setup_3d_plot(self):
        """Setup 3D plot parameters"""
        self.ax.set_xlabel('X (m)')
        self.ax.set_ylabel('Y (m)')
        self.ax.set_zlabel('Z (m)')
        self.ax.set_title('Robot Visualization')
        
        # Set limits
        self.ax.set_xlim([0, 1])
        self.ax.set_ylim([-0.5, 0.5])
        self.ax.set_zlim([0, 0.5])
    
    def _create_status_bar(self):
        """Create status bar at bottom"""
        self.status_bar = ttk.Label(self.root, text="Ready", relief=tk.SUNKEN)
        self.status_bar.pack(side=tk.BOTTOM, fill=tk.X)
    
    def _setup_control_variables(self):
        """Initialize control variables"""
        self.control_vars = {
            'joint_angles': [0.0] * 7,
            'target_position': [0.5, 0.0, 0.1],
            'grip_force': 0.5,
            'writing_pressure': 0.3,
            'pen_gripped': False
        }
    
    def _on_joint_change(self, joint_idx: int, value: str):
        """Handle joint angle change"""
        try:
            angle_deg = float(value)
            angle_rad = np.deg2rad(angle_deg)
            
            # Update control variable
            self.control_vars['joint_angles'][joint_idx] = angle_rad
            
            # Update display
            self.joint_scales[joint_idx][1].config(text=f"{angle_deg:.1f}°")
            
            # Move robot
            joint_angles = np.array(self.control_vars['joint_angles'])
            self.robot.state.joint_angles = joint_angles
            
            # Update forward kinematics
            self.robot.state.end_effector_pose, self.robot.state.pen_position = \
                self.robot.forward_kinematics(joint_angles)
            
        except ValueError:
            pass
    
    def _move_to_position(self):
        """Move robot to specified position"""
        try:
            target = [self.x_var.get(), self.y_var.get(), self.z_var.get()]
            trajectory = self.robot.move_to_position(np.array(target))
            
            if trajectory:
                self._log_status(f"Moved to position {target}")
                # Update joint displays
                for i, (scale, label) in enumerate(self.joint_scales):
                    angle_deg = np.rad2deg(self.robot.state.joint_angles[i])
                    scale.set(angle_deg)
                    label.config(text=f"{angle_deg:.1f}°")
            else:
                self._log_status("Failed to reach target position")
                
        except Exception as e:
            self._log_status(f"Error: {e}")
    
    def _on_grip_force_change(self, value: str):
        """Handle grip force change"""
        try:
            force = float(value)
            self.control_vars['grip_force'] = force
            if self.control_vars['pen_gripped']:
                self.gripper.set_grip_force(force)
        except ValueError:
            pass
    
    def _on_pressure_change(self, value: str):
        """Handle writing pressure change"""
        try:
            pressure = float(value)
            self.control_vars['writing_pressure'] = pressure
            if self.control_vars['pen_gripped']:
                self.gripper.set_writing_pressure(pressure)
        except ValueError:
            pass
    
    def _grip_pen(self):
        """Grip pen at current position"""
        success = self.gripper.grip_pen(self.robot.state.pen_position)
        if success:
            self.control_vars['pen_gripped'] = True
            self.gripper.set_grip_force(self.control_vars['grip_force'])
            self._log_status("Pen gripped successfully")
        else:
            self._log_status("Failed to grip pen")
    
    def _release_pen(self):
        """Release gripped pen"""
        self.gripper.release_pen()
        self.control_vars['pen_gripped'] = False
        self._log_status("Pen released")
    
    def _start_writing(self):
        """Start writing text"""
        text = self.text_var.get()
        if not text:
            messagebox.showwarning("Warning", "Please enter text to write")
            return
        
        if not self.control_vars['pen_gripped']:
            messagebox.showwarning("Warning", "Please grip pen first")
            return
        
        self._log_status(f"Starting to write: '{text}'")
        
        # Generate simple trajectory (placeholder)
        self.current_trajectory = self._generate_writing_trajectory(text)
        
        # Start writing animation
        self.animation_active = True
        self._execute_trajectory()
    
    def _stop_writing(self):
        """Stop current writing operation"""
        self.animation_active = False
        self._log_status("Writing stopped")
    
    def _generate_writing_trajectory(self, text: str) -> list:
        """Generate simple writing trajectory"""
        points = []
        start_x, start_y, z = 0.45, 0.0, 0.0
        
        for i, char in enumerate(text):
            char_x = start_x + i * 0.02
            # Simple up-down motion for each character
            points.extend([
                [char_x, start_y, z + 0.01],  # Lift pen
                [char_x, start_y, z],         # Lower pen
                [char_x, start_y + 0.02, z],  # Draw up
                [char_x, start_y, z],         # Draw down
            ])
        
        return points
    
    def _execute_trajectory(self):
        """Execute writing trajectory"""
        if not self.animation_active or not self.current_trajectory:
            return
        
        # Move to next point
        if self.current_trajectory:
            next_point = self.current_trajectory.pop(0)
            self.robot.move_to_position(np.array(next_point), duration=0.2)
            
            # Schedule next movement
            self.root.after(200, self._execute_trajectory)
    
    def _update_plot(self, frame):
        """Update 3D visualization"""
        self.ax.clear()
        self._setup_3d_plot()
        
        # Plot robot arm (simplified)
        arm_positions = self._get_arm_positions()
        if len(arm_positions) > 1:
            arm_positions = np.array(arm_positions)
            self.ax.plot(arm_positions[:, 0], arm_positions[:, 1], arm_positions[:, 2], 
                        'b-', linewidth=3, label='Robot Arm')
        
        # Plot end-effector
        end_pos = self.robot.state.pen_position
        self.ax.scatter(end_pos[0], end_pos[1], end_pos[2], 
                       c='red', s=100, marker='o', label='End-Effector')
        
        # Plot writing trajectory
        if self.current_trajectory:
            traj = np.array(self.current_trajectory)
            self.ax.plot(traj[:, 0], traj[:, 1], traj[:, 2], 
                        'g--', alpha=0.7, label='Trajectory')
        
        self.ax.legend()
        return []
    
    def _get_arm_positions(self) -> list:
        """Get simplified arm link positions"""
        # Simplified representation
        positions = [[0, 0, 0]]  # Base
        
        # Add some intermediate points based on joint angles
        current_pos = np.array([0, 0, 0])
        for i, (length, angle) in enumerate(zip(self.robot.link_lengths, self.robot.state.joint_angles)):
            # Simple approximation
            if i % 2 == 0:
                current_pos += length * np.array([np.cos(angle), np.sin(angle), 0])
            else:
                current_pos += length * np.array([np.cos(angle), 0, np.sin(angle)])
            positions.append(current_pos.copy())
        
        return positions
    
    def _reset_robot(self):
        """Reset robot to home position"""
        self.robot.reset()
        self.gripper.reset()
        self.control_vars['pen_gripped'] = False
        
        # Update GUI
        for i, (scale, label) in enumerate(self.joint_scales):
            scale.set(0)
            label.config(text="0°")
        
        self._log_status("Robot reset to home position")
    
    def _emergency_stop(self):
        """Emergency stop"""
        self.robot.emergency_stop()
        self.animation_active = False
        self._log_status("EMERGENCY STOP ACTIVATED!")
    
    def _calibrate_robot(self):
        """Calibrate robot (placeholder)"""
        self._log_status("Robot calibration completed")
    
    def _log_status(self, message: str):
        """Log message to status display"""
        timestamp = time.strftime("%H:%M:%S")
        full_message = f"[{timestamp}] {message}\n"
        
        self.status_text.insert(tk.END, full_message)
        self.status_text.see(tk.END)
        
        # Update status bar
        self.status_bar.config(text=message)
        
        logger.info(message)
    
    def _load_config_dialog(self):
        """Load configuration file dialog"""
        filename = filedialog.askopenfilename(
            title="Load Configuration",
            filetypes=[("YAML files", "*.yaml"), ("All files", "*.*")]
        )
        if filename:
            self.config = self._load_config(filename)
            self._log_status(f"Loaded config from {filename}")
    
    def _save_config_dialog(self):
        """Save configuration file dialog"""
        filename = filedialog.asksaveasfilename(
            title="Save Configuration",
            defaultextension=".yaml",
            filetypes=[("YAML files", "*.yaml"), ("All files", "*.*")]
        )
        if filename:
            with open(filename, 'w') as f:
                yaml.dump(self.config, f, default_flow_style=False)
            self._log_status(f"Saved config to {filename}")
    
    def _show_about(self):
        """Show about dialog"""
        messagebox.showinfo(
            "About",
            "Robotic Handwriting AI\n\n"
            "Interactive control interface for AI-powered\n"
            "robotic handwriting system.\n\n"
            "Version 1.0.0"
        )
    
    def _on_closing(self):
        """Handle window closing"""
        self.animation_active = False
        if hasattr(self, 'animation'):
            self.animation.event_source.stop()
        self.root.destroy()
    
    def run(self):
        """Run the GUI application"""
        self.create_gui()
        self.is_running = True
        
        # Initial status
        self._log_status("Interactive handwriting system ready")
        
        # Setup close handler
        self.root.protocol("WM_DELETE_WINDOW", self._on_closing)
        
        # Start main loop
        try:
            self.root.mainloop()
        except KeyboardInterrupt:
            logger.info("Application interrupted")
        finally:
            self.is_running = False


def main():
    """Main function"""
    parser = argparse.ArgumentParser(description='Interactive Robotic Handwriting Interface')
    parser.add_argument('--config', type=str, default='config/config.yaml',
                       help='Configuration file path')
    parser.add_argument('--simulation-mode', action='store_true',
                       help='Run in simulation mode (no real hardware)')
    
    args = parser.parse_args()
    
    try:
        # Create and run GUI
        gui = HandwritingGUI(args.config)
        gui.run()
        
        return 0
        
    except Exception as e:
        logger.error(f"Application failed: {e}")
        return 1


if __name__ == '__main__':
    sys.exit(main())