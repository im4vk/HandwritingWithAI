"""
Virtual Robot Arm Implementation
===============================

A 7-DOF virtual robot arm designed for handwriting tasks with
human-like kinematics and dynamics.
"""

import numpy as np
import torch
import torch.nn as nn
from typing import List, Tuple, Optional, Dict, Any
from dataclasses import dataclass
import logging

from .kinematics import KinematicChain
from .dexterous_hand import DexterousHand

logger = logging.getLogger(__name__)


@dataclass
class RobotState:
    """Current state of the robot"""
    joint_angles: np.ndarray  # 7 joint angles (rad)
    joint_velocities: np.ndarray  # 7 joint velocities (rad/s)
    joint_accelerations: np.ndarray  # 7 joint accelerations (rad/s²)
    end_effector_pose: np.ndarray  # [x, y, z, rx, ry, rz]
    pen_position: np.ndarray  # 3D position of pen tip
    pen_pressure: float  # Writing pressure (0-1)
    timestamp: float  # Current time


@dataclass
class RobotLimits:
    """Physical limits and constraints"""
    joint_limits: List[Tuple[float, float]]  # Min/max for each joint
    velocity_limits: List[float]  # Max velocity for each joint
    acceleration_limits: List[float]  # Max acceleration for each joint
    workspace_limits: Tuple[Tuple[float, float], ...]  # [(x_min, x_max), ...]
    max_force: float  # Maximum end-effector force


class VirtualRobotArm:
    """
    Virtual 7-DOF robot arm for handwriting simulation.
    
    Features:
    - Human-like kinematics (similar to human arm proportions)
    - Realistic joint limits and dynamics
    - Integrated pen gripper
    - Physics-based motion simulation
    - Safety constraints and collision detection
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize virtual robot arm.
        
        Args:
            config: Robot configuration dictionary
        """
        self.config = config
        self.name = config.get('name', 'WriteBot')
        self.num_joints = 7  # 7-DOF robot arm
        
        # Initialize kinematics
        self._setup_kinematics(config)
        
        # Initialize limits and constraints
        self._setup_limits(config)
        
        # Initialize hand/gripper
        self.hand = DexterousHand(config.get('hand', {}))
        
        # Current state
        self.state = RobotState(
            joint_angles=np.zeros(7),
            joint_velocities=np.zeros(7),
            joint_accelerations=np.zeros(7),
            end_effector_pose=np.zeros(6),
            pen_position=np.zeros(3),
            pen_pressure=0.0,
            timestamp=0.0
        )
        
        # Control parameters
        self.control_frequency = 100.0  # Hz
        self.dt = 1.0 / self.control_frequency
        
        # Safety monitoring
        self.safety_enabled = True
        self.emergency_stop = False
        
        logger.info(f"Initialized {self.name} with 7-DOF configuration")
    
    def _setup_kinematics(self, config: Dict[str, Any]) -> None:
        """Setup kinematic chain and parameters"""
        kinematics_config = config.get('kinematics', {})
        
        # Link lengths (from base to end-effector)
        self.link_lengths = np.array(kinematics_config.get(
            'link_lengths', [0.15, 0.30, 0.25, 0.20, 0.10, 0.08, 0.05]
        ))
        
        # DH parameters for 7-DOF arm (modified for human-like proportions)
        self.dh_params = self._compute_dh_parameters()
        
        # Initialize kinematic solver
        self.kinematics = KinematicChain(self.dh_params)
        
        # Pen offset from wrist (tool center point)
        pen_config = config.get('pen', {})
        self.pen_offset = np.array([0, 0, pen_config.get('length', 0.15)])
    
    def _setup_limits(self, config: Dict[str, Any]) -> None:
        """Setup joint limits and constraints"""
        kinematics_config = config.get('kinematics', {})
        
        # Joint limits (rad) - more reasonable defaults
        joint_limits_deg = config.get('joint_limits', [
            [-180, 180], [-90, 90], [-180, 180], [-90, 90],
            [-180, 180], [-90, 90], [-180, 180]
        ])
        
        # Ensure joint_limits_deg is properly formatted and convert to radians safely
        joint_limits = []
        for lim in joint_limits_deg:
            if isinstance(lim, (list, tuple)) and len(lim) == 2:
                try:
                    min_deg = float(lim[0])
                    max_deg = float(lim[1])
                    joint_limits.append((np.deg2rad(min_deg), np.deg2rad(max_deg)))
                except (ValueError, TypeError):
                    # Fallback to default if conversion fails
                    joint_limits.append((np.deg2rad(-180), np.deg2rad(180)))
            else:
                # Fallback to default if format is incorrect
                joint_limits.append((np.deg2rad(-180), np.deg2rad(180)))
        
        # Velocity limits (rad/s)
        velocity_limits = config.get('velocity_limits', [3.14, 3.14, 3.14, 3.14, 6.28, 6.28, 6.28])
        
        # Acceleration limits (rad/s²)
        acceleration_limits = config.get('acceleration_limits', [10.0, 10.0, 10.0, 10.0, 20.0, 20.0, 20.0])
        
        # Workspace limits (m) - much larger workspace
        workspace = config.get('workspace_limits', {
            'x': [-0.5, 0.5],
            'y': [-0.5, 0.5], 
            'z': [0.0, 1.0]
        })
        workspace_limits = (
            (workspace['x'][0], workspace['x'][1]),
            (workspace['y'][0], workspace['y'][1]),
            (workspace['z'][0], workspace['z'][1])
        )
        
        # Force limits
        max_force = config.get('max_force', 20.0)
        
        # Create RobotLimits object
        self.limits = RobotLimits(
            joint_limits=joint_limits,
            velocity_limits=velocity_limits,
            acceleration_limits=acceleration_limits,
            workspace_limits=workspace_limits,
            max_force=max_force
        )
        
        # Also keep individual attributes for backward compatibility
        self.joint_limits = joint_limits
        self.velocity_limits = np.array(velocity_limits)
        self.acceleration_limits = np.array(acceleration_limits)
        self.workspace_limits = workspace_limits
        self.max_force = max_force
    
    def _compute_dh_parameters(self) -> np.ndarray:
        """
        Compute Denavit-Hartenberg parameters for 7-DOF arm.
        
        Returns:
            DH parameters matrix [a, alpha, d, theta] for each joint
        """
        # Human-inspired 7-DOF configuration
        # Joint 1: Shoulder pan (rotation around torso)
        # Joint 2: Shoulder lift (up/down motion)  
        # Joint 3: Shoulder roll (internal/external rotation)
        # Joint 4: Elbow (flexion/extension)
        # Joint 5: Wrist 1 (rotation)
        # Joint 6: Wrist 2 (up/down)
        # Joint 7: Wrist 3 (tool rotation)
        
        dh_params = np.array([
            [0,        -np.pi/2,  self.link_lengths[0], 0],  # Shoulder pan
            [self.link_lengths[1], 0,  0,                0],  # Shoulder lift
            [0,        -np.pi/2,  0,                    0],  # Shoulder roll
            [0,         np.pi/2,  self.link_lengths[2], 0],  # Elbow
            [0,        -np.pi/2,  self.link_lengths[3], 0],  # Wrist 1
            [0,         np.pi/2,  self.link_lengths[4], 0],  # Wrist 2
            [0,         0,        self.link_lengths[5], 0],  # Wrist 3
        ])
        
        return dh_params
    
    def forward_kinematics(self, joint_angles: np.ndarray) -> np.ndarray:
        """
        Compute forward kinematics.
        
        Args:
            joint_angles: 7 joint angles (rad)
            
        Returns:
            end_effector_pose: [x, y, z, rx, ry, rz]
        """
        # Validate input
        if len(joint_angles) != 7:
            raise ValueError("Expected 7 joint angles")
        
        # Check joint limits
        if self.safety_enabled:
            for i, (angle, (min_limit, max_limit)) in enumerate(zip(joint_angles, self.joint_limits)):
                if not (min_limit <= angle <= max_limit):
                    logger.warning(f"Joint {i+1} angle {np.rad2deg(angle):.1f}° "
                                 f"exceeds limits [{np.rad2deg(min_limit):.1f}, "
                                 f"{np.rad2deg(max_limit):.1f}]")
        
        # Compute end-effector pose
        end_effector_pose = self.kinematics.forward_kinematics(joint_angles)
        
        # Update robot state
        self.state.joint_angles = joint_angles.copy()
        self.state.end_effector_pose = end_effector_pose.copy()
        
        # Compute pen tip position (accounting for pen offset)
        pen_position = self._compute_pen_position(end_effector_pose)
        self.state.pen_position = pen_position.copy()
        
        return end_effector_pose
    
    def inverse_kinematics(self, target_pose: np.ndarray, 
                         initial_guess: Optional[np.ndarray] = None) -> Optional[np.ndarray]:
        """
        Compute inverse kinematics.
        
        Args:
            target_pose: Desired end-effector pose [x, y, z, rx, ry, rz]
            initial_guess: Initial joint configuration for optimization
            
        Returns:
            joint_angles: 7 joint angles (rad) or None if no solution
        """
        if initial_guess is None:
            initial_guess = self.state.joint_angles.copy()
        
        # Check workspace limits
        if self.safety_enabled and not self._check_workspace(target_pose[:3]):
            logger.warning(f"Target position {target_pose[:3]} outside workspace")
            return None
        
        # Solve inverse kinematics
        joint_angles = self.kinematics.inverse_kinematics(
            target_pose, initial_guess, self.joint_limits
        )
        
        return joint_angles
    
    def _compute_pen_position(self, end_effector_pose: np.ndarray) -> np.ndarray:
        """Compute pen tip position from end-effector pose"""
        # Extract position and orientation
        position = end_effector_pose[:3]
        orientation = end_effector_pose[3:]
        
        # Convert orientation to rotation matrix
        rotation_matrix = self._euler_to_rotation_matrix(orientation)
        
        # Apply pen offset
        pen_position = position + rotation_matrix @ self.pen_offset
        
        return pen_position
    
    def _euler_to_rotation_matrix(self, euler_angles: np.ndarray) -> np.ndarray:
        """Convert Euler angles (rx, ry, rz) to rotation matrix"""
        rx, ry, rz = euler_angles
        
        # Rotation matrices
        Rx = np.array([[1, 0, 0],
                       [0, np.cos(rx), -np.sin(rx)],
                       [0, np.sin(rx), np.cos(rx)]])
        
        Ry = np.array([[np.cos(ry), 0, np.sin(ry)],
                       [0, 1, 0],
                       [-np.sin(ry), 0, np.cos(ry)]])
        
        Rz = np.array([[np.cos(rz), -np.sin(rz), 0],
                       [np.sin(rz), np.cos(rz), 0],
                       [0, 0, 1]])
        
        return Rz @ Ry @ Rx
    
    def _check_workspace(self, position: np.ndarray) -> bool:
        """Check if position is within workspace limits"""
        x, y, z = position
        
        x_limits, y_limits, z_limits = self.workspace_limits
        
        return (x_limits[0] <= x <= x_limits[1] and
                y_limits[0] <= y <= y_limits[1] and
                z_limits[0] <= z <= z_limits[1])
    
    def move_to_joint_angles(self, target_angles: np.ndarray, 
                           duration: float = 1.0) -> List[RobotState]:
        """
        Move robot to target joint configuration.
        
        Args:
            target_angles: Target joint angles (rad)
            duration: Movement duration (seconds)
            
        Returns:
            trajectory: List of robot states during movement
        """
        # Generate smooth trajectory
        trajectory = self._generate_joint_trajectory(
            self.state.joint_angles, target_angles, duration
        )
        
        # Execute trajectory
        robot_states = []
        for angles, velocities, accelerations, t in trajectory:
            # Update state
            self.state.joint_angles = angles
            self.state.joint_velocities = velocities
            self.state.joint_accelerations = accelerations
            self.state.timestamp = t
            
            # Compute forward kinematics
            self.state.end_effector_pose, self.state.pen_position = \
                self.forward_kinematics(angles)
            
            robot_states.append(self._copy_state())
        
        return robot_states
    
    def move_to_position(self, target_position: np.ndarray,
                        target_orientation: Optional[np.ndarray] = None,
                        duration: float = 1.0) -> Optional[List[RobotState]]:
        """
        Move robot to target Cartesian position.
        
        Args:
            target_position: Target position [x, y, z]
            target_orientation: Target orientation [rx, ry, rz] (optional)
            duration: Movement duration (seconds)
            
        Returns:
            trajectory: List of robot states or None if unreachable
        """
        # Use current orientation if not specified
        if target_orientation is None:
            target_orientation = self.state.end_effector_pose[3:]
        
        # Create target pose
        target_pose = np.concatenate([target_position, target_orientation])
        
        # Solve inverse kinematics
        target_angles = self.inverse_kinematics(target_pose)
        if target_angles is None:
            logger.error(f"Cannot reach target position {target_position}")
            return None
        
        # Execute movement
        return self.move_to_joint_angles(target_angles, duration)
    
    def _generate_joint_trajectory(self, start_angles: np.ndarray,
                                 end_angles: np.ndarray,
                                 duration: float) -> List[Tuple[np.ndarray, np.ndarray, np.ndarray, float]]:
        """Generate smooth joint trajectory using quintic polynomials"""
        num_points = int(duration * self.control_frequency)
        t_array = np.linspace(0, duration, num_points)
        
        trajectory = []
        for t in t_array:
            # Quintic polynomial trajectory
            s = self._quintic_polynomial(t / duration)
            ds_dt = self._quintic_polynomial_derivative(t / duration) / duration
            d2s_dt2 = self._quintic_polynomial_second_derivative(t / duration) / (duration ** 2)
            
            # Interpolate joint angles
            angles = start_angles + s * (end_angles - start_angles)
            velocities = ds_dt * (end_angles - start_angles)
            accelerations = d2s_dt2 * (end_angles - start_angles)
            
            # Check limits
            if self.safety_enabled:
                velocities = np.clip(velocities, -self.velocity_limits, self.velocity_limits)
                accelerations = np.clip(accelerations, -self.acceleration_limits, self.acceleration_limits)
            
            trajectory.append((angles, velocities, accelerations, t))
        
        return trajectory
    
    @staticmethod
    def _quintic_polynomial(s: float) -> float:
        """Quintic polynomial for smooth trajectories (0->1)"""
        return 10 * s**3 - 15 * s**4 + 6 * s**5
    
    @staticmethod
    def _quintic_polynomial_derivative(s: float) -> float:
        """First derivative of quintic polynomial"""
        return 30 * s**2 - 60 * s**3 + 30 * s**4
    
    @staticmethod
    def _quintic_polynomial_second_derivative(s: float) -> float:
        """Second derivative of quintic polynomial"""
        return 60 * s - 180 * s**2 + 120 * s**3
    
    def set_pen_pressure(self, pressure: float) -> None:
        """Set pen writing pressure (0-1)"""
        self.state.pen_pressure = np.clip(pressure, 0.0, 1.0)
        self.hand.set_grip_force(pressure * self.max_force)
    
    def get_jacobian(self, joint_angles: Optional[np.ndarray] = None) -> np.ndarray:
        """Get Jacobian matrix for current or specified configuration"""
        if joint_angles is None:
            joint_angles = self.state.joint_angles
        return self.kinematics.compute_jacobian(joint_angles)
    
    def get_manipulability(self, joint_angles: Optional[np.ndarray] = None) -> float:
        """Compute manipulability index"""
        jacobian = self.get_jacobian(joint_angles)
        # Use velocity manipulability index
        return np.sqrt(np.linalg.det(jacobian @ jacobian.T))
    
    def emergency_stop(self) -> None:
        """Trigger emergency stop"""
        self.emergency_stop = True
        self.state.joint_velocities.fill(0)
        self.state.joint_accelerations.fill(0)
        logger.warning("Emergency stop activated!")
    
    def reset(self) -> None:
        """Reset robot to home position"""
        self.emergency_stop = False
        self.state.joint_angles.fill(0)
        self.state.joint_velocities.fill(0)
        self.state.joint_accelerations.fill(0)
        self.state.pen_pressure = 0.0
        self.state.timestamp = 0.0
        
        # Update forward kinematics
        self.state.end_effector_pose, self.state.pen_position = \
            self.forward_kinematics(self.state.joint_angles)
        
        logger.info("Robot reset to home position")
    
    def _copy_state(self) -> RobotState:
        """Create a copy of current state"""
        return RobotState(
            joint_angles=self.state.joint_angles.copy(),
            joint_velocities=self.state.joint_velocities.copy(),
            joint_accelerations=self.state.joint_accelerations.copy(),
            end_effector_pose=self.state.end_effector_pose.copy(),
            pen_position=self.state.pen_position.copy(),
            pen_pressure=self.state.pen_pressure,
            timestamp=self.state.timestamp
        )
    
    def get_state_dict(self) -> Dict[str, Any]:
        """Get current state as dictionary"""
        return {
            'joint_angles': self.state.joint_angles.tolist(),
            'joint_velocities': self.state.joint_velocities.tolist(),
            'joint_accelerations': self.state.joint_accelerations.tolist(),
            'end_effector_pose': self.state.end_effector_pose.tolist(),
            'pen_position': self.state.pen_position.tolist(),
            'pen_pressure': self.state.pen_pressure,
            'timestamp': self.state.timestamp,
            'manipulability': self.get_manipulability()
        }
    
    def __str__(self) -> str:
        return f"VirtualRobotArm(name='{self.name}', dof=7, pen_pos={self.state.pen_position})"
    
    def __repr__(self) -> str:
        return self.__str__()