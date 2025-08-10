"""
Handwriting-specific simulation environment.

This module implements a specialized environment for robotic handwriting tasks,
including paper surface modeling, pen dynamics, and handwriting quality metrics.
"""

import numpy as np
from typing import Dict, Any, Tuple, Optional, List
import logging

from .base_environment import BaseEnvironment
from .physics_engines import MuJoCoEngine, PyBulletEngine, EnhancedMockEngine
from .environment_config import EnvironmentConfig

logger = logging.getLogger(__name__)


class HandwritingEnvironment(BaseEnvironment):
    """
    Specialized environment for robotic handwriting simulation.
    
    This environment includes:
    - Robot arm with pen end-effector
    - Paper surface with contact dynamics
    - Handwriting quality evaluation
    - Trajectory tracking and recording
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the handwriting environment.
        
        Args:
            config: Configuration dictionary with handwriting-specific parameters
        """
        super().__init__(config)
        
        # Handwriting-specific configuration
        self.paper_size = config.get('paper_size', (0.21, 0.297))  # A4 in meters
        self.paper_position = config.get('paper_position', [0.5, 0.0, 0.01])
        self.pen_tip_radius = config.get('pen_tip_radius', 0.0005)  # 0.5mm
        self.contact_threshold = config.get('contact_threshold', 0.001)
        
        # Physics engine selection
        engine_type = config.get('physics_engine', 'mujoco')
        if engine_type.lower() == 'mujoco':
            self.physics_engine = MuJoCoEngine(config)
        elif engine_type.lower() == 'pybullet':
            self.physics_engine = PyBulletEngine(config)
        elif engine_type.lower() == 'enhanced_mock':
            self.physics_engine = EnhancedMockEngine(config)
        else:
            raise ValueError(f"Unsupported physics engine: {engine_type}") # incase of simple physics engine
        
        # Handwriting state tracking
        self.pen_trajectory = []
        self.contact_points = []
        self.writing_pressure = []
        self.handwriting_quality_metrics = {}
        
        # Observation and action spaces
        self.setup_spaces()
        
        # Reward computation
        self.target_trajectory = config.get('target_trajectory', None)
        self.reward_weights = config.get('reward_weights', {
            'trajectory_tracking': 1.0,
            'smoothness': 0.5,
            'pressure_control': 0.3,
            'speed_consistency': 0.2
        })
    
    @property
    def current_pen_position(self) -> np.ndarray:
        """Get current pen position from physics engine."""
        if hasattr(self, 'physics_engine') and self.physics_engine:
            return self.physics_engine.get_pen_position()
        else:
            return np.array([0.0, 0.0, 0.0])
    
    def setup_spaces(self):
        """Setup observation and action spaces for the environment."""
        # Action space: [dx, dy, dz, pressure] relative movements + pressure
        self.action_dim = 4
        self.action_bounds = np.array([
            [-0.01, 0.01],   # dx (10mm max)
            [-0.01, 0.01],   # dy (10mm max)
            [-0.005, 0.005], # dz (5mm max)
            [0.0, 1.0]       # pressure (normalized)
        ])
        
        # Observation space: robot state + pen state + environment state
        self.observation_dim = 15  # Position(3) + velocity(3) + orientation(4) + contact(1) + target(3) + error(1)
    
    def initialize(self) -> bool:
        """
        Initialize the handwriting simulation environment.
        
        Returns:
            bool: True if initialization successful
        """
        try:
            # Initialize physics engine
            if not self.physics_engine.initialize():
                logger.error("Failed to initialize physics engine")
                return False
            
            # Load robot model
            robot_model_path = self.config.get('robot_model_path', 'models/robot_arm.xml')
            if not self.physics_engine.load_robot_model(robot_model_path):
                logger.error(f"Failed to load robot model: {robot_model_path}")
                return False
            
            # Setup paper surface
            self.setup_paper_surface()
            
            # Setup pen end-effector
            self.setup_pen_end_effector()
            
            # Initialize tracking variables
            self.reset_tracking()
            
            self.is_initialized = True
            logger.info("Handwriting environment initialized successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize environment: {e}")
            return False
    
    def setup_paper_surface(self):
        """Setup the paper surface in the simulation."""
        paper_config = {
            'size': self.paper_size,
            'position': self.paper_position,
            'friction': self.config.get('paper_friction', 0.8),
            'color': [1.0, 1.0, 1.0, 1.0]  # White paper
        }
        self.physics_engine.create_paper_surface(paper_config)
    
    def setup_pen_end_effector(self):
        """Setup the pen end-effector properties."""
        pen_config = {
            'tip_radius': self.pen_tip_radius,
            'length': self.config.get('pen_length', 0.15),
            'mass': self.config.get('pen_mass', 0.02),
            'color': [0.0, 0.0, 1.0, 1.0]  # Blue pen
        }
        self.physics_engine.setup_pen_end_effector(pen_config)
    
    def reset(self) -> np.ndarray:
        """
        Reset the environment to initial handwriting state.
        
        Returns:
            np.ndarray: Initial observation
        """
        # Reset physics simulation
        self.physics_engine.reset()
        
        # Reset tracking variables
        self.reset_tracking()
        
        # Set initial pen position above paper
        initial_position = [
            self.paper_position[0] - self.paper_size[0]/4,  # Start at left side
            self.paper_position[1] - self.paper_size[1]/4,  # Start at top
            self.paper_position[2] + 0.01  # 1cm above paper
        ]
        self.physics_engine.set_pen_position(initial_position)
        
        # Reset step counter
        self.current_step = 0
        self.is_running = True
        
        # Get initial observation
        observation = self.get_observation()
        self.last_observation = observation
        
        logger.info(f"Environment reset - Episode {self.episode_count}")
        self.episode_count += 1
        
        return observation
    
    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, Dict[str, Any]]:
        """
        Execute one simulation step for handwriting.
        
        Args:
            action: [dx, dy, dz, pressure] - relative movement and pressure
            
        Returns:
            Tuple of (observation, reward, done, info)
        """
        if not self.validate_action(action):
            raise ValueError(f"Invalid action: {action}")
        
        # Clip action to bounds
        action = np.clip(action, self.action_bounds[:, 0], self.action_bounds[:, 1])
        
        # Execute action in physics simulation
        movement = action[:3]  # dx, dy, dz
        pressure = action[3]   # pressure
        
        self.physics_engine.move_pen_relative(movement, pressure)
        self.physics_engine.step_simulation()
        
        # Update tracking data
        self.update_tracking()
        
        # Get new observation
        observation = self.get_observation()
        
        # Compute reward
        reward = self.compute_reward(action, observation)
        
        # Check if episode is done
        done = self.is_episode_done()
        
        # Prepare info dictionary
        info = self.get_step_info()
        
        # Update state
        self.current_step += 1
        self.last_action = action
        self.last_observation = observation
        
        # Log step if enabled
        self.log_step(action, reward, done)
        
        return observation, reward, done, info
    
    def get_observation(self) -> np.ndarray:
        """
        Get current observation from the handwriting environment.
        
        Returns:
            np.ndarray: Current observation vector
        """
        # Get pen state from physics engine
        pen_position = self.physics_engine.get_pen_position()
        pen_velocity = self.physics_engine.get_pen_velocity()
        pen_orientation = self.physics_engine.get_pen_orientation()
        
        # Contact detection
        is_in_contact = self.physics_engine.is_pen_in_contact()
        
        # Target information (if available)
        if self.target_trajectory and self.current_step < len(self.target_trajectory):
            target_position = self.target_trajectory[self.current_step]
            position_error = np.linalg.norm(pen_position - target_position)
        else:
            target_position = pen_position
            position_error = 0.0
        
        # Combine into observation vector
        observation = np.concatenate([
            pen_position,      # 3D position
            pen_velocity,      # 3D velocity
            pen_orientation,   # 4D quaternion
            [is_in_contact],   # Contact state
            target_position,   # Target position
            [position_error]   # Position error
        ])
        
        return observation.astype(np.float32)
    
    def compute_reward(self, action: np.ndarray, observation: np.ndarray) -> float:
        """
        Compute reward for handwriting task.
        
        Args:
            action: Action taken
            observation: Current observation
            
        Returns:
            float: Reward value
        """
        reward = 0.0
        
        # Trajectory tracking reward
        if self.target_trajectory and self.current_step < len(self.target_trajectory):
            position_error = observation[-1]  # Last element is position error
            trajectory_reward = -position_error * self.reward_weights['trajectory_tracking']
            reward += trajectory_reward
        
        # Smoothness reward (penalize large movements)
        movement_magnitude = np.linalg.norm(action[:3])
        smoothness_reward = -movement_magnitude * self.reward_weights['smoothness']
        reward += smoothness_reward
        
        # Pressure control reward
        pressure = action[3]
        if 0.2 <= pressure <= 0.8:  # Ideal pressure range
            pressure_reward = self.reward_weights['pressure_control']
        else:
            pressure_reward = -abs(pressure - 0.5) * self.reward_weights['pressure_control']
        reward += pressure_reward
        
        # Speed consistency reward
        if len(self.pen_trajectory) > 1:
            current_speed = np.linalg.norm(observation[3:6])  # Velocity magnitude
            if hasattr(self, 'last_speed'):
                speed_consistency = -abs(current_speed - self.last_speed) * self.reward_weights['speed_consistency']
                reward += speed_consistency
            self.last_speed = current_speed
        
        return reward
    
    def is_episode_done(self) -> bool:
        """
        Check if the handwriting episode is complete.
        
        Returns:
            bool: True if episode should end
        """
        # Max steps reached
        if self.current_step >= self.max_episode_steps:
            return True
        
        # Pen moved too far from paper
        pen_position = self.physics_engine.get_pen_position()
        if pen_position[2] > self.paper_position[2] + 0.05:  # 5cm above paper
            return True
        
        # Target trajectory completed
        if self.target_trajectory and self.current_step >= len(self.target_trajectory):
            return True
        
        return False
    
    def update_tracking(self):
        """Update handwriting trajectory tracking."""
        pen_position = self.physics_engine.get_pen_position()
        is_in_contact = self.physics_engine.is_pen_in_contact()
        pressure = self.physics_engine.get_contact_force()
        
        # Record trajectory point
        self.pen_trajectory.append(pen_position.copy())
        
        # Record contact points (only when pen touches paper)
        if is_in_contact:
            self.contact_points.append(pen_position.copy())
            self.writing_pressure.append(pressure)
    
    def reset_tracking(self):
        """Reset all tracking variables."""
        self.pen_trajectory = []
        self.contact_points = []
        self.writing_pressure = []
        self.handwriting_quality_metrics = {}
    
    def get_step_info(self) -> Dict[str, Any]:
        """Get additional information about the current step."""
        return {
            'pen_position': self.physics_engine.get_pen_position(),
            'pen_velocity': self.physics_engine.get_pen_velocity(),
            'is_in_contact': self.physics_engine.is_pen_in_contact(),
            'contact_force': self.physics_engine.get_contact_force(),
            'trajectory_length': len(self.pen_trajectory),
            'contact_points': len(self.contact_points),
            'episode_step': self.current_step
        }
    
    def get_state(self) -> Dict[str, Any]:
        """Get complete handwriting environment state."""
        base_state = super().get_environment_info()
        handwriting_state = {
            'pen_trajectory': self.pen_trajectory,
            'contact_points': self.contact_points,
            'writing_pressure': self.writing_pressure,
            'physics_state': self.physics_engine.get_state(),
            'target_trajectory': self.target_trajectory,
            'current_target_index': self.current_step
        }
        return {**base_state, **handwriting_state}
    
    def set_state(self, state: Dict[str, Any]) -> bool:
        """Set handwriting environment state."""
        try:
            # Set physics engine state
            if 'physics_state' in state:
                self.physics_engine.set_state(state['physics_state'])
            
            # Restore tracking data
            self.pen_trajectory = state.get('pen_trajectory', [])
            self.contact_points = state.get('contact_points', [])
            self.writing_pressure = state.get('writing_pressure', [])
            self.current_step = state.get('current_step', 0)
            
            return True
        except Exception as e:
            logger.error(f"Failed to set environment state: {e}")
            return False
    
    def set_target_trajectory(self, trajectory: List[np.ndarray]):
        """
        Set target trajectory for the handwriting task.
        
        Args:
            trajectory: List of 3D positions defining the target path
        """
        self.target_trajectory = trajectory
        logger.info(f"Set target trajectory with {len(trajectory)} points")
    
    def get_handwriting_quality_metrics(self) -> Dict[str, float]:
        """
        Compute handwriting quality metrics.
        
        Returns:
            Dict containing various quality metrics
        """
        if len(self.contact_points) < 2:
            return {}
        
        contact_points = np.array(self.contact_points)
        
        # Smoothness (variance in acceleration)
        if len(contact_points) > 2:
            velocities = np.diff(contact_points, axis=0)
            accelerations = np.diff(velocities, axis=0)
            smoothness = 1.0 / (1.0 + np.var(accelerations))
        else:
            smoothness = 0.0
        
        # Pressure consistency
        if len(self.writing_pressure) > 1:
            pressure_variance = np.var(self.writing_pressure)
            pressure_consistency = 1.0 / (1.0 + pressure_variance)
        else:
            pressure_consistency = 0.0
        
        # Line consistency (straightness for straight lines)
        if len(contact_points) > 2:
            # Fit line and compute deviations
            start_point = contact_points[0]
            end_point = contact_points[-1]
            line_vector = end_point - start_point
            line_length = np.linalg.norm(line_vector)
            
            if line_length > 0:
                deviations = []
                for point in contact_points[1:-1]:
                    # Distance from point to line
                    point_vector = point - start_point
                    projection = np.dot(point_vector, line_vector) / line_length
                    closest_point = start_point + projection * (line_vector / line_length)
                    deviation = np.linalg.norm(point - closest_point)
                    deviations.append(deviation)
                
                line_consistency = 1.0 / (1.0 + np.mean(deviations)) if deviations else 0.0
            else:
                line_consistency = 0.0
        else:
            line_consistency = 0.0
        
        return {
            'smoothness': smoothness,
            'pressure_consistency': pressure_consistency,
            'line_consistency': line_consistency,
            'total_writing_time': self.current_step * self.timestep,
            'writing_distance': self.get_writing_distance(),
            'contact_ratio': len(self.contact_points) / max(len(self.pen_trajectory), 1)
        }
    
    def get_writing_distance(self) -> float:
        """Calculate total distance of pen contact with paper."""
        if len(self.contact_points) < 2:
            return 0.0
        
        contact_points = np.array(self.contact_points)
        distances = np.linalg.norm(np.diff(contact_points, axis=0), axis=1)
        return np.sum(distances)
    
    def close(self):
        """Close the handwriting environment."""
        if hasattr(self, 'physics_engine'):
            self.physics_engine.close()
        
        self.is_running = False
        self.is_initialized = False
        logger.info("Handwriting environment closed")