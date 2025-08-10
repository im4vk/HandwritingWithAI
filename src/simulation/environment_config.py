"""
Environment configuration utilities for robotic handwriting simulation.

This module provides configuration management and validation for
simulation environments and physics engines.
"""

import numpy as np
from typing import Dict, Any, Optional, List, Union
from dataclasses import dataclass, field
import logging

logger = logging.getLogger(__name__)


@dataclass
class EnvironmentConfig:
    """
    Configuration class for handwriting simulation environments.
    
    Contains all necessary parameters for setting up physics simulation,
    robot models, and environment constraints.
    """
    
    # Physics engine settings
    physics_engine: str = "mujoco"  # "mujoco" or "pybullet"
    timestep: float = 0.001  # Simulation timestep in seconds
    gravity: List[float] = field(default_factory=lambda: [0.0, 0.0, -9.81])
    enable_visualization: bool = False
    
    # Robot model settings
    robot_model_path: Optional[str] = None
    robot_base_position: List[float] = field(default_factory=lambda: [0.0, 0.0, 0.1])
    robot_joint_limits: Dict[str, List[float]] = field(default_factory=dict)
    
    # Paper surface settings
    paper_size: List[float] = field(default_factory=lambda: [0.21, 0.297])  # A4 size in meters
    paper_position: List[float] = field(default_factory=lambda: [0.5, 0.0, 0.01])
    paper_friction: float = 0.8
    paper_thickness: float = 0.001
    
    # Pen settings
    pen_length: float = 0.15  # 15cm pen
    pen_mass: float = 0.02  # 20g pen
    pen_tip_radius: float = 0.0005  # 0.5mm tip radius
    
    # Contact detection settings
    contact_threshold: float = 0.001  # 1mm contact threshold
    max_contact_force: float = 10.0  # Maximum allowable contact force
    
    # Episode settings
    max_episode_steps: int = 1000
    episode_timeout: float = 30.0  # seconds
    
    # Observation space settings
    observation_includes_velocity: bool = True
    observation_includes_acceleration: bool = False
    observation_includes_forces: bool = True
    observation_history_length: int = 1
    
    # Action space settings
    action_type: str = "relative_position"  # "relative_position", "absolute_position", "velocity", "force"
    action_bounds: Dict[str, List[float]] = field(default_factory=lambda: {
        "position": [[-0.01, 0.01], [-0.01, 0.01], [-0.005, 0.005]],  # dx, dy, dz limits
        "pressure": [0.0, 1.0]  # Pressure range (normalized)
    })
    
    # Reward function settings
    reward_weights: Dict[str, float] = field(default_factory=lambda: {
        "trajectory_tracking": 1.0,
        "smoothness": 0.5,
        "pressure_control": 0.3,
        "speed_consistency": 0.2,
        "contact_stability": 0.1
    })
    
    # Target trajectory settings
    target_trajectory_type: str = "none"  # "none", "line", "circle", "text", "custom"
    target_trajectory_params: Dict[str, Any] = field(default_factory=dict)
    
    # Logging settings
    enable_logging: bool = False
    log_level: str = "INFO"
    log_trajectory: bool = True
    log_forces: bool = False
    
    # Safety settings
    workspace_bounds: Dict[str, List[float]] = field(default_factory=lambda: {
        "x": [0.2, 0.8],
        "y": [-0.3, 0.3],
        "z": [0.0, 0.2]
    })
    max_velocity: float = 0.1  # m/s
    max_acceleration: float = 1.0  # m/s²
    emergency_stop_threshold: float = 0.05  # Distance for emergency stop
    
    def __post_init__(self):
        """Validate configuration after initialization."""
        self.validate()
    
    def validate(self) -> bool:
        """
        Validate configuration parameters.
        
        Returns:
            bool: True if configuration is valid
            
        Raises:
            ValueError: If configuration is invalid
        """
        # Validate physics engine
        if self.physics_engine not in ["mujoco", "pybullet"]:
            raise ValueError(f"Unsupported physics engine: {self.physics_engine}")
        
        # Validate timestep
        if self.timestep <= 0 or self.timestep > 0.1:
            raise ValueError(f"Invalid timestep: {self.timestep}")
        
        # Validate paper size
        if len(self.paper_size) != 2 or any(s <= 0 for s in self.paper_size):
            raise ValueError(f"Invalid paper size: {self.paper_size}")
        
        # Validate positions
        if len(self.paper_position) != 3:
            raise ValueError(f"Invalid paper position: {self.paper_position}")
        
        if len(self.robot_base_position) != 3:
            raise ValueError(f"Invalid robot base position: {self.robot_base_position}")
        
        # Validate pen parameters
        if self.pen_length <= 0:
            raise ValueError(f"Invalid pen length: {self.pen_length}")
        
        if self.pen_mass <= 0:
            raise ValueError(f"Invalid pen mass: {self.pen_mass}")
        
        if self.pen_tip_radius <= 0:
            raise ValueError(f"Invalid pen tip radius: {self.pen_tip_radius}")
        
        # Validate contact threshold
        if self.contact_threshold <= 0:
            raise ValueError(f"Invalid contact threshold: {self.contact_threshold}")
        
        # Validate episode settings
        if self.max_episode_steps <= 0:
            raise ValueError(f"Invalid max episode steps: {self.max_episode_steps}")
        
        if self.episode_timeout <= 0:
            raise ValueError(f"Invalid episode timeout: {self.episode_timeout}")
        
        # Validate action type
        valid_action_types = ["relative_position", "absolute_position", "velocity", "force"]
        if self.action_type not in valid_action_types:
            raise ValueError(f"Invalid action type: {self.action_type}")
        
        # Validate workspace bounds
        for axis, bounds in self.workspace_bounds.items():
            if len(bounds) != 2 or bounds[0] >= bounds[1]:
                raise ValueError(f"Invalid workspace bounds for {axis}: {bounds}")
        
        # Validate safety limits
        if self.max_velocity <= 0:
            raise ValueError(f"Invalid max velocity: {self.max_velocity}")
        
        if self.max_acceleration <= 0:
            raise ValueError(f"Invalid max acceleration: {self.max_acceleration}")
        
        logger.info("Configuration validation passed")
        return True
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert configuration to dictionary.
        
        Returns:
            Dict containing all configuration parameters
        """
        import json
        from dataclasses import asdict
        
        config_dict = asdict(self)
        
        # Convert numpy arrays to lists if any
        def convert_numpy(obj):
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, dict):
                return {k: convert_numpy(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_numpy(item) for item in obj]
            return obj
        
        return convert_numpy(config_dict)
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'EnvironmentConfig':
        """
        Create configuration from dictionary.
        
        Args:
            config_dict: Dictionary containing configuration parameters
            
        Returns:
            EnvironmentConfig instance
        """
        return cls(**config_dict)
    
    @classmethod
    def from_yaml(cls, yaml_path: str) -> 'EnvironmentConfig':
        """
        Load configuration from YAML file.
        
        Args:
            yaml_path: Path to YAML configuration file
            
        Returns:
            EnvironmentConfig instance
        """
        try:
            import yaml
            
            with open(yaml_path, 'r') as f:
                config_dict = yaml.safe_load(f)
            
            return cls.from_dict(config_dict)
            
        except ImportError:
            logger.error("PyYAML not installed. Cannot load YAML configuration.")
            raise
        except Exception as e:
            logger.error(f"Failed to load configuration from {yaml_path}: {e}")
            raise
    
    def save_yaml(self, yaml_path: str):
        """
        Save configuration to YAML file.
        
        Args:
            yaml_path: Path to save YAML configuration file
        """
        try:
            import yaml
            
            config_dict = self.to_dict()
            
            with open(yaml_path, 'w') as f:
                yaml.dump(config_dict, f, default_flow_style=False, indent=2)
            
            logger.info(f"Configuration saved to {yaml_path}")
            
        except ImportError:
            logger.error("PyYAML not installed. Cannot save YAML configuration.")
            raise
        except Exception as e:
            logger.error(f"Failed to save configuration to {yaml_path}: {e}")
            raise
    
    def create_target_trajectory(self) -> Optional[List[np.ndarray]]:
        """
        Create target trajectory based on configuration.
        
        Returns:
            List of 3D positions defining the target trajectory
        """
        if self.target_trajectory_type == "none":
            return None
        
        elif self.target_trajectory_type == "line":
            return self._create_line_trajectory()
        
        elif self.target_trajectory_type == "circle":
            return self._create_circle_trajectory()
        
        elif self.target_trajectory_type == "text":
            return self._create_text_trajectory()
        
        elif self.target_trajectory_type == "custom":
            return self.target_trajectory_params.get("points", None)
        
        else:
            logger.warning(f"Unknown trajectory type: {self.target_trajectory_type}")
            return None
    
    def _create_line_trajectory(self) -> List[np.ndarray]:
        """Create a straight line trajectory."""
        params = self.target_trajectory_params
        start = np.array(params.get("start", [0.4, -0.1, 0.02]))
        end = np.array(params.get("end", [0.6, 0.1, 0.02]))
        num_points = params.get("num_points", 100)
        
        trajectory = []
        for i in range(num_points):
            t = i / (num_points - 1)
            point = start + t * (end - start)
            trajectory.append(point)
        
        return trajectory
    
    def _create_circle_trajectory(self) -> List[np.ndarray]:
        """Create a circular trajectory."""
        params = self.target_trajectory_params
        center = np.array(params.get("center", [0.5, 0.0, 0.02]))
        radius = params.get("radius", 0.05)
        num_points = params.get("num_points", 100)
        
        trajectory = []
        for i in range(num_points):
            angle = 2 * np.pi * i / num_points
            x = center[0] + radius * np.cos(angle)
            y = center[1] + radius * np.sin(angle)
            z = center[2]
            trajectory.append(np.array([x, y, z]))
        
        return trajectory
    
    def _create_text_trajectory(self) -> List[np.ndarray]:
        """Create a trajectory for writing text."""
        params = self.target_trajectory_params
        text = params.get("text", "Hello")
        start_pos = np.array(params.get("start_position", [0.4, 0.0, 0.02]))
        char_width = params.get("character_width", 0.02)
        char_height = params.get("character_height", 0.03)
        
        # Simplified text trajectory generation
        # In practice, this would use more sophisticated font rendering
        trajectory = []
        current_x = start_pos[0]
        
        for char in text:
            # Simple character shapes (very basic)
            if char == 'H':
                # Vertical line, horizontal line, vertical line
                char_points = [
                    [current_x, start_pos[1] - char_height/2, start_pos[2]],
                    [current_x, start_pos[1] + char_height/2, start_pos[2]],
                    [current_x, start_pos[1], start_pos[2]],
                    [current_x + char_width, start_pos[1], start_pos[2]],
                    [current_x + char_width, start_pos[1] - char_height/2, start_pos[2]],
                    [current_x + char_width, start_pos[1] + char_height/2, start_pos[2]]
                ]
            else:
                # Default simple line for other characters
                char_points = [
                    [current_x, start_pos[1] - char_height/2, start_pos[2]],
                    [current_x + char_width, start_pos[1] + char_height/2, start_pos[2]]
                ]
            
            trajectory.extend([np.array(point) for point in char_points])
            current_x += char_width * 1.5  # Character spacing
        
        return trajectory
    
    def get_observation_space_info(self) -> Dict[str, Any]:
        """
        Get information about the observation space.
        
        Returns:
            Dict containing observation space details
        """
        obs_dim = 3  # Position
        
        if self.observation_includes_velocity:
            obs_dim += 3  # Velocity
        
        if self.observation_includes_acceleration:
            obs_dim += 3  # Acceleration
        
        obs_dim += 4  # Orientation (quaternion)
        obs_dim += 1  # Contact state
        
        if self.observation_includes_forces:
            obs_dim += 3  # Contact forces
        
        obs_dim += 3  # Target position
        obs_dim += 1  # Position error
        
        # Multiply by history length
        obs_dim *= self.observation_history_length
        
        return {
            "dimension": obs_dim,
            "includes_velocity": self.observation_includes_velocity,
            "includes_acceleration": self.observation_includes_acceleration,
            "includes_forces": self.observation_includes_forces,
            "history_length": self.observation_history_length
        }
    
    def get_action_space_info(self) -> Dict[str, Any]:
        """
        Get information about the action space.
        
        Returns:
            Dict containing action space details
        """
        if self.action_type == "relative_position":
            action_dim = 4  # dx, dy, dz, pressure
        elif self.action_type == "absolute_position":
            action_dim = 4  # x, y, z, pressure
        elif self.action_type == "velocity":
            action_dim = 4  # vx, vy, vz, pressure
        elif self.action_type == "force":
            action_dim = 6  # fx, fy, fz, tx, ty, tz
        else:
            action_dim = 4  # Default
        
        return {
            "dimension": action_dim,
            "type": self.action_type,
            "bounds": self.action_bounds
        }


def create_default_config() -> EnvironmentConfig:
    """
    Create default environment configuration.
    
    Returns:
        EnvironmentConfig with default settings
    """
    return EnvironmentConfig()


def create_mujoco_config() -> EnvironmentConfig:
    """
    Create MuJoCo-specific environment configuration.
    
    Returns:
        EnvironmentConfig optimized for MuJoCo
    """
    config = EnvironmentConfig()
    config.physics_engine = "mujoco"
    config.timestep = 0.001  # 1ms for high precision
    config.enable_visualization = False  # Better performance
    return config


def create_pybullet_config() -> EnvironmentConfig:
    """
    Create PyBullet-specific environment configuration.
    
    Returns:
        EnvironmentConfig optimized for PyBullet
    """
    config = EnvironmentConfig()
    config.physics_engine = "pybullet"
    config.timestep = 0.005  # 5ms for reasonable performance
    config.enable_visualization = True  # Good visualization in PyBullet
    return config


def create_training_config() -> EnvironmentConfig:
    """
    Create configuration optimized for training.
    
    Returns:
        EnvironmentConfig optimized for RL training
    """
    config = EnvironmentConfig()
    config.enable_visualization = False
    config.enable_logging = False
    config.max_episode_steps = 500  # Shorter episodes for training
    config.timestep = 0.002  # Slightly larger timestep for speed
    return config


def create_evaluation_config() -> EnvironmentConfig:
    """
    Create configuration optimized for evaluation.
    
    Returns:
        EnvironmentConfig optimized for evaluation/testing
    """
    config = EnvironmentConfig()
    config.enable_visualization = True
    config.enable_logging = True
    config.log_trajectory = True
    config.max_episode_steps = 2000  # Longer episodes for evaluation
    config.timestep = 0.001  # High precision for evaluation
    return config