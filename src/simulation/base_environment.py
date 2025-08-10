"""
Base environment class for robotic handwriting simulations.

This module defines the abstract base class for all simulation environments,
providing a consistent interface for different physics engines and robot models.
"""

import numpy as np
from abc import ABC, abstractmethod
from typing import Dict, Any, Tuple, Optional, List
import logging

logger = logging.getLogger(__name__)


class BaseEnvironment(ABC):
    """
    Abstract base class for robotic handwriting simulation environments.
    
    This class defines the interface that all simulation environments must implement,
    providing methods for initialization, stepping, resetting, and state management.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the base environment.
        
        Args:
            config: Configuration dictionary containing environment parameters
        """
        self.config = config
        self.timestep = config.get('timestep', 0.001)  # Default 1ms timestep
        self.max_episode_steps = config.get('max_episode_steps', 1000)
        self.current_step = 0
        self.episode_count = 0
        
        # State tracking
        self.is_initialized = False
        self.is_running = False
        self.last_action = None
        self.last_observation = None
        
        # Logging
        self.enable_logging = config.get('enable_logging', False)
        if self.enable_logging:
            self.setup_logging()
    
    @abstractmethod
    def initialize(self) -> bool:
        """
        Initialize the simulation environment.
        
        Returns:
            bool: True if initialization successful, False otherwise
        """
        pass
    
    @abstractmethod
    def reset(self) -> np.ndarray:
        """
        Reset the environment to initial state.
        
        Returns:
            np.ndarray: Initial observation
        """
        pass
    
    @abstractmethod
    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, Dict[str, Any]]:
        """
        Execute one simulation step.
        
        Args:
            action: Action to execute
            
        Returns:
            Tuple containing:
                - observation: Current state observation
                - reward: Reward for the action
                - done: Whether episode is complete
                - info: Additional information dictionary
        """
        pass
    
    @abstractmethod
    def get_observation(self) -> np.ndarray:
        """
        Get current observation from the environment.
        
        Returns:
            np.ndarray: Current observation
        """
        pass
    
    @abstractmethod
    def get_state(self) -> Dict[str, Any]:
        """
        Get current simulation state.
        
        Returns:
            Dict containing complete simulation state
        """
        pass
    
    @abstractmethod
    def set_state(self, state: Dict[str, Any]) -> bool:
        """
        Set simulation state.
        
        Args:
            state: State dictionary to set
            
        Returns:
            bool: True if state was set successfully
        """
        pass
    
    @abstractmethod
    def close(self):
        """Close and cleanup the environment."""
        pass
    
    def setup_logging(self):
        """Setup logging for environment events."""
        if not hasattr(self, 'env_logger'):
            self.env_logger = logging.getLogger(f"{self.__class__.__name__}")
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            self.env_logger.addHandler(handler)
            self.env_logger.setLevel(logging.INFO)
    
    def log_step(self, action: np.ndarray, reward: float, done: bool):
        """Log information about a simulation step."""
        if self.enable_logging and hasattr(self, 'env_logger'):
            self.env_logger.info(
                f"Step {self.current_step}: Action shape={action.shape}, "
                f"Reward={reward:.4f}, Done={done}"
            )
    
    def get_environment_info(self) -> Dict[str, Any]:
        """
        Get general information about the environment.
        
        Returns:
            Dict containing environment metadata
        """
        return {
            'class_name': self.__class__.__name__,
            'timestep': self.timestep,
            'max_episode_steps': self.max_episode_steps,
            'current_step': self.current_step,
            'episode_count': self.episode_count,
            'is_initialized': self.is_initialized,
            'is_running': self.is_running,
            'config': self.config
        }
    
    def validate_action(self, action: np.ndarray) -> bool:
        """
        Validate that an action is within acceptable bounds.
        
        Args:
            action: Action to validate
            
        Returns:
            bool: True if action is valid
        """
        if action is None:
            return False
        
        if not isinstance(action, np.ndarray):
            return False
        
        # Check for NaN or infinite values
        if np.any(np.isnan(action)) or np.any(np.isinf(action)):
            return False
        
        return True
    
    def __enter__(self):
        """Context manager entry."""
        if not self.is_initialized:
            self.initialize()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.close()