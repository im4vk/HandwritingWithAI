"""
Base visualizer class for robotic handwriting visualization.

This module defines the abstract base class for all visualization components,
providing a consistent interface for different visualization types.
"""

import numpy as np
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, List, Tuple, Union
import logging

logger = logging.getLogger(__name__)


class BaseVisualizer(ABC):
    """
    Abstract base class for visualization components.
    
    Provides a common interface for different types of visualizers including
    real-time displays, trajectory plots, and performance dashboards.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the base visualizer.
        
        Args:
            config: Configuration dictionary containing visualization parameters
        """
        self.config = config
        self.is_initialized = False
        self.is_active = False
        
        # Display settings
        self.width = config.get('width', 800)
        self.height = config.get('height', 600)
        self.dpi = config.get('dpi', 100)
        self.background_color = config.get('background_color', 'white')
        
        # Update settings
        self.update_rate = config.get('update_rate', 30)  # FPS
        self.auto_scale = config.get('auto_scale', True)
        
        # Data storage
        self.data_buffer = []
        self.max_buffer_size = config.get('max_buffer_size', 1000)
        
        # State tracking
        self.frame_count = 0
        self.last_update_time = 0
        
        # Logging
        self.enable_logging = config.get('enable_logging', False)
        if self.enable_logging:
            self.setup_logging()
    
    @abstractmethod
    def initialize(self) -> bool:
        """
        Initialize the visualizer.
        
        Returns:
            bool: True if initialization successful
        """
        pass
    
    @abstractmethod
    def update(self, data: Dict[str, Any]) -> bool:
        """
        Update the visualization with new data.
        
        Args:
            data: Data dictionary to visualize
            
        Returns:
            bool: True if update successful
        """
        pass
    
    @abstractmethod
    def render(self) -> bool:
        """
        Render the current visualization.
        
        Returns:
            bool: True if rendering successful
        """
        pass
    
    @abstractmethod
    def close(self):
        """Close and cleanup the visualizer."""
        pass
    
    def setup_logging(self):
        """Setup logging for visualizer events."""
        if not hasattr(self, 'viz_logger'):
            self.viz_logger = logging.getLogger(f"{self.__class__.__name__}")
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            self.viz_logger.addHandler(handler)
            self.viz_logger.setLevel(logging.INFO)
    
    def add_data(self, data: Dict[str, Any]):
        """
        Add data to the visualization buffer.
        
        Args:
            data: Data to add to buffer
        """
        self.data_buffer.append(data)
        
        # Maintain buffer size
        if len(self.data_buffer) > self.max_buffer_size:
            self.data_buffer.pop(0)
    
    def clear_buffer(self):
        """Clear the data buffer."""
        self.data_buffer.clear()
        if self.enable_logging:
            self.viz_logger.info("Data buffer cleared")
    
    def get_buffer_data(self, key: Optional[str] = None) -> Union[List[Any], List[Dict[str, Any]]]:
        """
        Get data from buffer.
        
        Args:
            key: Specific data key to extract (if None, returns all data)
            
        Returns:
            List of data values or complete data dictionaries
        """
        if key is None:
            return self.data_buffer
        else:
            return [data.get(key) for data in self.data_buffer if key in data]
    
    def save_frame(self, filename: str) -> bool:
        """
        Save current frame to file.
        
        Args:
            filename: Output filename
            
        Returns:
            bool: True if save successful
        """
        try:
            # This should be implemented by subclasses
            if self.enable_logging:
                self.viz_logger.info(f"Frame saved to {filename}")
            return True
        except Exception as e:
            if self.enable_logging:
                self.viz_logger.error(f"Failed to save frame: {e}")
            return False
    
    def set_update_rate(self, fps: float):
        """
        Set visualization update rate.
        
        Args:
            fps: Frames per second
        """
        if fps > 0:
            self.update_rate = fps
            if self.enable_logging:
                self.viz_logger.info(f"Update rate set to {fps} FPS")
    
    def should_update(self, current_time: float) -> bool:
        """
        Check if visualization should update based on update rate.
        
        Args:
            current_time: Current timestamp
            
        Returns:
            bool: True if should update
        """
        if self.update_rate <= 0:
            return True
        
        time_threshold = 1.0 / self.update_rate
        if current_time - self.last_update_time >= time_threshold:
            self.last_update_time = current_time
            return True
        
        return False
    
    def get_visualization_info(self) -> Dict[str, Any]:
        """
        Get information about the visualizer.
        
        Returns:
            Dict containing visualizer metadata
        """
        return {
            'class_name': self.__class__.__name__,
            'width': self.width,
            'height': self.height,
            'dpi': self.dpi,
            'update_rate': self.update_rate,
            'is_initialized': self.is_initialized,
            'is_active': self.is_active,
            'frame_count': self.frame_count,
            'buffer_size': len(self.data_buffer),
            'max_buffer_size': self.max_buffer_size,
            'config': self.config
        }
    
    def reset(self):
        """Reset the visualizer to initial state."""
        self.clear_buffer()
        self.frame_count = 0
        self.last_update_time = 0
        if self.enable_logging:
            self.viz_logger.info("Visualizer reset")
    
    def validate_data(self, data: Dict[str, Any]) -> bool:
        """
        Validate input data.
        
        Args:
            data: Data to validate
            
        Returns:
            bool: True if data is valid
        """
        if not isinstance(data, dict):
            return False
        
        # Check for required fields (to be defined by subclasses)
        required_fields = getattr(self, 'required_fields', [])
        for field in required_fields:
            if field not in data:
                if self.enable_logging:
                    self.viz_logger.warning(f"Missing required field: {field}")
                return False
        
        return True
    
    def process_data(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process and transform data for visualization.
        
        Args:
            data: Raw input data
            
        Returns:
            Dict containing processed data
        """
        # Base implementation - just return as is
        # Subclasses should override for specific processing
        return data
    
    def get_statistics(self) -> Dict[str, Any]:
        """
        Get visualization statistics.
        
        Returns:
            Dict containing performance and usage statistics
        """
        return {
            'total_frames': self.frame_count,
            'buffer_usage': len(self.data_buffer) / self.max_buffer_size,
            'average_update_rate': self.frame_count / max(self.last_update_time, 1),
            'is_active': self.is_active,
            'is_initialized': self.is_initialized
        }
    
    def __enter__(self):
        """Context manager entry."""
        if not self.is_initialized:
            self.initialize()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.close()


class VisualizationManager:
    """
    Manager class for coordinating multiple visualizers.
    
    Handles initialization, updates, and synchronization of multiple
    visualization components.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the visualization manager.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.visualizers = {}
        self.is_active = False
        
        # Synchronization settings
        self.sync_updates = config.get('sync_updates', True)
        self.master_update_rate = config.get('master_update_rate', 30)
        
        # Logging
        self.enable_logging = config.get('enable_logging', False)
        if self.enable_logging:
            self.logger = logging.getLogger(f"{self.__class__.__name__}")
    
    def add_visualizer(self, name: str, visualizer: BaseVisualizer):
        """
        Add a visualizer to the manager.
        
        Args:
            name: Name identifier for the visualizer
            visualizer: Visualizer instance to add
        """
        self.visualizers[name] = visualizer
        if self.enable_logging:
            self.logger.info(f"Added visualizer: {name}")
    
    def remove_visualizer(self, name: str):
        """
        Remove a visualizer from the manager.
        
        Args:
            name: Name of visualizer to remove
        """
        if name in self.visualizers:
            self.visualizers[name].close()
            del self.visualizers[name]
            if self.enable_logging:
                self.logger.info(f"Removed visualizer: {name}")
    
    def initialize_all(self) -> bool:
        """
        Initialize all visualizers.
        
        Returns:
            bool: True if all initializations successful
        """
        success = True
        for name, visualizer in self.visualizers.items():
            if not visualizer.initialize():
                success = False
                if self.enable_logging:
                    self.logger.error(f"Failed to initialize visualizer: {name}")
        
        if success:
            self.is_active = True
            if self.enable_logging:
                self.logger.info("All visualizers initialized successfully")
        
        return success
    
    def update_all(self, data: Dict[str, Any]) -> bool:
        """
        Update all visualizers with new data.
        
        Args:
            data: Data to send to all visualizers
            
        Returns:
            bool: True if all updates successful
        """
        if not self.is_active:
            return False
        
        success = True
        for name, visualizer in self.visualizers.items():
            try:
                if not visualizer.update(data):
                    success = False
                    if self.enable_logging:
                        self.logger.warning(f"Update failed for visualizer: {name}")
            except Exception as e:
                success = False
                if self.enable_logging:
                    self.logger.error(f"Update error for visualizer {name}: {e}")
        
        return success
    
    def render_all(self) -> bool:
        """
        Render all visualizers.
        
        Returns:
            bool: True if all renders successful
        """
        if not self.is_active:
            return False
        
        success = True
        for name, visualizer in self.visualizers.items():
            try:
                if not visualizer.render():
                    success = False
                    if self.enable_logging:
                        self.logger.warning(f"Render failed for visualizer: {name}")
            except Exception as e:
                success = False
                if self.enable_logging:
                    self.logger.error(f"Render error for visualizer {name}: {e}")
        
        return success
    
    def close_all(self):
        """Close all visualizers."""
        for name, visualizer in self.visualizers.items():
            try:
                visualizer.close()
            except Exception as e:
                if self.enable_logging:
                    self.logger.error(f"Close error for visualizer {name}: {e}")
        
        self.visualizers.clear()
        self.is_active = False
        if self.enable_logging:
            self.logger.info("All visualizers closed")
    
    def get_visualizer(self, name: str) -> Optional[BaseVisualizer]:
        """
        Get a specific visualizer by name.
        
        Args:
            name: Name of visualizer to retrieve
            
        Returns:
            BaseVisualizer instance or None if not found
        """
        return self.visualizers.get(name)
    
    def get_all_statistics(self) -> Dict[str, Dict[str, Any]]:
        """
        Get statistics for all visualizers.
        
        Returns:
            Dict mapping visualizer names to their statistics
        """
        stats = {}
        for name, visualizer in self.visualizers.items():
            stats[name] = visualizer.get_statistics()
        return stats