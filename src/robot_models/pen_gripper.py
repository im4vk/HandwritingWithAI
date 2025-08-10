"""
Pen Gripper Module
=================

Specialized end-effector for pen gripping and writing control.
Simplified alternative to full dexterous hand for handwriting tasks.
"""

import numpy as np
from typing import Dict, Any, Optional
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)


@dataclass
class PenState:
    """State of the gripped pen"""
    position: np.ndarray  # 3D position of pen grip point
    orientation: np.ndarray  # Pen orientation [rx, ry, rz]
    tip_position: np.ndarray  # 3D position of pen tip
    pressure: float  # Writing pressure (0-1)
    is_gripped: bool  # Whether pen is gripped


class PenGripper:
    """
    Specialized pen gripper for handwriting tasks.
    
    Features:
    - Simple 3-finger gripper optimized for pen holding
    - Pressure control for writing force
    - Pen orientation adjustment
    - Writing surface contact detection
    - Grip stability monitoring
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize pen gripper.
        
        Args:
            config: Gripper configuration dictionary
        """
        self.config = config
        
        # Gripper parameters
        self.finger_length = config.get('finger_length', 0.03)
        self.max_grip_force = config.get('max_grip_force', 20.0)
        self.min_grip_force = config.get('min_grip_force', 1.0)
        
        # Pen parameters
        self.pen_length = config.get('pen_length', 0.15)
        self.pen_radius = config.get('pen_radius', 0.003)
        self.pen_weight = config.get('pen_weight', 0.02)
        
        # Writing parameters
        self.max_writing_pressure = config.get('max_writing_pressure', 10.0)
        self.min_writing_pressure = config.get('min_writing_pressure', 0.5)
        
        # Current state
        self.state = PenState(
            position=np.zeros(3),
            orientation=np.array([0, 0, -np.pi/2]),  # Pointing down
            tip_position=np.zeros(3),
            pressure=0.0,
            is_gripped=False
        )
        
        # Grip configuration (3-finger pinch grip)
        self.grip_width = 0.02  # Distance between fingers
        self.grip_offset = 0.05  # Distance from pen tip to grip point
        
        # Control parameters
        self.pressure_sensitivity = 0.1
        self.orientation_limits = np.array([np.pi/4, np.pi/4, np.pi])  # rx, ry, rz limits
        
        logger.info("Initialized pen gripper")
    
    def grip_pen(self, pen_position: np.ndarray, 
                 pen_orientation: Optional[np.ndarray] = None) -> bool:
        """
        Grip pen at specified position and orientation.
        
        Args:
            pen_position: 3D position where to grip the pen
            pen_orientation: Pen orientation [rx, ry, rz] (optional)
            
        Returns:
            success: True if grip successful
        """
        try:
            # Use default orientation if not specified
            if pen_orientation is None:
                pen_orientation = np.array([0, 0, -np.pi/2])  # Pointing down
            
            # Validate orientation limits
            if not self._check_orientation_limits(pen_orientation):
                logger.warning("Pen orientation exceeds gripper limits")
                return False
            
            # Update state
            self.state.position = pen_position.copy()
            self.state.orientation = pen_orientation.copy()
            self.state.is_gripped = True
            
            # Compute pen tip position
            self._update_pen_tip_position()
            
            # Set initial grip force
            self.set_grip_force(0.5)  # 50% grip strength
            
            logger.info(f"Gripped pen at position {pen_position}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to grip pen: {e}")
            return False
    
    def release_pen(self) -> None:
        """Release the gripped pen"""
        self.state.is_gripped = False
        self.state.pressure = 0.0
        logger.info("Released pen")
    
    def set_grip_force(self, force_ratio: float) -> None:
        """
        Set grip force as ratio of maximum force.
        
        Args:
            force_ratio: Grip force ratio (0-1)
        """
        force_ratio = np.clip(force_ratio, 0.0, 1.0)
        actual_force = self.min_grip_force + force_ratio * (self.max_grip_force - self.min_grip_force)
        
        # Update internal force (could be used for physics simulation)
        self._grip_force = actual_force
        
        if self.state.is_gripped:
            logger.debug(f"Set grip force to {actual_force:.1f}N ({force_ratio*100:.1f}%)")
    
    def set_writing_pressure(self, pressure_ratio: float) -> None:
        """
        Set writing pressure for contact with writing surface.
        
        Args:
            pressure_ratio: Writing pressure ratio (0-1)
        """
        if not self.state.is_gripped:
            logger.warning("Cannot set writing pressure - pen not gripped")
            return
        
        pressure_ratio = np.clip(pressure_ratio, 0.0, 1.0)
        self.state.pressure = pressure_ratio
        
        # Convert to actual force
        actual_pressure = self.min_writing_pressure + \
                         pressure_ratio * (self.max_writing_pressure - self.min_writing_pressure)
        
        logger.debug(f"Set writing pressure to {actual_pressure:.1f}N ({pressure_ratio*100:.1f}%)")
    
    def adjust_orientation(self, new_orientation: np.ndarray) -> bool:
        """
        Adjust pen orientation while maintaining grip.
        
        Args:
            new_orientation: New pen orientation [rx, ry, rz]
            
        Returns:
            success: True if adjustment successful
        """
        if not self.state.is_gripped:
            logger.warning("Cannot adjust orientation - pen not gripped")
            return False
        
        # Check orientation limits
        if not self._check_orientation_limits(new_orientation):
            logger.warning("Requested orientation exceeds gripper limits")
            return False
        
        # Update orientation
        self.state.orientation = new_orientation.copy()
        
        # Update pen tip position
        self._update_pen_tip_position()
        
        logger.debug(f"Adjusted pen orientation to {np.rad2deg(new_orientation)}")
        return True
    
    def move_to_position(self, new_position: np.ndarray) -> bool:
        """
        Move gripper (and pen) to new position.
        
        Args:
            new_position: New gripper position
            
        Returns:
            success: True if movement successful
        """
        if not self.state.is_gripped:
            logger.warning("Cannot move - pen not gripped")
            return False
        
        # Update position
        self.state.position = new_position.copy()
        
        # Update pen tip position
        self._update_pen_tip_position()
        
        return True
    
    def _update_pen_tip_position(self) -> None:
        """Update pen tip position based on current grip position and orientation"""
        if not self.state.is_gripped:
            self.state.tip_position = np.zeros(3)
            return
        
        # Convert orientation to direction vector
        direction = self._orientation_to_direction(self.state.orientation)
        
        # Pen tip is at grip position + pen_length in pen direction
        # Accounting for grip offset from actual pen tip
        effective_length = self.pen_length - self.grip_offset
        self.state.tip_position = self.state.position + effective_length * direction
    
    def _orientation_to_direction(self, orientation: np.ndarray) -> np.ndarray:
        """Convert Euler angles to unit direction vector"""
        rx, ry, rz = orientation
        
        # Create rotation matrix
        cos_rx, sin_rx = np.cos(rx), np.sin(rx)
        cos_ry, sin_ry = np.cos(ry), np.sin(ry)
        cos_rz, sin_rz = np.cos(rz), np.sin(rz)
        
        # Rotation matrix (ZYX convention)
        R = np.array([
            [cos_ry * cos_rz, -cos_ry * sin_rz, sin_ry],
            [sin_rx * sin_ry * cos_rz + cos_rx * sin_rz,
             -sin_rx * sin_ry * sin_rz + cos_rx * cos_rz, -sin_rx * cos_ry],
            [-cos_rx * sin_ry * cos_rz + sin_rx * sin_rz,
             cos_rx * sin_ry * sin_rz + sin_rx * cos_rz, cos_rx * cos_ry]
        ])
        
        # Default pen direction is along negative z-axis (pointing down)
        default_direction = np.array([0, 0, -1])
        
        # Apply rotation
        direction = R @ default_direction
        
        return direction / np.linalg.norm(direction)
    
    def _check_orientation_limits(self, orientation: np.ndarray) -> bool:
        """Check if orientation is within gripper limits"""
        return np.all(np.abs(orientation) <= self.orientation_limits)
    
    def get_pen_tip_position(self) -> np.ndarray:
        """Get current pen tip position"""
        return self.state.tip_position.copy()
    
    def get_pen_direction(self) -> np.ndarray:
        """Get current pen direction vector"""
        if not self.state.is_gripped:
            return np.array([0, 0, -1])  # Default down direction
        
        return self._orientation_to_direction(self.state.orientation)
    
    def is_touching_surface(self, surface_z: float, tolerance: float = 0.001) -> bool:
        """
        Check if pen is touching the writing surface.
        
        Args:
            surface_z: Z-coordinate of writing surface
            tolerance: Contact detection tolerance
            
        Returns:
            is_touching: True if pen tip is touching surface
        """
        if not self.state.is_gripped:
            return False
        
        pen_tip_z = self.state.tip_position[2]
        return abs(pen_tip_z - surface_z) <= tolerance
    
    def compute_writing_force(self, surface_z: float) -> float:
        """
        Compute writing force based on pen position and pressure setting.
        
        Args:
            surface_z: Z-coordinate of writing surface
            
        Returns:
            force: Writing force in Newtons
        """
        if not self.state.is_gripped or not self.is_touching_surface(surface_z):
            return 0.0
        
        # Base force from pressure setting
        base_force = self.state.pressure * self.max_writing_pressure
        
        # Add pen weight contribution
        gravity_force = self.pen_weight * 9.81  # Weight in Newtons
        
        # Total force (pressure + gravity component)
        pen_direction = self.get_pen_direction()
        gravity_contribution = abs(pen_direction[2]) * gravity_force  # Vertical component
        
        total_force = base_force + gravity_contribution
        
        return total_force
    
    def get_grip_stability(self) -> float:
        """
        Assess grip stability (0-1).
        
        Returns:
            stability: Grip stability score
        """
        if not self.state.is_gripped:
            return 0.0
        
        stability_factors = []
        
        # Grip force adequacy
        if hasattr(self, '_grip_force'):
            force_ratio = (self._grip_force - self.min_grip_force) / \
                         (self.max_grip_force - self.min_grip_force)
            # Optimal around 50% grip force
            force_stability = 1.0 - 2.0 * abs(force_ratio - 0.5)
            stability_factors.append(max(0.0, force_stability))
        
        # Orientation stability (closer to vertical is more stable)
        vertical_angle = abs(self.state.orientation[1])  # Pitch angle
        orientation_stability = max(0.0, 1.0 - 2.0 * vertical_angle / np.pi)
        stability_factors.append(orientation_stability)
        
        # Overall stability
        return np.mean(stability_factors) if stability_factors else 0.0
    
    def get_state_dict(self) -> Dict[str, Any]:
        """Get gripper state as dictionary"""
        return {
            'is_gripped': self.state.is_gripped,
            'position': self.state.position.tolist(),
            'orientation': self.state.orientation.tolist(),
            'orientation_degrees': np.rad2deg(self.state.orientation).tolist(),
            'tip_position': self.state.tip_position.tolist(),
            'writing_pressure': self.state.pressure,
            'pen_direction': self.get_pen_direction().tolist(),
            'grip_stability': self.get_grip_stability(),
            'grip_force': getattr(self, '_grip_force', 0.0)
        }
    
    def reset(self) -> None:
        """Reset gripper to default state"""
        self.release_pen()
        self.state.position.fill(0)
        self.state.orientation = np.array([0, 0, -np.pi/2])
        self.state.tip_position.fill(0)
        logger.info("Pen gripper reset")
    
    def __str__(self) -> str:
        return f"PenGripper(gripped={self.state.is_gripped}, pressure={self.state.pressure:.2f})"