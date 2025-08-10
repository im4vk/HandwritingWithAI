"""
Dexterous Hand Model
===================

Multi-finger hand simulation for pen gripping and writing control.
"""

import numpy as np
from typing import Dict, List, Tuple, Any
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)


@dataclass
class FingerState:
    """State of a single finger"""
    joint_angles: np.ndarray  # Joint angles for this finger
    joint_velocities: np.ndarray  # Joint velocities
    tip_position: np.ndarray  # 3D position of fingertip
    contact_force: float  # Force applied by this finger


@dataclass
class HandState:
    """Complete hand state"""
    fingers: List[FingerState]  # State of each finger
    grip_force: float  # Total grip force
    pen_orientation: np.ndarray  # Orientation of gripped pen
    is_gripping: bool  # Whether hand is gripping something


class DexterousHand:
    """
    Dexterous hand model with multiple fingers for pen manipulation.
    
    Features:
    - 5-finger configuration (thumb + 4 fingers)
    - Individual finger joint control
    - Force-based grip control
    - Pen orientation management
    - Collision detection between fingers
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize dexterous hand.
        
        Args:
            config: Hand configuration dictionary
        """
        self.config = config
        
        # Hand parameters
        self.num_fingers = config.get('fingers', 5)
        self.joints_per_finger = config.get('joints_per_finger', 3)
        self.finger_length = config.get('finger_length', 0.08)
        
        # Grip parameters
        self.max_grip_force = config.get('grip_force_range', [0.0, 50.0])[1]
        self.min_grip_force = config.get('grip_force_range', [0.0, 50.0])[0]
        
        # Initialize finger models
        self._setup_fingers()
        
        # Current state
        self.state = HandState(
            fingers=[self._create_finger_state(i) for i in range(self.num_fingers)],
            grip_force=0.0,
            pen_orientation=np.array([0, 0, 1]),  # Default pointing down
            is_gripping=False
        )
        
        # Pen gripping configuration
        self.pen_grip_fingers = [0, 1, 2]  # Thumb, index, middle fingers
        self.grip_positions = self._compute_grip_positions()
        
        logger.info(f"Initialized dexterous hand with {self.num_fingers} fingers")
    
    def _setup_fingers(self) -> None:
        """Setup individual finger models"""
        # Finger configurations (relative to palm)
        self.finger_configs = []
        
        # Thumb (opposable)
        thumb_config = {
            'name': 'thumb',
            'base_position': np.array([-0.02, -0.03, 0]),
            'base_orientation': np.array([0, 0, np.pi/4]),  # 45 degrees
            'joint_limits': [(-np.pi/3, np.pi/3), (-np.pi/2, np.pi/2), (-np.pi/4, np.pi/4)],
            'link_lengths': [0.025, 0.03, 0.025]
        }
        self.finger_configs.append(thumb_config)
        
        # Index finger
        index_config = {
            'name': 'index',
            'base_position': np.array([0.02, -0.01, 0]),
            'base_orientation': np.array([0, 0, 0]),
            'joint_limits': [(-np.pi/6, np.pi/2), (-np.pi/6, np.pi/2), (-np.pi/6, np.pi/2)],
            'link_lengths': [0.035, 0.025, 0.02]
        }
        self.finger_configs.append(index_config)
        
        # Middle finger
        middle_config = {
            'name': 'middle',
            'base_position': np.array([0.025, 0.01, 0]),
            'base_orientation': np.array([0, 0, 0]),
            'joint_limits': [(-np.pi/6, np.pi/2), (-np.pi/6, np.pi/2), (-np.pi/6, np.pi/2)],
            'link_lengths': [0.04, 0.03, 0.022]
        }
        self.finger_configs.append(middle_config)
        
        # Ring finger
        ring_config = {
            'name': 'ring',
            'base_position': np.array([0.02, 0.03, 0]),
            'base_orientation': np.array([0, 0, 0]),
            'joint_limits': [(-np.pi/6, np.pi/2), (-np.pi/6, np.pi/2), (-np.pi/6, np.pi/2)],
            'link_lengths': [0.035, 0.025, 0.02]
        }
        self.finger_configs.append(ring_config)
        
        # Pinky finger
        pinky_config = {
            'name': 'pinky',
            'base_position': np.array([0.015, 0.045, 0]),
            'base_orientation': np.array([0, 0, 0]),
            'joint_limits': [(-np.pi/6, np.pi/2), (-np.pi/6, np.pi/2), (-np.pi/6, np.pi/2)],
            'link_lengths': [0.025, 0.02, 0.015]
        }
        self.finger_configs.append(pinky_config)
    
    def _create_finger_state(self, finger_idx: int) -> FingerState:
        """Create initial state for a finger"""
        return FingerState(
            joint_angles=np.zeros(self.joints_per_finger),
            joint_velocities=np.zeros(self.joints_per_finger),
            tip_position=self._compute_fingertip_position(finger_idx, np.zeros(self.joints_per_finger)),
            contact_force=0.0
        )
    
    def _compute_fingertip_position(self, finger_idx: int, joint_angles: np.ndarray) -> np.ndarray:
        """Compute fingertip position for given joint angles"""
        config = self.finger_configs[finger_idx]
        
        # Start from finger base
        position = config['base_position'].copy()
        orientation = config['base_orientation'].copy()
        
        # Apply each joint transformation
        for i, (angle, length) in enumerate(zip(joint_angles, config['link_lengths'])):
            # Simplified finger kinematics (each joint bends finger forward)
            orientation[1] += angle  # Pitch rotation
            
            # Move along finger direction
            direction = np.array([
                np.cos(orientation[2]) * np.cos(orientation[1]),
                np.sin(orientation[2]) * np.cos(orientation[1]),
                np.sin(orientation[1])
            ])
            
            position += length * direction
        
        return position
    
    def _compute_grip_positions(self) -> List[np.ndarray]:
        """Compute optimal finger positions for pen gripping"""
        # Define grip positions relative to pen
        grip_positions = []
        
        # Thumb position (side grip)
        grip_positions.append(np.array([-0.01, -0.02, 0.05]))
        
        # Index finger position (front grip)
        grip_positions.append(np.array([0.01, -0.015, 0.05]))
        
        # Middle finger position (support)
        grip_positions.append(np.array([0.015, 0, 0.048]))
        
        return grip_positions
    
    def grip_pen(self, pen_position: np.ndarray, pen_orientation: np.ndarray) -> bool:
        """
        Grip a pen at specified position and orientation.
        
        Args:
            pen_position: 3D position of pen
            pen_orientation: Orientation of pen [rx, ry, rz]
            
        Returns:
            success: True if grip successful
        """
        try:
            # Compute required finger configurations for grip
            target_configurations = self._compute_grip_configurations(pen_position, pen_orientation)
            
            # Move fingers to grip positions
            for finger_idx, target_angles in target_configurations.items():
                if finger_idx < len(self.state.fingers):
                    self.state.fingers[finger_idx].joint_angles = target_angles
                    self.state.fingers[finger_idx].tip_position = \
                        self._compute_fingertip_position(finger_idx, target_angles)
            
            # Update hand state
            self.state.is_gripping = True
            self.state.pen_orientation = pen_orientation
            self.state.grip_force = self.max_grip_force * 0.5  # 50% grip force
            
            # Distribute force among gripping fingers
            force_per_finger = self.state.grip_force / len(self.pen_grip_fingers)
            for finger_idx in self.pen_grip_fingers:
                self.state.fingers[finger_idx].contact_force = force_per_finger
            
            logger.info("Successfully gripped pen")
            return True
            
        except Exception as e:
            logger.error(f"Failed to grip pen: {e}")
            return False
    
    def _compute_grip_configurations(self, pen_position: np.ndarray, 
                                   pen_orientation: np.ndarray) -> Dict[int, np.ndarray]:
        """Compute joint angles needed to grip pen"""
        configurations = {}
        
        # For each gripping finger, compute required joint angles
        for i, finger_idx in enumerate(self.pen_grip_fingers):
            # Target position for this finger
            target_pos = pen_position + self.grip_positions[i]
            
            # Simple inverse kinematics for finger
            joint_angles = self._finger_inverse_kinematics(finger_idx, target_pos)
            configurations[finger_idx] = joint_angles
        
        return configurations
    
    def _finger_inverse_kinematics(self, finger_idx: int, target_position: np.ndarray) -> np.ndarray:
        """Simple inverse kinematics for individual finger"""
        config = self.finger_configs[finger_idx]
        
        # Vector from finger base to target
        base_to_target = target_position - config['base_position']
        target_distance = np.linalg.norm(base_to_target)
        
        # Total finger length
        total_length = sum(config['link_lengths'])
        
        # Check if target is reachable
        if target_distance > total_length:
            # Extend as far as possible
            joint_angles = np.array([np.pi/4, np.pi/4, np.pi/4])
        else:
            # Simple heuristic: distribute bend across joints
            # More sophisticated IK could be implemented
            bend_factor = target_distance / total_length
            joint_angles = np.array([
                bend_factor * np.pi/6,
                bend_factor * np.pi/4,
                bend_factor * np.pi/6
            ])
        
        # Clamp to joint limits
        for i, (angle, (min_limit, max_limit)) in enumerate(zip(joint_angles, config['joint_limits'])):
            joint_angles[i] = np.clip(angle, min_limit, max_limit)
        
        return joint_angles
    
    def release_pen(self) -> None:
        """Release gripped pen"""
        # Reset finger positions
        for finger in self.state.fingers:
            finger.joint_angles.fill(0)
            finger.joint_velocities.fill(0)
            finger.contact_force = 0.0
        
        # Update positions
        for i, finger in enumerate(self.state.fingers):
            finger.tip_position = self._compute_fingertip_position(i, finger.joint_angles)
        
        # Update hand state
        self.state.is_gripping = False
        self.state.grip_force = 0.0
        
        logger.info("Released pen")
    
    def set_grip_force(self, force: float) -> None:
        """
        Set grip force for pen holding.
        
        Args:
            force: Desired grip force (0-max_grip_force)
        """
        self.state.grip_force = np.clip(force, self.min_grip_force, self.max_grip_force)
        
        if self.state.is_gripping:
            # Distribute force among gripping fingers
            force_per_finger = self.state.grip_force / len(self.pen_grip_fingers)
            for finger_idx in self.pen_grip_fingers:
                self.state.fingers[finger_idx].contact_force = force_per_finger
    
    def adjust_pen_orientation(self, new_orientation: np.ndarray) -> bool:
        """
        Adjust pen orientation while maintaining grip.
        
        Args:
            new_orientation: New pen orientation [rx, ry, rz]
            
        Returns:
            success: True if adjustment successful
        """
        if not self.state.is_gripping:
            logger.warning("Cannot adjust pen orientation - not gripping")
            return False
        
        try:
            # Compute rotation needed
            rotation_diff = new_orientation - self.state.pen_orientation
            
            # Check if rotation is within finger capabilities
            max_rotation = np.pi / 6  # 30 degrees
            if np.linalg.norm(rotation_diff) > max_rotation:
                logger.warning("Requested pen rotation exceeds finger capabilities")
                return False
            
            # Adjust finger positions to accommodate new orientation
            # (Simplified implementation - would need more sophisticated finger coordination)
            self.state.pen_orientation = new_orientation
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to adjust pen orientation: {e}")
            return False
    
    def get_pen_tip_position(self, pen_length: float = 0.15) -> np.ndarray:
        """
        Get current position of pen tip.
        
        Args:
            pen_length: Length of pen from grip point to tip
            
        Returns:
            pen_tip_position: 3D position of pen tip
        """
        if not self.state.is_gripping:
            return np.zeros(3)
        
        # Compute pen tip position based on orientation and length
        pen_direction = self._orientation_to_direction(self.state.pen_orientation)
        
        # Grip center (average of gripping finger positions)
        grip_center = np.mean([
            self.state.fingers[i].tip_position for i in self.pen_grip_fingers
        ], axis=0)
        
        # Pen tip is at grip center + length in pen direction
        pen_tip = grip_center + pen_length * pen_direction
        
        return pen_tip
    
    def _orientation_to_direction(self, orientation: np.ndarray) -> np.ndarray:
        """Convert orientation to unit direction vector"""
        rx, ry, rz = orientation
        
        # Convert Euler angles to direction vector
        direction = np.array([
            np.cos(ry) * np.cos(rz),
            np.cos(ry) * np.sin(rz),
            np.sin(ry)
        ])
        
        return direction / np.linalg.norm(direction)
    
    def check_finger_collisions(self) -> List[Tuple[int, int]]:
        """
        Check for collisions between fingers.
        
        Returns:
            collisions: List of (finger1_idx, finger2_idx) pairs in collision
        """
        collisions = []
        collision_threshold = 0.01  # 1cm
        
        for i in range(len(self.state.fingers)):
            for j in range(i + 1, len(self.state.fingers)):
                finger1_pos = self.state.fingers[i].tip_position
                finger2_pos = self.state.fingers[j].tip_position
                
                distance = np.linalg.norm(finger1_pos - finger2_pos)
                if distance < collision_threshold:
                    collisions.append((i, j))
        
        return collisions
    
    def get_grip_quality(self) -> float:
        """
        Assess quality of current grip (0-1).
        
        Returns:
            quality: Grip quality score
        """
        if not self.state.is_gripping:
            return 0.0
        
        quality_factors = []
        
        # Force distribution balance
        forces = [self.state.fingers[i].contact_force for i in self.pen_grip_fingers]
        force_balance = 1.0 - np.std(forces) / (np.mean(forces) + 1e-6)
        quality_factors.append(force_balance)
        
        # Finger position spread (good triangular grip)
        positions = [self.state.fingers[i].tip_position for i in self.pen_grip_fingers]
        if len(positions) >= 3:
            # Compute area of triangle formed by first 3 fingers
            p1, p2, p3 = positions[:3]
            area = 0.5 * np.linalg.norm(np.cross(p2 - p1, p3 - p1))
            # Normalize by expected area for good grip
            expected_area = 0.001  # 1 square cm
            spread_quality = min(area / expected_area, 1.0)
            quality_factors.append(spread_quality)
        
        # No finger collisions
        collisions = self.check_finger_collisions()
        collision_penalty = max(0, 1.0 - 0.5 * len(collisions))
        quality_factors.append(collision_penalty)
        
        # Overall quality (geometric mean)
        if quality_factors:
            return np.power(np.prod(quality_factors), 1.0 / len(quality_factors))
        else:
            return 0.0
    
    def get_state_dict(self) -> Dict[str, Any]:
        """Get hand state as dictionary"""
        return {
            'is_gripping': self.state.is_gripping,
            'grip_force': self.state.grip_force,
            'pen_orientation': self.state.pen_orientation.tolist(),
            'fingers': [
                {
                    'joint_angles': finger.joint_angles.tolist(),
                    'tip_position': finger.tip_position.tolist(),
                    'contact_force': finger.contact_force
                }
                for finger in self.state.fingers
            ],
            'grip_quality': self.get_grip_quality(),
            'collisions': self.check_finger_collisions()
        }
    
    def reset(self) -> None:
        """Reset hand to default configuration"""
        self.release_pen()
        logger.info("Hand reset to default configuration")
    
    def __str__(self) -> str:
        return f"DexterousHand(fingers={self.num_fingers}, gripping={self.state.is_gripping})"