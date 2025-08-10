"""
Movement Primitives for Handwriting
===================================

Basic movement primitives that can be combined to create
complex handwriting trajectories. Based on motor control theory
and empirical observations of human handwriting patterns.
"""

import numpy as np
from typing import Dict, Any, List, Tuple, Optional, Union
from dataclasses import dataclass
from enum import Enum
import logging

logger = logging.getLogger(__name__)


class PrimitiveType(Enum):
    """Types of movement primitives."""
    LINE = "line"
    ARC = "arc"
    CIRCLE = "circle"
    SPIRAL = "spiral"
    LOOP = "loop"
    HOOK = "hook"
    STROKE = "stroke"
    CURVE = "curve"


@dataclass
class StrokePrimitive:
    """
    Basic stroke primitive with parameters.
    
    Attributes:
        primitive_type: Type of primitive
        parameters: Primitive-specific parameters
        duration: Duration in seconds
        start_position: Starting position (relative)
        end_position: Ending position (relative)
        velocity_profile: Velocity profile type
    """
    primitive_type: PrimitiveType
    parameters: Dict[str, Any]
    duration: float = 0.5
    start_position: Optional[np.ndarray] = None
    end_position: Optional[np.ndarray] = None
    velocity_profile: str = "bell_shaped"
    
    def __post_init__(self):
        """Initialize default positions if not provided."""
        if self.start_position is None:
            self.start_position = np.array([0.0, 0.0])
        
        if self.end_position is None:
            # Set default end position based on primitive type
            if self.primitive_type == PrimitiveType.LINE:
                length = self.parameters.get('length', 0.01)
                angle = self.parameters.get('angle', 0.0)
                self.end_position = self.start_position + length * np.array([np.cos(angle), np.sin(angle)])
            else:
                self.end_position = self.start_position.copy()


class MovementPrimitive:
    """
    Generator for basic movement primitives used in handwriting.
    
    Implements various types of strokes, curves, and geometric shapes
    that form the building blocks of handwritten characters.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize movement primitive generator.
        
        Args:
            config: Generator configuration
        """
        self.config = config
        
        # Generation parameters
        self.sampling_rate = config.get('sampling_rate', 100)  # Hz
        self.default_duration = config.get('default_duration', 0.5)  # seconds
        self.velocity_scaling = config.get('velocity_scaling', 1.0)
        
        # Smoothness parameters
        self.smoothing_factor = config.get('smoothing_factor', 0.1)
        self.minimum_jerk = config.get('minimum_jerk', True)
        
        logger.info("Initialized MovementPrimitive generator")
    
    def generate_line_primitive(self,
                              start_pos: np.ndarray,
                              end_pos: np.ndarray,
                              duration: Optional[float] = None,
                              velocity_profile: str = "bell_shaped") -> Dict[str, np.ndarray]:
        """
        Generate straight line primitive.
        
        Args:
            start_pos: Starting position [x, y]
            end_pos: Ending position [x, y]
            duration: Duration in seconds
            velocity_profile: Velocity profile type
            
        Returns:
            primitive_data: Generated primitive trajectory
        """
        if duration is None:
            duration = self.default_duration
        
        # Generate time points
        dt = 1.0 / self.sampling_rate
        time_points = np.arange(0, duration, dt)
        n_points = len(time_points)
        
        # Generate position trajectory
        positions = np.zeros((n_points, 2))
        for i, t in enumerate(time_points):
            alpha = t / duration  # Interpolation factor
            positions[i] = start_pos + alpha * (end_pos - start_pos)
        
        # Generate velocity profile
        velocities = self._generate_velocity_profile(
            positions, time_points, velocity_profile
        )
        
        # Compute accelerations
        accelerations = np.diff(velocities, axis=0, prepend=velocities[:1]) / dt
        
        return {
            'positions': positions,
            'velocities': velocities,
            'accelerations': accelerations,
            'time': time_points,
            'primitive_type': 'line'
        }
    
    def generate_arc_primitive(self,
                             center: np.ndarray,
                             radius: float,
                             start_angle: float,
                             end_angle: float,
                             duration: Optional[float] = None,
                             velocity_profile: str = "bell_shaped") -> Dict[str, np.ndarray]:
        """
        Generate circular arc primitive.
        
        Args:
            center: Arc center [x, y]
            radius: Arc radius
            start_angle: Starting angle in radians
            end_angle: Ending angle in radians
            duration: Duration in seconds
            velocity_profile: Velocity profile type
            
        Returns:
            primitive_data: Generated primitive trajectory
        """
        if duration is None:
            duration = self.default_duration
        
        # Generate time points
        dt = 1.0 / self.sampling_rate
        time_points = np.arange(0, duration, dt)
        n_points = len(time_points)
        
        # Generate angular trajectory
        total_angle = end_angle - start_angle
        angles = np.zeros(n_points)
        
        for i, t in enumerate(time_points):
            alpha = t / duration
            angles[i] = start_angle + alpha * total_angle
        
        # Convert to Cartesian coordinates
        positions = np.zeros((n_points, 2))
        positions[:, 0] = center[0] + radius * np.cos(angles)
        positions[:, 1] = center[1] + radius * np.sin(angles)
        
        # Generate velocity profile
        velocities = self._generate_velocity_profile(
            positions, time_points, velocity_profile
        )
        
        # Compute accelerations
        accelerations = np.diff(velocities, axis=0, prepend=velocities[:1]) / dt
        
        return {
            'positions': positions,
            'velocities': velocities,
            'accelerations': accelerations,
            'time': time_points,
            'primitive_type': 'arc',
            'angles': angles
        }
    
    def generate_circle_primitive(self,
                                center: np.ndarray,
                                radius: float,
                                duration: Optional[float] = None,
                                clockwise: bool = False) -> Dict[str, np.ndarray]:
        """
        Generate complete circle primitive.
        
        Args:
            center: Circle center [x, y]
            radius: Circle radius
            duration: Duration in seconds
            clockwise: Direction of drawing
            
        Returns:
            primitive_data: Generated primitive trajectory
        """
        start_angle = 0.0
        end_angle = -2 * np.pi if clockwise else 2 * np.pi
        
        return self.generate_arc_primitive(
            center, radius, start_angle, end_angle, duration
        )
    
    def generate_spiral_primitive(self,
                                center: np.ndarray,
                                initial_radius: float,
                                final_radius: float,
                                num_turns: float,
                                duration: Optional[float] = None) -> Dict[str, np.ndarray]:
        """
        Generate spiral primitive.
        
        Args:
            center: Spiral center [x, y]
            initial_radius: Starting radius
            final_radius: Ending radius
            num_turns: Number of complete turns
            duration: Duration in seconds
            
        Returns:
            primitive_data: Generated primitive trajectory
        """
        if duration is None:
            duration = self.default_duration
        
        # Generate time points
        dt = 1.0 / self.sampling_rate
        time_points = np.arange(0, duration, dt)
        n_points = len(time_points)
        
        # Generate spiral trajectory
        positions = np.zeros((n_points, 2))
        
        total_angle = 2 * np.pi * num_turns
        
        for i, t in enumerate(time_points):
            alpha = t / duration
            
            # Radius varies linearly
            current_radius = initial_radius + alpha * (final_radius - initial_radius)
            
            # Angle varies linearly
            current_angle = alpha * total_angle
            
            positions[i, 0] = center[0] + current_radius * np.cos(current_angle)
            positions[i, 1] = center[1] + current_radius * np.sin(current_angle)
        
        # Generate velocity profile
        velocities = self._generate_velocity_profile(
            positions, time_points, "bell_shaped"
        )
        
        # Compute accelerations
        accelerations = np.diff(velocities, axis=0, prepend=velocities[:1]) / dt
        
        return {
            'positions': positions,
            'velocities': velocities,
            'accelerations': accelerations,
            'time': time_points,
            'primitive_type': 'spiral'
        }
    
    def generate_loop_primitive(self,
                              start_pos: np.ndarray,
                              loop_height: float,
                              loop_width: float,
                              duration: Optional[float] = None) -> Dict[str, np.ndarray]:
        """
        Generate loop primitive (common in cursive writing).
        
        Args:
            start_pos: Starting position [x, y]
            loop_height: Height of the loop
            loop_width: Width of the loop
            duration: Duration in seconds
            
        Returns:
            primitive_data: Generated primitive trajectory
        """
        if duration is None:
            duration = self.default_duration
        
        # Generate time points
        dt = 1.0 / self.sampling_rate
        time_points = np.arange(0, duration, dt)
        n_points = len(time_points)
        
        # Loop is essentially an oval/ellipse
        positions = np.zeros((n_points, 2))
        
        for i, t in enumerate(time_points):
            # Parameter from 0 to 2π
            theta = (t / duration) * 2 * np.pi
            
            # Ellipse equations
            x = start_pos[0] + (loop_width / 2) * np.cos(theta)
            y = start_pos[1] + (loop_height / 2) * np.sin(theta)
            
            positions[i] = [x, y]
        
        # Generate velocity profile
        velocities = self._generate_velocity_profile(
            positions, time_points, "bell_shaped"
        )
        
        # Compute accelerations
        accelerations = np.diff(velocities, axis=0, prepend=velocities[:1]) / dt
        
        return {
            'positions': positions,
            'velocities': velocities,
            'accelerations': accelerations,
            'time': time_points,
            'primitive_type': 'loop'
        }
    
    def generate_hook_primitive(self,
                              start_pos: np.ndarray,
                              end_pos: np.ndarray,
                              hook_amplitude: float,
                              duration: Optional[float] = None) -> Dict[str, np.ndarray]:
        """
        Generate hook primitive (J-shaped curve).
        
        Args:
            start_pos: Starting position [x, y]
            end_pos: Ending position [x, y] 
            hook_amplitude: Amplitude of the hook curve
            duration: Duration in seconds
            
        Returns:
            primitive_data: Generated primitive trajectory
        """
        if duration is None:
            duration = self.default_duration
        
        # Generate time points
        dt = 1.0 / self.sampling_rate
        time_points = np.arange(0, duration, dt)
        n_points = len(time_points)
        
        # Generate hook trajectory using parametric curve
        positions = np.zeros((n_points, 2))
        
        direction = end_pos - start_pos
        length = np.linalg.norm(direction)
        unit_direction = direction / (length + 1e-8)
        
        # Perpendicular direction for hook curvature
        perp_direction = np.array([-unit_direction[1], unit_direction[0]])
        
        for i, t in enumerate(time_points):
            alpha = t / duration
            
            # Base linear trajectory
            linear_pos = start_pos + alpha * direction
            
            # Hook curvature (sine wave that starts and ends at zero)
            hook_factor = np.sin(np.pi * alpha) * alpha * (1 - alpha)
            hook_offset = hook_amplitude * hook_factor * perp_direction
            
            positions[i] = linear_pos + hook_offset
        
        # Generate velocity profile
        velocities = self._generate_velocity_profile(
            positions, time_points, "bell_shaped"
        )
        
        # Compute accelerations
        accelerations = np.diff(velocities, axis=0, prepend=velocities[:1]) / dt
        
        return {
            'positions': positions,
            'velocities': velocities,
            'accelerations': accelerations,
            'time': time_points,
            'primitive_type': 'hook'
        }
    
    def generate_curve_primitive(self,
                               control_points: List[np.ndarray],
                               duration: Optional[float] = None,
                               curve_type: str = "quadratic") -> Dict[str, np.ndarray]:
        """
        Generate smooth curve primitive using control points.
        
        Args:
            control_points: List of control points
            duration: Duration in seconds
            curve_type: Type of curve ("quadratic", "cubic")
            
        Returns:
            primitive_data: Generated primitive trajectory
        """
        if duration is None:
            duration = self.default_duration
        
        # Generate time points
        dt = 1.0 / self.sampling_rate
        time_points = np.arange(0, duration, dt)
        n_points = len(time_points)
        
        if curve_type == "quadratic" and len(control_points) >= 3:
            positions = self._generate_quadratic_bezier(control_points[:3], time_points)
        elif curve_type == "cubic" and len(control_points) >= 4:
            positions = self._generate_cubic_bezier(control_points[:4], time_points)
        else:
            # Fallback to linear interpolation
            positions = self._generate_linear_interpolation(control_points, time_points)
        
        # Generate velocity profile
        velocities = self._generate_velocity_profile(
            positions, time_points, "bell_shaped"
        )
        
        # Compute accelerations
        accelerations = np.diff(velocities, axis=0, prepend=velocities[:1]) / dt
        
        return {
            'positions': positions,
            'velocities': velocities,
            'accelerations': accelerations,
            'time': time_points,
            'primitive_type': 'curve'
        }
    
    def _generate_velocity_profile(self,
                                 positions: np.ndarray,
                                 time_points: np.ndarray,
                                 profile_type: str) -> np.ndarray:
        """
        Generate velocity profile for given positions.
        
        Args:
            positions: Position trajectory [n_points, 2]
            time_points: Time points
            profile_type: Type of velocity profile
            
        Returns:
            velocities: Velocity profile [n_points, 2]
        """
        n_points = len(positions)
        dt = time_points[1] - time_points[0] if len(time_points) > 1 else 1.0 / self.sampling_rate
        
        if profile_type == "constant":
            # Constant velocity
            if n_points > 1:
                total_displacement = positions[-1] - positions[0]
                total_time = time_points[-1] - time_points[0]
                constant_velocity = total_displacement / total_time
                velocities = np.tile(constant_velocity, (n_points, 1))
            else:
                velocities = np.zeros((n_points, 2))
        
        elif profile_type == "bell_shaped":
            # Bell-shaped velocity profile (minimum jerk)
            velocities = self._generate_minimum_jerk_velocity(positions, time_points)
        
        elif profile_type == "trapezoidal":
            # Trapezoidal velocity profile
            velocities = self._generate_trapezoidal_velocity(positions, time_points)
        
        else:
            # Default: compute from position differences
            velocities = np.diff(positions, axis=0, prepend=positions[:1]) / dt
        
        # Apply velocity scaling
        velocities *= self.velocity_scaling
        
        return velocities
    
    def _generate_minimum_jerk_velocity(self,
                                      positions: np.ndarray,
                                      time_points: np.ndarray) -> np.ndarray:
        """Generate minimum jerk velocity profile."""
        n_points = len(positions)
        duration = time_points[-1] - time_points[0] if len(time_points) > 1 else 1.0
        
        if n_points < 2:
            return np.zeros((n_points, 2))
        
        # Start and end positions
        start_pos = positions[0]
        end_pos = positions[-1]
        displacement = end_pos - start_pos
        
        velocities = np.zeros((n_points, 2))
        
        for i, t in enumerate(time_points):
            tau = t / duration  # Normalized time [0, 1]
            
            # Minimum jerk velocity profile
            if tau <= 0:
                vel_factor = 0
            elif tau >= 1:
                vel_factor = 0
            else:
                # Fifth-order polynomial for minimum jerk
                vel_factor = (displacement / duration) * (30 * tau**4 - 60 * tau**3 + 30 * tau**2)
            
            velocities[i] = vel_factor
        
        return velocities
    
    def _generate_trapezoidal_velocity(self,
                                     positions: np.ndarray,
                                     time_points: np.ndarray) -> np.ndarray:
        """Generate trapezoidal velocity profile."""
        n_points = len(positions)
        duration = time_points[-1] - time_points[0] if len(time_points) > 1 else 1.0
        
        if n_points < 2:
            return np.zeros((n_points, 2))
        
        # Start and end positions
        start_pos = positions[0]
        end_pos = positions[-1]
        displacement = end_pos - start_pos
        distance = np.linalg.norm(displacement)
        
        if distance == 0:
            return np.zeros((n_points, 2))
        
        direction = displacement / distance
        
        # Trapezoidal profile parameters
        accel_time = duration * 0.3  # 30% acceleration
        decel_time = duration * 0.3  # 30% deceleration
        const_time = duration * 0.4  # 40% constant velocity
        
        # Maximum velocity
        max_velocity = distance / (accel_time/2 + const_time + decel_time/2)
        
        velocities = np.zeros((n_points, 2))
        
        for i, t in enumerate(time_points):
            if t <= accel_time:
                # Acceleration phase
                vel_magnitude = max_velocity * (t / accel_time)
            elif t <= accel_time + const_time:
                # Constant velocity phase
                vel_magnitude = max_velocity
            else:
                # Deceleration phase
                remaining_time = duration - t
                vel_magnitude = max_velocity * (remaining_time / decel_time)
            
            velocities[i] = vel_magnitude * direction
        
        return velocities
    
    def _generate_quadratic_bezier(self,
                                 control_points: List[np.ndarray],
                                 time_points: np.ndarray) -> np.ndarray:
        """Generate quadratic Bezier curve."""
        P0, P1, P2 = control_points
        n_points = len(time_points)
        duration = time_points[-1] - time_points[0] if len(time_points) > 1 else 1.0
        
        positions = np.zeros((n_points, 2))
        
        for i, t in enumerate(time_points):
            tau = t / duration  # Normalized time [0, 1]
            tau = np.clip(tau, 0, 1)
            
            # Quadratic Bezier formula
            pos = (1 - tau)**2 * P0 + 2 * (1 - tau) * tau * P1 + tau**2 * P2
            positions[i] = pos
        
        return positions
    
    def _generate_cubic_bezier(self,
                             control_points: List[np.ndarray],
                             time_points: np.ndarray) -> np.ndarray:
        """Generate cubic Bezier curve."""
        P0, P1, P2, P3 = control_points
        n_points = len(time_points)
        duration = time_points[-1] - time_points[0] if len(time_points) > 1 else 1.0
        
        positions = np.zeros((n_points, 2))
        
        for i, t in enumerate(time_points):
            tau = t / duration  # Normalized time [0, 1]
            tau = np.clip(tau, 0, 1)
            
            # Cubic Bezier formula
            pos = ((1 - tau)**3 * P0 +
                   3 * (1 - tau)**2 * tau * P1 +
                   3 * (1 - tau) * tau**2 * P2 +
                   tau**3 * P3)
            positions[i] = pos
        
        return positions
    
    def _generate_linear_interpolation(self,
                                     control_points: List[np.ndarray],
                                     time_points: np.ndarray) -> np.ndarray:
        """Generate linear interpolation through control points."""
        n_points = len(time_points)
        duration = time_points[-1] - time_points[0] if len(time_points) > 1 else 1.0
        
        positions = np.zeros((n_points, 2))
        
        if len(control_points) < 2:
            if len(control_points) == 1:
                positions[:] = control_points[0]
            return positions
        
        # Create parameter values for control points
        n_segments = len(control_points) - 1
        segment_duration = duration / n_segments
        
        for i, t in enumerate(time_points):
            # Find which segment we're in
            segment_idx = int(t / segment_duration)
            segment_idx = min(segment_idx, n_segments - 1)
            
            # Local time within segment
            segment_start_time = segment_idx * segment_duration
            local_t = (t - segment_start_time) / segment_duration
            local_t = np.clip(local_t, 0, 1)
            
            # Interpolate between control points
            start_point = control_points[segment_idx]
            end_point = control_points[segment_idx + 1]
            
            positions[i] = start_point + local_t * (end_point - start_point)
        
        return positions
    
    def generate_character_trajectory(self,
                                    stroke_primitives: List[StrokePrimitive],
                                    start_position: np.ndarray,
                                    style_params: Dict[str, Any]) -> Dict[str, np.ndarray]:
        """
        Generate complete character trajectory from stroke primitives.
        
        Args:
            stroke_primitives: List of stroke primitives
            start_position: Starting position
            style_params: Style parameters
            
        Returns:
            character_trajectory: Combined trajectory for character
        """
        all_positions = []
        all_velocities = []
        all_timestamps = []
        all_stroke_ids = []
        
        current_position = start_position.copy()
        current_time = 0.0
        
        for stroke_idx, primitive in enumerate(stroke_primitives):
            # Generate primitive trajectory
            if primitive.primitive_type == PrimitiveType.LINE:
                end_pos = current_position + primitive.parameters.get('displacement', np.array([0.01, 0]))
                stroke_data = self.generate_line_primitive(
                    current_position, end_pos, primitive.duration
                )
            
            elif primitive.primitive_type == PrimitiveType.ARC:
                center = current_position + primitive.parameters.get('center_offset', np.array([0, 0.005]))
                radius = primitive.parameters.get('radius', 0.005)
                start_angle = primitive.parameters.get('start_angle', 0)
                end_angle = primitive.parameters.get('end_angle', np.pi)
                
                stroke_data = self.generate_arc_primitive(
                    center, radius, start_angle, end_angle, primitive.duration
                )
            
            elif primitive.primitive_type == PrimitiveType.CIRCLE:
                center = current_position + primitive.parameters.get('center_offset', np.array([0.005, 0]))
                radius = primitive.parameters.get('radius', 0.005)
                
                stroke_data = self.generate_circle_primitive(
                    center, radius, primitive.duration
                )
            
            elif primitive.primitive_type == PrimitiveType.LOOP:
                height = primitive.parameters.get('height', 0.01)
                width = primitive.parameters.get('width', 0.005)
                
                stroke_data = self.generate_loop_primitive(
                    current_position, height, width, primitive.duration
                )
            
            else:
                # Default to line primitive
                end_pos = current_position + np.array([0.01, 0])
                stroke_data = self.generate_line_primitive(
                    current_position, end_pos, primitive.duration
                )
            
            # Add time offset
            stroke_timestamps = stroke_data['time'] + current_time
            
            # Append to trajectory
            all_positions.append(stroke_data['positions'])
            all_velocities.append(stroke_data['velocities'])
            all_timestamps.append(stroke_timestamps)
            all_stroke_ids.extend([stroke_idx] * len(stroke_data['positions']))
            
            # Update current position and time
            if len(stroke_data['positions']) > 0:
                current_position = stroke_data['positions'][-1]
                current_time = stroke_timestamps[-1] + 0.05  # Small gap between strokes
        
        # Combine all strokes
        combined_positions = np.concatenate(all_positions) if all_positions else np.array([]).reshape(0, 2)
        combined_velocities = np.concatenate(all_velocities) if all_velocities else np.array([]).reshape(0, 2)
        combined_timestamps = np.concatenate(all_timestamps) if all_timestamps else np.array([])
        
        return {
            'points': combined_positions,
            'velocities': combined_velocities,
            'timestamps': combined_timestamps,
            'stroke_ids': np.array(all_stroke_ids)
        }
    
    def create_character_primitives(self, character: str) -> List[StrokePrimitive]:
        """
        Create stroke primitives for a specific character.
        
        Args:
            character: Character to create primitives for
            
        Returns:
            primitives: List of stroke primitives
        """
        # Simplified primitive definitions for common characters
        primitive_definitions = {
            'a': [
                StrokePrimitive(PrimitiveType.ARC, {'radius': 0.005, 'start_angle': 0, 'end_angle': np.pi}),
                StrokePrimitive(PrimitiveType.LINE, {'displacement': np.array([0, 0.01])})
            ],
            'b': [
                StrokePrimitive(PrimitiveType.LINE, {'displacement': np.array([0, 0.015])}),
                StrokePrimitive(PrimitiveType.ARC, {'radius': 0.003, 'start_angle': 0, 'end_angle': np.pi}),
                StrokePrimitive(PrimitiveType.ARC, {'radius': 0.003, 'start_angle': np.pi, 'end_angle': 2*np.pi})
            ],
            'c': [
                StrokePrimitive(PrimitiveType.ARC, {'radius': 0.005, 'start_angle': np.pi/4, 'end_angle': 7*np.pi/4})
            ],
            'e': [
                StrokePrimitive(PrimitiveType.LINE, {'displacement': np.array([0.008, 0])}),
                StrokePrimitive(PrimitiveType.ARC, {'radius': 0.004, 'start_angle': 0, 'end_angle': -np.pi})
            ],
            'l': [
                StrokePrimitive(PrimitiveType.LINE, {'displacement': np.array([0, 0.015])})
            ],
            'o': [
                StrokePrimitive(PrimitiveType.CIRCLE, {'radius': 0.005})
            ]
        }
        
        return primitive_definitions.get(character.lower(), [])
    
    def optimize_primitive_parameters(self,
                                    primitives: List[StrokePrimitive],
                                    target_trajectory: np.ndarray,
                                    style_params: Dict[str, Any]) -> List[StrokePrimitive]:
        """
        Optimize primitive parameters to match target trajectory.
        
        Args:
            primitives: Initial primitives
            target_trajectory: Target trajectory to match
            style_params: Style parameters
            
        Returns:
            optimized_primitives: Optimized primitives
        """
        # Simplified optimization - in practice would use more sophisticated methods
        optimized_primitives = []
        
        for primitive in primitives:
            # Create a copy to modify
            opt_primitive = StrokePrimitive(
                primitive.primitive_type,
                primitive.parameters.copy(),
                primitive.duration,
                primitive.start_position.copy() if primitive.start_position is not None else None,
                primitive.end_position.copy() if primitive.end_position is not None else None,
                primitive.velocity_profile
            )
            
            # Simple scaling based on style parameters
            if 'letter_height' in style_params:
                scale_factor = style_params['letter_height'] / 0.01  # Normalize to 1cm
                
                # Scale relevant parameters
                if 'radius' in opt_primitive.parameters:
                    opt_primitive.parameters['radius'] *= scale_factor
                
                if 'displacement' in opt_primitive.parameters:
                    opt_primitive.parameters['displacement'] *= scale_factor
                
                if 'height' in opt_primitive.parameters:
                    opt_primitive.parameters['height'] *= scale_factor
                
                if 'width' in opt_primitive.parameters:
                    opt_primitive.parameters['width'] *= scale_factor
            
            # Adjust duration based on speed factor
            if 'speed_factor' in style_params:
                opt_primitive.duration /= style_params['speed_factor']
            
            optimized_primitives.append(opt_primitive)
        
        return optimized_primitives