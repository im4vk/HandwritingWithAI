"""
Main Trajectory Generator
========================

High-level interface for generating handwriting trajectories using
various models and techniques.
"""

import numpy as np
from typing import Dict, Any, List, Tuple, Optional, Union
from dataclasses import dataclass
import logging
from enum import Enum

from .sigma_lognormal import SigmaLognormalGenerator, LognormalParameter
from .biomechanical_models import BiomechanicalModel, MusculoskeletalModel, AdaptiveMotorControl
from .bezier_curves import BezierCurveGenerator
from .movement_primitives import MovementPrimitive, StrokePrimitive

logger = logging.getLogger(__name__)


class TrajectoryType(Enum):
    """Types of trajectory generation methods."""
    SIGMA_LOGNORMAL = "sigma_lognormal"
    BIOMECHANICAL = "biomechanical"
    BEZIER = "bezier"
    MOVEMENT_PRIMITIVES = "movement_primitives"
    HYBRID = "hybrid"


@dataclass
class HandwritingPath:
    """
    Represents a handwriting path with metadata.
    
    Attributes:
        points: Path points [n_points, 2]
        velocities: Velocity profile [n_points, 2]
        accelerations: Acceleration profile [n_points, 2] 
        timestamps: Time stamps [n_points]
        pen_states: Pen up/down states [n_points]
        pressures: Pen pressures [n_points]
        stroke_ids: Stroke identifiers [n_points]
        character_ids: Character identifiers [n_points]
        metadata: Additional metadata
    """
    points: np.ndarray
    velocities: Optional[np.ndarray] = None
    accelerations: Optional[np.ndarray] = None
    timestamps: Optional[np.ndarray] = None
    pen_states: Optional[np.ndarray] = None
    pressures: Optional[np.ndarray] = None
    stroke_ids: Optional[np.ndarray] = None
    character_ids: Optional[np.ndarray] = None
    metadata: Optional[Dict[str, Any]] = None
    
    def __post_init__(self):
        """Validate and initialize default values."""
        n_points = len(self.points)
        
        if self.velocities is None:
            self.velocities = np.zeros((n_points, 2))
        
        if self.accelerations is None:
            self.accelerations = np.zeros((n_points, 2))
        
        if self.timestamps is None:
            self.timestamps = np.linspace(0, 1, n_points)
        
        if self.pen_states is None:
            self.pen_states = np.ones(n_points)  # Pen down by default
        
        if self.pressures is None:
            self.pressures = np.ones(n_points) * 0.5  # Default pressure
        
        if self.stroke_ids is None:
            self.stroke_ids = np.zeros(n_points, dtype=int)
        
        if self.character_ids is None:
            self.character_ids = np.zeros(n_points, dtype=int)
        
        if self.metadata is None:
            self.metadata = {}


class TrajectoryGenerator:
    """
    Main trajectory generator that combines multiple generation methods.
    
    Provides a unified interface for generating handwriting trajectories
    using different algorithms and models.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize trajectory generator.
        
        Args:
            config: Generator configuration
        """
        self.config = config
        
        # Initialize component generators
        self._initialize_generators()
        
        # Generation parameters
        self.default_method = TrajectoryType(config.get('default_method', 'sigma_lognormal'))
        self.sampling_rate = config.get('sampling_rate', 100)  # Hz
        self.smoothing_enabled = config.get('smoothing_enabled', True)
        self.biomechanical_constraints = config.get('biomechanical_constraints', True)
        
        # Style parameters
        self.style_config = config.get('style', {})
        
        logger.info(f"Initialized TrajectoryGenerator with method: {self.default_method.value}")
    
    def _initialize_generators(self) -> None:
        """Initialize component generators."""
        # Sigma-lognormal generator
        self.sigma_lognormal = SigmaLognormalGenerator(
            self.config.get('sigma_lognormal', {})
        )
        
        # Biomechanical models
        self.biomechanical_model = BiomechanicalModel(
            self.config.get('biomechanical', {})
        )
        self.musculoskeletal_model = MusculoskeletalModel(
            self.config.get('musculoskeletal', {})
        )
        self.adaptive_control = AdaptiveMotorControl(
            self.config.get('adaptive_control', {})
        )
        
        # Bezier curve generator
        self.bezier_generator = BezierCurveGenerator(
            self.config.get('bezier', {})
        )
        
        # Movement primitives
        self.movement_primitives = MovementPrimitive(
            self.config.get('movement_primitives', {})
        )
    
    def generate_text_trajectory(self, 
                               text: str,
                               method: Optional[TrajectoryType] = None,
                               style_params: Optional[Dict[str, Any]] = None) -> HandwritingPath:
        """
        Generate trajectory for writing text.
        
        Args:
            text: Text to write
            method: Generation method to use
            style_params: Style parameters
            
        Returns:
            trajectory: Generated handwriting trajectory
        """
        if method is None:
            method = self.default_method
        
        if style_params is None:
            style_params = self.style_config
        
        logger.info(f"Generating trajectory for text: '{text}' using {method.value}")
        
        # Generate character trajectories
        character_paths = []
        current_position = np.array([0.0, 0.0])
        
        for char_idx, char in enumerate(text):
            if char == ' ':
                # Handle spaces
                current_position[0] += style_params.get('word_spacing', 0.015)
                continue
            
            # Generate character trajectory
            char_path = self._generate_character_trajectory(
                char, current_position, char_idx, method, style_params
            )
            
            if char_path is not None:
                character_paths.append(char_path)
                
                # Update position for next character
                if len(char_path.points) > 0:
                    current_position = char_path.points[-1].copy()
                    current_position[0] += style_params.get('letter_spacing', 0.005)
        
        # Combine character trajectories
        combined_path = self._combine_character_paths(character_paths)
        
        # Apply post-processing
        if self.smoothing_enabled:
            combined_path = self._apply_smoothing(combined_path)
        
        if self.biomechanical_constraints:
            combined_path = self._apply_biomechanical_constraints(combined_path)
        
        return combined_path
    
    def _generate_character_trajectory(self,
                                     character: str,
                                     start_position: np.ndarray,
                                     char_idx: int,
                                     method: TrajectoryType,
                                     style_params: Dict[str, Any]) -> Optional[HandwritingPath]:
        """
        Generate trajectory for a single character.
        
        Args:
            character: Character to generate
            start_position: Starting position
            char_idx: Character index in text
            method: Generation method
            style_params: Style parameters
            
        Returns:
            character_path: Character trajectory
        """
        try:
            if method == TrajectoryType.SIGMA_LOGNORMAL:
                return self._generate_sigma_lognormal_character(
                    character, start_position, char_idx, style_params
                )
            
            elif method == TrajectoryType.BIOMECHANICAL:
                return self._generate_biomechanical_character(
                    character, start_position, char_idx, style_params
                )
            
            elif method == TrajectoryType.BEZIER:
                return self._generate_bezier_character(
                    character, start_position, char_idx, style_params
                )
            
            elif method == TrajectoryType.MOVEMENT_PRIMITIVES:
                return self._generate_primitive_character(
                    character, start_position, char_idx, style_params
                )
            
            elif method == TrajectoryType.HYBRID:
                return self._generate_hybrid_character(
                    character, start_position, char_idx, style_params
                )
            
            else:
                logger.warning(f"Unknown generation method: {method}")
                return None
        
        except Exception as e:
            logger.error(f"Error generating character '{character}': {e}")
            return None
    
    def _generate_sigma_lognormal_character(self,
                                          character: str,
                                          start_position: np.ndarray,
                                          char_idx: int,
                                          style_params: Dict[str, Any]) -> HandwritingPath:
        """Generate character using sigma-lognormal model."""
        # Use sigma-lognormal generator
        trajectory_data = self.sigma_lognormal.synthesize_handwriting(
            character, style_params
        )
        
        # Extract data
        points = trajectory_data['positions'] + start_position
        velocities = trajectory_data['velocities']
        accelerations = trajectory_data['accelerations']
        timestamps = trajectory_data['time']
        
        # Create path
        return HandwritingPath(
            points=points,
            velocities=velocities,
            accelerations=accelerations,
            timestamps=timestamps,
            character_ids=np.full(len(points), char_idx),
            metadata={'method': 'sigma_lognormal', 'character': character}
        )
    
    def _generate_biomechanical_character(self,
                                        character: str,
                                        start_position: np.ndarray,
                                        char_idx: int,
                                        style_params: Dict[str, Any]) -> HandwritingPath:
        """Generate character using biomechanical model."""
        # Get character template
        template = self._get_character_template(character, style_params)
        target_trajectory = template + start_position
        
        # Use adaptive motor control
        execution_results = self.adaptive_control.execute_movement(target_trajectory)
        
        points = execution_results['executed_trajectory']
        velocities = execution_results['velocities']
        
        # Compute accelerations
        dt = 1.0 / self.sampling_rate
        accelerations = np.diff(velocities, axis=0, prepend=velocities[:1]) / dt
        timestamps = np.arange(len(points)) * dt
        
        return HandwritingPath(
            points=points,
            velocities=velocities,
            accelerations=accelerations,
            timestamps=timestamps,
            character_ids=np.full(len(points), char_idx),
            metadata={'method': 'biomechanical', 'character': character}
        )
    
    def _generate_bezier_character(self,
                                 character: str,
                                 start_position: np.ndarray,
                                 char_idx: int,
                                 style_params: Dict[str, Any]) -> HandwritingPath:
        """Generate character using Bezier curves."""
        # Get character control points
        control_points = self._get_character_bezier_points(character, style_params)
        
        # Generate Bezier curve
        bezier_path = self.bezier_generator.generate_character_curve(
            control_points, start_position, style_params
        )
        
        return HandwritingPath(
            points=bezier_path['points'],
            velocities=bezier_path['velocities'],
            timestamps=bezier_path['timestamps'],
            character_ids=np.full(len(bezier_path['points']), char_idx),
            metadata={'method': 'bezier', 'character': character}
        )
    
    def _generate_primitive_character(self,
                                    character: str,
                                    start_position: np.ndarray,
                                    char_idx: int,
                                    style_params: Dict[str, Any]) -> HandwritingPath:
        """Generate character using movement primitives."""
        # Get character stroke primitives
        stroke_primitives = self._get_character_primitives(character)
        
        # Generate trajectory from primitives
        primitive_path = self.movement_primitives.generate_character_trajectory(
            stroke_primitives, start_position, style_params
        )
        
        return HandwritingPath(
            points=primitive_path['points'],
            velocities=primitive_path['velocities'],
            timestamps=primitive_path['timestamps'],
            stroke_ids=primitive_path['stroke_ids'],
            character_ids=np.full(len(primitive_path['points']), char_idx),
            metadata={'method': 'movement_primitives', 'character': character}
        )
    
    def _generate_hybrid_character(self,
                                 character: str,
                                 start_position: np.ndarray,
                                 char_idx: int,
                                 style_params: Dict[str, Any]) -> HandwritingPath:
        """Generate character using hybrid approach."""
        # Use Bezier for shape, sigma-lognormal for dynamics
        bezier_path = self._generate_bezier_character(
            character, start_position, char_idx, style_params
        )
        
        # Apply sigma-lognormal velocity profile
        sigma_lognormal_data = self.sigma_lognormal.synthesize_handwriting(
            character, style_params
        )
        
        # Combine shape from Bezier with dynamics from sigma-lognormal
        hybrid_points = bezier_path.points
        hybrid_velocities = self._interpolate_velocities(
            sigma_lognormal_data['velocities'], len(hybrid_points)
        )
        
        # Recompute accelerations
        dt = 1.0 / self.sampling_rate
        hybrid_accelerations = np.diff(hybrid_velocities, axis=0, prepend=hybrid_velocities[:1]) / dt
        
        return HandwritingPath(
            points=hybrid_points,
            velocities=hybrid_velocities,
            accelerations=hybrid_accelerations,
            timestamps=bezier_path.timestamps,
            character_ids=np.full(len(hybrid_points), char_idx),
            metadata={'method': 'hybrid', 'character': character}
        )
    
    def _get_character_template(self, character: str, style_params: Dict[str, Any]) -> np.ndarray:
        """Get basic character template points."""
        # Simplified character templates (in practice, would use font data)
        templates = {
            'a': np.array([[0, 0], [0.3, 0], [0.5, 0.5], [0.7, 0], [1, 0], [1, 1], [0.7, 1], [0.3, 0.5]]),
            'b': np.array([[0, 0], [0, 1], [0.6, 1], [0.8, 0.8], [0.6, 0.5], [0.8, 0.3], [0.6, 0], [0, 0]]),
            'c': np.array([[0.8, 0.2], [0.3, 0], [0, 0.5], [0.3, 1], [0.8, 0.8]]),
            'e': np.array([[0, 0.5], [0.8, 0.5], [0.8, 0.2], [0.2, 0], [0, 0.5], [0.8, 1], [0.2, 1]]),
            'h': np.array([[0, 0], [0, 1], [0, 0.5], [0.8, 0.5], [0.8, 0], [0.8, 1]]),
            'l': np.array([[0, 0], [0, 1], [0.8, 1]]),
            'o': np.array([[0.5, 0], [0.8, 0.2], [1, 0.5], [0.8, 0.8], [0.5, 1], [0.2, 0.8], [0, 0.5], [0.2, 0.2], [0.5, 0]]),
            'r': np.array([[0, 0], [0, 1], [0.6, 1], [0.8, 0.8], [0.6, 0.5], [0, 0.5], [0.6, 0.5], [0.8, 0]]),
            'w': np.array([[0, 1], [0.2, 0], [0.4, 0.5], [0.6, 0], [0.8, 1]]),
            ' ': np.array([[0, 0.5], [0.5, 0.5]]),  # Space placeholder
        }
        
        template = templates.get(character.lower(), templates.get(' '))
        
        # Scale by letter height
        letter_height = style_params.get('letter_height', 0.01)
        scaled_template = template * letter_height
        
        # Apply slant
        slant_angle = style_params.get('slant_angle', 0.0) * np.pi / 180
        if slant_angle != 0:
            slant_matrix = np.array([[1, np.tan(slant_angle)], [0, 1]])
            scaled_template = scaled_template @ slant_matrix.T
        
        return scaled_template
    
    def _get_character_bezier_points(self, character: str, style_params: Dict[str, Any]) -> List:
        """Get Bezier control points for character."""
        # Simplified - would be more sophisticated in practice
        template = self._get_character_template(character, style_params)
        
        # Convert template points to Bezier control points
        control_points = []
        
        for i in range(0, len(template) - 1, 2):
            if i + 1 < len(template):
                start = template[i]
                end = template[i + 1]
                
                # Simple cubic Bezier with control points
                ctrl1 = start + (end - start) * 0.33
                ctrl2 = start + (end - start) * 0.67
                
                control_points.append([start, ctrl1, ctrl2, end])
        
        return control_points
    
    def _get_character_primitives(self, character: str) -> List[StrokePrimitive]:
        """Get movement primitives for character."""
        # Simplified stroke primitives
        primitives = {
            'a': [
                StrokePrimitive('arc', {'start_angle': 0, 'end_angle': np.pi}),
                StrokePrimitive('line', {'direction': 'vertical'})
            ],
            'b': [
                StrokePrimitive('line', {'direction': 'vertical'}),
                StrokePrimitive('arc', {'start_angle': 0, 'end_angle': np.pi}),
                StrokePrimitive('arc', {'start_angle': np.pi, 'end_angle': 2*np.pi})
            ],
            'o': [
                StrokePrimitive('circle', {'radius': 0.5})
            ]
        }
        
        return primitives.get(character.lower(), [])
    
    def _combine_character_paths(self, character_paths: List[HandwritingPath]) -> HandwritingPath:
        """Combine multiple character paths into single trajectory."""
        if not character_paths:
            return HandwritingPath(points=np.array([]).reshape(0, 2))
        
        # Concatenate all data
        all_points = []
        all_velocities = []
        all_accelerations = []
        all_timestamps = []
        all_pen_states = []
        all_pressures = []
        all_stroke_ids = []
        all_character_ids = []
        
        current_time_offset = 0.0
        
        for path in character_paths:
            # Add time offset
            timestamps = path.timestamps + current_time_offset
            all_timestamps.append(timestamps)
            
            all_points.append(path.points)
            all_velocities.append(path.velocities)
            all_accelerations.append(path.accelerations)
            all_pen_states.append(path.pen_states)
            all_pressures.append(path.pressures)
            all_stroke_ids.append(path.stroke_ids)
            all_character_ids.append(path.character_ids)
            
            if len(timestamps) > 0:
                current_time_offset = timestamps[-1] + 0.05  # Small gap between characters
        
        # Concatenate arrays
        combined_path = HandwritingPath(
            points=np.concatenate(all_points) if all_points else np.array([]).reshape(0, 2),
            velocities=np.concatenate(all_velocities) if all_velocities else np.array([]).reshape(0, 2),
            accelerations=np.concatenate(all_accelerations) if all_accelerations else np.array([]).reshape(0, 2),
            timestamps=np.concatenate(all_timestamps) if all_timestamps else np.array([]),
            pen_states=np.concatenate(all_pen_states) if all_pen_states else np.array([]),
            pressures=np.concatenate(all_pressures) if all_pressures else np.array([]),
            stroke_ids=np.concatenate(all_stroke_ids) if all_stroke_ids else np.array([]),
            character_ids=np.concatenate(all_character_ids) if all_character_ids else np.array([]),
            metadata={'method': 'combined', 'num_characters': len(character_paths)}
        )
        
        return combined_path
    
    def _apply_smoothing(self, path: HandwritingPath) -> HandwritingPath:
        """Apply smoothing to trajectory."""
        from scipy.signal import savgol_filter
        
        if len(path.points) < 5:  # Need minimum points for smoothing
            return path
        
        # Smooth positions
        window_length = min(5, len(path.points) - 1)
        if window_length % 2 == 0:
            window_length -= 1
        
        try:
            smoothed_points = np.column_stack([
                savgol_filter(path.points[:, 0], window_length, 3),
                savgol_filter(path.points[:, 1], window_length, 3)
            ])
            
            # Recompute velocities and accelerations
            dt = np.mean(np.diff(path.timestamps)) if len(path.timestamps) > 1 else 1.0 / self.sampling_rate
            smoothed_velocities = np.diff(smoothed_points, axis=0, prepend=smoothed_points[:1]) / dt
            smoothed_accelerations = np.diff(smoothed_velocities, axis=0, prepend=smoothed_velocities[:1]) / dt
            
            # Create smoothed path
            smoothed_path = HandwritingPath(
                points=smoothed_points,
                velocities=smoothed_velocities,
                accelerations=smoothed_accelerations,
                timestamps=path.timestamps,
                pen_states=path.pen_states,
                pressures=path.pressures,
                stroke_ids=path.stroke_ids,
                character_ids=path.character_ids,
                metadata=path.metadata
            )
            
            return smoothed_path
        
        except Exception as e:
            logger.warning(f"Smoothing failed: {e}, returning original path")
            return path
    
    def _apply_biomechanical_constraints(self, path: HandwritingPath) -> HandwritingPath:
        """Apply biomechanical constraints to trajectory."""
        constrained_points, constrained_velocities = self.biomechanical_model.apply_biomechanical_constraints(
            path.points, path.velocities
        )
        
        # Recompute accelerations
        dt = np.mean(np.diff(path.timestamps)) if len(path.timestamps) > 1 else 1.0 / self.sampling_rate
        constrained_accelerations = np.diff(constrained_velocities, axis=0, prepend=constrained_velocities[:1]) / dt
        
        constrained_path = HandwritingPath(
            points=constrained_points,
            velocities=constrained_velocities,
            accelerations=constrained_accelerations,
            timestamps=path.timestamps,
            pen_states=path.pen_states,
            pressures=path.pressures,
            stroke_ids=path.stroke_ids,
            character_ids=path.character_ids,
            metadata=path.metadata
        )
        
        return constrained_path
    
    def _interpolate_velocities(self, source_velocities: np.ndarray, target_length: int) -> np.ndarray:
        """Interpolate velocities to match target length."""
        if len(source_velocities) == target_length:
            return source_velocities
        
        # Linear interpolation
        source_indices = np.linspace(0, len(source_velocities) - 1, len(source_velocities))
        target_indices = np.linspace(0, len(source_velocities) - 1, target_length)
        
        interpolated_velocities = np.zeros((target_length, 2))
        for i in range(2):  # x and y components
            interpolated_velocities[:, i] = np.interp(target_indices, source_indices, source_velocities[:, i])
        
        return interpolated_velocities
    
    def generate_custom_trajectory(self,
                                 waypoints: np.ndarray,
                                 method: Optional[TrajectoryType] = None,
                                 constraints: Optional[Dict[str, Any]] = None) -> HandwritingPath:
        """
        Generate trajectory through custom waypoints.
        
        Args:
            waypoints: Waypoints to pass through [n_points, 2]
            method: Generation method
            constraints: Additional constraints
            
        Returns:
            trajectory: Generated trajectory
        """
        if method is None:
            method = self.default_method
        
        logger.info(f"Generating custom trajectory through {len(waypoints)} waypoints")
        
        # Generate trajectory based on method
        if method == TrajectoryType.BEZIER:
            trajectory_data = self.bezier_generator.generate_trajectory_through_points(waypoints)
        elif method == TrajectoryType.SIGMA_LOGNORMAL:
            # Use sigma-lognormal to connect waypoints
            trajectory_data = self._connect_waypoints_sigma_lognormal(waypoints)
        else:
            # Use biomechanical model
            execution_results = self.adaptive_control.execute_movement(waypoints)
            trajectory_data = {
                'points': execution_results['executed_trajectory'],
                'velocities': execution_results['velocities'],
                'timestamps': np.arange(len(execution_results['executed_trajectory'])) / self.sampling_rate
            }
        
        # Create path
        path = HandwritingPath(
            points=trajectory_data['points'],
            velocities=trajectory_data.get('velocities', np.zeros((len(trajectory_data['points']), 2))),
            timestamps=trajectory_data.get('timestamps', np.arange(len(trajectory_data['points'])) / self.sampling_rate),
            metadata={'method': method.value, 'waypoints': waypoints}
        )
        
        # Apply constraints if specified
        if constraints:
            path = self._apply_custom_constraints(path, constraints)
        
        return path
    
    def _connect_waypoints_sigma_lognormal(self, waypoints: np.ndarray) -> Dict[str, np.ndarray]:
        """Connect waypoints using sigma-lognormal segments."""
        all_points = []
        all_velocities = []
        all_timestamps = []
        
        current_time = 0.0
        
        for i in range(len(waypoints) - 1):
            start_point = waypoints[i]
            end_point = waypoints[i + 1]
            
            # Create simple lognormal parameters for this segment
            displacement = end_point - start_point
            duration = 0.5  # Fixed duration per segment
            
            param = LognormalParameter(
                t0=0.0,
                mu=np.log(duration / 2),
                sigma=0.3,
                D=displacement
            )
            
            # Generate segment
            segment_data = self.sigma_lognormal.generate_trajectory([param], duration, start_point)
            
            # Add time offset
            segment_timestamps = segment_data['time'] + current_time
            
            all_points.append(segment_data['positions'])
            all_velocities.append(segment_data['velocities'])
            all_timestamps.append(segment_timestamps)
            
            current_time = segment_timestamps[-1] + 0.1  # Small gap
        
        return {
            'points': np.concatenate(all_points) if all_points else waypoints,
            'velocities': np.concatenate(all_velocities) if all_velocities else np.zeros((len(waypoints), 2)),
            'timestamps': np.concatenate(all_timestamps) if all_timestamps else np.arange(len(waypoints))
        }
    
    def _apply_custom_constraints(self, path: HandwritingPath, constraints: Dict[str, Any]) -> HandwritingPath:
        """Apply custom constraints to trajectory."""
        # Apply velocity constraints
        if 'max_velocity' in constraints:
            max_vel = constraints['max_velocity']
            speeds = np.linalg.norm(path.velocities, axis=1)
            scale_factors = np.minimum(1.0, max_vel / (speeds + 1e-8))
            path.velocities *= scale_factors.reshape(-1, 1)
        
        # Apply acceleration constraints
        if 'max_acceleration' in constraints:
            max_acc = constraints['max_acceleration']
            dt = np.mean(np.diff(path.timestamps)) if len(path.timestamps) > 1 else 1.0 / self.sampling_rate
            accelerations = np.diff(path.velocities, axis=0) / dt
            acc_magnitudes = np.linalg.norm(accelerations, axis=1)
            acc_scale_factors = np.minimum(1.0, max_acc / (acc_magnitudes + 1e-8))
            
            # Apply scaling
            for i in range(len(acc_scale_factors)):
                if acc_scale_factors[i] < 1.0:
                    path.velocities[i+1] = path.velocities[i] + accelerations[i] * acc_scale_factors[i] * dt
            
            # Recompute positions
            path.points[0] = path.points[0]  # Keep first point
            for i in range(1, len(path.points)):
                path.points[i] = path.points[i-1] + path.velocities[i] * dt
        
        return path