"""
Bezier Curve Generation for Handwriting
=======================================

Implementation of Bezier curves for generating smooth handwriting trajectories.
Provides both cubic and higher-order Bezier curves with velocity and acceleration
profiles suitable for robotic handwriting.
"""

import numpy as np
from typing import Dict, Any, List, Tuple, Optional, Union
from dataclasses import dataclass
import logging
from scipy.special import comb

logger = logging.getLogger(__name__)


@dataclass
class ControlPoint:
    """
    Bezier curve control point.
    
    Attributes:
        position: 2D position [x, y]
        weight: Weight for rational Bezier curves
        tangent: Tangent direction (optional)
    """
    position: np.ndarray
    weight: float = 1.0
    tangent: Optional[np.ndarray] = None
    
    def __post_init__(self):
        """Validate control point."""
        if len(self.position) != 2:
            raise ValueError("Position must be 2D")
        
        if self.tangent is not None and len(self.tangent) != 2:
            raise ValueError("Tangent must be 2D")


class BezierCurveGenerator:
    """
    Generator for Bezier curves with handwriting-specific features.
    
    Supports cubic and higher-order Bezier curves with automatic
    control point generation and velocity profile computation.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize Bezier curve generator.
        
        Args:
            config: Generator configuration
        """
        self.config = config
        
        # Curve parameters
        self.default_resolution = config.get('resolution', 100)
        self.smoothness_factor = config.get('smoothness_factor', 0.5)
        self.velocity_scaling = config.get('velocity_scaling', 1.0)
        self.curvature_based_speed = config.get('curvature_based_speed', True)
        
        # Handwriting specific parameters
        self.pen_lift_height = config.get('pen_lift_height', 0.005)  # 5mm
        self.stroke_connection_threshold = config.get('stroke_connection_threshold', 0.002)  # 2mm
        
        logger.info("Initialized BezierCurveGenerator")
    
    def generate_cubic_bezier(self,
                            control_points: List[np.ndarray],
                            num_points: Optional[int] = None) -> Dict[str, np.ndarray]:
        """
        Generate cubic Bezier curve from 4 control points.
        
        Args:
            control_points: 4 control points [P0, P1, P2, P3]
            num_points: Number of points to generate
            
        Returns:
            curve_data: Dictionary with points, velocities, curvature, etc.
        """
        if len(control_points) != 4:
            raise ValueError("Cubic Bezier requires exactly 4 control points")
        
        if num_points is None:
            num_points = self.default_resolution
        
        # Parameter values
        t_values = np.linspace(0, 1, num_points)
        
        # Bezier curve points
        points = self._evaluate_cubic_bezier(control_points, t_values)
        
        # First derivative (velocity direction)
        velocities = self._evaluate_cubic_bezier_derivative(control_points, t_values, order=1)
        
        # Second derivative (acceleration)
        accelerations = self._evaluate_cubic_bezier_derivative(control_points, t_values, order=2)
        
        # Compute additional properties
        curvature = self._compute_curvature(velocities, accelerations)
        arc_length = self._compute_arc_length(points)
        
        # Compute speed profile based on curvature
        if self.curvature_based_speed:
            speed_profile = self._compute_curvature_based_speed(curvature, velocities)
        else:
            speed_profile = np.linalg.norm(velocities, axis=1)
        
        # Scale velocities by speed profile
        velocity_directions = velocities / (np.linalg.norm(velocities, axis=1, keepdims=True) + 1e-8)
        scaled_velocities = velocity_directions * speed_profile.reshape(-1, 1) * self.velocity_scaling
        
        return {
            'points': points,
            'velocities': scaled_velocities,
            'accelerations': accelerations,
            'curvature': curvature,
            'arc_length': arc_length,
            'parameter_values': t_values,
            'control_points': control_points
        }
    
    def _evaluate_cubic_bezier(self, control_points: List[np.ndarray], t_values: np.ndarray) -> np.ndarray:
        """Evaluate cubic Bezier curve at parameter values."""
        P0, P1, P2, P3 = control_points
        
        # Bezier basis functions
        B0 = (1 - t_values)**3
        B1 = 3 * t_values * (1 - t_values)**2
        B2 = 3 * t_values**2 * (1 - t_values)
        B3 = t_values**3
        
        # Curve points
        points = (B0.reshape(-1, 1) * P0 +
                 B1.reshape(-1, 1) * P1 +
                 B2.reshape(-1, 1) * P2 +
                 B3.reshape(-1, 1) * P3)
        
        return points
    
    def _evaluate_cubic_bezier_derivative(self,
                                        control_points: List[np.ndarray],
                                        t_values: np.ndarray,
                                        order: int = 1) -> np.ndarray:
        """Evaluate derivatives of cubic Bezier curve."""
        P0, P1, P2, P3 = control_points
        
        if order == 1:
            # First derivative
            D0 = 3 * (P1 - P0)
            D1 = 3 * (P2 - P1)
            D2 = 3 * (P3 - P2)
            
            B0 = (1 - t_values)**2
            B1 = 2 * t_values * (1 - t_values)
            B2 = t_values**2
            
            derivatives = (B0.reshape(-1, 1) * D0 +
                          B1.reshape(-1, 1) * D1 +
                          B2.reshape(-1, 1) * D2)
        
        elif order == 2:
            # Second derivative
            D0 = 6 * (P2 - 2*P1 + P0)
            D1 = 6 * (P3 - 2*P2 + P1)
            
            B0 = (1 - t_values)
            B1 = t_values
            
            derivatives = (B0.reshape(-1, 1) * D0 +
                          B1.reshape(-1, 1) * D1)
        
        else:
            raise ValueError("Only first and second derivatives are supported")
        
        return derivatives
    
    def _compute_curvature(self, velocities: np.ndarray, accelerations: np.ndarray) -> np.ndarray:
        """Compute curvature from velocity and acceleration vectors."""
        # Curvature formula: |v x a| / |v|^3
        cross_product = velocities[:, 0] * accelerations[:, 1] - velocities[:, 1] * accelerations[:, 0]
        speed_cubed = (np.linalg.norm(velocities, axis=1) + 1e-8) ** 3
        
        curvature = np.abs(cross_product) / speed_cubed
        return curvature
    
    def _compute_arc_length(self, points: np.ndarray) -> float:
        """Compute total arc length of curve."""
        if len(points) < 2:
            return 0.0
        
        segment_lengths = np.linalg.norm(np.diff(points, axis=0), axis=1)
        return np.sum(segment_lengths)
    
    def _compute_curvature_based_speed(self, curvature: np.ndarray, velocities: np.ndarray) -> np.ndarray:
        """Compute speed profile based on curvature (slower on tight curves)."""
        base_speed = np.linalg.norm(velocities, axis=1)
        
        # Speed reduction factor based on curvature
        max_curvature = np.max(curvature) if len(curvature) > 0 else 1.0
        normalized_curvature = curvature / (max_curvature + 1e-8)
        
        # Speed factor: slower on high curvature sections
        speed_factor = 1.0 / (1.0 + 2.0 * normalized_curvature)
        
        # Apply minimum speed threshold
        min_speed_factor = 0.1
        speed_factor = np.maximum(speed_factor, min_speed_factor)
        
        return base_speed * speed_factor
    
    def generate_character_curve(self,
                               control_points_sets: List[List[np.ndarray]],
                               start_position: np.ndarray,
                               style_params: Dict[str, Any]) -> Dict[str, np.ndarray]:
        """
        Generate Bezier curves for a character (multiple strokes).
        
        Args:
            control_points_sets: List of control point sets (one per stroke)
            start_position: Starting position
            style_params: Style parameters
            
        Returns:
            character_curve: Combined curve data for character
        """
        all_points = []
        all_velocities = []
        all_timestamps = []
        stroke_ids = []
        
        current_position = start_position.copy()
        current_time = 0.0
        
        for stroke_idx, control_points in enumerate(control_points_sets):
            if len(control_points) < 4:
                # Pad with duplicate points if needed
                while len(control_points) < 4:
                    control_points.append(control_points[-1])
            
            # Adjust first control point to current position
            control_points[0] = current_position
            
            # Generate stroke curve
            stroke_curve = self.generate_cubic_bezier(control_points)
            
            # Add pen lift between strokes (except for first stroke)
            if stroke_idx > 0:
                lift_trajectory = self._generate_pen_lift_trajectory(
                    current_position, control_points[0]
                )
                
                if lift_trajectory is not None:
                    all_points.append(lift_trajectory['points'])
                    all_velocities.append(lift_trajectory['velocities'])
                    
                    lift_duration = len(lift_trajectory['points']) * 0.01  # 10ms per point
                    lift_timestamps = np.linspace(current_time, current_time + lift_duration, 
                                                len(lift_trajectory['points']))
                    all_timestamps.append(lift_timestamps)
                    stroke_ids.extend([-1] * len(lift_trajectory['points']))  # -1 for pen lifts
                    
                    current_time = lift_timestamps[-1]
            
            # Add stroke points
            all_points.append(stroke_curve['points'])
            all_velocities.append(stroke_curve['velocities'])
            
            # Generate timestamps
            stroke_duration = style_params.get('stroke_duration', 0.5)
            stroke_timestamps = np.linspace(current_time, current_time + stroke_duration,
                                          len(stroke_curve['points']))
            all_timestamps.append(stroke_timestamps)
            stroke_ids.extend([stroke_idx] * len(stroke_curve['points']))
            
            # Update current position and time
            current_position = stroke_curve['points'][-1]
            current_time = stroke_timestamps[-1]
        
        # Combine all strokes
        combined_points = np.concatenate(all_points) if all_points else np.array([]).reshape(0, 2)
        combined_velocities = np.concatenate(all_velocities) if all_velocities else np.array([]).reshape(0, 2)
        combined_timestamps = np.concatenate(all_timestamps) if all_timestamps else np.array([])
        
        return {
            'points': combined_points,
            'velocities': combined_velocities,
            'timestamps': combined_timestamps,
            'stroke_ids': np.array(stroke_ids)
        }
    
    def _generate_pen_lift_trajectory(self,
                                    start_pos: np.ndarray,
                                    end_pos: np.ndarray) -> Optional[Dict[str, np.ndarray]]:
        """Generate trajectory for pen lift between strokes."""
        distance = np.linalg.norm(end_pos - start_pos)
        
        if distance < self.stroke_connection_threshold:
            # Don't lift pen for very short movements
            return None
        
        # Create lift trajectory: up -> move -> down
        lift_height = self.pen_lift_height
        
        # Control points for pen lift
        mid_pos = (start_pos + end_pos) / 2
        lift_points = [
            start_pos,
            start_pos + np.array([0, lift_height]),
            end_pos + np.array([0, lift_height]),
            end_pos
        ]
        
        # Generate curve
        lift_curve = self.generate_cubic_bezier(lift_points, num_points=20)
        
        return {
            'points': lift_curve['points'],
            'velocities': lift_curve['velocities'] * 0.5  # Slower pen lifts
        }
    
    def generate_trajectory_through_points(self,
                                         waypoints: np.ndarray,
                                         method: str = 'spline') -> Dict[str, np.ndarray]:
        """
        Generate smooth trajectory through given waypoints.
        
        Args:
            waypoints: Points to pass through [n_points, 2]
            method: Interpolation method ('spline', 'piecewise')
            
        Returns:
            trajectory: Generated trajectory
        """
        if len(waypoints) < 2:
            raise ValueError("At least 2 waypoints required")
        
        if method == 'spline':
            return self._generate_spline_trajectory(waypoints)
        elif method == 'piecewise':
            return self._generate_piecewise_trajectory(waypoints)
        else:
            raise ValueError(f"Unknown method: {method}")
    
    def _generate_spline_trajectory(self, waypoints: np.ndarray) -> Dict[str, np.ndarray]:
        """Generate trajectory using cubic spline interpolation."""
        n_waypoints = len(waypoints)
        
        if n_waypoints == 2:
            # Simple linear interpolation for 2 points
            control_points = [
                waypoints[0],
                waypoints[0] + (waypoints[1] - waypoints[0]) / 3,
                waypoints[0] + 2 * (waypoints[1] - waypoints[0]) / 3,
                waypoints[1]
            ]
            return self.generate_cubic_bezier(control_points)
        
        # Generate control points for cubic spline
        control_points_sets = []
        
        for i in range(n_waypoints - 1):
            start_point = waypoints[i]
            end_point = waypoints[i + 1]
            
            # Estimate tangent directions
            if i == 0:
                # First segment
                tangent_start = waypoints[1] - waypoints[0]
            else:
                tangent_start = (waypoints[i + 1] - waypoints[i - 1]) / 2
            
            if i == n_waypoints - 2:
                # Last segment
                tangent_end = waypoints[-1] - waypoints[-2]
            else:
                tangent_end = (waypoints[i + 2] - waypoints[i]) / 2
            
            # Generate control points
            segment_length = np.linalg.norm(end_point - start_point)
            control_factor = segment_length * self.smoothness_factor / 3
            
            ctrl1 = start_point + tangent_start * control_factor
            ctrl2 = end_point - tangent_end * control_factor
            
            control_points_sets.append([start_point, ctrl1, ctrl2, end_point])
        
        # Generate curves for each segment
        all_points = []
        all_velocities = []
        all_timestamps = []
        
        current_time = 0.0
        
        for control_points in control_points_sets:
            segment_curve = self.generate_cubic_bezier(control_points)
            
            all_points.append(segment_curve['points'])
            all_velocities.append(segment_curve['velocities'])
            
            # Generate timestamps
            segment_duration = segment_curve['arc_length'] / np.mean(
                np.linalg.norm(segment_curve['velocities'], axis=1)
            )
            segment_timestamps = np.linspace(current_time, current_time + segment_duration,
                                           len(segment_curve['points']))
            all_timestamps.append(segment_timestamps)
            
            current_time = segment_timestamps[-1]
        
        return {
            'points': np.concatenate(all_points),
            'velocities': np.concatenate(all_velocities),
            'timestamps': np.concatenate(all_timestamps)
        }
    
    def _generate_piecewise_trajectory(self, waypoints: np.ndarray) -> Dict[str, np.ndarray]:
        """Generate trajectory using piecewise cubic Bezier curves."""
        all_points = []
        all_velocities = []
        all_timestamps = []
        
        current_time = 0.0
        
        for i in range(len(waypoints) - 1):
            start_point = waypoints[i]
            end_point = waypoints[i + 1]
            
            # Simple control points (can be improved)
            direction = end_point - start_point
            ctrl1 = start_point + direction * 0.25
            ctrl2 = start_point + direction * 0.75
            
            control_points = [start_point, ctrl1, ctrl2, end_point]
            
            # Generate segment
            segment_curve = self.generate_cubic_bezier(control_points)
            
            all_points.append(segment_curve['points'])
            all_velocities.append(segment_curve['velocities'])
            
            # Timestamps
            segment_duration = 0.5  # Fixed duration per segment
            segment_timestamps = np.linspace(current_time, current_time + segment_duration,
                                           len(segment_curve['points']))
            all_timestamps.append(segment_timestamps)
            
            current_time = segment_timestamps[-1]
        
        return {
            'points': np.concatenate(all_points),
            'velocities': np.concatenate(all_velocities),
            'timestamps': np.concatenate(all_timestamps)
        }
    
    def optimize_control_points(self,
                              target_points: np.ndarray,
                              initial_control_points: List[np.ndarray],
                              max_iterations: int = 100) -> List[np.ndarray]:
        """
        Optimize control points to best fit target trajectory.
        
        Args:
            target_points: Target trajectory points
            initial_control_points: Initial control point guess
            max_iterations: Maximum optimization iterations
            
        Returns:
            optimized_control_points: Optimized control points
        """
        from scipy.optimize import minimize
        
        def objective(control_point_vector):
            # Reshape vector to control points
            control_points = []
            for i in range(0, len(control_point_vector), 2):
                control_points.append(np.array([control_point_vector[i], control_point_vector[i+1]]))
            
            # Generate curve
            curve_data = self.generate_cubic_bezier(control_points, len(target_points))
            
            # Compute error
            error = np.mean(np.linalg.norm(curve_data['points'] - target_points, axis=1))
            return error
        
        # Convert initial control points to vector
        initial_vector = []
        for cp in initial_control_points:
            initial_vector.extend([cp[0], cp[1]])
        
        # Optimize
        try:
            result = minimize(objective, initial_vector, method='BFGS',
                            options={'maxiter': max_iterations})
            
            if result.success:
                # Convert back to control points
                optimized_vector = result.x
                optimized_control_points = []
                for i in range(0, len(optimized_vector), 2):
                    optimized_control_points.append(
                        np.array([optimized_vector[i], optimized_vector[i+1]])
                    )
                return optimized_control_points
            else:
                logger.warning("Control point optimization failed")
                return initial_control_points
        
        except Exception as e:
            logger.warning(f"Control point optimization error: {e}")
            return initial_control_points
    
    def compute_curve_properties(self, curve_data: Dict[str, np.ndarray]) -> Dict[str, float]:
        """
        Compute geometric properties of a curve.
        
        Args:
            curve_data: Curve data from generate_cubic_bezier
            
        Returns:
            properties: Dictionary of curve properties
        """
        points = curve_data['points']
        velocities = curve_data['velocities']
        curvature = curve_data['curvature']
        
        if len(points) == 0:
            return {}
        
        # Basic properties
        arc_length = curve_data['arc_length']
        
        # Speed statistics
        speeds = np.linalg.norm(velocities, axis=1)
        mean_speed = np.mean(speeds)
        max_speed = np.max(speeds)
        speed_variation = np.std(speeds) / (mean_speed + 1e-8)
        
        # Curvature statistics
        mean_curvature = np.mean(curvature)
        max_curvature = np.max(curvature)
        
        # Smoothness (inverse of jerk)
        if len(velocities) >= 2:
            accelerations = np.diff(velocities, axis=0)
            jerk = np.diff(accelerations, axis=0)
            jerk_magnitude = np.linalg.norm(jerk, axis=1)
            smoothness = 1.0 / (1.0 + np.mean(jerk_magnitude))
        else:
            smoothness = 1.0
        
        # Bounding box
        min_pos = np.min(points, axis=0)
        max_pos = np.max(points, axis=0)
        bounding_box_area = (max_pos[0] - min_pos[0]) * (max_pos[1] - min_pos[1])
        
        return {
            'arc_length': arc_length,
            'mean_speed': mean_speed,
            'max_speed': max_speed,
            'speed_variation': speed_variation,
            'mean_curvature': mean_curvature,
            'max_curvature': max_curvature,
            'smoothness': smoothness,
            'bounding_box_area': bounding_box_area,
            'num_points': len(points)
        }