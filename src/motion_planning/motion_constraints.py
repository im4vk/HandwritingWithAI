"""
Motion Constraints
=================

Constraints for robot motion including joint limits, velocity limits,
acceleration limits, and collision avoidance.
"""

import numpy as np
from typing import Dict, Any, List, Tuple, Optional, Union
from dataclasses import dataclass
from enum import Enum
import logging

from .forward_kinematics import ForwardKinematics

logger = logging.getLogger(__name__)


class ConstraintType(Enum):
    """Types of motion constraints."""
    JOINT_POSITION = "joint_position"
    JOINT_VELOCITY = "joint_velocity"
    JOINT_ACCELERATION = "joint_acceleration"
    JOINT_JERK = "joint_jerk"
    CARTESIAN_POSITION = "cartesian_position"
    CARTESIAN_VELOCITY = "cartesian_velocity"
    COLLISION_AVOIDANCE = "collision_avoidance"
    SINGULARITY_AVOIDANCE = "singularity_avoidance"


@dataclass
class JointLimits:
    """
    Joint limits for robot motion.
    
    Attributes:
        position_limits: Position limits [(min, max), ...] for each joint
        velocity_limits: Velocity limits [max_vel, ...] for each joint
        acceleration_limits: Acceleration limits [max_acc, ...] for each joint
        jerk_limits: Jerk limits [max_jerk, ...] for each joint
        joint_names: Names of joints
    """
    position_limits: List[Tuple[float, float]]
    velocity_limits: List[float]
    acceleration_limits: List[float]
    jerk_limits: List[float]
    joint_names: Optional[List[str]] = None
    
    def __post_init__(self):
        """Validate joint limits."""
        n_joints = len(self.position_limits)
        
        if len(self.velocity_limits) != n_joints:
            raise ValueError("Velocity limits length must match position limits")
        if len(self.acceleration_limits) != n_joints:
            raise ValueError("Acceleration limits length must match position limits")
        if len(self.jerk_limits) != n_joints:
            raise ValueError("Jerk limits length must match position limits")
        
        if self.joint_names and len(self.joint_names) != n_joints:
            raise ValueError("Joint names length must match number of joints")


@dataclass
class CartesianLimits:
    """
    Cartesian space limits for end-effector motion.
    
    Attributes:
        position_limits: Position limits [(x_min, x_max), (y_min, y_max), (z_min, z_max)]
        velocity_limits: Maximum velocity in each direction [vx_max, vy_max, vz_max]
        acceleration_limits: Maximum acceleration [ax_max, ay_max, az_max]
        orientation_limits: Orientation limits (if applicable)
    """
    position_limits: List[Tuple[float, float]]
    velocity_limits: List[float]
    acceleration_limits: List[float]
    orientation_limits: Optional[List[Tuple[float, float]]] = None


class CollisionChecker:
    """
    Collision checking for robot motion.
    """
    
    def __init__(self, config: Dict[str, Any], fk: ForwardKinematics):
        """
        Initialize collision checker.
        
        Args:
            config: Collision checker configuration
            fk: Forward kinematics solver
        """
        self.config = config
        self.fk = fk
        
        # Environment obstacles
        self.sphere_obstacles = config.get('sphere_obstacles', [])
        self.box_obstacles = config.get('box_obstacles', [])
        self.cylinder_obstacles = config.get('cylinder_obstacles', [])
        
        # Robot self-collision parameters
        self.enable_self_collision = config.get('enable_self_collision', True)
        self.link_radii = config.get('link_radii', [0.05] * fk.num_joints)
        self.min_link_distance = config.get('min_link_distance', 0.02)
        
        # Collision checking resolution
        self.resolution = config.get('resolution', 0.01)
        
        logger.info("Initialized CollisionChecker")
    
    def check_point_collision(self, point: np.ndarray) -> bool:
        """
        Check if a point is in collision with obstacles.
        
        Args:
            point: 3D point to check [x, y, z]
            
        Returns:
            in_collision: True if point is in collision
        """
        # Check sphere obstacles
        for obstacle in self.sphere_obstacles:
            center = np.array(obstacle['center'])
            radius = obstacle['radius']
            
            if np.linalg.norm(point - center) < radius:
                return True
        
        # Check box obstacles
        for obstacle in self.box_obstacles:
            min_corner = np.array(obstacle['min'])
            max_corner = np.array(obstacle['max'])
            
            if np.all(point >= min_corner) and np.all(point <= max_corner):
                return True
        
        # Check cylinder obstacles
        for obstacle in self.cylinder_obstacles:
            center = np.array(obstacle['center'])
            radius = obstacle['radius']
            height = obstacle['height']
            axis = obstacle.get('axis', 'z')
            
            if axis == 'z':
                # Check if within cylinder radius and height
                dist_2d = np.linalg.norm(point[:2] - center[:2])
                if (dist_2d < radius and 
                    center[2] <= point[2] <= center[2] + height):
                    return True
        
        return False
    
    def check_configuration_collision(self, joint_angles: np.ndarray) -> bool:
        """
        Check if robot configuration is in collision.
        
        Args:
            joint_angles: Joint configuration to check
            
        Returns:
            in_collision: True if configuration is in collision
        """
        # Get link positions
        link_transforms = self.fk.compute_link_transforms(joint_angles)
        link_positions = [transform.position for transform in link_transforms]
        
        # Check each link against obstacles
        for i, link_pos in enumerate(link_positions):
            # Check point collision (simplified)
            if self.check_point_collision(link_pos):
                return True
            
            # Check sphere around link
            link_radius = self.link_radii[i] if i < len(self.link_radii) else 0.05
            
            for obstacle in self.sphere_obstacles:
                center = np.array(obstacle['center'])
                obstacle_radius = obstacle['radius']
                
                distance = np.linalg.norm(link_pos - center)
                if distance < (link_radius + obstacle_radius):
                    return True
        
        # Check self-collision
        if self.enable_self_collision:
            return self._check_self_collision(link_positions)
        
        return False
    
    def _check_self_collision(self, link_positions: List[np.ndarray]) -> bool:
        """Check for self-collision between robot links."""
        for i in range(len(link_positions)):
            for j in range(i + 2, len(link_positions)):  # Skip adjacent links
                distance = np.linalg.norm(link_positions[i] - link_positions[j])
                min_distance = self.link_radii[i] + self.link_radii[j] + self.min_link_distance
                
                if distance < min_distance:
                    return True
        
        return False
    
    def check_path_collision(self, path: List[np.ndarray]) -> Tuple[bool, List[int]]:
        """
        Check path for collisions.
        
        Args:
            path: Path as list of joint configurations
            
        Returns:
            has_collision: True if path has collisions
            collision_indices: Indices of configurations in collision
        """
        collision_indices = []
        
        for i, config in enumerate(path):
            if self.check_configuration_collision(config):
                collision_indices.append(i)
        
        # Check intermediate configurations between waypoints
        for i in range(len(path) - 1):
            start_config = path[i]
            end_config = path[i + 1]
            
            # Interpolate between configurations
            distance = np.linalg.norm(end_config - start_config)
            num_steps = max(int(distance / self.resolution), 1)
            
            for step in range(1, num_steps):
                alpha = step / num_steps
                intermediate_config = start_config + alpha * (end_config - start_config)
                
                if self.check_configuration_collision(intermediate_config):
                    collision_indices.append(i + alpha)  # Fractional index
        
        return len(collision_indices) > 0, collision_indices


class MotionConstraints:
    """
    Complete motion constraints system for robot motion planning.
    """
    
    def __init__(self, config: Dict[str, Any], fk: ForwardKinematics):
        """
        Initialize motion constraints.
        
        Args:
            config: Constraints configuration
            fk: Forward kinematics solver
        """
        self.config = config
        self.fk = fk
        
        # Initialize joint limits
        self.joint_limits = self._initialize_joint_limits(config)
        
        # Initialize Cartesian limits
        self.cartesian_limits = self._initialize_cartesian_limits(config)
        
        # Initialize collision checker
        collision_config = config.get('collision_checker', {})
        self.collision_checker = CollisionChecker(collision_config, fk)
        
        # Constraint checking parameters
        self.check_collisions = config.get('check_collisions', True)
        self.check_singularities = config.get('check_singularities', True)
        self.singularity_threshold = config.get('singularity_threshold', 0.01)
        
        logger.info("Initialized MotionConstraints")
    
    def _initialize_joint_limits(self, config: Dict[str, Any]) -> JointLimits:
        """Initialize joint limits from configuration."""
        joint_config = config.get('joint_limits', {})
        
        # Default limits for 7-DOF robot
        default_position_limits = [(-2.8973, 2.8973)] * 7  # Franka Panda-like limits
        default_velocity_limits = [2.1750, 2.1750, 2.1750, 2.1750, 2.6100, 2.6100, 2.6100]
        default_acceleration_limits = [15.0, 7.5, 10.0, 12.5, 15.0, 20.0, 20.0]
        default_jerk_limits = [7500.0, 3750.0, 5000.0, 6250.0, 7500.0, 10000.0, 10000.0]
        
        position_limits = joint_config.get('position_limits', default_position_limits)
        velocity_limits = joint_config.get('velocity_limits', default_velocity_limits)
        acceleration_limits = joint_config.get('acceleration_limits', default_acceleration_limits)
        jerk_limits = joint_config.get('jerk_limits', default_jerk_limits)
        joint_names = joint_config.get('joint_names', [f'joint_{i+1}' for i in range(7)])
        
        return JointLimits(
            position_limits=position_limits,
            velocity_limits=velocity_limits,
            acceleration_limits=acceleration_limits,
            jerk_limits=jerk_limits,
            joint_names=joint_names
        )
    
    def _initialize_cartesian_limits(self, config: Dict[str, Any]) -> CartesianLimits:
        """Initialize Cartesian limits from configuration."""
        cartesian_config = config.get('cartesian_limits', {})
        
        # Default Cartesian limits
        default_position_limits = [(-1.0, 1.0), (-1.0, 1.0), (0.0, 2.0)]  # x, y, z
        default_velocity_limits = [1.7, 1.7, 1.7]  # m/s
        default_acceleration_limits = [13.0, 13.0, 13.0]  # m/s²
        default_orientation_limits = [(-np.pi, np.pi)] * 3  # roll, pitch, yaw
        
        return CartesianLimits(
            position_limits=cartesian_config.get('position_limits', default_position_limits),
            velocity_limits=cartesian_config.get('velocity_limits', default_velocity_limits),
            acceleration_limits=cartesian_config.get('acceleration_limits', default_acceleration_limits),
            orientation_limits=cartesian_config.get('orientation_limits', default_orientation_limits)
        )
    
    def check_joint_position_limits(self, joint_angles: np.ndarray) -> Tuple[bool, List[str]]:
        """
        Check joint position limits.
        
        Args:
            joint_angles: Joint angles to check
            
        Returns:
            within_limits: True if all joints are within limits
            violations: List of violation descriptions
        """
        violations = []
        
        for i, (angle, limits) in enumerate(zip(joint_angles, self.joint_limits.position_limits)):
            joint_name = self.joint_limits.joint_names[i] if self.joint_limits.joint_names else f"joint_{i+1}"
            
            if angle < limits[0]:
                violations.append(f"{joint_name}: {angle:.4f} < {limits[0]:.4f}")
            elif angle > limits[1]:
                violations.append(f"{joint_name}: {angle:.4f} > {limits[1]:.4f}")
        
        return len(violations) == 0, violations
    
    def check_joint_velocity_limits(self, joint_velocities: np.ndarray) -> Tuple[bool, List[str]]:
        """Check joint velocity limits."""
        violations = []
        
        for i, (velocity, limit) in enumerate(zip(joint_velocities, self.joint_limits.velocity_limits)):
            joint_name = self.joint_limits.joint_names[i] if self.joint_limits.joint_names else f"joint_{i+1}"
            
            if abs(velocity) > limit:
                violations.append(f"{joint_name}: |{velocity:.4f}| > {limit:.4f}")
        
        return len(violations) == 0, violations
    
    def check_joint_acceleration_limits(self, joint_accelerations: np.ndarray) -> Tuple[bool, List[str]]:
        """Check joint acceleration limits."""
        violations = []
        
        for i, (acceleration, limit) in enumerate(zip(joint_accelerations, self.joint_limits.acceleration_limits)):
            joint_name = self.joint_limits.joint_names[i] if self.joint_limits.joint_names else f"joint_{i+1}"
            
            if abs(acceleration) > limit:
                violations.append(f"{joint_name}: |{acceleration:.4f}| > {limit:.4f}")
        
        return len(violations) == 0, violations
    
    def check_cartesian_position_limits(self, joint_angles: np.ndarray) -> Tuple[bool, List[str]]:
        """Check Cartesian position limits for end-effector."""
        violations = []
        
        # Get end-effector position
        transform = self.fk.forward_kinematics(joint_angles)
        position = transform.position
        
        # Check position limits
        axes = ['x', 'y', 'z']
        for i, (pos, limits) in enumerate(zip(position, self.cartesian_limits.position_limits)):
            if pos < limits[0]:
                violations.append(f"{axes[i]}: {pos:.4f} < {limits[0]:.4f}")
            elif pos > limits[1]:
                violations.append(f"{axes[i]}: {pos:.4f} > {limits[1]:.4f}")
        
        return len(violations) == 0, violations
    
    def check_singularity(self, joint_angles: np.ndarray) -> Tuple[bool, float]:
        """
        Check if configuration is near singularity.
        
        Args:
            joint_angles: Joint configuration to check
            
        Returns:
            is_singular: True if near singularity
            manipulability: Manipulability measure
        """
        try:
            # Compute Jacobian
            jacobian = self.fk.compute_jacobian(joint_angles)
            
            # Compute manipulability (determinant of J*J^T)
            manipulability = np.sqrt(np.linalg.det(jacobian @ jacobian.T))
            
            is_singular = manipulability < self.singularity_threshold
            
            return is_singular, manipulability
        
        except Exception as e:
            logger.warning(f"Singularity check failed: {e}")
            return True, 0.0
    
    def check_all_constraints(self, joint_angles: np.ndarray,
                            joint_velocities: Optional[np.ndarray] = None,
                            joint_accelerations: Optional[np.ndarray] = None) -> Dict[str, Any]:
        """
        Check all constraints for given joint configuration.
        
        Args:
            joint_angles: Joint configuration
            joint_velocities: Joint velocities (optional)
            joint_accelerations: Joint accelerations (optional)
            
        Returns:
            constraint_check: Complete constraint check result
        """
        result = {
            'valid': True,
            'violations': [],
            'warnings': []
        }
        
        # Check joint position limits
        pos_valid, pos_violations = self.check_joint_position_limits(joint_angles)
        if not pos_valid:
            result['valid'] = False
            result['violations'].extend(pos_violations)
        
        # Check joint velocity limits
        if joint_velocities is not None:
            vel_valid, vel_violations = self.check_joint_velocity_limits(joint_velocities)
            if not vel_valid:
                result['valid'] = False
                result['violations'].extend(vel_violations)
        
        # Check joint acceleration limits
        if joint_accelerations is not None:
            acc_valid, acc_violations = self.check_joint_acceleration_limits(joint_accelerations)
            if not acc_valid:
                result['valid'] = False
                result['violations'].extend(acc_violations)
        
        # Check Cartesian position limits
        cart_valid, cart_violations = self.check_cartesian_position_limits(joint_angles)
        if not cart_valid:
            result['valid'] = False
            result['violations'].extend(cart_violations)
        
        # Check singularities
        if self.check_singularities:
            is_singular, manipulability = self.check_singularity(joint_angles)
            result['manipulability'] = manipulability
            
            if is_singular:
                result['warnings'].append(f"Near singularity: manipulability = {manipulability:.6f}")
        
        # Check collisions
        if self.check_collisions:
            in_collision = self.collision_checker.check_configuration_collision(joint_angles)
            result['in_collision'] = in_collision
            
            if in_collision:
                result['valid'] = False
                result['violations'].append("Configuration in collision")
        
        return result
    
    def check_trajectory_constraints(self, joint_trajectory: np.ndarray,
                                   timestamps: np.ndarray) -> Dict[str, Any]:
        """
        Check constraints for entire trajectory.
        
        Args:
            joint_trajectory: Joint trajectory [n_points, n_joints]
            timestamps: Time stamps [n_points]
            
        Returns:
            trajectory_check: Trajectory constraint check result
        """
        n_points = len(joint_trajectory)
        
        # Compute velocities and accelerations
        dt = np.diff(timestamps)
        velocities = np.diff(joint_trajectory, axis=0) / dt.reshape(-1, 1)
        accelerations = np.diff(velocities, axis=0) / dt[1:].reshape(-1, 1)
        
        # Check each point
        violations_per_point = []
        collision_points = []
        singular_points = []
        
        for i, config in enumerate(joint_trajectory):
            # Get velocities and accelerations for this point
            vel = velocities[i-1] if i > 0 else np.zeros(joint_trajectory.shape[1])
            acc = accelerations[i-2] if i > 1 else np.zeros(joint_trajectory.shape[1])
            
            # Check constraints
            check_result = self.check_all_constraints(config, vel, acc)
            
            if not check_result['valid']:
                violations_per_point.append((i, check_result['violations']))
            
            if check_result.get('in_collision', False):
                collision_points.append(i)
            
            if 'manipulability' in check_result and check_result['manipulability'] < self.singularity_threshold:
                singular_points.append(i)
        
        # Overall statistics
        total_violations = sum(len(violations) for _, violations in violations_per_point)
        collision_percentage = len(collision_points) / n_points * 100
        singularity_percentage = len(singular_points) / n_points * 100
        
        return {
            'valid': total_violations == 0 and len(collision_points) == 0,
            'total_violations': total_violations,
            'violations_per_point': violations_per_point,
            'collision_points': collision_points,
            'collision_percentage': collision_percentage,
            'singular_points': singular_points,
            'singularity_percentage': singularity_percentage,
            'max_velocity': np.max(np.abs(velocities)) if len(velocities) > 0 else 0.0,
            'max_acceleration': np.max(np.abs(accelerations)) if len(accelerations) > 0 else 0.0,
            'trajectory_duration': timestamps[-1] - timestamps[0]
        }
    
    def get_safety_margins(self, joint_angles: np.ndarray) -> Dict[str, float]:
        """
        Get safety margins for current configuration.
        
        Args:
            joint_angles: Current joint configuration
            
        Returns:
            margins: Safety margins for various constraints
        """
        margins = {}
        
        # Joint position margins
        joint_margins = []
        for i, (angle, limits) in enumerate(zip(joint_angles, self.joint_limits.position_limits)):
            range_size = limits[1] - limits[0]
            center = (limits[0] + limits[1]) / 2
            margin = 1.0 - abs(angle - center) / (range_size / 2)
            joint_margins.append(margin)
        
        margins['min_joint_margin'] = min(joint_margins)
        margins['avg_joint_margin'] = np.mean(joint_margins)
        
        # Manipulability margin
        if self.check_singularities:
            _, manipulability = self.check_singularity(joint_angles)
            margins['manipulability'] = manipulability
            margins['singularity_margin'] = manipulability / self.singularity_threshold
        
        # Cartesian workspace margin
        transform = self.fk.forward_kinematics(joint_angles)
        position = transform.position
        
        workspace_margins = []
        for i, (pos, limits) in enumerate(zip(position, self.cartesian_limits.position_limits)):
            range_size = limits[1] - limits[0]
            center = (limits[0] + limits[1]) / 2
            margin = 1.0 - abs(pos - center) / (range_size / 2)
            workspace_margins.append(margin)
        
        margins['min_workspace_margin'] = min(workspace_margins)
        margins['avg_workspace_margin'] = np.mean(workspace_margins)
        
        return margins
    
    def suggest_constraint_relaxation(self, violations: List[str]) -> List[str]:
        """
        Suggest constraint relaxations based on violations.
        
        Args:
            violations: List of constraint violations
            
        Returns:
            suggestions: List of suggested relaxations
        """
        suggestions = []
        
        # Analyze violation patterns
        joint_violations = [v for v in violations if 'joint_' in v]
        cartesian_violations = [v for v in violations if any(axis in v for axis in ['x:', 'y:', 'z:'])]
        collision_violations = [v for v in violations if 'collision' in v]
        
        if joint_violations:
            suggestions.append("Consider expanding joint limits for frequently violated joints")
        
        if cartesian_violations:
            suggestions.append("Consider expanding workspace limits or repositioning robot base")
        
        if collision_violations:
            suggestions.append("Consider modifying obstacle configuration or robot path")
        
        # Specific suggestions based on violation frequency
        joint_counts = {}
        for violation in joint_violations:
            for i in range(len(self.joint_limits.joint_names)):
                joint_name = self.joint_limits.joint_names[i]
                if joint_name in violation:
                    joint_counts[joint_name] = joint_counts.get(joint_name, 0) + 1
        
        most_violated_joint = max(joint_counts, key=joint_counts.get) if joint_counts else None
        if most_violated_joint:
            suggestions.append(f"Joint {most_violated_joint} is most frequently violated - check mechanical design")
        
        return suggestions