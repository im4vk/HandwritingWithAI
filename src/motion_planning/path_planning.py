"""
Path Planning Algorithms
=======================

Path planning algorithms for generating collision-free paths
and smooth trajectories for robotic handwriting.
"""

import numpy as np
from typing import Dict, Any, List, Tuple, Optional, Union, Callable
from dataclasses import dataclass
from enum import Enum
import logging
from collections import deque
import random
from scipy.spatial.distance import cdist
from scipy.spatial import cKDTree

from .forward_kinematics import ForwardKinematics
from .inverse_kinematics import InverseKinematics

logger = logging.getLogger(__name__)


class PlanningMethod(Enum):
    """Path planning methods."""
    RRT = "rrt"
    RRT_STAR = "rrt_star"
    RRT_CONNECT = "rrt_connect"
    POTENTIAL_FIELD = "potential_field"
    DIJKSTRA = "dijkstra"
    A_STAR = "a_star"


@dataclass
class PathPlanningResult:
    """
    Result of path planning computation.
    
    Attributes:
        path: Planned path as list of configurations
        cost: Path cost
        success: Whether planning succeeded
        planning_time: Time taken for planning
        nodes_explored: Number of nodes explored
        method: Planning method used
    """
    path: List[np.ndarray]
    cost: float
    success: bool
    planning_time: float
    nodes_explored: int
    method: str
    
    def get_path_array(self) -> np.ndarray:
        """Convert path to numpy array."""
        if not self.path:
            return np.array([]).reshape(0, -1)
        return np.array(self.path)


class CollisionChecker:
    """
    Collision checking for robot configurations.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize collision checker.
        
        Args:
            config: Collision checker configuration
        """
        self.config = config
        
        # Workspace limits
        self.workspace_limits = config.get('workspace_limits', {
            'x': [-1.0, 1.0],
            'y': [-1.0, 1.0], 
            'z': [0.0, 2.0]
        })
        
        # Obstacles (simplified as spheres and boxes)
        self.sphere_obstacles = config.get('sphere_obstacles', [])
        self.box_obstacles = config.get('box_obstacles', [])
        
        # Self-collision parameters
        self.enable_self_collision = config.get('enable_self_collision', False)
        self.min_link_distance = config.get('min_link_distance', 0.1)
        
        logger.info("Initialized CollisionChecker")
    
    def check_configuration(self, joint_angles: np.ndarray,
                          fk: ForwardKinematics) -> bool:
        """
        Check if configuration is collision-free.
        
        Args:
            joint_angles: Joint configuration to check
            fk: Forward kinematics solver
            
        Returns:
            is_valid: True if configuration is collision-free
        """
        # Check workspace limits
        if not self._check_workspace_limits(joint_angles, fk):
            return False
        
        # Check obstacle collisions
        if not self._check_obstacle_collisions(joint_angles, fk):
            return False
        
        # Check self-collisions
        if self.enable_self_collision and not self._check_self_collisions(joint_angles, fk):
            return False
        
        return True
    
    def _check_workspace_limits(self, joint_angles: np.ndarray,
                              fk: ForwardKinematics) -> bool:
        """Check if end-effector is within workspace limits."""
        transform = fk.forward_kinematics(joint_angles)
        pos = transform.position
        
        x_limits = self.workspace_limits['x']
        y_limits = self.workspace_limits['y']
        z_limits = self.workspace_limits['z']
        
        if (pos[0] < x_limits[0] or pos[0] > x_limits[1] or
            pos[1] < y_limits[0] or pos[1] > y_limits[1] or
            pos[2] < z_limits[0] or pos[2] > z_limits[1]):
            return False
        
        return True
    
    def _check_obstacle_collisions(self, joint_angles: np.ndarray,
                                 fk: ForwardKinematics) -> bool:
        """Check collisions with obstacles."""
        # Get link positions
        link_transforms = fk.compute_link_transforms(joint_angles)
        link_positions = [transform.position for transform in link_transforms]
        
        # Check sphere obstacles
        for obstacle in self.sphere_obstacles:
            center = np.array(obstacle['center'])
            radius = obstacle['radius']
            
            for link_pos in link_positions:
                distance = np.linalg.norm(link_pos - center)
                if distance < radius:
                    return False
        
        # Check box obstacles
        for obstacle in self.box_obstacles:
            min_corner = np.array(obstacle['min'])
            max_corner = np.array(obstacle['max'])
            
            for link_pos in link_positions:
                if (np.all(link_pos >= min_corner) and np.all(link_pos <= max_corner)):
                    return False
        
        return True
    
    def _check_self_collisions(self, joint_angles: np.ndarray,
                             fk: ForwardKinematics) -> bool:
        """Check self-collisions between robot links."""
        # Get link positions
        link_transforms = fk.compute_link_transforms(joint_angles)
        link_positions = [transform.position for transform in link_transforms]
        
        # Check distances between non-adjacent links
        for i in range(len(link_positions)):
            for j in range(i + 2, len(link_positions)):  # Skip adjacent links
                distance = np.linalg.norm(link_positions[i] - link_positions[j])
                if distance < self.min_link_distance:
                    return False
        
        return True
    
    def check_path(self, path: List[np.ndarray], fk: ForwardKinematics,
                  resolution: float = 0.1) -> bool:
        """
        Check if entire path is collision-free.
        
        Args:
            path: Path as list of configurations
            fk: Forward kinematics solver
            resolution: Resolution for path interpolation
            
        Returns:
            is_valid: True if entire path is collision-free
        """
        if len(path) < 2:
            return True
        
        for i in range(len(path) - 1):
            start_config = path[i]
            end_config = path[i + 1]
            
            # Interpolate between configurations
            distance = np.linalg.norm(end_config - start_config)
            num_steps = max(int(distance / resolution), 1)
            
            for step in range(num_steps + 1):
                alpha = step / num_steps
                interpolated_config = start_config + alpha * (end_config - start_config)
                
                if not self.check_configuration(interpolated_config, fk):
                    return False
        
        return True


class RRTPlanner:
    """
    Rapidly-Exploring Random Tree (RRT) path planner.
    """
    
    def __init__(self, config: Dict[str, Any], collision_checker: CollisionChecker):
        """
        Initialize RRT planner.
        
        Args:
            config: Planner configuration
            collision_checker: Collision checker
        """
        self.config = config
        self.collision_checker = collision_checker
        
        # RRT parameters
        self.max_iterations = config.get('max_iterations', 5000)
        self.step_size = config.get('step_size', 0.1)
        self.goal_bias = config.get('goal_bias', 0.1)
        self.goal_tolerance = config.get('goal_tolerance', 0.05)
        
        # Joint limits for sampling
        self.joint_limits = config.get('joint_limits', [(-np.pi, np.pi)] * 7)
        
        logger.info("Initialized RRTPlanner")
    
    def plan(self, start_config: np.ndarray, goal_config: np.ndarray,
            fk: ForwardKinematics) -> PathPlanningResult:
        """
        Plan path from start to goal using RRT.
        
        Args:
            start_config: Start configuration
            goal_config: Goal configuration
            fk: Forward kinematics solver
            
        Returns:
            result: Path planning result
        """
        import time
        start_time = time.time()
        
        # Initialize tree
        tree = RRTTree()
        tree.add_node(start_config, parent=None)
        
        # Check if start and goal are valid
        if not self.collision_checker.check_configuration(start_config, fk):
            return PathPlanningResult([], float('inf'), False, 0.0, 0, "rrt")
        
        if not self.collision_checker.check_configuration(goal_config, fk):
            return PathPlanningResult([], float('inf'), False, 0.0, 0, "rrt")
        
        # RRT main loop
        for iteration in range(self.max_iterations):
            # Sample random configuration
            if random.random() < self.goal_bias:
                sample_config = goal_config
            else:
                sample_config = self._sample_random_configuration()
            
            # Find nearest node in tree
            nearest_node = tree.find_nearest(sample_config)
            
            # Extend tree toward sample
            new_config = self._extend_toward(nearest_node.config, sample_config)
            
            # Check if new configuration is valid
            if self.collision_checker.check_configuration(new_config, fk):
                # Check if path to new configuration is valid
                if self.collision_checker.check_path([nearest_node.config, new_config], fk):
                    # Add new node to tree
                    new_node = tree.add_node(new_config, nearest_node)
                    
                    # Check if we reached the goal
                    if np.linalg.norm(new_config - goal_config) < self.goal_tolerance:
                        # Reconstruct path
                        path = tree.reconstruct_path(new_node)
                        cost = tree.compute_path_cost(path)
                        
                        return PathPlanningResult(
                            path=path,
                            cost=cost,
                            success=True,
                            planning_time=time.time() - start_time,
                            nodes_explored=iteration + 1,
                            method="rrt"
                        )
        
        # Failed to find path
        return PathPlanningResult(
            path=[],
            cost=float('inf'),
            success=False,
            planning_time=time.time() - start_time,
            nodes_explored=self.max_iterations,
            method="rrt"
        )
    
    def _sample_random_configuration(self) -> np.ndarray:
        """Sample random valid configuration."""
        config = np.array([
            random.uniform(limits[0], limits[1])
            for limits in self.joint_limits
        ])
        return config
    
    def _extend_toward(self, from_config: np.ndarray, to_config: np.ndarray) -> np.ndarray:
        """Extend from one configuration toward another."""
        direction = to_config - from_config
        distance = np.linalg.norm(direction)
        
        if distance <= self.step_size:
            return to_config
        else:
            unit_direction = direction / distance
            return from_config + self.step_size * unit_direction


class RRTNode:
    """Node in RRT tree."""
    
    def __init__(self, config: np.ndarray, parent: Optional['RRTNode'] = None):
        """Initialize RRT node."""
        self.config = config
        self.parent = parent
        self.children = []
        self.cost = 0.0
        
        if parent is not None:
            parent.children.append(self)
            self.cost = parent.cost + np.linalg.norm(config - parent.config)


class RRTTree:
    """RRT tree data structure."""
    
    def __init__(self):
        """Initialize RRT tree."""
        self.nodes = []
        self.kdtree = None
    
    def add_node(self, config: np.ndarray, parent: Optional[RRTNode] = None) -> RRTNode:
        """Add node to tree."""
        node = RRTNode(config, parent)
        self.nodes.append(node)
        
        # Rebuild KDTree for efficient nearest neighbor search
        if len(self.nodes) > 1:
            configs = [node.config for node in self.nodes]
            self.kdtree = cKDTree(configs)
        
        return node
    
    def find_nearest(self, config: np.ndarray) -> RRTNode:
        """Find nearest node to given configuration."""
        if len(self.nodes) == 1:
            return self.nodes[0]
        
        if self.kdtree is not None:
            _, nearest_idx = self.kdtree.query(config)
            return self.nodes[nearest_idx]
        else:
            # Fallback to linear search
            distances = [np.linalg.norm(config - node.config) for node in self.nodes]
            nearest_idx = np.argmin(distances)
            return self.nodes[nearest_idx]
    
    def reconstruct_path(self, goal_node: RRTNode) -> List[np.ndarray]:
        """Reconstruct path from root to goal node."""
        path = []
        current_node = goal_node
        
        while current_node is not None:
            path.append(current_node.config)
            current_node = current_node.parent
        
        path.reverse()
        return path
    
    def compute_path_cost(self, path: List[np.ndarray]) -> float:
        """Compute cost of path."""
        if len(path) < 2:
            return 0.0
        
        cost = 0.0
        for i in range(len(path) - 1):
            cost += np.linalg.norm(path[i + 1] - path[i])
        
        return cost


class PotentialFieldPlanner:
    """
    Potential field path planner.
    """
    
    def __init__(self, config: Dict[str, Any], collision_checker: CollisionChecker):
        """
        Initialize potential field planner.
        
        Args:
            config: Planner configuration
            collision_checker: Collision checker
        """
        self.config = config
        self.collision_checker = collision_checker
        
        # Potential field parameters
        self.max_iterations = config.get('max_iterations', 1000)
        self.step_size = config.get('step_size', 0.01)
        self.goal_tolerance = config.get('goal_tolerance', 0.05)
        
        # Potential function parameters
        self.attractive_gain = config.get('attractive_gain', 1.0)
        self.repulsive_gain = config.get('repulsive_gain', 1.0)
        self.influence_distance = config.get('influence_distance', 0.5)
        
        logger.info("Initialized PotentialFieldPlanner")
    
    def plan(self, start_config: np.ndarray, goal_config: np.ndarray,
            fk: ForwardKinematics) -> PathPlanningResult:
        """
        Plan path using potential field method.
        
        Args:
            start_config: Start configuration
            goal_config: Goal configuration
            fk: Forward kinematics solver
            
        Returns:
            result: Path planning result
        """
        import time
        start_time = time.time()
        
        current_config = start_config.copy()
        path = [current_config.copy()]
        
        for iteration in range(self.max_iterations):
            # Check if goal is reached
            if np.linalg.norm(current_config - goal_config) < self.goal_tolerance:
                cost = self._compute_path_cost(path)
                return PathPlanningResult(
                    path=path,
                    cost=cost,
                    success=True,
                    planning_time=time.time() - start_time,
                    nodes_explored=iteration + 1,
                    method="potential_field"
                )
            
            # Compute potential field gradient
            gradient = self._compute_gradient(current_config, goal_config, fk)
            
            # Update configuration
            next_config = current_config - self.step_size * gradient
            
            # Check if new configuration is valid
            if self.collision_checker.check_configuration(next_config, fk):
                current_config = next_config
                path.append(current_config.copy())
            else:
                # Add random component to escape local minima
                random_direction = np.random.normal(0, 0.1, len(current_config))
                next_config = current_config + random_direction
                
                if self.collision_checker.check_configuration(next_config, fk):
                    current_config = next_config
                    path.append(current_config.copy())
        
        # Failed to reach goal
        cost = self._compute_path_cost(path)
        return PathPlanningResult(
            path=path,
            cost=cost,
            success=False,
            planning_time=time.time() - start_time,
            nodes_explored=self.max_iterations,
            method="potential_field"
        )
    
    def _compute_gradient(self, config: np.ndarray, goal_config: np.ndarray,
                         fk: ForwardKinematics) -> np.ndarray:
        """Compute potential field gradient."""
        # Attractive force toward goal
        attractive_force = self.attractive_gain * (config - goal_config)
        
        # Repulsive forces from obstacles (simplified)
        repulsive_force = np.zeros_like(config)
        
        # Get current end-effector position
        transform = fk.forward_kinematics(config)
        ee_pos = transform.position
        
        # Repulsive forces from sphere obstacles
        for obstacle in self.collision_checker.sphere_obstacles:
            center = np.array(obstacle['center'])
            radius = obstacle['radius']
            
            distance = np.linalg.norm(ee_pos - center)
            if distance < self.influence_distance:
                # Compute repulsive force in configuration space
                direction = (ee_pos - center) / distance
                magnitude = self.repulsive_gain * (1.0 / distance - 1.0 / self.influence_distance) / (distance ** 2)
                
                # Map to configuration space using Jacobian
                jacobian = fk.compute_jacobian(config)
                repulsive_force += jacobian.T @ (magnitude * direction)
        
        return attractive_force + repulsive_force
    
    def _compute_path_cost(self, path: List[np.ndarray]) -> float:
        """Compute cost of path."""
        if len(path) < 2:
            return 0.0
        
        cost = 0.0
        for i in range(len(path) - 1):
            cost += np.linalg.norm(path[i + 1] - path[i])
        
        return cost


class PathPlanner:
    """
    Main path planning interface combining multiple algorithms.
    """
    
    def __init__(self, config: Dict[str, Any], fk: ForwardKinematics, ik: InverseKinematics):
        """
        Initialize path planner.
        
        Args:
            config: Configuration dictionary
            fk: Forward kinematics solver
            ik: Inverse kinematics solver
        """
        self.config = config
        self.fk = fk
        self.ik = ik
        
        # Initialize collision checker
        collision_config = config.get('collision_checker', {})
        self.collision_checker = CollisionChecker(collision_config)
        
        # Initialize planners
        rrt_config = config.get('rrt', {})
        self.rrt_planner = RRTPlanner(rrt_config, self.collision_checker)
        
        potential_config = config.get('potential_field', {})
        self.potential_planner = PotentialFieldPlanner(potential_config, self.collision_checker)
        
        # Default method
        self.default_method = PlanningMethod(config.get('default_method', 'rrt'))
        
        logger.info("Initialized PathPlanner with multiple algorithms")
    
    def plan_joint_space_path(self, start_config: np.ndarray, goal_config: np.ndarray,
                             method: Optional[PlanningMethod] = None) -> PathPlanningResult:
        """
        Plan path in joint space between two configurations.
        
        Args:
            start_config: Start joint configuration
            goal_config: Goal joint configuration
            method: Planning method to use
            
        Returns:
            result: Path planning result
        """
        if method is None:
            method = self.default_method
        
        if method == PlanningMethod.RRT:
            return self.rrt_planner.plan(start_config, goal_config, self.fk)
        elif method == PlanningMethod.POTENTIAL_FIELD:
            return self.potential_planner.plan(start_config, goal_config, self.fk)
        else:
            raise ValueError(f"Unsupported planning method: {method}")
    
    def plan_cartesian_path(self, start_pose: np.ndarray, goal_pose: np.ndarray,
                          initial_config: Optional[np.ndarray] = None,
                          method: Optional[PlanningMethod] = None) -> PathPlanningResult:
        """
        Plan path between Cartesian poses.
        
        Args:
            start_pose: Start pose (4x4 matrix or position)
            goal_pose: Goal pose (4x4 matrix or position)
            initial_config: Initial joint configuration for IK
            method: Planning method to use
            
        Returns:
            result: Path planning result
        """
        # Convert poses to joint configurations
        if start_pose.shape == (3,):
            # Position only
            start_matrix = np.eye(4)
            start_matrix[:3, 3] = start_pose
            start_pose = start_matrix
        
        if goal_pose.shape == (3,):
            # Position only
            goal_matrix = np.eye(4)
            goal_matrix[:3, 3] = goal_pose
            goal_pose = goal_matrix
        
        # Solve IK for start and goal poses
        start_ik_result = self.ik.solve(start_pose, initial_config)
        if not start_ik_result.success:
            return PathPlanningResult([], float('inf'), False, 0.0, 0, "cartesian_ik_failed")
        
        goal_ik_result = self.ik.solve(goal_pose, start_ik_result.joint_angles)
        if not goal_ik_result.success:
            return PathPlanningResult([], float('inf'), False, 0.0, 0, "cartesian_ik_failed")
        
        # Plan in joint space
        return self.plan_joint_space_path(start_ik_result.joint_angles, goal_ik_result.joint_angles, method)
    
    def plan_handwriting_trajectory(self, cartesian_trajectory: np.ndarray,
                                  initial_config: Optional[np.ndarray] = None,
                                  method: Optional[PlanningMethod] = None) -> Dict[str, Any]:
        """
        Plan robot trajectory for handwriting path.
        
        Args:
            cartesian_trajectory: Handwriting trajectory [n_points, 3]
            initial_config: Initial joint configuration
            method: Planning method for connecting segments
            
        Returns:
            trajectory_result: Complete trajectory planning result
        """
        n_points = len(cartesian_trajectory)
        joint_trajectory = np.zeros((n_points, self.fk.num_joints))
        planning_results = []
        
        current_config = initial_config if initial_config is not None else np.zeros(self.fk.num_joints)
        
        # Convert first point
        start_matrix = np.eye(4)
        start_matrix[:3, 3] = cartesian_trajectory[0]
        
        start_ik = self.ik.solve(start_matrix, current_config)
        if start_ik.success:
            joint_trajectory[0] = start_ik.joint_angles
            current_config = start_ik.joint_angles
        else:
            logger.warning("Failed to find IK solution for first trajectory point")
            joint_trajectory[0] = current_config
        
        # Plan path between consecutive points
        for i in range(1, n_points):
            target_matrix = np.eye(4)
            target_matrix[:3, 3] = cartesian_trajectory[i]
            
            # Solve IK for target point
            target_ik = self.ik.solve(target_matrix, current_config)
            
            if target_ik.success:
                # Plan path to target configuration
                if method is not None and np.linalg.norm(target_ik.joint_angles - current_config) > 0.1:
                    # Use path planning for large movements
                    path_result = self.plan_joint_space_path(current_config, target_ik.joint_angles, method)
                    planning_results.append(path_result)
                    
                    if path_result.success and len(path_result.path) > 1:
                        joint_trajectory[i] = path_result.path[-1]
                    else:
                        joint_trajectory[i] = target_ik.joint_angles
                else:
                    # Direct IK solution for small movements
                    joint_trajectory[i] = target_ik.joint_angles
                
                current_config = joint_trajectory[i]
            else:
                # Keep previous configuration if IK fails
                joint_trajectory[i] = current_config
                logger.warning(f"Failed to find IK solution for trajectory point {i}")
        
        return {
            'joint_trajectory': joint_trajectory,
            'cartesian_trajectory': cartesian_trajectory,
            'planning_results': planning_results,
            'success_rate': len([r for r in planning_results if r.success]) / max(len(planning_results), 1),
            'total_planning_time': sum(r.planning_time for r in planning_results)
        }
    
    def smooth_path(self, path: List[np.ndarray], iterations: int = 10) -> List[np.ndarray]:
        """
        Smooth path using simple averaging.
        
        Args:
            path: Original path
            iterations: Number of smoothing iterations
            
        Returns:
            smoothed_path: Smoothed path
        """
        if len(path) < 3:
            return path
        
        smoothed_path = [config.copy() for config in path]
        
        for _ in range(iterations):
            new_path = [smoothed_path[0]]  # Keep start point
            
            for i in range(1, len(smoothed_path) - 1):
                # Average with neighbors
                smoothed_config = (smoothed_path[i-1] + smoothed_path[i] + smoothed_path[i+1]) / 3.0
                
                # Check if smoothed configuration is valid
                if self.collision_checker.check_configuration(smoothed_config, self.fk):
                    new_path.append(smoothed_config)
                else:
                    new_path.append(smoothed_path[i])  # Keep original if invalid
            
            new_path.append(smoothed_path[-1])  # Keep end point
            smoothed_path = new_path
        
        return smoothed_path
    
    def validate_path(self, path: List[np.ndarray]) -> Dict[str, Any]:
        """
        Validate path for collision-free and kinematic feasibility.
        
        Args:
            path: Path to validate
            
        Returns:
            validation_result: Validation results
        """
        if not path:
            return {'valid': False, 'issues': ['Empty path']}
        
        issues = []
        warnings = []
        
        # Check each configuration
        for i, config in enumerate(path):
            if not self.collision_checker.check_configuration(config, self.fk):
                issues.append(f"Configuration {i} in collision")
        
        # Check path continuity
        max_step_size = 0.2  # radians
        for i in range(len(path) - 1):
            step_size = np.linalg.norm(path[i+1] - path[i])
            if step_size > max_step_size:
                warnings.append(f"Large step between configurations {i} and {i+1}: {step_size:.3f}")
        
        # Check path segments
        if not self.collision_checker.check_path(path, self.fk):
            issues.append("Path contains collision segments")
        
        return {
            'valid': len(issues) == 0,
            'issues': issues,
            'warnings': warnings,
            'path_length': len(path),
            'total_distance': sum(np.linalg.norm(path[i+1] - path[i]) for i in range(len(path)-1))
        }