"""
Trajectory Optimization
======================

Trajectory optimization algorithms for smooth and efficient
robot motion with various objectives and constraints.
"""

import numpy as np
from typing import Dict, Any, List, Tuple, Optional, Union, Callable
from dataclasses import dataclass
from enum import Enum
import logging
from scipy.optimize import minimize, differential_evolution
from scipy.interpolate import interp1d, CubicSpline
import cvxpy as cp

from .forward_kinematics import ForwardKinematics
from .motion_constraints import MotionConstraints

logger = logging.getLogger(__name__)


class OptimizationObjective(Enum):
    """Trajectory optimization objectives."""
    MINIMUM_TIME = "minimum_time"
    MINIMUM_JERK = "minimum_jerk"
    MINIMUM_ENERGY = "minimum_energy"
    MAXIMUM_SMOOTHNESS = "maximum_smoothness"
    COMBINED = "combined"


@dataclass
class OptimizationConstraints:
    """
    Constraints for trajectory optimization.
    
    Attributes:
        joint_limits: Joint position limits [(min, max), ...]
        velocity_limits: Joint velocity limits [max_vel, ...]
        acceleration_limits: Joint acceleration limits [max_acc, ...]
        jerk_limits: Joint jerk limits [max_jerk, ...]
        waypoint_constraints: Must-pass waypoints [(time, config), ...]
        collision_constraints: Collision avoidance constraints
        smoothness_constraints: Smoothness requirements
    """
    joint_limits: Optional[List[Tuple[float, float]]] = None
    velocity_limits: Optional[List[float]] = None
    acceleration_limits: Optional[List[float]] = None
    jerk_limits: Optional[List[float]] = None
    waypoint_constraints: Optional[List[Tuple[float, np.ndarray]]] = None
    collision_constraints: bool = True
    smoothness_constraints: bool = True


@dataclass
class OptimizationResult:
    """
    Result of trajectory optimization.
    
    Attributes:
        optimized_trajectory: Optimized joint trajectory [n_points, n_joints]
        timestamps: Time stamps [n_points]
        velocities: Joint velocities [n_points, n_joints]
        accelerations: Joint accelerations [n_points, n_joints]
        objective_value: Final objective function value
        success: Whether optimization succeeded
        iterations: Number of optimization iterations
        computation_time: Time taken for optimization
        constraints_satisfied: Whether all constraints are satisfied
    """
    optimized_trajectory: np.ndarray
    timestamps: np.ndarray
    velocities: np.ndarray
    accelerations: np.ndarray
    objective_value: float
    success: bool
    iterations: int
    computation_time: float
    constraints_satisfied: bool


class TrajectoryOptimizer:
    """
    Main trajectory optimization class with multiple algorithms.
    """
    
    def __init__(self, config: Dict[str, Any], fk: ForwardKinematics):
        """
        Initialize trajectory optimizer.
        
        Args:
            config: Optimizer configuration
            fk: Forward kinematics solver
        """
        self.config = config
        self.fk = fk
        
        # Optimization parameters
        self.max_iterations = config.get('max_iterations', 1000)
        self.tolerance = config.get('tolerance', 1e-6)
        self.use_cvxpy = config.get('use_cvxpy', True)
        
        # Objective weights
        self.time_weight = config.get('time_weight', 1.0)
        self.jerk_weight = config.get('jerk_weight', 0.1)
        self.energy_weight = config.get('energy_weight', 0.01)
        self.smoothness_weight = config.get('smoothness_weight', 0.1)
        
        # Motion constraints
        constraints_config = config.get('motion_constraints', {})
        self.motion_constraints = MotionConstraints(constraints_config, fk)
        
        logger.info("Initialized TrajectoryOptimizer")
    
    def optimize_trajectory(self, initial_trajectory: np.ndarray,
                          timestamps: np.ndarray,
                          objective: OptimizationObjective = OptimizationObjective.MINIMUM_JERK,
                          constraints: Optional[OptimizationConstraints] = None) -> OptimizationResult:
        """
        Optimize trajectory with specified objective and constraints.
        
        Args:
            initial_trajectory: Initial trajectory [n_points, n_joints]
            timestamps: Time stamps [n_points]
            objective: Optimization objective
            constraints: Optimization constraints
            
        Returns:
            result: Optimization result
        """
        import time
        start_time = time.time()
        
        n_points, n_joints = initial_trajectory.shape
        
        # Set default constraints
        if constraints is None:
            constraints = self._get_default_constraints(n_joints)
        
        # Choose optimization method based on objective
        if objective == OptimizationObjective.MINIMUM_JERK and self.use_cvxpy:
            result = self._optimize_minimum_jerk_cvx(initial_trajectory, timestamps, constraints)
        elif objective == OptimizationObjective.MINIMUM_TIME:
            result = self._optimize_minimum_time(initial_trajectory, timestamps, constraints)
        elif objective == OptimizationObjective.MINIMUM_ENERGY:
            result = self._optimize_minimum_energy(initial_trajectory, timestamps, constraints)
        elif objective == OptimizationObjective.MAXIMUM_SMOOTHNESS:
            result = self._optimize_smoothness(initial_trajectory, timestamps, constraints)
        elif objective == OptimizationObjective.COMBINED:
            result = self._optimize_combined_objective(initial_trajectory, timestamps, constraints)
        else:
            # Fallback to scipy-based minimum jerk
            result = self._optimize_minimum_jerk_scipy(initial_trajectory, timestamps, constraints)
        
        # Set computation time
        result.computation_time = time.time() - start_time
        
        return result
    
    def _get_default_constraints(self, n_joints: int) -> OptimizationConstraints:
        """Get default optimization constraints."""
        return OptimizationConstraints(
            joint_limits=[(-np.pi, np.pi)] * n_joints,
            velocity_limits=[2.0] * n_joints,  # rad/s
            acceleration_limits=[5.0] * n_joints,  # rad/s²
            jerk_limits=[50.0] * n_joints,  # rad/s³
            collision_constraints=True,
            smoothness_constraints=True
        )
    
    def _optimize_minimum_jerk_cvx(self, initial_trajectory: np.ndarray,
                                  timestamps: np.ndarray,
                                  constraints: OptimizationConstraints) -> OptimizationResult:
        """Optimize for minimum jerk using CVXPY."""
        try:
            n_points, n_joints = initial_trajectory.shape
            dt = np.mean(np.diff(timestamps))
            
            # Decision variables
            trajectory = cp.Variable((n_points, n_joints))
            
            # Objective: minimize jerk (third derivative)
            jerk_cost = 0
            for i in range(3, n_points):
                # Finite difference approximation of jerk
                jerk = (trajectory[i] - 3*trajectory[i-1] + 3*trajectory[i-2] - trajectory[i-3]) / (dt**3)
                jerk_cost += cp.sum_squares(jerk)
            
            objective = cp.Minimize(jerk_cost)
            
            # Constraints
            constraints_list = []
            
            # Keep start and end points fixed
            constraints_list.append(trajectory[0] == initial_trajectory[0])
            constraints_list.append(trajectory[-1] == initial_trajectory[-1])
            
            # Joint limits
            if constraints.joint_limits:
                for j, (min_val, max_val) in enumerate(constraints.joint_limits):
                    constraints_list.append(trajectory[:, j] >= min_val)
                    constraints_list.append(trajectory[:, j] <= max_val)
            
            # Velocity limits
            if constraints.velocity_limits:
                for i in range(n_points - 1):
                    velocity = (trajectory[i+1] - trajectory[i]) / dt
                    for j, max_vel in enumerate(constraints.velocity_limits):
                        constraints_list.append(velocity[j] >= -max_vel)
                        constraints_list.append(velocity[j] <= max_vel)
            
            # Acceleration limits
            if constraints.acceleration_limits:
                for i in range(n_points - 2):
                    acceleration = (trajectory[i+2] - 2*trajectory[i+1] + trajectory[i]) / (dt**2)
                    for j, max_acc in enumerate(constraints.acceleration_limits):
                        constraints_list.append(acceleration[j] >= -max_acc)
                        constraints_list.append(acceleration[j] <= max_acc)
            
            # Waypoint constraints
            if constraints.waypoint_constraints:
                for time_point, waypoint_config in constraints.waypoint_constraints:
                    # Find closest time index
                    time_idx = np.argmin(np.abs(timestamps - time_point))
                    constraints_list.append(trajectory[time_idx] == waypoint_config)
            
            # Solve optimization problem
            problem = cp.Problem(objective, constraints_list)
            problem.solve(verbose=False)
            
            if problem.status == cp.OPTIMAL:
                optimized_traj = trajectory.value
                success = True
                objective_value = problem.value
            else:
                logger.warning(f"CVXPY optimization failed with status: {problem.status}")
                optimized_traj = initial_trajectory
                success = False
                objective_value = float('inf')
            
        except Exception as e:
            logger.warning(f"CVXPY optimization error: {e}")
            optimized_traj = initial_trajectory
            success = False
            objective_value = float('inf')
        
        # Compute velocities and accelerations
        velocities = self._compute_velocities(optimized_traj, timestamps)
        accelerations = self._compute_accelerations(velocities, timestamps)
        
        # Check constraints satisfaction
        constraints_satisfied = self._check_constraints_satisfaction(
            optimized_traj, velocities, accelerations, timestamps, constraints
        )
        
        return OptimizationResult(
            optimized_trajectory=optimized_traj,
            timestamps=timestamps,
            velocities=velocities,
            accelerations=accelerations,
            objective_value=objective_value,
            success=success,
            iterations=0,  # CVXPY doesn't report iterations
            computation_time=0.0,  # Will be set by caller
            constraints_satisfied=constraints_satisfied
        )
    
    def _optimize_minimum_jerk_scipy(self, initial_trajectory: np.ndarray,
                                   timestamps: np.ndarray,
                                   constraints: OptimizationConstraints) -> OptimizationResult:
        """Optimize for minimum jerk using scipy."""
        n_points, n_joints = initial_trajectory.shape
        dt = np.mean(np.diff(timestamps))
        
        def objective_function(traj_flat):
            trajectory = traj_flat.reshape(n_points, n_joints)
            
            # Compute jerk
            jerk_cost = 0
            for i in range(3, n_points):
                jerk = (trajectory[i] - 3*trajectory[i-1] + 3*trajectory[i-2] + trajectory[i-3]) / (dt**3)
                jerk_cost += np.sum(jerk**2)
            
            return jerk_cost
        
        def constraint_function(traj_flat):
            trajectory = traj_flat.reshape(n_points, n_joints)
            violations = []
            
            # Velocity constraints
            if constraints.velocity_limits:
                velocities = self._compute_velocities(trajectory, timestamps)
                for j, max_vel in enumerate(constraints.velocity_limits):
                    violations.extend(max_vel - np.abs(velocities[:, j]))
            
            # Acceleration constraints
            if constraints.acceleration_limits:
                velocities = self._compute_velocities(trajectory, timestamps)
                accelerations = self._compute_accelerations(velocities, timestamps)
                for j, max_acc in enumerate(constraints.acceleration_limits):
                    violations.extend(max_acc - np.abs(accelerations[:, j]))
            
            return np.array(violations)
        
        # Set up bounds
        bounds = []
        for i in range(n_points):
            for j in range(n_joints):
                if i == 0 or i == n_points - 1:
                    # Fix start and end points
                    bounds.append((initial_trajectory[i, j], initial_trajectory[i, j]))
                else:
                    # Use joint limits
                    if constraints.joint_limits:
                        bounds.append(constraints.joint_limits[j])
                    else:
                        bounds.append((-np.pi, np.pi))
        
        # Initial guess
        x0 = initial_trajectory.flatten()
        
        # Optimization
        try:
            if len(constraint_function(x0)) > 0:
                # Constrained optimization
                constraint_dict = {'type': 'ineq', 'fun': constraint_function}
                result = minimize(objective_function, x0, method='SLSQP',
                                bounds=bounds, constraints=constraint_dict,
                                options={'maxiter': self.max_iterations})
            else:
                # Unconstrained optimization (with bounds)
                result = minimize(objective_function, x0, method='L-BFGS-B',
                                bounds=bounds, options={'maxiter': self.max_iterations})
            
            if result.success:
                optimized_traj = result.x.reshape(n_points, n_joints)
                success = True
                objective_value = result.fun
                iterations = result.nit
            else:
                optimized_traj = initial_trajectory
                success = False
                objective_value = float('inf')
                iterations = 0
        
        except Exception as e:
            logger.warning(f"Scipy optimization error: {e}")
            optimized_traj = initial_trajectory
            success = False
            objective_value = float('inf')
            iterations = 0
        
        # Compute derivatives
        velocities = self._compute_velocities(optimized_traj, timestamps)
        accelerations = self._compute_accelerations(velocities, timestamps)
        
        # Check constraints
        constraints_satisfied = self._check_constraints_satisfaction(
            optimized_traj, velocities, accelerations, timestamps, constraints
        )
        
        return OptimizationResult(
            optimized_trajectory=optimized_traj,
            timestamps=timestamps,
            velocities=velocities,
            accelerations=accelerations,
            objective_value=objective_value,
            success=success,
            iterations=iterations,
            computation_time=0.0,
            constraints_satisfied=constraints_satisfied
        )
    
    def _optimize_minimum_time(self, initial_trajectory: np.ndarray,
                             timestamps: np.ndarray,
                             constraints: OptimizationConstraints) -> OptimizationResult:
        """Optimize for minimum time using time scaling."""
        n_points = len(timestamps)
        
        def objective_function(time_scaling_factor):
            if time_scaling_factor <= 0:
                return float('inf')
            
            # Scale time
            scaled_timestamps = timestamps * time_scaling_factor
            
            # Compute velocities and accelerations with scaled time
            velocities = self._compute_velocities(initial_trajectory, scaled_timestamps)
            accelerations = self._compute_accelerations(velocities, scaled_timestamps)
            
            # Check if constraints are satisfied
            if not self._check_constraints_satisfaction(
                initial_trajectory, velocities, accelerations, scaled_timestamps, constraints
            ):
                return float('inf')
            
            return scaled_timestamps[-1]  # Total time
        
        # Optimize time scaling factor
        try:
            result = minimize(objective_function, 1.0, method='Brent',
                            options={'maxiter': self.max_iterations})
            
            if result.success:
                optimal_scaling = result.x
                scaled_timestamps = timestamps * optimal_scaling
                success = True
                objective_value = result.fun
            else:
                scaled_timestamps = timestamps
                success = False
                objective_value = float('inf')
        
        except Exception as e:
            logger.warning(f"Time optimization error: {e}")
            scaled_timestamps = timestamps
            success = False
            objective_value = float('inf')
        
        # Compute final derivatives
        velocities = self._compute_velocities(initial_trajectory, scaled_timestamps)
        accelerations = self._compute_accelerations(velocities, scaled_timestamps)
        
        constraints_satisfied = self._check_constraints_satisfaction(
            initial_trajectory, velocities, accelerations, scaled_timestamps, constraints
        )
        
        return OptimizationResult(
            optimized_trajectory=initial_trajectory,
            timestamps=scaled_timestamps,
            velocities=velocities,
            accelerations=accelerations,
            objective_value=objective_value,
            success=success,
            iterations=0,
            computation_time=0.0,
            constraints_satisfied=constraints_satisfied
        )
    
    def _optimize_minimum_energy(self, initial_trajectory: np.ndarray,
                               timestamps: np.ndarray,
                               constraints: OptimizationConstraints) -> OptimizationResult:
        """Optimize for minimum energy consumption."""
        # Simplified energy model based on joint velocities and accelerations
        def energy_objective(traj_flat):
            trajectory = traj_flat.reshape(initial_trajectory.shape)
            velocities = self._compute_velocities(trajectory, timestamps)
            accelerations = self._compute_accelerations(velocities, timestamps)
            
            # Energy = sum of squared velocities and accelerations (simplified)
            velocity_energy = np.sum(velocities**2)
            acceleration_energy = np.sum(accelerations**2)
            
            return velocity_energy + 0.1 * acceleration_energy
        
        # Use scipy optimization
        return self._scipy_optimization_wrapper(
            initial_trajectory, timestamps, constraints, energy_objective, "minimum_energy"
        )
    
    def _optimize_smoothness(self, initial_trajectory: np.ndarray,
                           timestamps: np.ndarray,
                           constraints: OptimizationConstraints) -> OptimizationResult:
        """Optimize for maximum smoothness."""
        def smoothness_objective(traj_flat):
            trajectory = traj_flat.reshape(initial_trajectory.shape)
            
            # Smoothness = minimize curvature and jerk
            curvature_cost = 0
            jerk_cost = 0
            
            dt = np.mean(np.diff(timestamps))
            
            # Second derivative (curvature measure)
            for i in range(2, len(trajectory)):
                curvature = (trajectory[i] - 2*trajectory[i-1] + trajectory[i-2]) / (dt**2)
                curvature_cost += np.sum(curvature**2)
            
            # Third derivative (jerk)
            for i in range(3, len(trajectory)):
                jerk = (trajectory[i] - 3*trajectory[i-1] + 3*trajectory[i-2] - trajectory[i-3]) / (dt**3)
                jerk_cost += np.sum(jerk**2)
            
            return curvature_cost + jerk_cost
        
        return self._scipy_optimization_wrapper(
            initial_trajectory, timestamps, constraints, smoothness_objective, "smoothness"
        )
    
    def _optimize_combined_objective(self, initial_trajectory: np.ndarray,
                                   timestamps: np.ndarray,
                                   constraints: OptimizationConstraints) -> OptimizationResult:
        """Optimize combined objective with multiple criteria."""
        def combined_objective(traj_flat):
            trajectory = traj_flat.reshape(initial_trajectory.shape)
            velocities = self._compute_velocities(trajectory, timestamps)
            accelerations = self._compute_accelerations(velocities, timestamps)
            
            dt = np.mean(np.diff(timestamps))
            
            # Time component
            time_cost = self.time_weight * timestamps[-1]
            
            # Jerk component
            jerk_cost = 0
            for i in range(3, len(trajectory)):
                jerk = (trajectory[i] - 3*trajectory[i-1] + 3*trajectory[i-2] - trajectory[i-3]) / (dt**3)
                jerk_cost += np.sum(jerk**2)
            jerk_cost *= self.jerk_weight
            
            # Energy component
            energy_cost = self.energy_weight * (np.sum(velocities**2) + 0.1 * np.sum(accelerations**2))
            
            # Smoothness component
            smoothness_cost = 0
            for i in range(2, len(trajectory)):
                curvature = (trajectory[i] - 2*trajectory[i-1] + trajectory[i-2]) / (dt**2)
                smoothness_cost += np.sum(curvature**2)
            smoothness_cost *= self.smoothness_weight
            
            return time_cost + jerk_cost + energy_cost + smoothness_cost
        
        return self._scipy_optimization_wrapper(
            initial_trajectory, timestamps, constraints, combined_objective, "combined"
        )
    
    def _scipy_optimization_wrapper(self, initial_trajectory: np.ndarray,
                                  timestamps: np.ndarray,
                                  constraints: OptimizationConstraints,
                                  objective_function: Callable,
                                  method_name: str) -> OptimizationResult:
        """Wrapper for scipy-based optimization."""
        n_points, n_joints = initial_trajectory.shape
        
        # Set up bounds
        bounds = []
        for i in range(n_points):
            for j in range(n_joints):
                if i == 0 or i == n_points - 1:
                    # Fix start and end points
                    bounds.append((initial_trajectory[i, j], initial_trajectory[i, j]))
                else:
                    if constraints.joint_limits:
                        bounds.append(constraints.joint_limits[j])
                    else:
                        bounds.append((-np.pi, np.pi))
        
        # Initial guess
        x0 = initial_trajectory.flatten()
        
        # Optimize
        try:
            result = minimize(objective_function, x0, method='L-BFGS-B',
                            bounds=bounds, options={'maxiter': self.max_iterations})
            
            if result.success:
                optimized_traj = result.x.reshape(n_points, n_joints)
                success = True
                objective_value = result.fun
                iterations = result.nit
            else:
                optimized_traj = initial_trajectory
                success = False
                objective_value = float('inf')
                iterations = 0
        
        except Exception as e:
            logger.warning(f"{method_name} optimization error: {e}")
            optimized_traj = initial_trajectory
            success = False
            objective_value = float('inf')
            iterations = 0
        
        # Compute derivatives
        velocities = self._compute_velocities(optimized_traj, timestamps)
        accelerations = self._compute_accelerations(velocities, timestamps)
        
        # Check constraints
        constraints_satisfied = self._check_constraints_satisfaction(
            optimized_traj, velocities, accelerations, timestamps, constraints
        )
        
        return OptimizationResult(
            optimized_trajectory=optimized_traj,
            timestamps=timestamps,
            velocities=velocities,
            accelerations=accelerations,
            objective_value=objective_value,
            success=success,
            iterations=iterations,
            computation_time=0.0,
            constraints_satisfied=constraints_satisfied
        )
    
    def _compute_velocities(self, trajectory: np.ndarray, timestamps: np.ndarray) -> np.ndarray:
        """Compute velocities from trajectory."""
        dt = np.diff(timestamps)
        velocities = np.zeros_like(trajectory)
        
        # Forward differences
        velocities[:-1] = np.diff(trajectory, axis=0) / dt.reshape(-1, 1)
        velocities[-1] = velocities[-2]  # Copy last velocity
        
        return velocities
    
    def _compute_accelerations(self, velocities: np.ndarray, timestamps: np.ndarray) -> np.ndarray:
        """Compute accelerations from velocities."""
        dt = np.diff(timestamps)
        accelerations = np.zeros_like(velocities)
        
        # Forward differences
        accelerations[:-1] = np.diff(velocities, axis=0) / dt.reshape(-1, 1)
        accelerations[-1] = accelerations[-2]  # Copy last acceleration
        
        return accelerations
    
    def _check_constraints_satisfaction(self, trajectory: np.ndarray,
                                      velocities: np.ndarray,
                                      accelerations: np.ndarray,
                                      timestamps: np.ndarray,
                                      constraints: OptimizationConstraints) -> bool:
        """Check if all constraints are satisfied."""
        # Joint limits
        if constraints.joint_limits:
            for j, (min_val, max_val) in enumerate(constraints.joint_limits):
                if np.any(trajectory[:, j] < min_val) or np.any(trajectory[:, j] > max_val):
                    return False
        
        # Velocity limits
        if constraints.velocity_limits:
            for j, max_vel in enumerate(constraints.velocity_limits):
                if np.any(np.abs(velocities[:, j]) > max_vel):
                    return False
        
        # Acceleration limits
        if constraints.acceleration_limits:
            for j, max_acc in enumerate(constraints.acceleration_limits):
                if np.any(np.abs(accelerations[:, j]) > max_acc):
                    return False
        
        # Jerk limits
        if constraints.jerk_limits:
            dt = np.mean(np.diff(timestamps))
            jerks = np.diff(accelerations, axis=0) / dt
            for j, max_jerk in enumerate(constraints.jerk_limits):
                if np.any(np.abs(jerks[:, j]) > max_jerk):
                    return False
        
        return True
    
    def interpolate_trajectory(self, waypoints: np.ndarray,
                             timestamps: np.ndarray,
                             target_timestamps: np.ndarray,
                             method: str = 'cubic') -> np.ndarray:
        """
        Interpolate trajectory to new timestamps.
        
        Args:
            waypoints: Waypoint configurations [n_waypoints, n_joints]
            timestamps: Original timestamps [n_waypoints]
            target_timestamps: Target timestamps [n_target]
            method: Interpolation method ('linear', 'cubic', 'spline')
            
        Returns:
            interpolated_trajectory: Interpolated trajectory [n_target, n_joints]
        """
        n_joints = waypoints.shape[1]
        interpolated = np.zeros((len(target_timestamps), n_joints))
        
        for j in range(n_joints):
            if method == 'linear':
                f = interp1d(timestamps, waypoints[:, j], kind='linear',
                           bounds_error=False, fill_value='extrapolate')
            elif method == 'cubic':
                f = interp1d(timestamps, waypoints[:, j], kind='cubic',
                           bounds_error=False, fill_value='extrapolate')
            elif method == 'spline':
                f = CubicSpline(timestamps, waypoints[:, j])
            else:
                raise ValueError(f"Unknown interpolation method: {method}")
            
            interpolated[:, j] = f(target_timestamps)
        
        return interpolated
    
    def resample_trajectory(self, trajectory: np.ndarray,
                          original_timestamps: np.ndarray,
                          target_sample_rate: float) -> Tuple[np.ndarray, np.ndarray]:
        """
        Resample trajectory to target sample rate.
        
        Args:
            trajectory: Original trajectory [n_points, n_joints]
            original_timestamps: Original timestamps [n_points]
            target_sample_rate: Target sample rate in Hz
            
        Returns:
            resampled_trajectory: Resampled trajectory
            new_timestamps: New timestamps
        """
        # Create new timestamp vector
        duration = original_timestamps[-1] - original_timestamps[0]
        dt = 1.0 / target_sample_rate
        new_timestamps = np.arange(original_timestamps[0], original_timestamps[-1] + dt/2, dt)
        
        # Interpolate trajectory
        resampled_trajectory = self.interpolate_trajectory(
            trajectory, original_timestamps, new_timestamps, method='cubic'
        )
        
        return resampled_trajectory, new_timestamps
    
    def validate_optimization_result(self, result: OptimizationResult,
                                   constraints: OptimizationConstraints) -> Dict[str, Any]:
        """
        Validate optimization result against constraints.
        
        Args:
            result: Optimization result to validate
            constraints: Constraints to check against
            
        Returns:
            validation_report: Detailed validation report
        """
        violations = []
        warnings = []
        
        trajectory = result.optimized_trajectory
        velocities = result.velocities
        accelerations = result.accelerations
        
        # Check joint limits
        if constraints.joint_limits:
            for j, (min_val, max_val) in enumerate(constraints.joint_limits):
                min_violation = np.min(trajectory[:, j] - min_val)
                max_violation = np.max(trajectory[:, j] - max_val)
                
                if min_violation < 0:
                    violations.append(f"Joint {j} below limit by {-min_violation:.4f}")
                if max_violation > 0:
                    violations.append(f"Joint {j} above limit by {max_violation:.4f}")
        
        # Check velocity limits
        if constraints.velocity_limits:
            for j, max_vel in enumerate(constraints.velocity_limits):
                max_actual_vel = np.max(np.abs(velocities[:, j]))
                if max_actual_vel > max_vel:
                    violations.append(f"Joint {j} velocity {max_actual_vel:.4f} exceeds limit {max_vel}")
        
        # Check acceleration limits
        if constraints.acceleration_limits:
            for j, max_acc in enumerate(constraints.acceleration_limits):
                max_actual_acc = np.max(np.abs(accelerations[:, j]))
                if max_actual_acc > max_acc:
                    violations.append(f"Joint {j} acceleration {max_actual_acc:.4f} exceeds limit {max_acc}")
        
        # Check smoothness
        if constraints.smoothness_constraints:
            dt = np.mean(np.diff(result.timestamps))
            jerks = np.diff(accelerations, axis=0) / dt
            max_jerk = np.max(np.abs(jerks))
            
            if max_jerk > 100.0:  # Arbitrary threshold
                warnings.append(f"High jerk detected: {max_jerk:.2f}")
        
        return {
            'valid': len(violations) == 0,
            'violations': violations,
            'warnings': warnings,
            'max_velocity': np.max(np.abs(velocities)),
            'max_acceleration': np.max(np.abs(accelerations)),
            'trajectory_duration': result.timestamps[-1] - result.timestamps[0],
            'objective_value': result.objective_value
        }