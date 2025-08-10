"""
Inverse Kinematics Implementation
=================================

Inverse kinematics solvers for the 7-DOF robot arm including
analytical, numerical, and optimization-based approaches.
"""

import numpy as np
from typing import Dict, Any, List, Tuple, Optional, Union, Callable
from dataclasses import dataclass
from enum import Enum
import logging
from scipy.optimize import minimize, least_squares
from scipy.spatial.transform import Rotation

from .forward_kinematics import ForwardKinematics, TransformationMatrix

logger = logging.getLogger(__name__)


class IKMethod(Enum):
    """Inverse kinematics solution methods."""
    JACOBIAN_PSEUDO_INVERSE = "jacobian_pseudo_inverse"
    DAMPED_LEAST_SQUARES = "damped_least_squares"
    LEVENBERG_MARQUARDT = "levenberg_marquardt"
    OPTIMIZATION = "optimization"
    ANALYTICAL = "analytical"


@dataclass
class IKSolution:
    """
    Result of inverse kinematics computation.
    
    Attributes:
        joint_angles: Solution joint angles [7]
        success: Whether a valid solution was found
        error: Final positioning error
        iterations: Number of iterations used
        method: IK method used
        computation_time: Time taken for computation
        multiple_solutions: List of alternative solutions
    """
    joint_angles: np.ndarray
    success: bool
    error: float
    iterations: int
    method: str
    computation_time: float = 0.0
    multiple_solutions: Optional[List[np.ndarray]] = None


class IKSolver:
    """
    Base class for inverse kinematics solvers.
    """
    
    def __init__(self, forward_kinematics: ForwardKinematics, config: Dict[str, Any]):
        """
        Initialize IK solver.
        
        Args:
            forward_kinematics: Forward kinematics solver
            config: Solver configuration
        """
        self.fk = forward_kinematics
        self.config = config
        
        # Solver parameters
        self.max_iterations = config.get('max_iterations', 100)
        self.tolerance = config.get('tolerance', 1e-6)
        self.step_size = config.get('step_size', 0.1)
        
        # Joint limits
        self.joint_limits = config.get('joint_limits', [(-np.pi, np.pi)] * 7)
        
        # Weights for position vs orientation
        self.position_weight = config.get('position_weight', 1.0)
        self.orientation_weight = config.get('orientation_weight', 0.1)
        
        logger.info(f"Initialized IKSolver with {self.fk.num_joints} DOF")
    
    def solve(self, target_pose: Union[np.ndarray, TransformationMatrix],
             initial_guess: Optional[np.ndarray] = None,
             method: IKMethod = IKMethod.DAMPED_LEAST_SQUARES) -> IKSolution:
        """
        Solve inverse kinematics for target pose.
        
        Args:
            target_pose: Target pose (4x4 matrix or TransformationMatrix)
            initial_guess: Initial joint configuration
            method: IK solution method
            
        Returns:
            solution: IK solution result
        """
        import time
        start_time = time.time()
        
        # Convert target pose to standard format
        if isinstance(target_pose, TransformationMatrix):
            target_matrix = target_pose.matrix
            target_position = target_pose.position
            target_orientation = target_pose.euler_angles
        else:
            target_matrix = target_pose
            target_position = target_matrix[:3, 3]
            target_orientation = TransformationMatrix(target_matrix).euler_angles
        
        # Initial guess
        if initial_guess is None:
            initial_guess = np.zeros(self.fk.num_joints)
        
        # Solve based on method
        if method == IKMethod.JACOBIAN_PSEUDO_INVERSE:
            result = self._solve_jacobian_pseudo_inverse(target_position, target_orientation, initial_guess)
        elif method == IKMethod.DAMPED_LEAST_SQUARES:
            result = self._solve_damped_least_squares(target_position, target_orientation, initial_guess)
        elif method == IKMethod.LEVENBERG_MARQUARDT:
            result = self._solve_levenberg_marquardt(target_position, target_orientation, initial_guess)
        elif method == IKMethod.OPTIMIZATION:
            result = self._solve_optimization(target_position, target_orientation, initial_guess)
        else:
            raise ValueError(f"Unsupported IK method: {method}")
        
        # Set computation time
        result.computation_time = time.time() - start_time
        result.method = method.value
        
        return result
    
    def _solve_jacobian_pseudo_inverse(self, target_pos: np.ndarray, 
                                     target_orient: np.ndarray,
                                     initial_guess: np.ndarray) -> IKSolution:
        """Solve IK using Jacobian pseudo-inverse method."""
        current_angles = initial_guess.copy()
        
        for iteration in range(self.max_iterations):
            # Current pose
            current_transform = self.fk.forward_kinematics(current_angles)
            current_pos = current_transform.position
            current_orient = current_transform.euler_angles
            
            # Pose error
            pos_error = target_pos - current_pos
            orient_error = target_orient - current_orient
            
            # Combine errors with weights
            pose_error = np.concatenate([
                self.position_weight * pos_error,
                self.orientation_weight * orient_error
            ])
            
            # Check convergence
            error_magnitude = np.linalg.norm(pose_error)
            if error_magnitude < self.tolerance:
                return IKSolution(
                    joint_angles=current_angles,
                    success=True,
                    error=error_magnitude,
                    iterations=iteration + 1,
                    method="jacobian_pseudo_inverse"
                )
            
            # Compute Jacobian
            jacobian = self.fk.compute_jacobian(current_angles)
            
            # Pseudo-inverse
            try:
                jacobian_pinv = np.linalg.pinv(jacobian)
                delta_angles = jacobian_pinv @ pose_error
            except np.linalg.LinAlgError:
                # Jacobian is singular
                break
            
            # Update joint angles with step size control
            current_angles += self.step_size * delta_angles
            
            # Apply joint limits
            current_angles = self._apply_joint_limits(current_angles)
        
        # Failed to converge
        final_transform = self.fk.forward_kinematics(current_angles)
        final_error = np.linalg.norm(target_pos - final_transform.position)
        
        return IKSolution(
            joint_angles=current_angles,
            success=False,
            error=final_error,
            iterations=self.max_iterations,
            method="jacobian_pseudo_inverse"
        )
    
    def _solve_damped_least_squares(self, target_pos: np.ndarray,
                                  target_orient: np.ndarray,
                                  initial_guess: np.ndarray) -> IKSolution:
        """Solve IK using damped least squares (Levenberg-Marquardt variant)."""
        current_angles = initial_guess.copy()
        damping_factor = self.config.get('damping_factor', 0.01)
        
        for iteration in range(self.max_iterations):
            # Current pose
            current_transform = self.fk.forward_kinematics(current_angles)
            current_pos = current_transform.position
            current_orient = current_transform.euler_angles
            
            # Pose error
            pos_error = target_pos - current_pos
            orient_error = self._normalize_angle_error(target_orient - current_orient)
            
            # Combine errors
            pose_error = np.concatenate([
                self.position_weight * pos_error,
                self.orientation_weight * orient_error
            ])
            
            # Check convergence
            error_magnitude = np.linalg.norm(pose_error)
            if error_magnitude < self.tolerance:
                return IKSolution(
                    joint_angles=current_angles,
                    success=True,
                    error=error_magnitude,
                    iterations=iteration + 1,
                    method="damped_least_squares"
                )
            
            # Compute Jacobian
            jacobian = self.fk.compute_jacobian(current_angles)
            
            # Damped least squares solution
            n_joints = jacobian.shape[1]
            damped_matrix = jacobian.T @ jacobian + damping_factor * np.eye(n_joints)
            
            try:
                delta_angles = np.linalg.solve(damped_matrix, jacobian.T @ pose_error)
            except np.linalg.LinAlgError:
                # Matrix is singular
                break
            
            # Update joint angles
            current_angles += self.step_size * delta_angles
            
            # Apply joint limits
            current_angles = self._apply_joint_limits(current_angles)
            
            # Adaptive damping
            new_transform = self.fk.forward_kinematics(current_angles)
            new_error = np.linalg.norm(target_pos - new_transform.position)
            
            if new_error > error_magnitude:
                # Increase damping if error increased
                damping_factor *= 2.0
            else:
                # Decrease damping if error decreased
                damping_factor *= 0.9
        
        # Failed to converge
        final_transform = self.fk.forward_kinematics(current_angles)
        final_error = np.linalg.norm(target_pos - final_transform.position)
        
        return IKSolution(
            joint_angles=current_angles,
            success=False,
            error=final_error,
            iterations=self.max_iterations,
            method="damped_least_squares"
        )
    
    def _solve_levenberg_marquardt(self, target_pos: np.ndarray,
                                 target_orient: np.ndarray,
                                 initial_guess: np.ndarray) -> IKSolution:
        """Solve IK using Levenberg-Marquardt optimization."""
        def residual_function(joint_angles):
            """Compute residual for optimization."""
            current_transform = self.fk.forward_kinematics(joint_angles)
            current_pos = current_transform.position
            current_orient = current_transform.euler_angles
            
            pos_error = target_pos - current_pos
            orient_error = self._normalize_angle_error(target_orient - current_orient)
            
            return np.concatenate([
                self.position_weight * pos_error,
                self.orientation_weight * orient_error
            ])
        
        def jacobian_function(joint_angles):
            """Compute Jacobian for optimization."""
            return self.fk.compute_jacobian(joint_angles)
        
        # Set up bounds
        bounds = (
            [limit[0] for limit in self.joint_limits],
            [limit[1] for limit in self.joint_limits]
        )
        
        try:
            # Solve using least_squares
            result = least_squares(
                residual_function,
                initial_guess,
                jac=jacobian_function,
                bounds=bounds,
                method='lm',
                max_nfev=self.max_iterations * 10,
                ftol=self.tolerance,
                xtol=self.tolerance
            )
            
            success = result.success
            final_angles = result.x
            final_error = np.linalg.norm(result.fun)
            iterations = result.nfev
            
        except Exception as e:
            logger.warning(f"Levenberg-Marquardt optimization failed: {e}")
            success = False
            final_angles = initial_guess
            final_error = float('inf')
            iterations = 0
        
        return IKSolution(
            joint_angles=final_angles,
            success=success,
            error=final_error,
            iterations=iterations,
            method="levenberg_marquardt"
        )
    
    def _solve_optimization(self, target_pos: np.ndarray,
                          target_orient: np.ndarray,
                          initial_guess: np.ndarray) -> IKSolution:
        """Solve IK using general optimization."""
        def objective_function(joint_angles):
            """Objective function for optimization."""
            current_transform = self.fk.forward_kinematics(joint_angles)
            current_pos = current_transform.position
            current_orient = current_transform.euler_angles
            
            pos_error = np.linalg.norm(target_pos - current_pos)
            orient_error = np.linalg.norm(self._normalize_angle_error(target_orient - current_orient))
            
            return self.position_weight * pos_error + self.orientation_weight * orient_error
        
        # Set up bounds
        bounds = [(limit[0], limit[1]) for limit in self.joint_limits]
        
        try:
            # Solve using minimize
            result = minimize(
                objective_function,
                initial_guess,
                method='L-BFGS-B',
                bounds=bounds,
                options={
                    'maxiter': self.max_iterations,
                    'ftol': self.tolerance,
                    'gtol': self.tolerance
                }
            )
            
            success = result.success and result.fun < self.tolerance
            final_angles = result.x
            final_error = result.fun
            iterations = result.nit
            
        except Exception as e:
            logger.warning(f"Optimization failed: {e}")
            success = False
            final_angles = initial_guess
            final_error = float('inf')
            iterations = 0
        
        return IKSolution(
            joint_angles=final_angles,
            success=success,
            error=final_error,
            iterations=iterations,
            method="optimization"
        )
    
    def _apply_joint_limits(self, joint_angles: np.ndarray) -> np.ndarray:
        """Apply joint limits to joint angles."""
        limited_angles = joint_angles.copy()
        
        for i, (angle, limits) in enumerate(zip(joint_angles, self.joint_limits)):
            limited_angles[i] = np.clip(angle, limits[0], limits[1])
        
        return limited_angles
    
    def _normalize_angle_error(self, angle_error: np.ndarray) -> np.ndarray:
        """Normalize angle errors to [-pi, pi]."""
        normalized = angle_error.copy()
        
        for i in range(len(normalized)):
            while normalized[i] > np.pi:
                normalized[i] -= 2 * np.pi
            while normalized[i] < -np.pi:
                normalized[i] += 2 * np.pi
        
        return normalized
    
    def solve_multiple_solutions(self, target_pose: Union[np.ndarray, TransformationMatrix],
                               num_attempts: int = 10,
                               method: IKMethod = IKMethod.DAMPED_LEAST_SQUARES) -> List[IKSolution]:
        """
        Find multiple IK solutions using different initial guesses.
        
        Args:
            target_pose: Target pose
            num_attempts: Number of random initial guesses
            method: IK solution method
            
        Returns:
            solutions: List of IK solutions
        """
        solutions = []
        
        for attempt in range(num_attempts):
            # Generate random initial guess within joint limits
            initial_guess = np.array([
                np.random.uniform(limits[0], limits[1])
                for limits in self.joint_limits
            ])
            
            # Solve IK
            solution = self.solve(target_pose, initial_guess, method)
            
            if solution.success:
                # Check if this is a new solution
                is_new_solution = True
                for existing_solution in solutions:
                    if np.allclose(solution.joint_angles, existing_solution.joint_angles, atol=0.01):
                        is_new_solution = False
                        break
                
                if is_new_solution:
                    solutions.append(solution)
        
        return solutions


class NumericalIKSolver(IKSolver):
    """
    Numerical inverse kinematics solver with advanced features.
    """
    
    def __init__(self, forward_kinematics: ForwardKinematics, config: Dict[str, Any]):
        """Initialize numerical IK solver."""
        super().__init__(forward_kinematics, config)
        
        # Additional numerical solver parameters
        self.null_space_optimization = config.get('null_space_optimization', True)
        self.singularity_avoidance = config.get('singularity_avoidance', True)
        self.singularity_threshold = config.get('singularity_threshold', 0.01)
        
        # Secondary objectives for null space
        self.joint_center_weights = config.get('joint_center_weights', np.ones(7))
        self.joint_velocity_weights = config.get('joint_velocity_weights', np.ones(7))
        
        logger.info("Initialized NumericalIKSolver with advanced features")
    
    def solve_with_constraints(self, target_pose: Union[np.ndarray, TransformationMatrix],
                              initial_guess: Optional[np.ndarray] = None,
                              position_only: bool = False,
                              avoid_singularities: bool = True) -> IKSolution:
        """
        Solve IK with additional constraints and features.
        
        Args:
            target_pose: Target pose
            initial_guess: Initial joint configuration
            position_only: Only constrain position, not orientation
            avoid_singularities: Avoid singular configurations
            
        Returns:
            solution: IK solution
        """
        # Adjust weights for position-only IK
        if position_only:
            original_orientation_weight = self.orientation_weight
            self.orientation_weight = 0.0
        
        # Solve with damped least squares and null space optimization
        solution = self._solve_with_null_space_optimization(target_pose, initial_guess, avoid_singularities)
        
        # Restore original weights
        if position_only:
            self.orientation_weight = original_orientation_weight
        
        return solution
    
    def _solve_with_null_space_optimization(self, target_pose: Union[np.ndarray, TransformationMatrix],
                                          initial_guess: Optional[np.ndarray],
                                          avoid_singularities: bool) -> IKSolution:
        """Solve IK with null space optimization for secondary objectives."""
        # Convert target pose
        if isinstance(target_pose, TransformationMatrix):
            target_pos = target_pose.position
            target_orient = target_pose.euler_angles
        else:
            target_matrix = target_pose
            target_pos = target_matrix[:3, 3]
            target_orient = TransformationMatrix(target_matrix).euler_angles
        
        # Initial guess
        if initial_guess is None:
            initial_guess = np.zeros(self.fk.num_joints)
        
        current_angles = initial_guess.copy()
        damping_factor = self.config.get('damping_factor', 0.01)
        
        for iteration in range(self.max_iterations):
            # Current pose
            current_transform = self.fk.forward_kinematics(current_angles)
            current_pos = current_transform.position
            current_orient = current_transform.euler_angles
            
            # Primary task error
            pos_error = target_pos - current_pos
            orient_error = self._normalize_angle_error(target_orient - current_orient)
            
            pose_error = np.concatenate([
                self.position_weight * pos_error,
                self.orientation_weight * orient_error
            ])
            
            # Check convergence
            error_magnitude = np.linalg.norm(pose_error)
            if error_magnitude < self.tolerance:
                return IKSolution(
                    joint_angles=current_angles,
                    success=True,
                    error=error_magnitude,
                    iterations=iteration + 1,
                    method="null_space_optimization"
                )
            
            # Compute Jacobian
            jacobian = self.fk.compute_jacobian(current_angles)
            
            # Check for singularities
            if avoid_singularities and self.singularity_avoidance:
                manipulability = np.sqrt(np.linalg.det(jacobian @ jacobian.T))
                if manipulability < self.singularity_threshold:
                    # Increase damping near singularities
                    damping_factor = max(damping_factor, 0.1)
            
            # Primary task solution (damped least squares)
            n_joints = jacobian.shape[1]
            damped_matrix = jacobian.T @ jacobian + damping_factor * np.eye(n_joints)
            
            try:
                primary_solution = np.linalg.solve(damped_matrix, jacobian.T @ pose_error)
            except np.linalg.LinAlgError:
                break
            
            # Null space optimization for secondary objectives
            if self.null_space_optimization:
                # Compute null space projector
                jacobian_pinv = np.linalg.pinv(jacobian)
                null_space_projector = np.eye(n_joints) - jacobian_pinv @ jacobian
                
                # Secondary objective: joint centering
                joint_centers = np.array([(limits[0] + limits[1]) / 2 for limits in self.joint_limits])
                centering_gradient = self.joint_center_weights * (joint_centers - current_angles)
                
                # Project secondary objective into null space
                null_space_solution = null_space_projector @ centering_gradient
                
                # Combine primary and secondary solutions
                delta_angles = primary_solution + 0.1 * null_space_solution
            else:
                delta_angles = primary_solution
            
            # Update joint angles
            current_angles += self.step_size * delta_angles
            
            # Apply joint limits
            current_angles = self._apply_joint_limits(current_angles)
        
        # Failed to converge
        final_transform = self.fk.forward_kinematics(current_angles)
        final_error = np.linalg.norm(target_pos - final_transform.position)
        
        return IKSolution(
            joint_angles=current_angles,
            success=False,
            error=final_error,
            iterations=self.max_iterations,
            method="null_space_optimization"
        )
    
    def solve_trajectory(self, target_trajectory: List[Union[np.ndarray, TransformationMatrix]],
                        initial_guess: Optional[np.ndarray] = None,
                        method: IKMethod = IKMethod.DAMPED_LEAST_SQUARES) -> List[IKSolution]:
        """
        Solve IK for a trajectory of poses.
        
        Args:
            target_trajectory: List of target poses
            initial_guess: Initial joint configuration for first pose
            method: IK solution method
            
        Returns:
            solutions: List of IK solutions for each pose
        """
        solutions = []
        current_guess = initial_guess
        
        for i, target_pose in enumerate(target_trajectory):
            # Solve IK for current pose
            solution = self.solve(target_pose, current_guess, method)
            solutions.append(solution)
            
            # Use current solution as initial guess for next pose
            if solution.success:
                current_guess = solution.joint_angles
            else:
                logger.warning(f"IK failed for trajectory point {i}")
                # Keep previous guess if solution failed
        
        return solutions
    
    def compute_ik_metrics(self, joint_angles: np.ndarray) -> Dict[str, float]:
        """
        Compute IK-related metrics for given configuration.
        
        Args:
            joint_angles: Joint configuration
            
        Returns:
            metrics: Dictionary of IK metrics
        """
        # Compute Jacobian
        jacobian = self.fk.compute_jacobian(joint_angles)
        
        # Manipulability
        manipulability = np.sqrt(np.linalg.det(jacobian @ jacobian.T))
        
        # Condition number
        try:
            condition_number = np.linalg.cond(jacobian)
        except np.linalg.LinAlgError:
            condition_number = float('inf')
        
        # Distance from joint limits
        joint_limit_distances = []
        for angle, limits in zip(joint_angles, self.joint_limits):
            range_size = limits[1] - limits[0]
            center = (limits[0] + limits[1]) / 2
            distance = abs(angle - center) / (range_size / 2)
            joint_limit_distances.append(distance)
        
        avg_joint_distance = np.mean(joint_limit_distances)
        max_joint_distance = np.max(joint_limit_distances)
        
        return {
            'manipulability': manipulability,
            'condition_number': condition_number,
            'avg_joint_limit_distance': avg_joint_distance,
            'max_joint_limit_distance': max_joint_distance,
            'is_singular': manipulability < self.singularity_threshold
        }


class InverseKinematics:
    """
    Main inverse kinematics interface combining multiple solvers.
    """
    
    def __init__(self, forward_kinematics: ForwardKinematics, config: Dict[str, Any]):
        """
        Initialize inverse kinematics system.
        
        Args:
            forward_kinematics: Forward kinematics solver
            config: Configuration dictionary
        """
        self.fk = forward_kinematics
        self.config = config
        
        # Initialize solvers
        self.basic_solver = IKSolver(forward_kinematics, config.get('basic_solver', {}))
        self.numerical_solver = NumericalIKSolver(forward_kinematics, config.get('numerical_solver', {}))
        
        # Default solver
        self.default_solver = config.get('default_solver', 'numerical')
        
        logger.info("Initialized InverseKinematics with multiple solvers")
    
    def solve(self, target_pose: Union[np.ndarray, TransformationMatrix],
             initial_guess: Optional[np.ndarray] = None,
             solver: str = None,
             **kwargs) -> IKSolution:
        """
        Solve inverse kinematics using specified solver.
        
        Args:
            target_pose: Target pose
            initial_guess: Initial joint configuration
            solver: Solver to use ('basic', 'numerical')
            **kwargs: Additional solver-specific arguments
            
        Returns:
            solution: IK solution
        """
        if solver is None:
            solver = self.default_solver
        
        if solver == 'basic':
            return self.basic_solver.solve(target_pose, initial_guess, **kwargs)
        elif solver == 'numerical':
            return self.numerical_solver.solve(target_pose, initial_guess, **kwargs)
        else:
            raise ValueError(f"Unknown solver: {solver}")
    
    def solve_cartesian_trajectory(self, cartesian_trajectory: np.ndarray,
                                 timestamps: Optional[np.ndarray] = None,
                                 initial_guess: Optional[np.ndarray] = None) -> Dict[str, Any]:
        """
        Convert Cartesian trajectory to joint trajectory.
        
        Args:
            cartesian_trajectory: Cartesian positions [n_points, 3]
            timestamps: Time stamps [n_points]
            initial_guess: Initial joint configuration
            
        Returns:
            joint_trajectory: Joint trajectory and metadata
        """
        n_points = len(cartesian_trajectory)
        joint_trajectories = np.zeros((n_points, self.fk.num_joints))
        successes = np.zeros(n_points, dtype=bool)
        errors = np.zeros(n_points)
        
        current_guess = initial_guess if initial_guess is not None else np.zeros(self.fk.num_joints)
        
        for i, target_pos in enumerate(cartesian_trajectory):
            # Create target transformation matrix (position only)
            target_matrix = np.eye(4)
            target_matrix[:3, 3] = target_pos
            
            # Solve IK
            solution = self.numerical_solver.solve_with_constraints(
                target_matrix, current_guess, position_only=True
            )
            
            joint_trajectories[i] = solution.joint_angles
            successes[i] = solution.success
            errors[i] = solution.error
            
            # Update guess for next iteration
            if solution.success:
                current_guess = solution.joint_angles
        
        return {
            'joint_angles': joint_trajectories,
            'timestamps': timestamps,
            'successes': successes,
            'errors': errors,
            'success_rate': np.mean(successes),
            'mean_error': np.mean(errors[successes]),
            'max_error': np.max(errors[successes]) if np.any(successes) else float('inf')
        }