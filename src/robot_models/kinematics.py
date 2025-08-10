"""
Kinematic Chain Implementation
=============================

Forward and inverse kinematics solver for the 7-DOF robotic arm
using Denavit-Hartenberg parameters and numerical optimization.
"""

import numpy as np
import scipy.optimize as opt
from typing import List, Tuple, Optional, Callable
import logging

logger = logging.getLogger(__name__)


class KinematicChain:
    """
    Kinematic chain solver for robotic manipulator.
    
    Features:
    - Forward kinematics using DH parameters
    - Inverse kinematics using numerical optimization
    - Jacobian computation for velocity control
    - Multiple solution handling for redundant arms
    """
    
    def __init__(self, dh_params: np.ndarray):
        """
        Initialize kinematic chain.
        
        Args:
            dh_params: DH parameters matrix [a, alpha, d, theta] for each joint
        """
        self.dh_params = dh_params
        self.num_joints = len(dh_params)
        self.tolerance = 1e-6
        self.max_iterations = 1000
        
        logger.info(f"Initialized kinematic chain with {self.num_joints} joints")
    
    def dh_transform(self, a: float, alpha: float, d: float, theta: float) -> np.ndarray:
        """
        Compute Denavit-Hartenberg transformation matrix.
        
        Args:
            a: Link length
            alpha: Link twist
            d: Joint offset
            theta: Joint angle
            
        Returns:
            4x4 transformation matrix
        """
        cos_theta = np.cos(theta)
        sin_theta = np.sin(theta)
        cos_alpha = np.cos(alpha)
        sin_alpha = np.sin(alpha)
        
        transform = np.array([
            [cos_theta, -sin_theta * cos_alpha,  sin_theta * sin_alpha, a * cos_theta],
            [sin_theta,  cos_theta * cos_alpha, -cos_theta * sin_alpha, a * sin_theta],
            [0,          sin_alpha,              cos_alpha,             d],
            [0,          0,                      0,                     1]
        ])
        
        return transform
    
    def forward_kinematics(self, joint_angles: np.ndarray) -> np.ndarray:
        """
        Compute forward kinematics.
        
        Args:
            joint_angles: Array of joint angles (rad)
            
        Returns:
            end_effector_pose: [x, y, z, rx, ry, rz] (position + Euler angles)
        """
        if len(joint_angles) != self.num_joints:
            raise ValueError(f"Expected {self.num_joints} joint angles, got {len(joint_angles)}")
        
        # Start with identity transformation
        T = np.eye(4)
        
        # Apply each joint transformation
        for i, angle in enumerate(joint_angles):
            a, alpha, d, theta_offset = self.dh_params[i]
            theta = angle + theta_offset
            
            # Compute transformation for this joint
            T_i = self.dh_transform(a, alpha, d, theta)
            
            # Accumulate transformation
            T = T @ T_i
        
        # Extract position
        position = T[:3, 3]
        
        # Extract orientation (convert rotation matrix to Euler angles)
        orientation = self._rotation_matrix_to_euler(T[:3, :3])
        
        return np.concatenate([position, orientation])
    
    def inverse_kinematics(self, target_pose: np.ndarray,
                          initial_guess: np.ndarray,
                          joint_limits: List[Tuple[float, float]]) -> Optional[np.ndarray]:
        """
        Solve inverse kinematics using numerical optimization.
        
        Args:
            target_pose: Desired end-effector pose [x, y, z, rx, ry, rz]
            initial_guess: Initial joint configuration
            joint_limits: Joint limits [(min, max), ...]
            
        Returns:
            joint_angles: Solution joint angles or None if no solution found
        """
        
        def objective_function(joint_angles: np.ndarray) -> float:
            """Objective function for optimization"""
            current_pose = self.forward_kinematics(joint_angles)
            
            # Position error (weighted more heavily)
            position_error = np.linalg.norm(current_pose[:3] - target_pose[:3])
            
            # Orientation error
            orientation_error = np.linalg.norm(current_pose[3:] - target_pose[3:])
            
            # Combined error with position priority
            return position_error + 0.1 * orientation_error
        
        def constraint_function(joint_angles: np.ndarray) -> np.ndarray:
            """Constraint function for joint limits"""
            constraints = []
            for i, (angle, (min_limit, max_limit)) in enumerate(zip(joint_angles, joint_limits)):
                constraints.append(angle - min_limit)  # angle >= min_limit
                constraints.append(max_limit - angle)  # angle <= max_limit
            return np.array(constraints)
        
        # Set up optimization bounds
        bounds = [(min_limit, max_limit) for min_limit, max_limit in joint_limits]
        
        # Set up constraints
        constraints = {
            'type': 'ineq',
            'fun': constraint_function
        }
        
        try:
            # Solve using multiple methods for robustness
            methods = ['SLSQP', 'L-BFGS-B']
            best_solution = None
            best_error = float('inf')
            
            for method in methods:
                try:
                    result = opt.minimize(
                        objective_function,
                        initial_guess,
                        method=method,
                        bounds=bounds,
                        constraints=constraints if method == 'SLSQP' else None,
                        options={'maxiter': self.max_iterations}
                    )
                    
                    if result.success and result.fun < best_error:
                        best_solution = result.x
                        best_error = result.fun
                        
                except Exception as e:
                    logger.debug(f"Optimization method {method} failed: {e}")
                    continue
            
            # Check if solution is good enough
            if best_solution is not None and best_error < self.tolerance:
                return best_solution
            else:
                logger.warning(f"IK solution found but error {best_error:.6f} > tolerance {self.tolerance}")
                return best_solution if best_solution is not None else None
                
        except Exception as e:
            logger.error(f"Inverse kinematics failed: {e}")
            return None
    
    def compute_jacobian(self, joint_angles: np.ndarray, delta: float = 1e-6) -> np.ndarray:
        """
        Compute Jacobian matrix using numerical differentiation.
        
        Args:
            joint_angles: Current joint configuration
            delta: Small perturbation for numerical differentiation
            
        Returns:
            jacobian: 6x7 Jacobian matrix [position_jacobian; orientation_jacobian]
        """
        jacobian = np.zeros((6, self.num_joints))
        
        # Current pose
        current_pose = self.forward_kinematics(joint_angles)
        
        # Compute partial derivatives for each joint
        for i in range(self.num_joints):
            # Perturb joint angle
            perturbed_angles = joint_angles.copy()
            perturbed_angles[i] += delta
            
            # Compute perturbed pose
            perturbed_pose = self.forward_kinematics(perturbed_angles)
            
            # Numerical derivative
            jacobian[:, i] = (perturbed_pose - current_pose) / delta
        
        return jacobian
    
    def compute_analytical_jacobian(self, joint_angles: np.ndarray) -> np.ndarray:
        """
        Compute analytical Jacobian (more accurate but more complex).
        
        Args:
            joint_angles: Current joint configuration
            
        Returns:
            jacobian: 6x7 analytical Jacobian matrix
        """
        # Compute transformation matrices for each joint
        transforms = []
        T = np.eye(4)
        
        for i, angle in enumerate(joint_angles):
            a, alpha, d, theta_offset = self.dh_params[i]
            theta = angle + theta_offset
            T_i = self.dh_transform(a, alpha, d, theta)
            T = T @ T_i
            transforms.append(T.copy())
        
        # End-effector position
        p_end = transforms[-1][:3, 3]
        
        # Initialize Jacobian
        jacobian = np.zeros((6, self.num_joints))
        
        # Compute Jacobian columns
        for i in range(self.num_joints):
            if i == 0:
                # First joint (relative to base frame)
                z_i = np.array([0, 0, 1])  # Base frame z-axis
                p_i = np.array([0, 0, 0])  # Base frame origin
            else:
                # i-th joint (relative to (i-1)-th frame)
                z_i = transforms[i-1][:3, 2]  # z-axis of (i-1)-th frame
                p_i = transforms[i-1][:3, 3]  # Origin of (i-1)-th frame
            
            # Linear velocity part (cross product)
            jacobian[:3, i] = np.cross(z_i, p_end - p_i)
            
            # Angular velocity part
            jacobian[3:, i] = z_i
        
        return jacobian
    
    def inverse_kinematics_jacobian(self, target_pose: np.ndarray,
                                  initial_guess: np.ndarray,
                                  max_iterations: int = 100,
                                  tolerance: float = 1e-6) -> Optional[np.ndarray]:
        """
        Solve inverse kinematics using Jacobian-based iterative method.
        
        Args:
            target_pose: Desired end-effector pose
            initial_guess: Initial joint configuration
            max_iterations: Maximum number of iterations
            tolerance: Convergence tolerance
            
        Returns:
            joint_angles: Solution or None if not converged
        """
        joint_angles = initial_guess.copy()
        
        for iteration in range(max_iterations):
            # Current pose
            current_pose = self.forward_kinematics(joint_angles)
            
            # Error
            error = target_pose - current_pose
            error_norm = np.linalg.norm(error)
            
            # Check convergence
            if error_norm < tolerance:
                logger.debug(f"IK converged in {iteration} iterations")
                return joint_angles
            
            # Compute Jacobian
            jacobian = self.compute_jacobian(joint_angles)
            
            # Pseudo-inverse for redundant manipulator
            try:
                jacobian_pinv = np.linalg.pinv(jacobian)
            except np.linalg.LinAlgError:
                logger.warning("Jacobian pseudo-inverse failed (singular matrix)")
                return None
            
            # Update joint angles
            delta_q = jacobian_pinv @ error
            
            # Apply damping for stability
            damping_factor = 0.1
            joint_angles += damping_factor * delta_q
        
        logger.warning(f"IK did not converge after {max_iterations} iterations")
        return None
    
    def _rotation_matrix_to_euler(self, R: np.ndarray) -> np.ndarray:
        """
        Convert rotation matrix to Euler angles (ZYX convention).
        
        Args:
            R: 3x3 rotation matrix
            
        Returns:
            euler_angles: [rx, ry, rz] in radians
        """
        # Extract Euler angles (ZYX convention)
        sy = np.sqrt(R[0, 0] ** 2 + R[1, 0] ** 2)
        
        singular = sy < 1e-6
        
        if not singular:
            rx = np.arctan2(R[2, 1], R[2, 2])
            ry = np.arctan2(-R[2, 0], sy)
            rz = np.arctan2(R[1, 0], R[0, 0])
        else:
            rx = np.arctan2(-R[1, 2], R[1, 1])
            ry = np.arctan2(-R[2, 0], sy)
            rz = 0
        
        return np.array([rx, ry, rz])
    
    def _euler_to_rotation_matrix(self, euler_angles: np.ndarray) -> np.ndarray:
        """
        Convert Euler angles to rotation matrix (ZYX convention).
        
        Args:
            euler_angles: [rx, ry, rz] in radians
            
        Returns:
            R: 3x3 rotation matrix
        """
        rx, ry, rz = euler_angles
        
        # Individual rotation matrices
        Rx = np.array([[1, 0, 0],
                       [0, np.cos(rx), -np.sin(rx)],
                       [0, np.sin(rx), np.cos(rx)]])
        
        Ry = np.array([[np.cos(ry), 0, np.sin(ry)],
                       [0, 1, 0],
                       [-np.sin(ry), 0, np.cos(ry)]])
        
        Rz = np.array([[np.cos(rz), -np.sin(rz), 0],
                       [np.sin(rz), np.cos(rz), 0],
                       [0, 0, 1]])
        
        return Rz @ Ry @ Rx
    
    def check_singularity(self, joint_angles: np.ndarray, threshold: float = 1e-3) -> bool:
        """
        Check if current configuration is near a kinematic singularity.
        
        Args:
            joint_angles: Joint configuration to check
            threshold: Singularity threshold
            
        Returns:
            is_singular: True if near singularity
        """
        jacobian = self.compute_jacobian(joint_angles)
        
        # Compute manipulability index (determinant of J*J^T)
        manipulability = np.sqrt(np.linalg.det(jacobian @ jacobian.T))
        
        return manipulability < threshold
    
    def get_multiple_solutions(self, target_pose: np.ndarray,
                             num_attempts: int = 10) -> List[np.ndarray]:
        """
        Find multiple IK solutions by trying different initial guesses.
        
        Args:
            target_pose: Desired end-effector pose
            num_attempts: Number of different initial guesses to try
            
        Returns:
            solutions: List of valid joint angle solutions
        """
        solutions = []
        
        for _ in range(num_attempts):
            # Random initial guess within joint limits
            initial_guess = np.random.uniform(-np.pi, np.pi, self.num_joints)
            
            # Solve IK
            solution = self.inverse_kinematics_jacobian(target_pose, initial_guess)
            
            if solution is not None:
                # Check if this is a new solution (not too close to existing ones)
                is_new = True
                for existing_solution in solutions:
                    if np.linalg.norm(solution - existing_solution) < 0.1:
                        is_new = False
                        break
                
                if is_new:
                    solutions.append(solution)
        
        return solutions


class DifferentialKinematics:
    """
    Differential kinematics utilities for velocity and acceleration control.
    """
    
    def __init__(self, kinematic_chain: KinematicChain):
        self.kinematics = kinematic_chain
    
    def velocity_kinematics(self, joint_angles: np.ndarray,
                           joint_velocities: np.ndarray) -> np.ndarray:
        """
        Compute end-effector velocity from joint velocities.
        
        Args:
            joint_angles: Current joint configuration
            joint_velocities: Joint velocities
            
        Returns:
            end_effector_velocity: [vx, vy, vz, wx, wy, wz]
        """
        jacobian = self.kinematics.compute_jacobian(joint_angles)
        return jacobian @ joint_velocities
    
    def inverse_velocity_kinematics(self, joint_angles: np.ndarray,
                                  desired_velocity: np.ndarray) -> np.ndarray:
        """
        Compute required joint velocities for desired end-effector velocity.
        
        Args:
            joint_angles: Current joint configuration
            desired_velocity: Desired end-effector velocity
            
        Returns:
            joint_velocities: Required joint velocities
        """
        jacobian = self.kinematics.compute_jacobian(joint_angles)
        jacobian_pinv = np.linalg.pinv(jacobian)
        return jacobian_pinv @ desired_velocity