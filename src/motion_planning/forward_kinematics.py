"""
Forward Kinematics Implementation
=================================

Forward kinematics for the 7-DOF robot arm using Denavit-Hartenberg
parameters and transformation matrices.
"""

import numpy as np
from typing import Dict, Any, List, Tuple, Optional, Union
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)


@dataclass
class DHParameters:
    """
    Denavit-Hartenberg parameters for a single joint.
    
    Attributes:
        a: Link length (distance along x-axis)
        alpha: Link twist (rotation about x-axis)
        d: Link offset (distance along z-axis)  
        theta: Joint angle (rotation about z-axis)
        joint_type: 'revolute' or 'prismatic'
    """
    a: float
    alpha: float
    d: float
    theta: float
    joint_type: str = 'revolute'
    
    def __post_init__(self):
        """Validate DH parameters."""
        if self.joint_type not in ['revolute', 'prismatic']:
            raise ValueError(f"Invalid joint type: {self.joint_type}")


@dataclass 
class TransformationMatrix:
    """
    Homogeneous transformation matrix with position and orientation.
    
    Attributes:
        matrix: 4x4 transformation matrix
        position: 3D position [x, y, z]
        rotation: 3x3 rotation matrix
        euler_angles: Euler angles [roll, pitch, yaw]
    """
    matrix: np.ndarray
    
    def __post_init__(self):
        """Extract position and rotation from matrix."""
        if self.matrix.shape != (4, 4):
            raise ValueError("Transformation matrix must be 4x4")
        
        self.position = self.matrix[:3, 3]
        self.rotation = self.matrix[:3, :3]
        self.euler_angles = self._rotation_to_euler(self.rotation)
    
    def _rotation_to_euler(self, R: np.ndarray) -> np.ndarray:
        """Convert rotation matrix to Euler angles (ZYX convention)."""
        # Extract Euler angles from rotation matrix
        sy = np.sqrt(R[0, 0]**2 + R[1, 0]**2)
        
        singular = sy < 1e-6
        
        if not singular:
            x = np.arctan2(R[2, 1], R[2, 2])
            y = np.arctan2(-R[2, 0], sy)
            z = np.arctan2(R[1, 0], R[0, 0])
        else:
            x = np.arctan2(-R[1, 2], R[1, 1])
            y = np.arctan2(-R[2, 0], sy)
            z = 0
        
        return np.array([x, y, z])


class ForwardKinematics:
    """
    Forward kinematics solver for 7-DOF robot arm.
    
    Computes end-effector pose from joint angles using
    Denavit-Hartenberg parameters and transformation matrices.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize forward kinematics solver.
        
        Args:
            config: Configuration with DH parameters
        """
        self.config = config
        
        # Initialize DH parameters for 7-DOF arm
        self.dh_params = self._initialize_dh_parameters(config)
        
        # Robot geometry
        self.num_joints = len(self.dh_params)
        self.base_frame = np.eye(4)
        self.tool_frame = self._get_tool_frame(config)
        
        # Caching for performance
        self._cache_enabled = config.get('cache_enabled', True)
        self._transformation_cache = {}
        
        logger.info(f"Initialized ForwardKinematics for {self.num_joints}-DOF robot")
    
    def _initialize_dh_parameters(self, config: Dict[str, Any]) -> List[DHParameters]:
        """Initialize DH parameters for 7-DOF robot arm."""
        # Standard DH parameters for 7-DOF arm (similar to Franka Panda)
        default_dh = [
            # Joint 1: Base rotation
            DHParameters(a=0.0, alpha=0.0, d=0.333, theta=0.0),
            # Joint 2: Shoulder
            DHParameters(a=0.0, alpha=-np.pi/2, d=0.0, theta=0.0),
            # Joint 3: Upper arm
            DHParameters(a=0.0, alpha=np.pi/2, d=0.316, theta=0.0),
            # Joint 4: Elbow
            DHParameters(a=0.0825, alpha=np.pi/2, d=0.0, theta=0.0),
            # Joint 5: Forearm
            DHParameters(a=-0.0825, alpha=-np.pi/2, d=0.384, theta=0.0),
            # Joint 6: Wrist 1
            DHParameters(a=0.0, alpha=np.pi/2, d=0.0, theta=0.0),
            # Joint 7: Wrist 2 (flange)
            DHParameters(a=0.088, alpha=np.pi/2, d=0.107, theta=0.0)
        ]
        
        # Override with config if provided
        if 'dh_parameters' in config:
            custom_dh = config['dh_parameters']
            for i, params in enumerate(custom_dh):
                if i < len(default_dh):
                    default_dh[i] = DHParameters(**params)
        
        return default_dh
    
    def _get_tool_frame(self, config: Dict[str, Any]) -> np.ndarray:
        """Get tool frame transformation (end-effector to pen tip)."""
        # Default tool frame (pen extending 15cm from flange)
        tool_offset = config.get('tool_offset', [0.0, 0.0, 0.15])  # [x, y, z]
        tool_rotation = config.get('tool_rotation', [0.0, 0.0, 0.0])  # [rx, ry, rz]
        
        # Create transformation matrix
        tool_frame = np.eye(4)
        tool_frame[:3, 3] = tool_offset
        
        # Apply rotation if specified
        if np.any(tool_rotation):
            R = self._euler_to_rotation(tool_rotation)
            tool_frame[:3, :3] = R
        
        return tool_frame
    
    def _euler_to_rotation(self, euler_angles: np.ndarray) -> np.ndarray:
        """Convert Euler angles to rotation matrix (ZYX convention)."""
        roll, pitch, yaw = euler_angles
        
        # Rotation matrices for each axis
        Rx = np.array([
            [1, 0, 0],
            [0, np.cos(roll), -np.sin(roll)],
            [0, np.sin(roll), np.cos(roll)]
        ])
        
        Ry = np.array([
            [np.cos(pitch), 0, np.sin(pitch)],
            [0, 1, 0],
            [-np.sin(pitch), 0, np.cos(pitch)]
        ])
        
        Rz = np.array([
            [np.cos(yaw), -np.sin(yaw), 0],
            [np.sin(yaw), np.cos(yaw), 0],
            [0, 0, 1]
        ])
        
        # Combined rotation (ZYX order)
        R = Rz @ Ry @ Rx
        return R
    
    def compute_dh_transform(self, dh: DHParameters, joint_value: float) -> np.ndarray:
        """
        Compute transformation matrix from DH parameters.
        
        Args:
            dh: DH parameters
            joint_value: Joint angle (revolute) or displacement (prismatic)
            
        Returns:
            transform: 4x4 transformation matrix
        """
        # Update joint variable
        if dh.joint_type == 'revolute':
            theta = dh.theta + joint_value
            d = dh.d
        else:  # prismatic
            theta = dh.theta
            d = dh.d + joint_value
        
        # DH transformation matrix
        cos_theta = np.cos(theta)
        sin_theta = np.sin(theta)
        cos_alpha = np.cos(dh.alpha)
        sin_alpha = np.sin(dh.alpha)
        
        transform = np.array([
            [cos_theta, -sin_theta * cos_alpha,  sin_theta * sin_alpha, dh.a * cos_theta],
            [sin_theta,  cos_theta * cos_alpha, -cos_theta * sin_alpha, dh.a * sin_theta],
            [0,          sin_alpha,              cos_alpha,             d],
            [0,          0,                      0,                     1]
        ])
        
        return transform
    
    def forward_kinematics(self, joint_angles: np.ndarray, 
                          include_tool: bool = True) -> TransformationMatrix:
        """
        Compute forward kinematics for given joint angles.
        
        Args:
            joint_angles: Joint angles [7] in radians
            include_tool: Whether to include tool frame transformation
            
        Returns:
            end_effector_transform: End-effector transformation
        """
        if len(joint_angles) != self.num_joints:
            raise ValueError(f"Expected {self.num_joints} joint angles, got {len(joint_angles)}")
        
        # Check cache
        cache_key = tuple(joint_angles) if self._cache_enabled else None
        if cache_key and cache_key in self._transformation_cache:
            return self._transformation_cache[cache_key]
        
        # Compute forward kinematics
        transform = self.base_frame.copy()
        
        for i, (dh, angle) in enumerate(zip(self.dh_params, joint_angles)):
            joint_transform = self.compute_dh_transform(dh, angle)
            transform = transform @ joint_transform
        
        # Apply tool frame if requested
        if include_tool:
            transform = transform @ self.tool_frame
        
        result = TransformationMatrix(transform)
        
        # Cache result
        if cache_key:
            self._transformation_cache[cache_key] = result
        
        return result
    
    def compute_jacobian(self, joint_angles: np.ndarray, 
                        numerical: bool = False) -> np.ndarray:
        """
        Compute Jacobian matrix for current joint configuration.
        
        Args:
            joint_angles: Joint angles [7] in radians
            numerical: Whether to use numerical differentiation
            
        Returns:
            jacobian: 6x7 Jacobian matrix [linear_velocity; angular_velocity]
        """
        if numerical:
            return self._compute_numerical_jacobian(joint_angles)
        else:
            return self._compute_analytical_jacobian(joint_angles)
    
    def _compute_analytical_jacobian(self, joint_angles: np.ndarray) -> np.ndarray:
        """Compute analytical Jacobian matrix."""
        jacobian = np.zeros((6, self.num_joints))
        
        # Get transformation matrices for each joint
        transforms = []
        current_transform = self.base_frame.copy()
        transforms.append(current_transform.copy())
        
        for i, (dh, angle) in enumerate(zip(self.dh_params, joint_angles)):
            joint_transform = self.compute_dh_transform(dh, angle)
            current_transform = current_transform @ joint_transform
            transforms.append(current_transform.copy())
        
        # End-effector position
        end_effector_pos = transforms[-1][:3, 3]
        
        # Compute Jacobian columns
        for i in range(self.num_joints):
            # Joint axis (z-axis of previous frame)
            joint_axis = transforms[i][:3, 2]
            
            # Joint position
            joint_pos = transforms[i][:3, 3]
            
            # Position difference
            pos_diff = end_effector_pos - joint_pos
            
            if self.dh_params[i].joint_type == 'revolute':
                # Revolute joint
                # Linear velocity: omega x (pe - pj)
                jacobian[:3, i] = np.cross(joint_axis, pos_diff)
                # Angular velocity: omega
                jacobian[3:, i] = joint_axis
            else:
                # Prismatic joint
                # Linear velocity: joint axis direction
                jacobian[:3, i] = joint_axis
                # Angular velocity: zero
                jacobian[3:, i] = np.zeros(3)
        
        return jacobian
    
    def _compute_numerical_jacobian(self, joint_angles: np.ndarray, 
                                  epsilon: float = 1e-6) -> np.ndarray:
        """Compute numerical Jacobian using finite differences."""
        jacobian = np.zeros((6, self.num_joints))
        
        # Current end-effector pose
        current_transform = self.forward_kinematics(joint_angles)
        current_pose = np.concatenate([current_transform.position, current_transform.euler_angles])
        
        # Compute partial derivatives
        for i in range(self.num_joints):
            # Perturbed joint angles
            joint_angles_plus = joint_angles.copy()
            joint_angles_plus[i] += epsilon
            
            # Forward kinematics for perturbed configuration
            perturbed_transform = self.forward_kinematics(joint_angles_plus)
            perturbed_pose = np.concatenate([perturbed_transform.position, perturbed_transform.euler_angles])
            
            # Finite difference
            jacobian[:, i] = (perturbed_pose - current_pose) / epsilon
        
        return jacobian
    
    def compute_link_transforms(self, joint_angles: np.ndarray) -> List[TransformationMatrix]:
        """
        Compute transformation matrices for all links.
        
        Args:
            joint_angles: Joint angles [7] in radians
            
        Returns:
            link_transforms: List of transformation matrices for each link
        """
        transforms = []
        current_transform = self.base_frame.copy()
        
        # Base transform
        transforms.append(TransformationMatrix(current_transform.copy()))
        
        # Compute transforms for each joint
        for dh, angle in zip(self.dh_params, joint_angles):
            joint_transform = self.compute_dh_transform(dh, angle)
            current_transform = current_transform @ joint_transform
            transforms.append(TransformationMatrix(current_transform.copy()))
        
        return transforms
    
    def compute_workspace_points(self, num_samples: int = 1000,
                               joint_limits: Optional[List[Tuple[float, float]]] = None) -> np.ndarray:
        """
        Sample workspace points by random joint configurations.
        
        Args:
            num_samples: Number of workspace points to sample
            joint_limits: Joint limits [(min, max), ...] for each joint
            
        Returns:
            workspace_points: Workspace points [num_samples, 3]
        """
        if joint_limits is None:
            # Default joint limits (approximate)
            joint_limits = [(-np.pi, np.pi)] * self.num_joints
        
        workspace_points = np.zeros((num_samples, 3))
        
        for i in range(num_samples):
            # Random joint configuration
            joint_angles = np.array([
                np.random.uniform(limits[0], limits[1]) 
                for limits in joint_limits
            ])
            
            # Forward kinematics
            transform = self.forward_kinematics(joint_angles)
            workspace_points[i] = transform.position
        
        return workspace_points
    
    def validate_configuration(self, joint_angles: np.ndarray,
                             joint_limits: Optional[List[Tuple[float, float]]] = None) -> Dict[str, Any]:
        """
        Validate joint configuration.
        
        Args:
            joint_angles: Joint angles to validate
            joint_limits: Joint limits for validation
            
        Returns:
            validation_result: Validation result with issues
        """
        issues = []
        warnings = []
        
        # Check array dimensions
        if len(joint_angles) != self.num_joints:
            issues.append(f"Expected {self.num_joints} joint angles, got {len(joint_angles)}")
            return {'valid': False, 'issues': issues, 'warnings': warnings}
        
        # Check for non-finite values
        if not np.all(np.isfinite(joint_angles)):
            issues.append("Non-finite joint angles detected")
        
        # Check joint limits
        if joint_limits is not None:
            for i, (angle, limits) in enumerate(zip(joint_angles, joint_limits)):
                if angle < limits[0] or angle > limits[1]:
                    issues.append(f"Joint {i+1} angle {angle:.3f} outside limits {limits}")
        
        # Check singularities (simplified)
        try:
            jacobian = self.compute_jacobian(joint_angles)
            condition_number = np.linalg.cond(jacobian)
            
            if condition_number > 1e6:
                warnings.append(f"High Jacobian condition number: {condition_number:.2e}")
        except Exception as e:
            warnings.append(f"Could not compute Jacobian: {e}")
        
        valid = len(issues) == 0
        
        return {
            'valid': valid,
            'issues': issues,
            'warnings': warnings,
            'jacobian_condition': condition_number if 'condition_number' in locals() else None
        }
    
    def clear_cache(self) -> None:
        """Clear transformation cache."""
        self._transformation_cache.clear()
        logger.debug("Cleared forward kinematics cache")
    
    def get_link_names(self) -> List[str]:
        """Get names of robot links."""
        return [
            'base_link',
            'link_1',
            'link_2', 
            'link_3',
            'link_4',
            'link_5',
            'link_6',
            'link_7',
            'end_effector'
        ]
    
    def visualize_robot(self, joint_angles: np.ndarray,
                       ax: Optional[Any] = None) -> None:
        """
        Visualize robot configuration (simplified).
        
        Args:
            joint_angles: Joint angles to visualize
            ax: Matplotlib axis (optional)
        """
        try:
            import matplotlib.pyplot as plt
            from mpl_toolkits.mplot3d import Axes3D
        except ImportError:
            logger.warning("Matplotlib not available for visualization")
            return
        
        if ax is None:
            fig = plt.figure(figsize=(10, 8))
            ax = fig.add_subplot(111, projection='3d')
        
        # Get link transforms
        link_transforms = self.compute_link_transforms(joint_angles)
        
        # Extract link positions
        link_positions = np.array([transform.position for transform in link_transforms])
        
        # Plot robot structure
        ax.plot(link_positions[:, 0], link_positions[:, 1], link_positions[:, 2], 
                'b-o', linewidth=3, markersize=8, label='Robot Links')
        
        # Plot end-effector
        end_pos = link_transforms[-1].position
        ax.scatter(end_pos[0], end_pos[1], end_pos[2], 
                  c='red', s=100, marker='s', label='End Effector')
        
        # Plot coordinate frames
        for i, transform in enumerate(link_transforms[::2]):  # Show every other frame
            pos = transform.position
            R = transform.rotation
            
            # Draw coordinate axes
            axis_length = 0.05
            for j, color in enumerate(['red', 'green', 'blue']):
                axis_end = pos + axis_length * R[:, j]
                ax.plot([pos[0], axis_end[0]], [pos[1], axis_end[1]], [pos[2], axis_end[2]], 
                       color=color, alpha=0.7)
        
        ax.set_xlabel('X (m)')
        ax.set_ylabel('Y (m)')
        ax.set_zlabel('Z (m)')
        ax.set_title('Robot Configuration')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Equal aspect ratio
        ax.set_box_aspect([1,1,1])
        
        if ax is None:
            plt.show()
    
    def export_urdf(self, file_path: str) -> None:
        """
        Export robot description as URDF (simplified).
        
        Args:
            file_path: Output URDF file path
        """
        urdf_content = """<?xml version="1.0"?>
<robot name="handwriting_robot">
"""
        
        # Add links and joints based on DH parameters
        for i, dh in enumerate(self.dh_params):
            link_name = f"link_{i}"
            joint_name = f"joint_{i}"
            
            # Link definition
            urdf_content += f"""
  <link name="{link_name}">
    <visual>
      <geometry>
        <cylinder radius="0.05" length="{abs(dh.d) if dh.d != 0 else 0.1}"/>
      </geometry>
      <material name="blue">
        <color rgba="0 0 1 1"/>
      </material>
    </visual>
  </link>
"""
            
            # Joint definition
            if i > 0:
                parent_link = f"link_{i-1}"
            else:
                parent_link = "base_link"
            
            urdf_content += f"""
  <joint name="{joint_name}" type="revolute">
    <parent link="{parent_link}"/>
    <child link="{link_name}"/>
    <origin xyz="{dh.a} 0 {dh.d}" rpy="0 0 {dh.alpha}"/>
    <axis xyz="0 0 1"/>
    <limit lower="-3.14" upper="3.14" effort="100" velocity="1"/>
  </joint>
"""
        
        urdf_content += "\n</robot>"
        
        # Write to file
        with open(file_path, 'w') as f:
            f.write(urdf_content)
        
        logger.info(f"Exported URDF to {file_path}")