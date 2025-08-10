"""
Biomechanical Models for Human-Like Motion
==========================================

Models of human motor control and biomechanics for generating
realistic handwriting motions that respect physiological constraints.
"""

import numpy as np
import torch
import torch.nn as nn
from typing import Dict, Any, List, Tuple, Optional, Union
from dataclasses import dataclass
import logging
from scipy.integrate import odeint
from scipy.optimize import minimize

logger = logging.getLogger(__name__)


@dataclass
class MuscleParameters:
    """Parameters for a muscle model."""
    max_force: float        # Maximum force generation capability
    optimal_length: float   # Optimal muscle length
    pennation_angle: float  # Pennation angle (radians)
    activation_time: float  # Time constant for activation
    deactivation_time: float # Time constant for deactivation


@dataclass
class JointParameters:
    """Parameters for a joint model."""
    damping: float          # Joint damping coefficient
    stiffness: float        # Joint stiffness
    friction: float         # Joint friction
    range_of_motion: Tuple[float, float]  # Min, max joint angles


class BiomechanicalModel:
    """
    Base class for biomechanical models of human motor control.
    
    Provides framework for modeling physiological constraints
    and generating human-like motion patterns.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize biomechanical model.
        
        Args:
            config: Model configuration
        """
        self.config = config
        self.dt = config.get('dt', 0.01)  # Time step
        self.noise_level = config.get('noise_level', 0.01)
        
        # Human motor control parameters
        self.reaction_time = config.get('reaction_time', 0.15)  # Motor command delay
        self.movement_time = config.get('movement_time', 0.8)   # Typical movement duration
        self.tremor_frequency = config.get('tremor_frequency', 8.0)  # Hz
        self.tremor_amplitude = config.get('tremor_amplitude', 0.0005)  # m
        
        logger.info("Initialized BiomechanicalModel")
    
    def generate_motor_command(self, 
                             target_position: np.ndarray,
                             current_position: np.ndarray,
                             current_velocity: np.ndarray) -> np.ndarray:
        """
        Generate motor command based on target and current state.
        
        Args:
            target_position: Desired position [x, y]
            current_position: Current position [x, y]
            current_velocity: Current velocity [vx, vy]
            
        Returns:
            motor_command: Motor command (acceleration) [ax, ay]
        """
        # Implement minimum jerk model (Flash & Hogan, 1985)
        position_error = target_position - current_position
        
        # PD controller with biomechanical constraints
        kp = 25.0  # Position gain
        kd = 10.0  # Velocity gain
        
        motor_command = kp * position_error - kd * current_velocity
        
        # Apply physiological limits
        max_acceleration = 5.0  # m/s²
        motor_command = np.clip(motor_command, -max_acceleration, max_acceleration)
        
        # Add motor noise (signal-dependent)
        signal_magnitude = np.linalg.norm(motor_command)
        noise_std = self.noise_level * (1.0 + 0.5 * signal_magnitude)
        motor_noise = np.random.normal(0, noise_std, 2)
        motor_command += motor_noise
        
        # Add physiological tremor
        tremor = self._generate_tremor()
        motor_command += tremor
        
        return motor_command
    
    def _generate_tremor(self) -> np.ndarray:
        """Generate physiological tremor component."""
        # Simple sinusoidal tremor model
        t = getattr(self, '_current_time', 0.0)
        tremor_x = self.tremor_amplitude * np.sin(2 * np.pi * self.tremor_frequency * t)
        tremor_y = self.tremor_amplitude * np.sin(2 * np.pi * self.tremor_frequency * t + np.pi/4)
        
        return np.array([tremor_x, tremor_y])
    
    def apply_biomechanical_constraints(self, 
                                      trajectory: np.ndarray,
                                      velocities: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Apply biomechanical constraints to trajectory.
        
        Args:
            trajectory: Input trajectory [n_points, 2]
            velocities: Input velocities [n_points, 2]
            
        Returns:
            constrained_trajectory: Constrained trajectory
            constrained_velocities: Constrained velocities
        """
        # Apply velocity limits
        max_velocity = 0.5  # m/s
        speed = np.linalg.norm(velocities, axis=1)
        speed_scale = np.minimum(1.0, max_velocity / (speed + 1e-8))
        constrained_velocities = velocities * speed_scale.reshape(-1, 1)
        
        # Apply acceleration limits
        dt = self.dt
        accelerations = np.diff(constrained_velocities, axis=0) / dt
        max_acceleration = 5.0  # m/s²
        acc_magnitude = np.linalg.norm(accelerations, axis=1)
        acc_scale = np.minimum(1.0, max_acceleration / (acc_magnitude + 1e-8))
        
        # Smooth out excessive accelerations
        for i in range(len(acc_scale)):
            if acc_scale[i] < 1.0:
                start_idx = max(0, i)
                end_idx = min(len(constrained_velocities) - 1, i + 2)
                constrained_velocities[start_idx:end_idx] *= acc_scale[i]
        
        # Recompute trajectory from constrained velocities
        constrained_trajectory = np.zeros_like(trajectory)
        constrained_trajectory[0] = trajectory[0]
        
        for i in range(1, len(trajectory)):
            constrained_trajectory[i] = constrained_trajectory[i-1] + constrained_velocities[i] * dt
        
        return constrained_trajectory, constrained_velocities
    
    def simulate_muscle_activation(self, 
                                 desired_force: float,
                                 muscle_params: MuscleParameters,
                                 current_activation: float = 0.0) -> float:
        """
        Simulate muscle activation dynamics.
        
        Args:
            desired_force: Desired force output
            muscle_params: Muscle parameters
            current_activation: Current activation level
            
        Returns:
            new_activation: Updated activation level
        """
        # Desired activation level
        desired_activation = desired_force / muscle_params.max_force
        desired_activation = np.clip(desired_activation, 0.0, 1.0)
        
        # First-order activation dynamics
        if desired_activation > current_activation:
            # Activation
            tau = muscle_params.activation_time
        else:
            # Deactivation
            tau = muscle_params.deactivation_time
        
        # Update activation
        activation_change = (desired_activation - current_activation) / tau * self.dt
        new_activation = current_activation + activation_change
        new_activation = np.clip(new_activation, 0.0, 1.0)
        
        return new_activation


class MusculoskeletalModel(BiomechanicalModel):
    """
    Musculoskeletal model for generating human-like arm movements.
    
    Models the major muscle groups and joints involved in handwriting.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize musculoskeletal model."""
        super().__init__(config)
        
        # Define muscle groups for 2D arm model
        self.muscles = self._initialize_muscles()
        self.joints = self._initialize_joints()
        
        # Current muscle activations
        self.muscle_activations = {name: 0.0 for name in self.muscles.keys()}
        
        logger.info("Initialized MusculoskeletalModel")
    
    def _initialize_muscles(self) -> Dict[str, MuscleParameters]:
        """Initialize muscle parameters."""
        muscles = {
            'shoulder_flexor': MuscleParameters(
                max_force=200.0, optimal_length=0.3, pennation_angle=0.0,
                activation_time=0.05, deactivation_time=0.1
            ),
            'shoulder_extensor': MuscleParameters(
                max_force=250.0, optimal_length=0.32, pennation_angle=0.0,
                activation_time=0.05, deactivation_time=0.1
            ),
            'elbow_flexor': MuscleParameters(
                max_force=150.0, optimal_length=0.25, pennation_angle=0.1,
                activation_time=0.04, deactivation_time=0.08
            ),
            'elbow_extensor': MuscleParameters(
                max_force=180.0, optimal_length=0.27, pennation_angle=0.1,
                activation_time=0.04, deactivation_time=0.08
            ),
            'wrist_flexor': MuscleParameters(
                max_force=80.0, optimal_length=0.15, pennation_angle=0.15,
                activation_time=0.03, deactivation_time=0.06
            ),
            'wrist_extensor': MuscleParameters(
                max_force=70.0, optimal_length=0.16, pennation_angle=0.15,
                activation_time=0.03, deactivation_time=0.06
            )
        }
        return muscles
    
    def _initialize_joints(self) -> Dict[str, JointParameters]:
        """Initialize joint parameters."""
        joints = {
            'shoulder': JointParameters(
                damping=2.0, stiffness=20.0, friction=0.1,
                range_of_motion=(-np.pi/2, np.pi/2)
            ),
            'elbow': JointParameters(
                damping=1.5, stiffness=15.0, friction=0.08,
                range_of_motion=(0, np.pi)
            ),
            'wrist': JointParameters(
                damping=1.0, stiffness=10.0, friction=0.05,
                range_of_motion=(-np.pi/3, np.pi/3)
            )
        }
        return joints
    
    def forward_kinematics(self, joint_angles: np.ndarray) -> np.ndarray:
        """
        Compute end-effector position from joint angles.
        
        Args:
            joint_angles: Joint angles [shoulder, elbow, wrist]
            
        Returns:
            end_effector_pos: End-effector position [x, y]
        """
        # Simplified 3-DOF arm model
        L1 = 0.3  # Upper arm length
        L2 = 0.25  # Forearm length
        L3 = 0.1   # Hand length
        
        q1, q2, q3 = joint_angles
        
        # Forward kinematics
        x = L1 * np.cos(q1) + L2 * np.cos(q1 + q2) + L3 * np.cos(q1 + q2 + q3)
        y = L1 * np.sin(q1) + L2 * np.sin(q1 + q2) + L3 * np.sin(q1 + q2 + q3)
        
        return np.array([x, y])
    
    def inverse_kinematics(self, target_position: np.ndarray,
                          initial_guess: np.ndarray = None) -> np.ndarray:
        """
        Compute joint angles for target end-effector position.
        
        Args:
            target_position: Target position [x, y]
            initial_guess: Initial guess for joint angles
            
        Returns:
            joint_angles: Joint angles [shoulder, elbow, wrist]
        """
        if initial_guess is None:
            initial_guess = np.array([0.0, np.pi/4, 0.0])
        
        def objective(joint_angles):
            current_pos = self.forward_kinematics(joint_angles)
            error = np.linalg.norm(current_pos - target_position)
            return error
        
        # Joint limits
        bounds = [
            self.joints['shoulder'].range_of_motion,
            self.joints['elbow'].range_of_motion,
            self.joints['wrist'].range_of_motion
        ]
        
        # Optimize
        try:
            result = minimize(objective, initial_guess, bounds=bounds, method='L-BFGS-B')
            if result.success:
                return result.x
            else:
                logger.warning("IK optimization failed, using initial guess")
                return initial_guess
        except Exception as e:
            logger.warning(f"IK error: {e}, using initial guess")
            return initial_guess
    
    def compute_muscle_forces(self, joint_angles: np.ndarray,
                            joint_torques: np.ndarray) -> Dict[str, float]:
        """
        Compute required muscle forces for given joint torques.
        
        Args:
            joint_angles: Current joint angles
            joint_torques: Desired joint torques
            
        Returns:
            muscle_forces: Required muscle forces
        """
        # Simplified muscle moment arms (would be more complex in reality)
        moment_arms = {
            'shoulder_flexor': 0.05,
            'shoulder_extensor': -0.05,
            'elbow_flexor': 0.04,
            'elbow_extensor': -0.04,
            'wrist_flexor': 0.03,
            'wrist_extensor': -0.03
        }
        
        # Map torques to muscle forces (simplified)
        muscle_forces = {}
        
        # Shoulder torque
        if joint_torques[0] > 0:
            muscle_forces['shoulder_flexor'] = joint_torques[0] / moment_arms['shoulder_flexor']
            muscle_forces['shoulder_extensor'] = 0.0
        else:
            muscle_forces['shoulder_flexor'] = 0.0
            muscle_forces['shoulder_extensor'] = abs(joint_torques[0]) / abs(moment_arms['shoulder_extensor'])
        
        # Elbow torque
        if joint_torques[1] > 0:
            muscle_forces['elbow_flexor'] = joint_torques[1] / moment_arms['elbow_flexor']
            muscle_forces['elbow_extensor'] = 0.0
        else:
            muscle_forces['elbow_flexor'] = 0.0
            muscle_forces['elbow_extensor'] = abs(joint_torques[1]) / abs(moment_arms['elbow_extensor'])
        
        # Wrist torque
        if joint_torques[2] > 0:
            muscle_forces['wrist_flexor'] = joint_torques[2] / moment_arms['wrist_flexor']
            muscle_forces['wrist_extensor'] = 0.0
        else:
            muscle_forces['wrist_flexor'] = 0.0
            muscle_forces['wrist_extensor'] = abs(joint_torques[2]) / abs(moment_arms['wrist_extensor'])
        
        return muscle_forces
    
    def simulate_trajectory(self, target_trajectory: np.ndarray,
                          duration: float) -> Dict[str, np.ndarray]:
        """
        Simulate musculoskeletal trajectory execution.
        
        Args:
            target_trajectory: Target trajectory [n_points, 2]
            duration: Total duration
            
        Returns:
            simulation_results: Dictionary with trajectories and muscle activations
        """
        n_points = len(target_trajectory)
        dt = duration / n_points
        
        # Initialize arrays
        actual_trajectory = np.zeros_like(target_trajectory)
        joint_angles_history = np.zeros((n_points, 3))
        muscle_activation_history = {name: np.zeros(n_points) for name in self.muscles.keys()}
        
        # Initial state
        current_joint_angles = np.array([0.0, np.pi/4, 0.0])
        actual_trajectory[0] = self.forward_kinematics(current_joint_angles)
        joint_angles_history[0] = current_joint_angles
        
        for i in range(1, n_points):
            target_pos = target_trajectory[i]
            
            # Inverse kinematics
            desired_joint_angles = self.inverse_kinematics(target_pos, current_joint_angles)
            
            # Compute required joint torques (simplified PD control)
            angle_error = desired_joint_angles - current_joint_angles
            joint_torques = 10.0 * angle_error  # Proportional control
            
            # Compute muscle forces
            muscle_forces = self.compute_muscle_forces(current_joint_angles, joint_torques)
            
            # Update muscle activations
            for muscle_name, desired_force in muscle_forces.items():
                current_activation = self.muscle_activations[muscle_name]
                new_activation = self.simulate_muscle_activation(
                    desired_force, self.muscles[muscle_name], current_activation
                )
                self.muscle_activations[muscle_name] = new_activation
                muscle_activation_history[muscle_name][i] = new_activation
            
            # Update joint angles (simplified integration)
            current_joint_angles += angle_error * 0.5  # Simplified update
            
            # Apply joint limits
            for j, joint_name in enumerate(['shoulder', 'elbow', 'wrist']):
                limits = self.joints[joint_name].range_of_motion
                current_joint_angles[j] = np.clip(current_joint_angles[j], limits[0], limits[1])
            
            # Forward kinematics
            actual_trajectory[i] = self.forward_kinematics(current_joint_angles)
            joint_angles_history[i] = current_joint_angles
        
        return {
            'actual_trajectory': actual_trajectory,
            'target_trajectory': target_trajectory,
            'joint_angles': joint_angles_history,
            'muscle_activations': muscle_activation_history
        }


class NeuralControlModel(BiomechanicalModel):
    """
    Neural control model based on motor cortex and cerebellar function.
    
    Models high-level motor planning and control as observed in
    neuroscience research on handwriting.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize neural control model."""
        super().__init__(config)
        
        # Neural control parameters
        self.planning_horizon = config.get('planning_horizon', 0.5)  # seconds
        self.adaptation_rate = config.get('adaptation_rate', 0.1)
        self.prediction_error_weight = config.get('prediction_error_weight', 0.2)
        
        # Internal models
        self.forward_model = ForwardModel(config.get('forward_model', {}))
        self.inverse_model = InverseModel(config.get('inverse_model', {}))
        
        logger.info("Initialized NeuralControlModel")
    
    def plan_movement(self, target_trajectory: np.ndarray) -> np.ndarray:
        """
        Plan movement using neural control principles.
        
        Args:
            target_trajectory: Target trajectory [n_points, 2]
            
        Returns:
            motor_plan: Motor plan (sequence of motor commands)
        """
        n_points = len(target_trajectory)
        motor_plan = np.zeros((n_points, 2))  # acceleration commands
        
        # Current state
        current_position = target_trajectory[0] if len(target_trajectory) > 0 else np.zeros(2)
        current_velocity = np.zeros(2)
        
        for i in range(n_points):
            target_position = target_trajectory[i]
            
            # Generate motor command
            motor_command = self.generate_motor_command(
                target_position, current_position, current_velocity
            )
            motor_plan[i] = motor_command
            
            # Update state prediction
            dt = self.dt
            current_velocity += motor_command * dt
            current_position += current_velocity * dt
            
            # Apply neural noise and adaptation
            motor_plan[i] = self._apply_neural_adaptations(motor_plan[i], i)
        
        return motor_plan
    
    def _apply_neural_adaptations(self, motor_command: np.ndarray, time_step: int) -> np.ndarray:
        """Apply neural adaptations to motor command."""
        # Simulate learning and adaptation effects
        adapted_command = motor_command.copy()
        
        # Add temporal correlations (neural firing patterns)
        if time_step > 0:
            correlation_factor = 0.1
            adapted_command += correlation_factor * getattr(self, '_prev_command', np.zeros(2))
        
        self._prev_command = adapted_command
        
        # Add neural variability
        neural_noise = np.random.normal(0, 0.01, 2)
        adapted_command += neural_noise
        
        return adapted_command
    
    def simulate_cerebellar_correction(self, 
                                     predicted_trajectory: np.ndarray,
                                     actual_trajectory: np.ndarray) -> np.ndarray:
        """
        Simulate cerebellar error correction.
        
        Args:
            predicted_trajectory: Predicted trajectory
            actual_trajectory: Actual trajectory
            
        Returns:
            corrected_trajectory: Error-corrected trajectory
        """
        prediction_error = actual_trajectory - predicted_trajectory
        
        # Cerebellar learning (simplified)
        correction = self.prediction_error_weight * prediction_error
        
        # Apply correction with delay (neural processing time)
        corrected_trajectory = predicted_trajectory.copy()
        delay_steps = int(0.1 / self.dt)  # 100ms delay
        
        for i in range(delay_steps, len(corrected_trajectory)):
            corrected_trajectory[i] += correction[i - delay_steps]
        
        return corrected_trajectory


class ForwardModel(nn.Module):
    """
    Neural forward model for predicting movement outcomes.
    
    Learns to predict the sensory consequences of motor commands.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize forward model."""
        super().__init__()
        
        input_dim = config.get('input_dim', 4)  # position + velocity
        hidden_dim = config.get('hidden_dim', 64)
        output_dim = config.get('output_dim', 2)  # predicted position change
        
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )
        
        logger.info("Initialized ForwardModel")
    
    def forward(self, state_action: torch.Tensor) -> torch.Tensor:
        """
        Predict next state from current state and action.
        
        Args:
            state_action: Current state and action [batch, input_dim]
            
        Returns:
            predicted_state_change: Predicted state change [batch, output_dim]
        """
        return self.network(state_action)


class InverseModel(nn.Module):
    """
    Neural inverse model for motor command generation.
    
    Learns to generate motor commands that achieve desired movements.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize inverse model."""
        super().__init__()
        
        input_dim = config.get('input_dim', 4)  # current + desired position
        hidden_dim = config.get('hidden_dim', 64)
        output_dim = config.get('output_dim', 2)  # motor command
        
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
            nn.Tanh()  # Bounded motor commands
        )
        
        logger.info("Initialized InverseModel")
    
    def forward(self, current_desired_state: torch.Tensor) -> torch.Tensor:
        """
        Generate motor command for desired state change.
        
        Args:
            current_desired_state: Current and desired states [batch, input_dim]
            
        Returns:
            motor_command: Generated motor command [batch, output_dim]
        """
        return self.network(current_desired_state)


class AdaptiveMotorControl:
    """
    Adaptive motor control system that learns and improves over time.
    
    Combines multiple biomechanical models with learning mechanisms.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize adaptive motor control."""
        self.config = config
        
        # Component models
        self.biomechanical_model = BiomechanicalModel(config.get('biomechanical', {}))
        self.musculoskeletal_model = MusculoskeletalModel(config.get('musculoskeletal', {}))
        self.neural_model = NeuralControlModel(config.get('neural', {}))
        
        # Learning parameters
        self.learning_rate = config.get('learning_rate', 0.01)
        self.experience_buffer = []
        self.max_buffer_size = config.get('max_buffer_size', 1000)
        
        logger.info("Initialized AdaptiveMotorControl")
    
    def execute_movement(self, target_trajectory: np.ndarray) -> Dict[str, np.ndarray]:
        """
        Execute movement using adaptive control.
        
        Args:
            target_trajectory: Target trajectory
            
        Returns:
            execution_results: Complete movement execution results
        """
        # Neural planning
        motor_plan = self.neural_model.plan_movement(target_trajectory)
        
        # Musculoskeletal simulation
        msk_results = self.musculoskeletal_model.simulate_trajectory(
            target_trajectory, len(target_trajectory) * self.biomechanical_model.dt
        )
        
        # Apply biomechanical constraints
        constrained_trajectory, constrained_velocities = self.biomechanical_model.apply_biomechanical_constraints(
            msk_results['actual_trajectory'], 
            np.diff(msk_results['actual_trajectory'], axis=0, prepend=msk_results['actual_trajectory'][:1])
        )
        
        # Store experience for learning
        self._store_experience(target_trajectory, constrained_trajectory, motor_plan)
        
        return {
            'target_trajectory': target_trajectory,
            'planned_trajectory': msk_results['actual_trajectory'],
            'executed_trajectory': constrained_trajectory,
            'motor_plan': motor_plan,
            'joint_angles': msk_results['joint_angles'],
            'muscle_activations': msk_results['muscle_activations'],
            'velocities': constrained_velocities
        }
    
    def _store_experience(self, target: np.ndarray, actual: np.ndarray, commands: np.ndarray) -> None:
        """Store movement experience for learning."""
        experience = {
            'target': target,
            'actual': actual,
            'commands': commands,
            'error': np.mean(np.linalg.norm(target - actual, axis=1))
        }
        
        self.experience_buffer.append(experience)
        
        # Limit buffer size
        if len(self.experience_buffer) > self.max_buffer_size:
            self.experience_buffer.pop(0)
    
    def adapt_control(self) -> None:
        """Adapt control parameters based on experience."""
        if len(self.experience_buffer) < 10:
            return
        
        # Analyze recent performance
        recent_errors = [exp['error'] for exp in self.experience_buffer[-10:]]
        avg_error = np.mean(recent_errors)
        
        # Adaptive adjustments (simplified)
        if avg_error > 0.01:  # High error
            # Increase damping for stability
            for joint_params in self.musculoskeletal_model.joints.values():
                joint_params.damping *= 1.05
        else:  # Low error
            # Decrease damping for responsiveness
            for joint_params in self.musculoskeletal_model.joints.values():
                joint_params.damping *= 0.98
        
        logger.debug(f"Adapted control parameters, avg_error: {avg_error:.6f}")
    
    def get_performance_metrics(self) -> Dict[str, float]:
        """Get performance metrics from recent experience."""
        if not self.experience_buffer:
            return {}
        
        recent_experiences = self.experience_buffer[-50:]  # Last 50 movements
        
        errors = [exp['error'] for exp in recent_experiences]
        
        return {
            'mean_error': np.mean(errors),
            'std_error': np.std(errors),
            'max_error': np.max(errors),
            'min_error': np.min(errors),
            'num_experiences': len(self.experience_buffer)
        }