"""
Sigma-Lognormal Model Implementation
===================================

Implementation of the Sigma-Lognormal model for generating human-like
handwriting velocity profiles. This is a well-established model in
handwriting research that produces biomechanically plausible movements.

Reference:
- Plamondon, R., & Djioua, M. (2006). A multi-scale competition approach 
  for the automatic extraction of movements in handwriting.
"""

import numpy as np
import torch
import torch.nn as nn
from typing import Dict, Any, List, Tuple, Optional, Union
from dataclasses import dataclass
import logging
import scipy.special as special
from scipy.optimize import minimize

logger = logging.getLogger(__name__)


@dataclass
class LognormalParameter:
    """
    Parameters for a single lognormal component.
    
    Attributes:
        t0: Time onset of the movement
        mu: Mean of the underlying normal distribution  
        sigma: Standard deviation of the underlying normal distribution
        D: Displacement vector [dx, dy]
    """
    t0: float
    mu: float
    sigma: float
    D: np.ndarray  # 2D displacement vector
    
    def __post_init__(self):
        """Validate parameters."""
        if self.sigma <= 0:
            raise ValueError("Sigma must be positive")
        if len(self.D) != 2:
            raise ValueError("Displacement D must be 2D vector")


class SigmaLognormalGenerator:
    """
    Sigma-Lognormal velocity profile generator for handwriting trajectories.
    
    Generates human-like velocity profiles by superposing multiple lognormal
    velocity components, each representing a basic motor command.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize Sigma-Lognormal generator.
        
        Args:
            config: Generator configuration
        """
        self.config = config
        
        # Model parameters
        self.sampling_rate = config.get('sampling_rate', 100)  # Hz
        self.min_sigma = config.get('min_sigma', 0.1)
        self.max_sigma = config.get('max_sigma', 2.0)
        self.min_mu = config.get('min_mu', -2.0)
        self.max_mu = config.get('max_mu', 2.0)
        
        # Optimization parameters
        self.max_components = config.get('max_components', 10)
        self.convergence_threshold = config.get('convergence_threshold', 1e-6)
        self.max_iterations = config.get('max_iterations', 1000)
        
        # Noise parameters
        self.noise_level = config.get('noise_level', 0.01)
        
        logger.info("Initialized SigmaLognormalGenerator")
    
    def generate_trajectory(self, 
                          parameters: List[LognormalParameter],
                          duration: float,
                          start_position: np.ndarray = None) -> Dict[str, np.ndarray]:
        """
        Generate complete trajectory from lognormal parameters.
        
        Args:
            parameters: List of lognormal parameters
            duration: Total duration in seconds
            start_position: Starting position [x, y]
            
        Returns:
            trajectory: Dictionary containing positions, velocities, accelerations
        """
        if start_position is None:
            start_position = np.array([0.0, 0.0])
        
        # Generate time vector
        dt = 1.0 / self.sampling_rate
        time_points = np.arange(0, duration, dt)
        n_points = len(time_points)
        
        # Generate velocity profile
        velocity_profile = self.generate_velocity_profile(parameters, time_points)
        
        # Integrate to get positions
        positions = np.zeros((n_points, 2))
        positions[0] = start_position
        
        for i in range(1, n_points):
            positions[i] = positions[i-1] + velocity_profile[i] * dt
        
        # Compute accelerations (finite differences)
        accelerations = np.zeros_like(velocity_profile)
        accelerations[1:-1] = (velocity_profile[2:] - velocity_profile[:-2]) / (2 * dt)
        accelerations[0] = (velocity_profile[1] - velocity_profile[0]) / dt
        accelerations[-1] = (velocity_profile[-1] - velocity_profile[-2]) / dt
        
        # Add realistic noise
        if self.noise_level > 0:
            noise = np.random.normal(0, self.noise_level, positions.shape)
            positions += noise
        
        return {
            'time': time_points,
            'positions': positions,
            'velocities': velocity_profile,
            'accelerations': accelerations,
            'parameters': parameters
        }
    
    def generate_velocity_profile(self, 
                                parameters: List[LognormalParameter],
                                time_points: np.ndarray) -> np.ndarray:
        """
        Generate velocity profile from lognormal parameters.
        
        Args:
            parameters: List of lognormal parameters
            time_points: Time points to evaluate
            
        Returns:
            velocity_profile: Velocity vectors [n_points, 2]
        """
        n_points = len(time_points)
        velocity_profile = np.zeros((n_points, 2))
        
        for param in parameters:
            component_velocity = self._lognormal_velocity_component(
                time_points, param.t0, param.mu, param.sigma, param.D
            )
            velocity_profile += component_velocity
        
        return velocity_profile
    
    def _lognormal_velocity_component(self,
                                    time_points: np.ndarray,
                                    t0: float,
                                    mu: float, 
                                    sigma: float,
                                    D: np.ndarray) -> np.ndarray:
        """
        Compute single lognormal velocity component.
        
        Args:
            time_points: Time points
            t0: Time onset
            mu: Mean parameter
            sigma: Standard deviation parameter
            D: Displacement vector
            
        Returns:
            velocity_component: Velocity contribution [n_points, 2]
        """
        # Shift time by onset
        t_shifted = time_points - t0
        
        # Avoid division by zero and negative times
        valid_mask = t_shifted > 1e-8
        n_points = len(time_points)
        velocity_component = np.zeros((n_points, 2))
        
        if not np.any(valid_mask):
            return velocity_component
        
        t_valid = t_shifted[valid_mask]
        
        # Lognormal probability density function
        log_t = np.log(t_valid)
        exponent = -0.5 * ((log_t - mu) / sigma) ** 2
        normalization = 1.0 / (t_valid * sigma * np.sqrt(2 * np.pi))
        
        lognormal_pdf = normalization * np.exp(exponent)
        
        # Apply displacement vector
        for i in range(2):  # x and y components
            velocity_component[valid_mask, i] = D[i] * lognormal_pdf
        
        return velocity_component
    
    def extract_parameters(self, 
                         trajectory: np.ndarray,
                         initial_guess: Optional[List[LognormalParameter]] = None) -> List[LognormalParameter]:
        """
        Extract lognormal parameters from observed trajectory using optimization.
        
        Args:
            trajectory: Observed trajectory [n_points, 2] (positions)
            initial_guess: Initial parameter guess
            
        Returns:
            parameters: Extracted lognormal parameters
        """
        # Compute velocity from trajectory
        dt = 1.0 / self.sampling_rate
        velocity = np.diff(trajectory, axis=0) / dt
        time_points = np.arange(len(velocity)) * dt
        
        # Initial parameter estimation if not provided
        if initial_guess is None:
            initial_guess = self._estimate_initial_parameters(velocity, time_points)
        
        # Optimize parameters
        optimized_parameters = self._optimize_parameters(velocity, time_points, initial_guess)
        
        return optimized_parameters
    
    def _estimate_initial_parameters(self, 
                                   velocity: np.ndarray,
                                   time_points: np.ndarray) -> List[LognormalParameter]:
        """
        Estimate initial parameters using velocity profile analysis.
        
        Args:
            velocity: Velocity profile [n_points, 2]
            time_points: Time points
            
        Returns:
            initial_parameters: Initial parameter estimates
        """
        # Compute speed (magnitude of velocity)
        speed = np.linalg.norm(velocity, axis=1)
        
        # Find local maxima (potential stroke onsets)
        from scipy.signal import find_peaks
        peaks, _ = find_peaks(speed, height=np.max(speed) * 0.1, distance=int(0.1 * self.sampling_rate))
        
        parameters = []
        
        # Limit number of components
        n_components = min(len(peaks), self.max_components)
        if n_components == 0:
            n_components = 1
            peaks = [len(speed) // 2]  # Use middle point
        
        for i in range(n_components):
            if i < len(peaks):
                peak_idx = peaks[i]
            else:
                peak_idx = len(speed) // 2
            
            # Estimate parameters around this peak
            t0 = max(0, time_points[peak_idx] - 0.1)  # Onset slightly before peak
            
            # Estimate mu and sigma from local velocity profile
            local_start = max(0, peak_idx - int(0.2 * self.sampling_rate))
            local_end = min(len(speed), peak_idx + int(0.2 * self.sampling_rate))
            
            if local_end > local_start:
                local_times = time_points[local_start:local_end] - t0
                local_speeds = speed[local_start:local_end]
                
                # Rough estimation
                mu = np.log(time_points[peak_idx] - t0 + 1e-8)
                sigma = 0.5  # Default value
                
                # Displacement estimation
                if i + 1 < len(peaks):
                    end_idx = peaks[i + 1]
                else:
                    end_idx = len(velocity)
                
                displacement = np.sum(velocity[peak_idx:end_idx], axis=0) * dt
            else:
                mu = 0.0
                sigma = 0.5
                displacement = np.array([0.01, 0.01])  # Small default displacement
            
            # Clamp parameters to valid ranges
            mu = np.clip(mu, self.min_mu, self.max_mu)
            sigma = np.clip(sigma, self.min_sigma, self.max_sigma)
            
            param = LognormalParameter(t0=t0, mu=mu, sigma=sigma, D=displacement)
            parameters.append(param)
        
        return parameters
    
    def _optimize_parameters(self,
                           velocity: np.ndarray,
                           time_points: np.ndarray,
                           initial_parameters: List[LognormalParameter]) -> List[LognormalParameter]:
        """
        Optimize lognormal parameters to fit observed velocity.
        
        Args:
            velocity: Observed velocity [n_points, 2]
            time_points: Time points
            initial_parameters: Initial parameter estimates
            
        Returns:
            optimized_parameters: Optimized parameters
        """
        def objective(param_vector):
            """Objective function for optimization."""
            parameters = self._vector_to_parameters(param_vector)
            predicted_velocity = self.generate_velocity_profile(parameters, time_points)
            
            # Mean squared error
            mse = np.mean((velocity - predicted_velocity) ** 2)
            return mse
        
        # Convert parameters to optimization vector
        initial_vector = self._parameters_to_vector(initial_parameters)
        
        # Parameter bounds
        bounds = []
        for _ in initial_parameters:
            bounds.extend([
                (0, time_points[-1]),           # t0 bounds
                (self.min_mu, self.max_mu),     # mu bounds  
                (self.min_sigma, self.max_sigma), # sigma bounds
                (-10, 10),                      # Dx bounds
                (-10, 10)                       # Dy bounds
            ])
        
        # Optimize
        try:
            result = minimize(objective, initial_vector, bounds=bounds, 
                            method='L-BFGS-B', options={'maxiter': self.max_iterations})
            
            if result.success:
                optimized_vector = result.x
            else:
                logger.warning("Optimization failed, using initial parameters")
                optimized_vector = initial_vector
        
        except Exception as e:
            logger.warning(f"Optimization error: {e}, using initial parameters")
            optimized_vector = initial_vector
        
        # Convert back to parameters
        optimized_parameters = self._vector_to_parameters(optimized_vector)
        
        return optimized_parameters
    
    def _parameters_to_vector(self, parameters: List[LognormalParameter]) -> np.ndarray:
        """Convert parameter list to optimization vector."""
        vector = []
        for param in parameters:
            vector.extend([param.t0, param.mu, param.sigma, param.D[0], param.D[1]])
        return np.array(vector)
    
    def _vector_to_parameters(self, vector: np.ndarray) -> List[LognormalParameter]:
        """Convert optimization vector to parameter list."""
        parameters = []
        n_params = len(vector) // 5  # 5 values per parameter
        
        for i in range(n_params):
            start_idx = i * 5
            t0 = vector[start_idx]
            mu = vector[start_idx + 1]
            sigma = vector[start_idx + 2]
            D = np.array([vector[start_idx + 3], vector[start_idx + 4]])
            
            param = LognormalParameter(t0=t0, mu=mu, sigma=sigma, D=D)
            parameters.append(param)
        
        return parameters
    
    def synthesize_handwriting(self,
                             text: str,
                             style_parameters: Dict[str, float] = None) -> Dict[str, np.ndarray]:
        """
        Synthesize handwriting trajectory for given text.
        
        Args:
            text: Text to synthesize
            style_parameters: Style parameters (speed, slant, size, etc.)
            
        Returns:
            trajectory: Generated handwriting trajectory
        """
        if style_parameters is None:
            style_parameters = self._get_default_style()
        
        # Generate character trajectories
        character_trajectories = []
        current_position = np.array([0.0, 0.0])
        
        for char in text:
            if char == ' ':
                # Space - just advance position
                current_position[0] += style_parameters.get('letter_spacing', 0.01) * 2
                continue
            
            # Generate character trajectory
            char_traj = self._generate_character_trajectory(char, current_position, style_parameters)
            character_trajectories.append(char_traj)
            
            # Update position for next character
            if len(char_traj['positions']) > 0:
                current_position = char_traj['positions'][-1].copy()
                current_position[0] += style_parameters.get('letter_spacing', 0.01)
        
        # Combine all character trajectories
        if character_trajectories:
            combined_trajectory = self._combine_trajectories(character_trajectories)
        else:
            # Empty trajectory
            combined_trajectory = {
                'time': np.array([0.0]),
                'positions': np.array([[0.0, 0.0]]),
                'velocities': np.array([[0.0, 0.0]]),
                'accelerations': np.array([[0.0, 0.0]])
            }
        
        return combined_trajectory
    
    def _get_default_style(self) -> Dict[str, float]:
        """Get default handwriting style parameters."""
        return {
            'letter_height': 0.01,      # 1 cm
            'letter_spacing': 0.005,    # 5 mm
            'slant_angle': 0.0,         # degrees
            'speed_factor': 1.0,        # relative speed
            'pressure_variation': 0.2,   # pressure variation
            'smoothness': 0.8           # trajectory smoothness
        }
    
    def _generate_character_trajectory(self,
                                     character: str,
                                     start_position: np.ndarray,
                                     style_parameters: Dict[str, float]) -> Dict[str, np.ndarray]:
        """
        Generate trajectory for a single character.
        
        Args:
            character: Character to generate
            start_position: Starting position
            style_parameters: Style parameters
            
        Returns:
            trajectory: Character trajectory
        """
        # Get character template (simplified - in practice would use font data)
        char_template = self._get_character_template(character)
        
        # Scale and position template
        height = style_parameters.get('letter_height', 0.01)
        slant = style_parameters.get('slant_angle', 0.0) * np.pi / 180
        
        # Apply transformations
        scaled_template = char_template * height
        
        # Apply slant
        if slant != 0:
            slant_matrix = np.array([[1, np.tan(slant)], [0, 1]])
            scaled_template = scaled_template @ slant_matrix.T
        
        # Translate to start position
        scaled_template += start_position
        
        # Generate lognormal parameters for this character
        parameters = self._create_character_parameters(scaled_template, style_parameters)
        
        # Generate trajectory
        duration = len(scaled_template) / self.sampling_rate * style_parameters.get('speed_factor', 1.0)
        trajectory = self.generate_trajectory(parameters, duration, start_position)
        
        return trajectory
    
    def _get_character_template(self, character: str) -> np.ndarray:
        """
        Get basic template for character shape.
        
        Args:
            character: Character to get template for
            
        Returns:
            template: Character template points [n_points, 2]
        """
        # Simplified character templates (in practice, would use proper font data)
        templates = {
            'a': np.array([[0, 0], [0.3, 0], [0.5, 0.5], [0.7, 0], [1, 0], [1, 1], [0.7, 1], [0.3, 0.5]]),
            'b': np.array([[0, 0], [0, 1], [0.6, 1], [0.8, 0.8], [0.6, 0.5], [0.8, 0.3], [0.6, 0], [0, 0]]),
            'c': np.array([[0.8, 0.2], [0.3, 0], [0, 0.5], [0.3, 1], [0.8, 0.8]]),
            # Add more characters as needed...
        }
        
        # Default template for unknown characters
        default_template = np.array([[0, 0], [0.5, 0.5], [1, 0]])
        
        template = templates.get(character.lower(), default_template)
        
        # Normalize to unit height
        if len(template) > 0:
            min_y = np.min(template[:, 1])
            max_y = np.max(template[:, 1])
            if max_y > min_y:
                template[:, 1] = (template[:, 1] - min_y) / (max_y - min_y)
        
        return template
    
    def _create_character_parameters(self,
                                   template: np.ndarray,
                                   style_parameters: Dict[str, float]) -> List[LognormalParameter]:
        """
        Create lognormal parameters for character template.
        
        Args:
            template: Character template points
            style_parameters: Style parameters
            
        Returns:
            parameters: Lognormal parameters for character
        """
        parameters = []
        
        if len(template) < 2:
            return parameters
        
        # Create parameters for each stroke segment
        n_segments = len(template) - 1
        segment_duration = 0.5 / n_segments  # Total character duration ~0.5s
        
        for i in range(n_segments):
            start_point = template[i]
            end_point = template[i + 1]
            displacement = end_point - start_point
            
            # Parameter estimation
            t0 = i * segment_duration
            mu = np.log(segment_duration / 2)  # Peak at middle of segment
            sigma = 0.3 * style_parameters.get('smoothness', 0.8)  # Adjust by smoothness
            
            param = LognormalParameter(t0=t0, mu=mu, sigma=sigma, D=displacement)
            parameters.append(param)
        
        return parameters
    
    def _combine_trajectories(self, trajectories: List[Dict[str, np.ndarray]]) -> Dict[str, np.ndarray]:
        """
        Combine multiple character trajectories into single trajectory.
        
        Args:
            trajectories: List of character trajectories
            
        Returns:
            combined_trajectory: Combined trajectory
        """
        if not trajectories:
            return {}
        
        # Concatenate all arrays
        all_times = []
        all_positions = []
        all_velocities = []
        all_accelerations = []
        
        current_time_offset = 0.0
        
        for traj in trajectories:
            # Offset time to be continuous
            times = traj['time'] + current_time_offset
            all_times.append(times)
            all_positions.append(traj['positions'])
            all_velocities.append(traj['velocities'])
            all_accelerations.append(traj['accelerations'])
            
            if len(times) > 0:
                current_time_offset = times[-1] + 0.01  # Small gap between characters
        
        # Concatenate arrays
        combined = {
            'time': np.concatenate(all_times) if all_times else np.array([]),
            'positions': np.concatenate(all_positions) if all_positions else np.array([]).reshape(0, 2),
            'velocities': np.concatenate(all_velocities) if all_velocities else np.array([]).reshape(0, 2),
            'accelerations': np.concatenate(all_accelerations) if all_accelerations else np.array([]).reshape(0, 2)
        }
        
        return combined
    
    def compute_snr(self, trajectory: Dict[str, np.ndarray]) -> float:
        """
        Compute Signal-to-Noise Ratio of generated trajectory.
        
        Args:
            trajectory: Generated trajectory
            
        Returns:
            snr: Signal-to-noise ratio in dB
        """
        velocity = trajectory['velocities']
        
        if len(velocity) == 0:
            return 0.0
        
        # Signal power (mean squared velocity)
        signal_power = np.mean(np.sum(velocity ** 2, axis=1))
        
        # Estimate noise power from high-frequency components
        # (simplified - in practice would use more sophisticated methods)
        velocity_diff = np.diff(velocity, axis=0)
        noise_power = np.mean(np.sum(velocity_diff ** 2, axis=1)) / 4  # Rough estimate
        
        if noise_power == 0:
            return float('inf')
        
        snr_db = 10 * np.log10(signal_power / noise_power)
        return snr_db