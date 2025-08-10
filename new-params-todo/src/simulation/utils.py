"""
Utility functions for robotic handwriting simulation.

This module provides helper functions for simulation setup, state management,
trajectory processing, and performance monitoring.
"""

import numpy as np
from typing import Dict, Any, List, Optional, Tuple, Union
import time
import logging
from pathlib import Path

from .base_environment import BaseEnvironment
from .handwriting_environment import HandwritingEnvironment
from .environment_config import EnvironmentConfig

logger = logging.getLogger(__name__)


def setup_simulation(config: Union[Dict[str, Any], EnvironmentConfig, str]) -> HandwritingEnvironment:
    """
    Setup and initialize a handwriting simulation environment.
    
    Args:
        config: Configuration dictionary, EnvironmentConfig object, or path to config file
        
    Returns:
        HandwritingEnvironment: Initialized simulation environment
    """
    # Handle different config input types
    if isinstance(config, str):
        # Load from file
        config_obj = EnvironmentConfig.from_yaml(config)
    elif isinstance(config, dict):
        # Create from dictionary
        config_obj = EnvironmentConfig.from_dict(config)
    elif isinstance(config, EnvironmentConfig):
        # Use directly
        config_obj = config
    else:
        raise ValueError(f"Invalid config type: {type(config)}")
    
    # Create environment
    env = HandwritingEnvironment(config_obj.to_dict())
    
    # Initialize environment
    if not env.initialize():
        raise RuntimeError("Failed to initialize simulation environment")
    
    logger.info("Simulation environment setup completed")
    return env


def reset_simulation(env: BaseEnvironment) -> np.ndarray:
    """
    Reset simulation environment to initial state.
    
    Args:
        env: Simulation environment to reset
        
    Returns:
        np.ndarray: Initial observation after reset
    """
    try:
        observation = env.reset()
        logger.info("Simulation environment reset successfully")
        return observation
    except Exception as e:
        logger.error(f"Failed to reset simulation: {e}")
        raise


def step_simulation(env: BaseEnvironment, action: np.ndarray) -> Tuple[np.ndarray, float, bool, Dict[str, Any]]:
    """
    Execute one simulation step.
    
    Args:
        env: Simulation environment
        action: Action to execute
        
    Returns:
        Tuple of (observation, reward, done, info)
    """
    try:
        return env.step(action)
    except Exception as e:
        logger.error(f"Failed to step simulation: {e}")
        raise


def get_simulation_state(env: BaseEnvironment) -> Dict[str, Any]:
    """
    Get current simulation state.
    
    Args:
        env: Simulation environment
        
    Returns:
        Dict containing complete simulation state
    """
    try:
        return env.get_state()
    except Exception as e:
        logger.error(f"Failed to get simulation state: {e}")
        raise


def set_simulation_state(env: BaseEnvironment, state: Dict[str, Any]) -> bool:
    """
    Set simulation state.
    
    Args:
        env: Simulation environment
        state: State dictionary to set
        
    Returns:
        bool: True if state was set successfully
    """
    try:
        return env.set_state(state)
    except Exception as e:
        logger.error(f"Failed to set simulation state: {e}")
        return False


def run_episode(env: BaseEnvironment, 
                policy_fn: callable = None,
                max_steps: Optional[int] = None,
                render: bool = False,
                record_trajectory: bool = True) -> Dict[str, Any]:
    """
    Run a complete episode in the simulation.
    
    Args:
        env: Simulation environment
        policy_fn: Policy function that takes observation and returns action
        max_steps: Maximum number of steps (uses env default if None)
        render: Whether to render during episode
        record_trajectory: Whether to record trajectory data
        
    Returns:
        Dict containing episode results and statistics
    """
    if policy_fn is None:
        # Default random policy
        def policy_fn(obs):
            # Simple random action within bounds
            action = np.random.uniform(-0.005, 0.005, size=4)
            action[3] = np.random.uniform(0.0, 0.8)  # Pressure
            return action
    
    # Reset environment
    observation = env.reset()
    
    # Episode tracking
    total_reward = 0.0
    episode_length = 0
    trajectory = [] if record_trajectory else None
    actions = [] if record_trajectory else None
    rewards = [] if record_trajectory else None
    
    done = False
    start_time = time.time()
    
    max_steps = max_steps or env.max_episode_steps
    
    try:
        while not done and episode_length < max_steps:
            # Get action from policy
            action = policy_fn(observation)
            
            # Execute step
            next_observation, reward, done, info = env.step(action)
            
            # Record data
            total_reward += reward
            episode_length += 1
            
            if record_trajectory:
                trajectory.append(observation.copy())
                actions.append(action.copy())
                rewards.append(reward)
            
            # Render if requested
            if render and hasattr(env, 'render'):
                env.render()
            
            observation = next_observation
        
        # Record final observation
        if record_trajectory:
            trajectory.append(observation.copy())
        
        episode_time = time.time() - start_time
        
        # Compile episode results
        results = {
            'total_reward': total_reward,
            'episode_length': episode_length,
            'episode_time': episode_time,
            'success': done and episode_length < max_steps,
            'final_observation': observation,
            'steps_per_second': episode_length / episode_time if episode_time > 0 else 0
        }
        
        if record_trajectory:
            results.update({
                'trajectory': np.array(trajectory),
                'actions': np.array(actions),
                'rewards': np.array(rewards)
            })
        
        # Add environment-specific metrics if available
        if hasattr(env, 'get_handwriting_quality_metrics'):
            results['quality_metrics'] = env.get_handwriting_quality_metrics()
        
        logger.info(f"Episode completed: {episode_length} steps, reward={total_reward:.3f}")
        return results
        
    except Exception as e:
        logger.error(f"Episode failed: {e}")
        raise


def evaluate_policy(env: BaseEnvironment,
                   policy_fn: callable,
                   num_episodes: int = 10,
                   max_steps_per_episode: Optional[int] = None,
                   save_results: bool = False,
                   results_path: Optional[str] = None) -> Dict[str, Any]:
    """
    Evaluate a policy over multiple episodes.
    
    Args:
        env: Simulation environment
        policy_fn: Policy function to evaluate
        num_episodes: Number of episodes to run
        max_steps_per_episode: Maximum steps per episode
        save_results: Whether to save detailed results
        results_path: Path to save results (if save_results=True)
        
    Returns:
        Dict containing evaluation statistics
    """
    episode_results = []
    
    for episode in range(num_episodes):
        logger.info(f"Running evaluation episode {episode + 1}/{num_episodes}")
        
        try:
            result = run_episode(
                env, 
                policy_fn, 
                max_steps_per_episode, 
                render=False,
                record_trajectory=save_results
            )
            episode_results.append(result)
            
        except Exception as e:
            logger.error(f"Episode {episode + 1} failed: {e}")
            continue
    
    if not episode_results:
        raise RuntimeError("All evaluation episodes failed")
    
    # Compute statistics
    rewards = [r['total_reward'] for r in episode_results]
    lengths = [r['episode_length'] for r in episode_results]
    times = [r['episode_time'] for r in episode_results]
    success_rate = np.mean([r['success'] for r in episode_results])
    
    stats = {
        'num_episodes': len(episode_results),
        'mean_reward': np.mean(rewards),
        'std_reward': np.std(rewards),
        'min_reward': np.min(rewards),
        'max_reward': np.max(rewards),
        'mean_episode_length': np.mean(lengths),
        'std_episode_length': np.std(lengths),
        'mean_episode_time': np.mean(times),
        'success_rate': success_rate,
        'total_steps': np.sum(lengths),
        'total_time': np.sum(times)
    }
    
    # Add quality metrics if available
    quality_metrics = []
    for result in episode_results:
        if 'quality_metrics' in result:
            quality_metrics.append(result['quality_metrics'])
    
    if quality_metrics:
        # Aggregate quality metrics
        quality_stats = {}
        for key in quality_metrics[0].keys():
            values = [qm[key] for qm in quality_metrics if key in qm]
            if values:
                quality_stats[f'mean_{key}'] = np.mean(values)
                quality_stats[f'std_{key}'] = np.std(values)
        
        stats['quality_metrics'] = quality_stats
    
    # Save results if requested
    if save_results and results_path:
        save_evaluation_results(stats, episode_results, results_path)
    
    logger.info(f"Evaluation completed: {stats['mean_reward']:.3f} ± {stats['std_reward']:.3f} reward")
    return stats


def save_evaluation_results(stats: Dict[str, Any], 
                          episode_results: List[Dict[str, Any]], 
                          results_path: str):
    """
    Save evaluation results to file.
    
    Args:
        stats: Evaluation statistics
        episode_results: Individual episode results
        results_path: Path to save results
    """
    try:
        import json
        from pathlib import Path
        
        results_path = Path(results_path)
        results_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Prepare data for saving (convert numpy arrays to lists)
        def convert_for_json(obj):
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, dict):
                return {k: convert_for_json(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_for_json(item) for item in obj]
            return obj
        
        data = {
            'statistics': convert_for_json(stats),
            'episode_results': convert_for_json(episode_results),
            'timestamp': time.time()
        }
        
        with open(results_path, 'w') as f:
            json.dump(data, f, indent=2)
        
        logger.info(f"Evaluation results saved to {results_path}")
        
    except Exception as e:
        logger.error(f"Failed to save evaluation results: {e}")


def load_evaluation_results(results_path: str) -> Dict[str, Any]:
    """
    Load evaluation results from file.
    
    Args:
        results_path: Path to results file
        
    Returns:
        Dict containing loaded results
    """
    try:
        import json
        
        with open(results_path, 'r') as f:
            data = json.load(f)
        
        logger.info(f"Evaluation results loaded from {results_path}")
        return data
        
    except Exception as e:
        logger.error(f"Failed to load evaluation results: {e}")
        raise


def create_simple_trajectory(start: np.ndarray, 
                           end: np.ndarray, 
                           num_points: int = 100,
                           trajectory_type: str = "linear") -> List[np.ndarray]:
    """
    Create a simple trajectory between two points.
    
    Args:
        start: Starting position [x, y, z]
        end: Ending position [x, y, z]
        num_points: Number of trajectory points
        trajectory_type: Type of trajectory ("linear", "smooth", "arc")
        
    Returns:
        List of 3D trajectory points
    """
    trajectory = []
    
    if trajectory_type == "linear":
        # Simple linear interpolation
        for i in range(num_points):
            t = i / (num_points - 1)
            point = start + t * (end - start)
            trajectory.append(point)
    
    elif trajectory_type == "smooth":
        # Smooth trajectory using cubic interpolation
        # Add intermediate control points for smoothness
        mid_point = (start + end) / 2
        control1 = start + 0.3 * (mid_point - start)
        control2 = end - 0.3 * (end - mid_point)
        
        for i in range(num_points):
            t = i / (num_points - 1)
            # Cubic Bezier curve
            point = ((1-t)**3 * start + 
                    3*(1-t)**2*t * control1 + 
                    3*(1-t)*t**2 * control2 + 
                    t**3 * end)
            trajectory.append(point)
    
    elif trajectory_type == "arc":
        # Arc trajectory (semicircle)
        center = (start + end) / 2
        radius = np.linalg.norm(end - start) / 2
        direction = (end - start) / np.linalg.norm(end - start)
        perpendicular = np.array([-direction[1], direction[0], 0])
        
        for i in range(num_points):
            angle = np.pi * i / (num_points - 1)
            point = (center + 
                    radius * np.cos(angle) * direction + 
                    radius * np.sin(angle) * perpendicular)
            trajectory.append(point)
    
    else:
        raise ValueError(f"Unknown trajectory type: {trajectory_type}")
    
    return trajectory


def analyze_trajectory(trajectory: np.ndarray,
                      timestep: float = 0.001) -> Dict[str, Any]:
    """
    Analyze trajectory characteristics.
    
    Args:
        trajectory: Array of trajectory points [N, 3]
        timestep: Time between trajectory points
        
    Returns:
        Dict containing trajectory analysis
    """
    if len(trajectory) < 2:
        return {}
    
    # Compute velocities
    velocities = np.diff(trajectory, axis=0) / timestep
    speeds = np.linalg.norm(velocities, axis=1)
    
    # Compute accelerations
    if len(velocities) > 1:
        accelerations = np.diff(velocities, axis=0) / timestep
        acceleration_magnitudes = np.linalg.norm(accelerations, axis=1)
    else:
        accelerations = np.array([])
        acceleration_magnitudes = np.array([])
    
    # Compute curvature (simplified)
    curvatures = []
    if len(trajectory) >= 3:
        for i in range(1, len(trajectory) - 1):
            p1, p2, p3 = trajectory[i-1], trajectory[i], trajectory[i+1]
            v1 = p2 - p1
            v2 = p3 - p2
            if np.linalg.norm(v1) > 0 and np.linalg.norm(v2) > 0:
                angle = np.arccos(np.clip(np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2)), -1, 1))
                curvature = angle / np.linalg.norm(v1)
                curvatures.append(curvature)
    
    # Total distance
    distances = np.linalg.norm(np.diff(trajectory, axis=0), axis=1)
    total_distance = np.sum(distances)
    
    # Duration
    total_time = len(trajectory) * timestep
    
    analysis = {
        'total_distance': total_distance,
        'total_time': total_time,
        'mean_speed': np.mean(speeds) if len(speeds) > 0 else 0,
        'max_speed': np.max(speeds) if len(speeds) > 0 else 0,
        'speed_std': np.std(speeds) if len(speeds) > 0 else 0,
        'mean_acceleration': np.mean(acceleration_magnitudes) if len(acceleration_magnitudes) > 0 else 0,
        'max_acceleration': np.max(acceleration_magnitudes) if len(acceleration_magnitudes) > 0 else 0,
        'mean_curvature': np.mean(curvatures) if curvatures else 0,
        'max_curvature': np.max(curvatures) if curvatures else 0,
        'smoothness': 1.0 / (1.0 + np.var(acceleration_magnitudes)) if len(acceleration_magnitudes) > 0 else 0,
        'num_points': len(trajectory)
    }
    
    return analysis


def create_benchmark_trajectory(benchmark_type: str = "figure_eight",
                              scale: float = 0.05,
                              center: np.ndarray = None,
                              num_points: int = 200) -> List[np.ndarray]:
    """
    Create benchmark trajectories for evaluation.
    
    Args:
        benchmark_type: Type of benchmark ("figure_eight", "spiral", "square", "circle")
        scale: Scale factor for the trajectory
        center: Center position for the trajectory
        num_points: Number of trajectory points
        
    Returns:
        List of 3D trajectory points
    """
    if center is None:
        center = np.array([0.5, 0.0, 0.02])
    
    trajectory = []
    
    if benchmark_type == "figure_eight":
        for i in range(num_points):
            t = 2 * np.pi * i / num_points
            x = center[0] + scale * np.sin(t)
            y = center[1] + scale * np.sin(2*t)
            z = center[2]
            trajectory.append(np.array([x, y, z]))
    
    elif benchmark_type == "spiral":
        for i in range(num_points):
            t = 4 * np.pi * i / num_points
            r = scale * i / num_points
            x = center[0] + r * np.cos(t)
            y = center[1] + r * np.sin(t)
            z = center[2]
            trajectory.append(np.array([x, y, z]))
    
    elif benchmark_type == "square":
        side_points = num_points // 4
        # Bottom side
        for i in range(side_points):
            t = i / side_points
            x = center[0] - scale + 2 * scale * t
            y = center[1] - scale
            z = center[2]
            trajectory.append(np.array([x, y, z]))
        
        # Right side
        for i in range(side_points):
            t = i / side_points
            x = center[0] + scale
            y = center[1] - scale + 2 * scale * t
            z = center[2]
            trajectory.append(np.array([x, y, z]))
        
        # Top side
        for i in range(side_points):
            t = i / side_points
            x = center[0] + scale - 2 * scale * t
            y = center[1] + scale
            z = center[2]
            trajectory.append(np.array([x, y, z]))
        
        # Left side
        for i in range(num_points - 3 * side_points):
            t = i / (num_points - 3 * side_points)
            x = center[0] - scale
            y = center[1] + scale - 2 * scale * t
            z = center[2]
            trajectory.append(np.array([x, y, z]))
    
    elif benchmark_type == "circle":
        for i in range(num_points):
            t = 2 * np.pi * i / num_points
            x = center[0] + scale * np.cos(t)
            y = center[1] + scale * np.sin(t)
            z = center[2]
            trajectory.append(np.array([x, y, z]))
    
    else:
        raise ValueError(f"Unknown benchmark type: {benchmark_type}")
    
    return trajectory


def compute_trajectory_similarity(traj1: np.ndarray, 
                                traj2: np.ndarray,
                                method: str = "dtw") -> float:
    """
    Compute similarity between two trajectories.
    
    Args:
        traj1: First trajectory [N, 3]
        traj2: Second trajectory [M, 3]
        method: Similarity method ("dtw", "frechet", "hausdorff")
        
    Returns:
        float: Similarity score (higher = more similar)
    """
    if method == "dtw":
        # Dynamic Time Warping (simplified)
        return compute_dtw_similarity(traj1, traj2)
    
    elif method == "frechet":
        # Fréchet distance (simplified)
        return compute_frechet_similarity(traj1, traj2)
    
    elif method == "hausdorff":
        # Hausdorff distance
        return compute_hausdorff_similarity(traj1, traj2)
    
    else:
        raise ValueError(f"Unknown similarity method: {method}")


def compute_dtw_similarity(traj1: np.ndarray, traj2: np.ndarray) -> float:
    """Compute DTW similarity between trajectories."""
    # Simplified DTW implementation
    n, m = len(traj1), len(traj2)
    dtw_matrix = np.full((n + 1, m + 1), np.inf)
    dtw_matrix[0, 0] = 0
    
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            cost = np.linalg.norm(traj1[i-1] - traj2[j-1])
            dtw_matrix[i, j] = cost + min(
                dtw_matrix[i-1, j],      # Insertion
                dtw_matrix[i, j-1],      # Deletion
                dtw_matrix[i-1, j-1]     # Match
            )
    
    dtw_distance = dtw_matrix[n, m]
    # Convert to similarity (0-1 range)
    max_possible_distance = np.sqrt(3) * max(n, m)  # Rough estimate
    similarity = 1.0 / (1.0 + dtw_distance / max_possible_distance)
    return similarity


def compute_frechet_similarity(traj1: np.ndarray, traj2: np.ndarray) -> float:
    """Compute simplified Fréchet similarity."""
    # Simplified implementation - just use endpoint distances
    if len(traj1) == 0 or len(traj2) == 0:
        return 0.0
    
    start_dist = np.linalg.norm(traj1[0] - traj2[0])
    end_dist = np.linalg.norm(traj1[-1] - traj2[-1])
    
    # Add some intermediate point comparisons
    mid1 = len(traj1) // 2
    mid2 = len(traj2) // 2
    mid_dist = np.linalg.norm(traj1[mid1] - traj2[mid2])
    
    avg_dist = (start_dist + end_dist + mid_dist) / 3
    similarity = 1.0 / (1.0 + avg_dist)
    return similarity


def compute_hausdorff_similarity(traj1: np.ndarray, traj2: np.ndarray) -> float:
    """Compute Hausdorff similarity."""
    if len(traj1) == 0 or len(traj2) == 0:
        return 0.0
    
    # Directed Hausdorff distances
    def directed_hausdorff(A, B):
        return max(min(np.linalg.norm(a - b) for b in B) for a in A)
    
    h1 = directed_hausdorff(traj1, traj2)
    h2 = directed_hausdorff(traj2, traj1)
    hausdorff_dist = max(h1, h2)
    
    similarity = 1.0 / (1.0 + hausdorff_dist)
    return similarity