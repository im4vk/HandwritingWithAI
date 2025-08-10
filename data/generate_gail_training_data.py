#!/usr/bin/env python3
"""
GAIL Training Data Generator
===========================

This script generates expert demonstration data for training the GAIL
(Generative Adversarial Imitation Learning) model for robotic handwriting.
"""

import numpy as np
import json
import os
from typing import List, Dict, Any, Tuple
from pathlib import Path
import pickle


class GAILTrainingDataGenerator:
    """Generator for GAIL expert demonstration data."""
    
    def __init__(self, output_dir: str = "training"):
        """
        Initialize the GAIL data generator.
        
        Args:
            output_dir: Directory to save training data
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        # Load sample texts
        with open('sample_texts.json', 'r') as f:
            self.sample_texts = json.load(f)
        
        # Robot state dimensions
        self.state_dim = 15  # Position(3) + velocity(3) + orientation(4) + contact(1) + target(3) + error(1)
        self.action_dim = 4  # dx, dy, dz, pressure
        
        # Simulation parameters
        self.timestep = 0.001  # 1ms
        self.paper_height = 0.02  # Height above paper surface
    
    def generate_expert_trajectory(self, text: str, skill_level: str = "expert") -> Dict[str, Any]:
        """
        Generate an expert demonstration trajectory.
        
        Args:
            text: Text to write
            skill_level: Skill level ("novice", "intermediate", "expert")
            
        Returns:
            Dict containing states, actions, and metadata
        """
        # Skill-dependent parameters
        skill_params = {
            "novice": {"noise": 0.003, "smoothness": 0.5, "speed": 0.8},
            "intermediate": {"noise": 0.002, "smoothness": 0.7, "speed": 0.9},
            "expert": {"noise": 0.001, "smoothness": 0.9, "speed": 1.0}
        }
        
        params = skill_params.get(skill_level, skill_params["expert"])
        
        # Generate basic trajectory points (simplified character generation)
        trajectory_points = self.generate_text_trajectory(text, params)
        
        # Convert to state-action pairs
        states, actions = self.trajectory_to_state_action_pairs(trajectory_points, params)
        
        # Calculate rewards (for reference, GAIL doesn't use these directly)
        rewards = self.calculate_trajectory_rewards(states, actions)
        
        return {
            "text": text,
            "skill_level": skill_level,
            "states": np.array(states),
            "actions": np.array(actions),
            "rewards": np.array(rewards),
            "trajectory_points": np.array(trajectory_points),
            "metadata": {
                "num_steps": len(states),
                "total_time": len(states) * self.timestep,
                "skill_params": params,
                "trajectory_length": self.calculate_trajectory_length(trajectory_points)
            }
        }
    
    def generate_text_trajectory(self, text: str, params: Dict[str, float]) -> List[np.ndarray]:
        """Generate trajectory points for text."""
        points = []
        
        # Starting position
        current_pos = np.array([0.1, 0.15, self.paper_height])
        
        # Character dimensions
        char_width = 0.015
        char_height = 0.02
        
        for char in text:
            if char == ' ':
                # Space - just move position
                current_pos[0] += char_width * 0.8
            else:
                # Generate character trajectory
                char_points = self.generate_character_points(char, current_pos, char_width, char_height, params)
                points.extend(char_points)
                
                # Move to next character position
                current_pos[0] += char_width * 1.2
        
        return points
    
    def generate_character_points(self, char: str, start_pos: np.ndarray, 
                                width: float, height: float, params: Dict[str, float]) -> List[np.ndarray]:
        """Generate trajectory points for a single character."""
        points = []
        
        # Simplified character shapes (for demonstration)
        if char.upper() == 'A':
            # Triangle shape
            pts = [
                [0, 0], [0.5, 1], [1, 0], [0.3, 0.4], [0.7, 0.4]
            ]
        elif char.upper() == 'B':
            # B shape
            pts = [
                [0, 0], [0, 1], [0.7, 1], [0.7, 0.5], [0, 0.5], [0.7, 0.5], [0.7, 0], [0, 0]
            ]
        elif char.upper() == 'C':
            # C shape
            pts = [[1, 0.8], [0.2, 1], [0, 0.5], [0.2, 0], [1, 0.2]]
        else:
            # Default simple line
            pts = [[0, 0], [1, 1]]
        
        # Convert to actual coordinates and add smoothing
        for i, pt in enumerate(pts):
            x = start_pos[0] + pt[0] * width
            y = start_pos[1] + pt[1] * height
            z = start_pos[2]
            
            # Add noise based on skill level
            noise = params["noise"]
            x += np.random.normal(0, noise)
            y += np.random.normal(0, noise)
            z += np.random.normal(0, noise * 0.1)
            
            points.append(np.array([x, y, z]))
        
        # Add smoothing between points
        smoothed_points = self.smooth_trajectory(points, params["smoothness"])
        
        return smoothed_points
    
    def smooth_trajectory(self, points: List[np.ndarray], smoothness: float) -> List[np.ndarray]:
        """Apply smoothing to trajectory points."""
        if len(points) < 2:
            return points
        
        smoothed = []
        num_interpolated = max(1, int(10 * smoothness))
        
        for i in range(len(points) - 1):
            start = points[i]
            end = points[i + 1]
            
            for j in range(num_interpolated):
                t = j / num_interpolated
                # Cubic interpolation for smoother curves
                point = start + t * (end - start)
                smoothed.append(point)
        
        smoothed.append(points[-1])
        return smoothed
    
    def trajectory_to_state_action_pairs(self, trajectory: List[np.ndarray], 
                                       params: Dict[str, float]) -> Tuple[List[np.ndarray], List[np.ndarray]]:
        """Convert trajectory to state-action pairs for GAIL training."""
        states = []
        actions = []
        
        for i in range(len(trajectory) - 1):
            current_pos = trajectory[i]
            next_pos = trajectory[i + 1]
            
            # Calculate velocity
            if i > 0:
                velocity = (current_pos - trajectory[i-1]) / self.timestep
            else:
                velocity = np.zeros(3)
            
            # Mock orientation (quaternion)
            orientation = np.array([1, 0, 0, 0])  # Identity quaternion
            
            # Contact state (simplified - assume in contact when close to paper)
            is_in_contact = 1.0 if current_pos[2] < self.paper_height + 0.001 else 0.0
            
            # Target position (next point in trajectory)
            target_pos = next_pos
            
            # Position error
            position_error = np.linalg.norm(current_pos - target_pos)
            
            # Construct state vector
            state = np.concatenate([
                current_pos,      # Position (3)
                velocity,         # Velocity (3)
                orientation,      # Orientation (4)
                [is_in_contact],  # Contact (1)
                target_pos,       # Target (3)
                [position_error]  # Error (1)
            ])
            
            # Calculate action (relative movement + pressure)
            movement = next_pos - current_pos
            pressure = 0.7 if is_in_contact > 0.5 else 0.1  # High pressure when writing
            
            action = np.array([movement[0], movement[1], movement[2], pressure])
            
            states.append(state)
            actions.append(action)
        
        return states, actions
    
    def calculate_trajectory_rewards(self, states: List[np.ndarray], 
                                   actions: List[np.ndarray]) -> List[float]:
        """Calculate rewards for the trajectory (for reference)."""
        rewards = []
        
        for i, (state, action) in enumerate(zip(states, actions)):
            reward = 0.0
            
            # Reward for staying on target
            position_error = state[-1]  # Last element is position error
            reward -= position_error * 100  # Penalty for position error
            
            # Reward for smooth movement
            movement_magnitude = np.linalg.norm(action[:3])
            if movement_magnitude < 0.01:  # Reasonable movement
                reward += 1.0
            else:
                reward -= movement_magnitude * 10
            
            # Reward for appropriate pressure
            pressure = action[3]
            is_in_contact = state[7]  # Contact state
            
            if is_in_contact > 0.5:
                # Should have high pressure when in contact
                if 0.5 <= pressure <= 0.9:
                    reward += 2.0
                else:
                    reward -= abs(pressure - 0.7) * 5
            else:
                # Should have low pressure when not in contact
                if pressure < 0.3:
                    reward += 1.0
                else:
                    reward -= pressure * 3
            
            rewards.append(reward)
        
        return rewards
    
    def calculate_trajectory_length(self, trajectory: List[np.ndarray]) -> float:
        """Calculate total trajectory length."""
        if len(trajectory) < 2:
            return 0.0
        
        length = 0.0
        for i in range(len(trajectory) - 1):
            length += np.linalg.norm(trajectory[i + 1] - trajectory[i])
        
        return length
    
    def generate_diverse_dataset(self, num_demonstrations: int = 1000) -> List[Dict[str, Any]]:
        """Generate a diverse dataset of expert demonstrations."""
        demonstrations = []
        
        # Get all text samples
        all_texts = []
        for category, texts in self.sample_texts.items():
            all_texts.extend(texts)
        
        # Skill level distribution
        skill_levels = ["expert"] * 700 + ["intermediate"] * 250 + ["novice"] * 50
        
        for i in range(num_demonstrations):
            # Select random text and skill level
            text = np.random.choice(all_texts)
            skill_level = skill_levels[i % len(skill_levels)]
            
            # Generate demonstration
            demo = self.generate_expert_trajectory(text, skill_level)
            demo["demo_id"] = i
            
            demonstrations.append(demo)
            
            if (i + 1) % 100 == 0:
                print(f"Generated {i + 1}/{num_demonstrations} demonstrations")
        
        return demonstrations
    
    def save_gail_dataset(self, demonstrations: List[Dict[str, Any]], dataset_name: str):
        """Save dataset in GAIL-compatible format."""
        # Create directories
        dataset_dir = self.output_dir / dataset_name
        dataset_dir.mkdir(exist_ok=True)
        
        # Separate data by type
        all_states = []
        all_actions = []
        all_rewards = []
        metadata_list = []
        
        for demo in demonstrations:
            all_states.append(demo["states"])
            all_actions.append(demo["actions"])
            all_rewards.append(demo["rewards"])
            
            metadata_list.append({
                "demo_id": demo["demo_id"],
                "text": demo["text"],
                "skill_level": demo["skill_level"],
                "metadata": demo["metadata"]
            })
        
        # Combine all demonstrations
        combined_states = np.vstack(all_states)
        combined_actions = np.vstack(all_actions)
        combined_rewards = np.concatenate(all_rewards)
        
        # Save in multiple formats
        
        # 1. NumPy format (efficient for training)
        np.savez(dataset_dir / "expert_demonstrations.npz",
                states=combined_states,
                actions=combined_actions,
                rewards=combined_rewards)
        
        # 2. Pickle format (preserves structure)
        with open(dataset_dir / "demonstrations.pkl", "wb") as f:
            pickle.dump(demonstrations, f)
        
        # 3. Individual demonstration files
        demo_dir = dataset_dir / "individual_demos"
        demo_dir.mkdir(exist_ok=True)
        
        for demo in demonstrations:
            demo_file = demo_dir / f"demo_{demo['demo_id']:04d}.npz"
            np.savez(demo_file,
                    states=demo["states"],
                    actions=demo["actions"],
                    rewards=demo["rewards"],
                    trajectory=demo["trajectory_points"])
        
        # 4. Metadata JSON
        with open(dataset_dir / "metadata.json", "w") as f:
            json.dump(metadata_list, f, indent=2)
        
        # 5. Dataset statistics
        stats = {
            "num_demonstrations": len(demonstrations),
            "total_steps": len(combined_states),
            "state_dim": combined_states.shape[1],
            "action_dim": combined_actions.shape[1],
            "skill_level_distribution": {
                level: sum(1 for demo in demonstrations if demo["skill_level"] == level)
                for level in ["novice", "intermediate", "expert"]
            },
            "average_trajectory_length": float(np.mean([demo["metadata"]["trajectory_length"] 
                                                      for demo in demonstrations])),
            "average_steps_per_demo": float(np.mean([demo["metadata"]["num_steps"] 
                                                   for demo in demonstrations]))
        }
        
        with open(dataset_dir / "statistics.json", "w") as f:
            json.dump(stats, f, indent=2)
        
        print(f"\nGAIL dataset saved to {dataset_dir}")
        print(f"Total demonstrations: {stats['num_demonstrations']}")
        print(f"Total steps: {stats['total_steps']}")
        print(f"State dimension: {stats['state_dim']}")
        print(f"Action dimension: {stats['action_dim']}")
        print(f"Skill distribution: {stats['skill_level_distribution']}")
    
    def create_validation_set(self, demonstrations: List[Dict[str, Any]], 
                            validation_ratio: float = 0.2) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        """Split demonstrations into training and validation sets."""
        np.random.shuffle(demonstrations)
        
        split_idx = int(len(demonstrations) * (1 - validation_ratio))
        
        train_demos = demonstrations[:split_idx]
        val_demos = demonstrations[split_idx:]
        
        return train_demos, val_demos


def main():
    """Main function to generate GAIL training data."""
    print("Generating GAIL training data...")
    
    # Initialize generator
    generator = GAILTrainingDataGenerator()
    
    # Generate diverse demonstrations
    print("Generating expert demonstrations...")
    demonstrations = generator.generate_diverse_dataset(num_demonstrations=500)
    
    # Split into training and validation
    train_demos, val_demos = generator.create_validation_set(demonstrations)
    
    # Save datasets
    print("Saving training dataset...")
    generator.save_gail_dataset(train_demos, "gail_train")
    
    print("Saving validation dataset...")
    generator.save_gail_dataset(val_demos, "gail_validation")
    
    # Save complete dataset
    print("Saving complete dataset...")
    generator.save_gail_dataset(demonstrations, "gail_complete")
    
    print("\nGAIL training data generation completed!")
    print(f"Training demonstrations: {len(train_demos)}")
    print(f"Validation demonstrations: {len(val_demos)}")
    print(f"Total demonstrations: {len(demonstrations)}")


if __name__ == "__main__":
    main()