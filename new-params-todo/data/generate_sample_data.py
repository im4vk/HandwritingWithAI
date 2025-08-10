#!/usr/bin/env python3
"""
Sample Data Generation Script for Robotic Handwriting
====================================================

This script generates various types of sample data for training and testing
the robotic handwriting system, including synthetic trajectories, text samples,
and training datasets.
"""

import numpy as np
import json
import csv
import os
from typing import List, Dict, Any, Tuple
import matplotlib.pyplot as plt
from pathlib import Path


class HandwritingDataGenerator:
    """Generator for synthetic handwriting data and trajectories."""
    
    def __init__(self, output_dir: str = "datasets"):
        """
        Initialize the data generator.
        
        Args:
            output_dir: Directory to save generated data
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        # Character definitions (simplified stroke patterns)
        self.character_strokes = self.define_character_strokes()
        
        # Trajectory parameters
        self.paper_size = [0.21, 0.297]  # A4 size in meters
        self.writing_height = 0.02  # 2cm character height
        self.writing_width = 0.015  # 1.5cm character width
        self.line_spacing = 0.025  # 2.5cm line spacing
    
    def define_character_strokes(self) -> Dict[str, List[List[Tuple[float, float]]]]:
        """Define basic stroke patterns for characters."""
        strokes = {
            'A': [[(0, 0), (0.5, 1), (1, 0)], [(0.3, 0.4), (0.7, 0.4)]],
            'B': [[(0, 0), (0, 1), (0.7, 1), (0.7, 0.5), (0, 0.5)], [(0, 0.5), (0.7, 0.5), (0.7, 0), (0, 0)]],
            'C': [[(1, 0.8), (0.2, 1), (0, 0.5), (0.2, 0), (1, 0.2)]],
            'D': [[(0, 0), (0, 1), (0.8, 1), (1, 0.8), (1, 0.2), (0.8, 0), (0, 0)]],
            'E': [[(1, 0), (0, 0), (0, 1), (0.8, 1)], [(0, 0.5), (0.6, 0.5)]],
            'F': [[(0, 0), (0, 1), (0.8, 1)], [(0, 0.5), (0.6, 0.5)]],
            'G': [[(1, 0.8), (0.2, 1), (0, 0.5), (0.2, 0), (1, 0.2), (1, 0.5), (0.7, 0.5)]],
            'H': [[(0, 0), (0, 1)], [(1, 0), (1, 1)], [(0, 0.5), (1, 0.5)]],
            'I': [[(0.2, 0), (0.8, 0)], [(0.5, 0), (0.5, 1)], [(0.2, 1), (0.8, 1)]],
            'J': [[(0, 0.2), (0.2, 0), (0.8, 0), (1, 0.2), (1, 1), (0.8, 1)]],
            'K': [[(0, 0), (0, 1)], [(1, 1), (0, 0.5), (1, 0)]],
            'L': [[(0, 1), (0, 0), (1, 0)]],
            'M': [[(0, 0), (0, 1), (0.5, 0.5), (1, 1), (1, 0)]],
            'N': [[(0, 0), (0, 1), (1, 0), (1, 1)]],
            'O': [[(0.2, 1), (0, 0.8), (0, 0.2), (0.2, 0), (0.8, 0), (1, 0.2), (1, 0.8), (0.8, 1), (0.2, 1)]],
            'P': [[(0, 0), (0, 1), (0.8, 1), (1, 0.8), (1, 0.6), (0.8, 0.5), (0, 0.5)]],
            'Q': [[(0.2, 1), (0, 0.8), (0, 0.2), (0.2, 0), (0.8, 0), (1, 0.2), (1, 0.8), (0.8, 1), (0.2, 1)], [(0.6, 0.3), (1, 0)]],
            'R': [[(0, 0), (0, 1), (0.8, 1), (1, 0.8), (1, 0.6), (0.8, 0.5), (0, 0.5)], [(0.5, 0.5), (1, 0)]],
            'S': [[(1, 0.8), (0.8, 1), (0.2, 1), (0, 0.8), (0.2, 0.6), (0.8, 0.4), (1, 0.2), (0.8, 0), (0.2, 0), (0, 0.2)]],
            'T': [[(0, 1), (1, 1)], [(0.5, 1), (0.5, 0)]],
            'U': [[(0, 1), (0, 0.2), (0.2, 0), (0.8, 0), (1, 0.2), (1, 1)]],
            'V': [[(0, 1), (0.5, 0), (1, 1)]],
            'W': [[(0, 1), (0.25, 0), (0.5, 0.5), (0.75, 0), (1, 1)]],
            'X': [[(0, 1), (1, 0)], [(0, 0), (1, 1)]],
            'Y': [[(0, 1), (0.5, 0.5), (1, 1)], [(0.5, 0.5), (0.5, 0)]],
            'Z': [[(0, 1), (1, 1), (0, 0), (1, 0)]],
            ' ': [],  # Space character - no strokes
        }
        return strokes
    
    def generate_character_trajectory(self, char: str, start_pos: np.ndarray, 
                                    noise_level: float = 0.001) -> List[np.ndarray]:
        """
        Generate trajectory for a single character.
        
        Args:
            char: Character to generate
            start_pos: Starting position [x, y, z]
            noise_level: Amount of noise to add for realism
            
        Returns:
            List of 3D trajectory points
        """
        if char.upper() not in self.character_strokes:
            return []
        
        strokes = self.character_strokes[char.upper()]
        trajectory = []
        
        for stroke in strokes:
            # Convert normalized coordinates to actual positions
            stroke_points = []
            for point in stroke:
                # Scale to character size and offset by start position
                x = start_pos[0] + point[0] * self.writing_width
                y = start_pos[1] + point[1] * self.writing_height
                z = start_pos[2]
                
                # Add small amount of noise for realism
                if noise_level > 0:
                    x += np.random.normal(0, noise_level)
                    y += np.random.normal(0, noise_level)
                    z += np.random.normal(0, noise_level * 0.1)  # Less noise in Z
                
                stroke_points.append(np.array([x, y, z]))
            
            # Interpolate between stroke points for smoother trajectory
            if len(stroke_points) > 1:
                interpolated = self.interpolate_stroke(stroke_points)
                trajectory.extend(interpolated)
            
            # Add pen lift between strokes (move to higher Z)
            if len(strokes) > 1:
                lift_point = stroke_points[-1].copy()
                lift_point[2] += 0.005  # 5mm pen lift
                trajectory.append(lift_point)
        
        return trajectory
    
    def interpolate_stroke(self, points: List[np.ndarray], density: int = 10) -> List[np.ndarray]:
        """Interpolate between stroke points for smoother trajectory."""
        if len(points) < 2:
            return points
        
        interpolated = []
        for i in range(len(points) - 1):
            start = points[i]
            end = points[i + 1]
            
            for j in range(density):
                t = j / density
                point = start + t * (end - start)
                interpolated.append(point)
        
        # Add final point
        interpolated.append(points[-1])
        return interpolated
    
    def generate_word_trajectory(self, word: str, start_pos: np.ndarray, 
                               noise_level: float = 0.001) -> Tuple[List[np.ndarray], List[bool]]:
        """
        Generate trajectory for a word.
        
        Args:
            word: Word to generate
            start_pos: Starting position [x, y, z]
            noise_level: Amount of noise to add
            
        Returns:
            Tuple of (trajectory points, contact states)
        """
        trajectory = []
        contact_states = []
        current_pos = start_pos.copy()
        
        for char in word:
            char_trajectory = self.generate_character_trajectory(char, current_pos, noise_level)
            
            if char_trajectory:
                trajectory.extend(char_trajectory)
                # Assume pen is in contact for actual character strokes
                contact_states.extend([True] * len(char_trajectory))
                
                # Update position for next character
                current_pos[0] += self.writing_width * 1.2  # Character spacing
            else:
                # Space character - just move position
                current_pos[0] += self.writing_width * 0.8
        
        return trajectory, contact_states
    
    def generate_sentence_trajectory(self, sentence: str, start_pos: np.ndarray = None,
                                   noise_level: float = 0.001) -> Dict[str, Any]:
        """
        Generate trajectory for a complete sentence.
        
        Args:
            sentence: Sentence to generate
            start_pos: Starting position [x, y, z]
            noise_level: Amount of noise to add
            
        Returns:
            Dict containing trajectory data
        """
        if start_pos is None:
            start_pos = np.array([0.1, 0.1, 0.02])  # Default starting position
        
        words = sentence.split()
        full_trajectory = []
        full_contacts = []
        word_boundaries = []
        current_pos = start_pos.copy()
        
        for word_idx, word in enumerate(words):
            word_start_idx = len(full_trajectory)
            
            word_trajectory, word_contacts = self.generate_word_trajectory(
                word, current_pos, noise_level
            )
            
            full_trajectory.extend(word_trajectory)
            full_contacts.extend(word_contacts)
            
            word_end_idx = len(full_trajectory)
            word_boundaries.append({
                'word': word,
                'start_idx': word_start_idx,
                'end_idx': word_end_idx
            })
            
            # Move to next word position (add word spacing)
            if word_idx < len(words) - 1:
                current_pos[0] += len(word) * self.writing_width * 1.2 + self.writing_width * 0.5
                
                # Check if we need to wrap to next line
                if current_pos[0] > self.paper_size[0] - 0.05:  # 5cm margin
                    current_pos[0] = start_pos[0]
                    current_pos[1] += self.line_spacing
        
        return {
            'trajectory': np.array(full_trajectory),
            'contact_states': np.array(full_contacts),
            'word_boundaries': word_boundaries,
            'sentence': sentence,
            'start_position': start_pos,
            'noise_level': noise_level,
            'metadata': {
                'num_points': len(full_trajectory),
                'num_words': len(words),
                'writing_time': len(full_trajectory) * 0.01,  # Assume 10ms per point
                'paper_size': self.paper_size
            }
        }
    
    def generate_synthetic_dataset(self, num_samples: int = 100) -> List[Dict[str, Any]]:
        """Generate a dataset of synthetic handwriting samples."""
        samples = []
        
        # Sample sentences for generation
        sentences = [
            "Hello World",
            "The quick brown fox jumps over the lazy dog",
            "Artificial Intelligence",
            "Robotic Handwriting System",
            "Machine Learning",
            "Neural Networks",
            "Deep Learning",
            "Computer Vision",
            "Natural Language Processing",
            "Robotics and Automation",
            "Human Robot Interaction",
            "Biomechanical Modeling",
            "Trajectory Generation",
            "Motion Planning",
            "Control Systems",
            "Python Programming",
            "Scientific Computing",
            "Data Science",
            "Research and Development",
            "Innovation Technology"
        ]
        
        for i in range(num_samples):
            # Select random sentence
            sentence = np.random.choice(sentences)
            
            # Random starting position
            start_x = np.random.uniform(0.05, 0.15)
            start_y = np.random.uniform(0.05, 0.2)
            start_z = 0.02 + np.random.normal(0, 0.001)
            start_pos = np.array([start_x, start_y, start_z])
            
            # Random noise level
            noise_level = np.random.uniform(0.0005, 0.002)
            
            # Generate trajectory
            sample = self.generate_sentence_trajectory(sentence, start_pos, noise_level)
            sample['sample_id'] = i
            sample['timestamp'] = i * 1000  # Simulated timestamp
            
            samples.append(sample)
        
        return samples
    
    def generate_benchmark_trajectories(self) -> Dict[str, Dict[str, Any]]:
        """Generate standard benchmark trajectories for evaluation."""
        benchmarks = {}
        
        # 1. Simple line
        line_points = []
        for i in range(100):
            t = i / 99
            x = 0.1 + t * 0.1  # 10cm line
            y = 0.1
            z = 0.02
            line_points.append([x, y, z])
        
        benchmarks['line'] = {
            'trajectory': np.array(line_points),
            'contact_states': np.ones(100, dtype=bool),
            'description': 'Simple horizontal line',
            'difficulty': 'easy'
        }
        
        # 2. Circle
        circle_points = []
        radius = 0.03  # 3cm radius
        center = np.array([0.15, 0.15, 0.02])
        for i in range(200):
            angle = 2 * np.pi * i / 199
            x = center[0] + radius * np.cos(angle)
            y = center[1] + radius * np.sin(angle)
            z = center[2]
            circle_points.append([x, y, z])
        
        benchmarks['circle'] = {
            'trajectory': np.array(circle_points),
            'contact_states': np.ones(200, dtype=bool),
            'description': 'Perfect circle',
            'difficulty': 'medium'
        }
        
        # 3. Figure-8
        figure8_points = []
        scale = 0.025  # 2.5cm scale
        center = np.array([0.15, 0.15, 0.02])
        for i in range(300):
            t = 2 * np.pi * i / 299
            x = center[0] + scale * np.sin(t)
            y = center[1] + scale * np.sin(2*t)
            z = center[2]
            figure8_points.append([x, y, z])
        
        benchmarks['figure8'] = {
            'trajectory': np.array(figure8_points),
            'contact_states': np.ones(300, dtype=bool),
            'description': 'Figure-8 pattern',
            'difficulty': 'hard'
        }
        
        # 4. Spiral
        spiral_points = []
        center = np.array([0.15, 0.15, 0.02])
        for i in range(400):
            t = 4 * np.pi * i / 399
            r = 0.01 + (0.03 * i / 399)  # Expanding spiral
            x = center[0] + r * np.cos(t)
            y = center[1] + r * np.sin(t)
            z = center[2]
            spiral_points.append([x, y, z])
        
        benchmarks['spiral'] = {
            'trajectory': np.array(spiral_points),
            'contact_states': np.ones(400, dtype=bool),
            'description': 'Expanding spiral',
            'difficulty': 'hard'
        }
        
        return benchmarks
    
    def save_dataset(self, dataset: List[Dict[str, Any]], filename: str):
        """Save dataset to file in multiple formats."""
        # Save as JSON
        json_data = []
        for sample in dataset:
            json_sample = sample.copy()
            # Convert numpy arrays to lists for JSON serialization
            json_sample['trajectory'] = sample['trajectory'].tolist()
            json_sample['contact_states'] = sample['contact_states'].tolist()
            json_data.append(json_sample)
        
        json_path = self.output_dir / f"{filename}.json"
        with open(json_path, 'w') as f:
            json.dump(json_data, f, indent=2)
        
        # Save as CSV (simplified format)
        csv_path = self.output_dir / f"{filename}.csv"
        with open(csv_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['sample_id', 'sentence', 'num_points', 'writing_time', 'noise_level'])
            
            for sample in dataset:
                writer.writerow([
                    sample['sample_id'],
                    sample['sentence'],
                    sample['metadata']['num_points'],
                    sample['metadata']['writing_time'],
                    sample['noise_level']
                ])
        
        # Save individual trajectory files (NPZ format)
        traj_dir = self.output_dir / "trajectories" / filename
        traj_dir.mkdir(parents=True, exist_ok=True)
        
        for sample in dataset:
            traj_path = traj_dir / f"sample_{sample['sample_id']:03d}.npz"
            np.savez(traj_path,
                    trajectory=sample['trajectory'],
                    contact_states=sample['contact_states'],
                    **sample['metadata'])
        
        print(f"Dataset saved to {json_path}")
        print(f"Summary saved to {csv_path}")
        print(f"Individual trajectories saved to {traj_dir}")
    
    def save_benchmarks(self, benchmarks: Dict[str, Dict[str, Any]], filename: str = "benchmarks"):
        """Save benchmark trajectories."""
        json_data = {}
        for name, benchmark in benchmarks.items():
            json_data[name] = {
                'trajectory': benchmark['trajectory'].tolist(),
                'contact_states': benchmark['contact_states'].tolist(),
                'description': benchmark['description'],
                'difficulty': benchmark['difficulty']
            }
        
        json_path = self.output_dir / f"{filename}.json"
        with open(json_path, 'w') as f:
            json.dump(json_data, f, indent=2)
        
        # Save individual benchmark files
        bench_dir = self.output_dir / "trajectories" / filename
        bench_dir.mkdir(parents=True, exist_ok=True)
        
        for name, benchmark in benchmarks.items():
            bench_path = bench_dir / f"{name}.npz"
            np.savez(bench_path, **benchmark)
        
        print(f"Benchmarks saved to {json_path}")
        print(f"Individual benchmarks saved to {bench_dir}")
    
    def visualize_sample(self, sample: Dict[str, Any], save_path: str = None):
        """Visualize a handwriting sample."""
        trajectory = sample['trajectory']
        contact_states = sample['contact_states']
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # 2D trajectory plot
        contact_points = trajectory[contact_states]
        lift_points = trajectory[~contact_states]
        
        if len(contact_points) > 0:
            ax1.plot(contact_points[:, 0], contact_points[:, 1], 'b-', linewidth=2, label='Writing')
        if len(lift_points) > 0:
            ax1.scatter(lift_points[:, 0], lift_points[:, 1], c='red', s=10, label='Pen Lift')
        
        ax1.set_xlabel('X (m)')
        ax1.set_ylabel('Y (m)')
        ax1.set_title(f'Handwriting: "{sample["sentence"]}"')
        ax1.grid(True, alpha=0.3)
        ax1.legend()
        ax1.set_aspect('equal')
        
        # 3D trajectory plot
        ax2 = fig.add_subplot(122, projection='3d')
        if len(contact_points) > 0:
            ax2.plot(contact_points[:, 0], contact_points[:, 1], contact_points[:, 2], 'b-', linewidth=2, label='Writing')
        if len(lift_points) > 0:
            ax2.scatter(lift_points[:, 0], lift_points[:, 1], lift_points[:, 2], c='red', s=10, label='Pen Lift')
        
        ax2.set_xlabel('X (m)')
        ax2.set_ylabel('Y (m)')
        ax2.set_zlabel('Z (m)')
        ax2.set_title('3D Trajectory')
        ax2.legend()
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Visualization saved to {save_path}")
        else:
            plt.show()
        
        plt.close()


def main():
    """Main function to generate all sample data."""
    print("Generating sample handwriting datasets...")
    
    # Initialize generator
    generator = HandwritingDataGenerator()
    
    # Generate synthetic dataset
    print("Generating synthetic handwriting dataset...")
    synthetic_data = generator.generate_synthetic_dataset(num_samples=50)
    generator.save_dataset(synthetic_data, "synthetic_handwriting")
    
    # Generate benchmark trajectories
    print("Generating benchmark trajectories...")
    benchmarks = generator.generate_benchmark_trajectories()
    generator.save_benchmarks(benchmarks)
    
    # Generate training data for GAIL
    print("Generating training dataset for GAIL...")
    training_data = generator.generate_synthetic_dataset(num_samples=200)
    generator.save_dataset(training_data, "gail_training_data")
    
    # Generate test data
    print("Generating test dataset...")
    test_data = generator.generate_synthetic_dataset(num_samples=30)
    generator.save_dataset(test_data, "test_data")
    
    # Create sample visualizations
    print("Creating sample visualizations...")
    vis_dir = Path("datasets") / "visualizations"
    vis_dir.mkdir(exist_ok=True)
    
    # Visualize a few samples
    for i in range(min(3, len(synthetic_data))):
        sample = synthetic_data[i]
        vis_path = vis_dir / f"sample_{i:03d}.png"
        generator.visualize_sample(sample, str(vis_path))
    
    print("Sample data generation completed!")
    print(f"Generated {len(synthetic_data)} synthetic samples")
    print(f"Generated {len(training_data)} training samples")
    print(f"Generated {len(test_data)} test samples")
    print(f"Generated {len(benchmarks)} benchmark trajectories")


if __name__ == "__main__":
    main()