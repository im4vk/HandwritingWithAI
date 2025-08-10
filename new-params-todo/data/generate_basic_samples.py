#!/usr/bin/env python3
"""
Basic Sample Data Generator (No Dependencies)
===========================================

Simple data generator that creates sample handwriting data without
requiring external visualization libraries.
"""

import json
import csv
import os
import random
from pathlib import Path


def create_sample_trajectory(text: str, sample_id: int):
    """Create a simple trajectory for text."""
    trajectory = []
    contact_states = []
    
    # Starting position
    x, y, z = 0.1, 0.1, 0.02
    char_width = 0.015
    char_height = 0.02
    
    for char in text:
        if char == ' ':
            # Space - just move position
            x += char_width * 0.8
        else:
            # Simple character trajectory (just a few points per character)
            char_points = [
                [x, y, z],
                [x + char_width/2, y + char_height/2, z],
                [x + char_width, y, z]
            ]
            
            for point in char_points:
                # Add small random variation
                noise = 0.001
                point[0] += random.uniform(-noise, noise)
                point[1] += random.uniform(-noise, noise)
                point[2] += random.uniform(-noise*0.1, noise*0.1)
                
                trajectory.append(point)
                contact_states.append(True)
            
            x += char_width * 1.2
    
    return {
        'sample_id': sample_id,
        'sentence': text,
        'trajectory': trajectory,
        'contact_states': contact_states,
        'start_position': [0.1, 0.1, 0.02],
        'noise_level': 0.001,
        'metadata': {
            'num_points': len(trajectory),
            'num_words': len(text.split()),
            'writing_time': len(trajectory) * 0.01,
            'paper_size': [0.21, 0.297]
        }
    }


def generate_benchmark_trajectories():
    """Generate simple benchmark trajectories."""
    benchmarks = {}
    
    # Simple line
    line_points = []
    for i in range(50):
        t = i / 49
        x = 0.1 + t * 0.1
        y = 0.1
        z = 0.02
        line_points.append([x, y, z])
    
    benchmarks['line'] = {
        'trajectory': line_points,
        'contact_states': [True] * 50,
        'description': 'Simple horizontal line',
        'difficulty': 'easy'
    }
    
    # Simple circle (approximated with straight segments)
    circle_points = []
    center_x, center_y = 0.15, 0.15
    radius = 0.03
    for i in range(60):
        angle = 2 * 3.14159 * i / 59
        x = center_x + radius * (angle / 6.28318)  # Simplified
        y = center_y + radius * ((i % 20) / 20 - 0.5)  # Simplified
        z = 0.02
        circle_points.append([x, y, z])
    
    benchmarks['simple_pattern'] = {
        'trajectory': circle_points,
        'contact_states': [True] * 60,
        'description': 'Simple pattern',
        'difficulty': 'medium'
    }
    
    return benchmarks


def main():
    """Generate basic sample data."""
    print("Generating basic sample data...")
    
    # Create output directories
    datasets_dir = Path("datasets")
    datasets_dir.mkdir(exist_ok=True)
    
    trajectories_dir = Path("trajectories")
    trajectories_dir.mkdir(exist_ok=True)
    
    # Sample texts
    sample_texts = [
        "Hello World",
        "AI Robot",
        "Test Sample",
        "Simple Text",
        "Basic Writing",
        "Machine Learning",
        "Computer Science",
        "Technology",
        "Innovation",
        "Research"
    ]
    
    # Generate synthetic samples
    synthetic_samples = []
    for i, text in enumerate(sample_texts):
        sample = create_sample_trajectory(text, i)
        synthetic_samples.append(sample)
    
    # Save synthetic samples
    with open(datasets_dir / "synthetic_handwriting.json", 'w') as f:
        json.dump(synthetic_samples, f, indent=2)
    
    # Save as CSV summary
    with open(datasets_dir / "synthetic_summary.csv", 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['sample_id', 'sentence', 'num_points', 'writing_time'])
        
        for sample in synthetic_samples:
            writer.writerow([
                sample['sample_id'],
                sample['sentence'],
                sample['metadata']['num_points'],
                sample['metadata']['writing_time']
            ])
    
    # Generate benchmarks
    benchmarks = generate_benchmark_trajectories()
    
    with open(datasets_dir / "benchmarks.json", 'w') as f:
        json.dump(benchmarks, f, indent=2)
    
    # Create training data samples
    training_samples = []
    for i in range(20):
        text = random.choice(sample_texts)
        sample = create_sample_trajectory(text, i + 100)
        training_samples.append(sample)
    
    with open(datasets_dir / "training_samples.json", 'w') as f:
        json.dump(training_samples, f, indent=2)
    
    # Create test data
    test_samples = []
    test_texts = ["Test One", "Test Two", "Test Three", "Test Four", "Test Five"]
    for i, text in enumerate(test_texts):
        sample = create_sample_trajectory(text, i + 200)
        test_samples.append(sample)
    
    with open(datasets_dir / "test_samples.json", 'w') as f:
        json.dump(test_samples, f, indent=2)
    
    print(f"Generated {len(synthetic_samples)} synthetic samples")
    print(f"Generated {len(training_samples)} training samples")
    print(f"Generated {len(test_samples)} test samples")
    print(f"Generated {len(benchmarks)} benchmark trajectories")
    print("Sample data generation completed!")


if __name__ == "__main__":
    main()