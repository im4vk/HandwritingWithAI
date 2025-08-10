#!/usr/bin/env python3
"""
Fix corrupted data files
========================

This script fixes the corrupted synthetic_handwriting.json file and creates
proper data files for all demos.
"""

import json
import numpy as np
from pathlib import Path

def create_proper_synthetic_data():
    """Create a proper synthetic handwriting dataset."""
    print("🔧 Creating proper synthetic handwriting data...")
    
    # Sample sentences for handwriting
    sentences = [
        "Hello World",
        "Artificial Intelligence", 
        "Robot Writing",
        "Handwriting Analysis",
        "Machine Learning",
        "Neural Networks",
        "Deep Learning",
        "Computer Vision",
        "Natural Language",
        "Data Science",
        "Biomechanical Modeling",
        "Trajectory Planning",
        "Motion Control",
        "Human-Robot Interaction",
        "Adaptive Systems"
    ]
    
    samples = []
    
    for i, sentence in enumerate(sentences):
        # Generate realistic trajectory for each sentence
        num_points = len(sentence) * 8 + np.random.randint(-5, 15)  # Variable length
        
        # Create realistic handwriting trajectory
        trajectory = []
        x_start = 0.1
        y_base = 0.15
        
        for j in range(num_points):
            t = j / num_points
            # Realistic handwriting motion
            x = x_start + t * len(sentence) * 0.015 + np.random.normal(0, 0.002)
            y = y_base + 0.01 * np.sin(t * len(sentence) * 2) + np.random.normal(0, 0.001)
            z = 0.02 + np.random.normal(0, 0.0005)
            
            trajectory.append([x, y, z])
        
        # Create word boundaries
        words = sentence.split()
        word_boundaries = []
        char_count = 0
        
        for word in words:
            start_idx = int(char_count * num_points / len(sentence))
            char_count += len(word) + 1  # +1 for space
            end_idx = min(int(char_count * num_points / len(sentence)), num_points - 1)
            
            word_boundaries.append({
                "word": word,
                "start_idx": start_idx,
                "end_idx": end_idx
            })
        
        # Create writing pressure data
        pressure = [0.5 + 0.3 * np.sin(j * 0.2) + np.random.normal(0, 0.1) 
                   for j in range(num_points)]
        pressure = [max(0.0, min(1.0, p)) for p in pressure]  # Clamp to [0,1]
        
        # Create velocity data
        velocities = []
        for j in range(num_points - 1):
            dx = trajectory[j+1][0] - trajectory[j][0]
            dy = trajectory[j+1][1] - trajectory[j][1]
            dz = trajectory[j+1][2] - trajectory[j][2]
            vel = np.sqrt(dx*dx + dy*dy + dz*dz) / 0.01  # Assuming 0.01s per step
            velocities.append(vel)
        velocities.append(velocities[-1] if velocities else 0.0)  # Last point
        
        # Create pen contact data
        pen_contact = [True] * num_points  # Assume pen is always in contact for simplicity
        
        sample = {
            "sample_id": i + 100,
            "sentence": sentence,
            "trajectory": trajectory,
            "metadata": {
                "num_points": num_points,
                "writing_time": num_points * 0.01,
                "avg_velocity": np.mean(velocities),
                "max_velocity": np.max(velocities),
                "total_distance": sum(np.linalg.norm(np.array(trajectory[j+1]) - np.array(trajectory[j])) 
                                    for j in range(num_points - 1))
            },
            "pressure": pressure,
            "velocity": velocities,
            "pen_contact": pen_contact,
            "word_boundaries": word_boundaries,
            "start_position": trajectory[0]
        }
        
        samples.append(sample)
    
    return samples

def create_missing_preprocessor():
    """Create the missing HandwritingPreprocessor class."""
    print("🔧 Creating missing HandwritingPreprocessor class...")
    
    preprocessor_code = '''"""
Handwriting Data Preprocessing Module
===================================

Data preprocessing utilities for handwriting datasets.
"""

import numpy as np
from typing import List, Dict, Any, Tuple, Optional
import json
from pathlib import Path

class HandwritingPreprocessor:
    """
    Preprocessor for handwriting data with filtering, normalization,
    and feature extraction capabilities.
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        """
        Initialize preprocessor.
        
        Args:
            config: Preprocessing configuration
        """
        self.config = config or {}
        self.sampling_rate = self.config.get('sampling_rate', 100.0)
        self.filter_cutoff = self.config.get('filter_cutoff', 10.0)
        self.normalize_position = self.config.get('normalize_position', True)
        self.normalize_velocity = self.config.get('normalize_velocity', True)
    
    def preprocess_trajectory(self, trajectory: List[List[float]]) -> Dict[str, np.ndarray]:
        """
        Preprocess a single trajectory.
        
        Args:
            trajectory: Raw trajectory points [[x, y, z], ...]
            
        Returns:
            Preprocessed data with positions, velocities, accelerations
        """
        trajectory = np.array(trajectory)
        
        # Apply smoothing filter
        smoothed = self._apply_smoothing_filter(trajectory)
        
        # Calculate velocities
        velocities = self._calculate_velocities(smoothed)
        
        # Calculate accelerations
        accelerations = self._calculate_accelerations(velocities)
        
        # Normalize if requested
        if self.normalize_position:
            smoothed = self._normalize_positions(smoothed)
        
        if self.normalize_velocity:
            velocities = self._normalize_velocities(velocities)
        
        return {
            'positions': smoothed,
            'velocities': velocities,
            'accelerations': accelerations,
            'timestamps': np.arange(len(smoothed)) / self.sampling_rate
        }
    
    def _apply_smoothing_filter(self, trajectory: np.ndarray) -> np.ndarray:
        """Apply smoothing filter to trajectory."""
        # Simple moving average filter
        window_size = max(1, int(self.sampling_rate / self.filter_cutoff))
        
        if window_size <= 1:
            return trajectory
        
        smoothed = trajectory.copy()
        for i in range(len(trajectory)):
            start_idx = max(0, i - window_size // 2)
            end_idx = min(len(trajectory), i + window_size // 2 + 1)
            smoothed[i] = np.mean(trajectory[start_idx:end_idx], axis=0)
        
        return smoothed
    
    def _calculate_velocities(self, positions: np.ndarray) -> np.ndarray:
        """Calculate velocities from positions."""
        velocities = np.zeros_like(positions)
        dt = 1.0 / self.sampling_rate
        
        for i in range(1, len(positions)):
            velocities[i] = (positions[i] - positions[i-1]) / dt
        
        return velocities
    
    def _calculate_accelerations(self, velocities: np.ndarray) -> np.ndarray:
        """Calculate accelerations from velocities."""
        accelerations = np.zeros_like(velocities)
        dt = 1.0 / self.sampling_rate
        
        for i in range(1, len(velocities)):
            accelerations[i] = (velocities[i] - velocities[i-1]) / dt
        
        return accelerations
    
    def _normalize_positions(self, positions: np.ndarray) -> np.ndarray:
        """Normalize positions to [0, 1] range."""
        normalized = positions.copy()
        
        for dim in range(positions.shape[1]):
            min_val = np.min(positions[:, dim])
            max_val = np.max(positions[:, dim])
            
            if max_val > min_val:
                normalized[:, dim] = (positions[:, dim] - min_val) / (max_val - min_val)
        
        return normalized
    
    def _normalize_velocities(self, velocities: np.ndarray) -> np.ndarray:
        """Normalize velocities by maximum magnitude."""
        magnitudes = np.linalg.norm(velocities, axis=1)
        max_magnitude = np.max(magnitudes)
        
        if max_magnitude > 0:
            return velocities / max_magnitude
        else:
            return velocities
    
    def load_and_preprocess_dataset(self, dataset_path: str) -> List[Dict[str, Any]]:
        """
        Load and preprocess an entire dataset.
        
        Args:
            dataset_path: Path to dataset JSON file
            
        Returns:
            List of preprocessed samples
        """
        with open(dataset_path, 'r') as f:
            samples = json.load(f)
        
        preprocessed_samples = []
        
        for sample in samples:
            trajectory = sample['trajectory']
            preprocessed_data = self.preprocess_trajectory(trajectory)
            
            # Add preprocessed data to sample
            sample['preprocessed'] = preprocessed_data
            preprocessed_samples.append(sample)
        
        return preprocessed_samples
    
    def extract_features(self, trajectory: np.ndarray) -> Dict[str, float]:
        """
        Extract features from trajectory.
        
        Args:
            trajectory: Trajectory points
            
        Returns:
            Dictionary of extracted features
        """
        if len(trajectory) < 2:
            return {}
        
        # Basic features
        total_distance = np.sum(np.linalg.norm(np.diff(trajectory, axis=0), axis=1))
        total_time = len(trajectory) / self.sampling_rate
        avg_velocity = total_distance / total_time if total_time > 0 else 0
        
        # Velocity features
        velocities = self._calculate_velocities(trajectory)
        velocity_magnitudes = np.linalg.norm(velocities, axis=1)
        max_velocity = np.max(velocity_magnitudes)
        velocity_variance = np.var(velocity_magnitudes)
        
        # Acceleration features
        accelerations = self._calculate_accelerations(velocities)
        acceleration_magnitudes = np.linalg.norm(accelerations, axis=1)
        max_acceleration = np.max(acceleration_magnitudes)
        
        # Jerk features (rate of change of acceleration)
        jerks = self._calculate_accelerations(accelerations)
        jerk_magnitudes = np.linalg.norm(jerks, axis=1)
        avg_jerk = np.mean(jerk_magnitudes)
        
        return {
            'total_distance': total_distance,
            'total_time': total_time,
            'avg_velocity': avg_velocity,
            'max_velocity': max_velocity,
            'velocity_variance': velocity_variance,
            'max_acceleration': max_acceleration,
            'avg_jerk': avg_jerk,
            'num_points': len(trajectory),
            'smoothness': 1.0 / (1.0 + avg_jerk) if avg_jerk > 0 else 1.0
        }
'''
    
    # Write to the preprocessing module
    preprocessing_file = Path("src/data_processing/preprocessing.py")
    with open(preprocessing_file, 'w') as f:
        f.write(preprocessor_code)
    
    print(f"✅ Created HandwritingPreprocessor in {preprocessing_file}")

def fix_demo_components():
    """Fix the demo_components.py to use proper error handling."""
    print("🔧 Fixing demo_components.py with better error handling...")
    
    demo_file = Path("demo_components.py")
    
    # Read current content
    with open(demo_file, 'r') as f:
        content = f.read()
    
    # Replace the problematic data loading section
    old_data_loading = '''def demo_data_loading():
    """Demonstrate data loading capabilities."""
    print("📊 DATA LOADING DEMO")
    print("=" * 40)
    
    # Load sample data
    data_file = Path("data/datasets/synthetic_handwriting.json")
    if data_file.exists():
        with open(data_file, 'r') as f:
            samples = json.load(f)'''
    
    new_data_loading = '''def demo_data_loading():
    """Demonstrate data loading capabilities."""
    print("📊 DATA LOADING DEMO")
    print("=" * 40)
    
    # Try multiple data files with fallback
    data_files = [
        "data/datasets/test_samples.json",
        "data/datasets/synthetic_handwriting.json",
        "data/datasets/training_samples.json"
    ]
    
    samples = None
    for data_file in data_files:
        data_path = Path(data_file)
        if data_path.exists():
            try:
                with open(data_path, 'r') as f:
                    samples = json.load(f)
                print(f"✅ Successfully loaded {len(samples)} samples from {data_file}")
                break
            except json.JSONDecodeError as e:
                print(f"⚠️  JSON error in {data_file}: {e}")
                continue
            except Exception as e:
                print(f"⚠️  Error loading {data_file}: {e}")
                continue
    
    if samples:'''
    
    if old_data_loading in content:
        content = content.replace(old_data_loading, new_data_loading)
        
        with open(demo_file, 'w') as f:
            f.write(content)
        
        print(f"✅ Fixed {demo_file} with better error handling")
    else:
        print("⚠️  Could not find the exact pattern to replace in demo_components.py")

def main():
    """Fix all data and preprocessing issues."""
    print("🔧 FIXING ALL DATA AND PREPROCESSING ISSUES")
    print("=" * 60)
    
    try:
        # Fix synthetic data
        samples = create_proper_synthetic_data()
        
        # Save fixed synthetic data
        output_file = Path("data/datasets/synthetic_handwriting.json")
        with open(output_file, 'w') as f:
            json.dump(samples, f, indent=2)
        
        print(f"✅ Created proper synthetic_handwriting.json with {len(samples)} samples")
        
        # Create missing preprocessor
        create_missing_preprocessor()
        
        # Fix demo components
        fix_demo_components()
        
        print("\n🎉 ALL FIXES COMPLETED!")
        print("✅ Fixed synthetic_handwriting.json")
        print("✅ Created HandwritingPreprocessor class")  
        print("✅ Fixed demo_components.py error handling")
        
    except Exception as e:
        print(f"❌ Error during fixing: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()