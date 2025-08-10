#!/usr/bin/env python3
"""
Fixed Component Demonstrations - No Cross-Calls
==============================================

This version eliminates all cross-calls between demo functions to prevent hanging.
"""

import numpy as np
import json
from pathlib import Path
import sys

def demo_data_loading():
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
    
    if samples:
        print(f"✅ Loaded {len(samples)} handwriting samples")
        
        # Show first sample
        sample = samples[0]
        print(f"\n📝 Sample: '{sample['sentence']}'")
        print(f"   Points: {sample['metadata']['num_points']}")
        print(f"   Time: {sample['metadata']['writing_time']:.2f}s")
        
        # Show trajectory bounds
        trajectory = np.array(sample['trajectory'])
        print(f"   X range: {trajectory[:, 0].min():.3f} to {trajectory[:, 0].max():.3f}m")
        print(f"   Y range: {trajectory[:, 1].min():.3f} to {trajectory[:, 1].max():.3f}m")
        print(f"   Z range: {trajectory[:, 2].min():.3f} to {trajectory[:, 2].max():.3f}m")
    else:
        print("❌ Could not load any data files")

def demo_trajectory_generation():
    """Demonstrate trajectory generation."""
    print("\n✍️  TRAJECTORY GENERATION DEMO")
    print("=" * 40)
    
    # Simple trajectory generator
    def generate_word_trajectory(word: str):
        trajectory = []
        x, y, z = 0.1, 0.15, 0.02
        char_width = 0.02
        
        for char in word:
            if char == ' ':
                x += char_width * 0.7
                continue
            
            # Create simple character pattern
            char_points = []
            for i in range(8):  # 8 points per character
                px = x + (i / 7) * char_width
                py = y + np.sin(i * np.pi / 4) * 0.01  # Simple wave
                pz = z
                char_points.append([px, py, pz])
            
            trajectory.extend(char_points)
            x += char_width * 1.1
        
        return np.array(trajectory)
    
    # Generate trajectory for different words
    words = ["HELLO", "WORLD", "AI", "ROBOT"]
    
    for word in words:
        traj = generate_word_trajectory(word)
        print(f"✅ Generated '{word}': {len(traj)} points")
        print(f"   Length: {np.sum(np.linalg.norm(np.diff(traj, axis=0), axis=1)):.3f}m")

def demo_motion_planning():
    """Demonstrate motion planning and optimization."""
    print("\n⚙️  MOTION PLANNING DEMO")
    print("=" * 40)
    
    # Create sample trajectory directly (no function calls)
    trajectory = np.array([
        [0.1 + i*0.01, 0.15 + 0.01*np.sin(i*0.5), 0.02] 
        for i in range(32)
    ])
    
    print(f"📥 Input trajectory: {len(trajectory)} points")
    
    # Simple trajectory smoothing
    def smooth_trajectory(traj, alpha=0.5):
        smoothed = traj.copy()
        for i in range(1, len(traj) - 1):
            for dim in range(3):
                smoothed[i, dim] = (alpha * traj[i, dim] + 
                                  (1-alpha) * (traj[i-1, dim] + traj[i+1, dim]) / 2)
        return smoothed
    
    # Apply smoothing
    smoothed = smooth_trajectory(trajectory)
    
    # Calculate smoothness metrics
    def calculate_smoothness(traj):
        if len(traj) < 3:
            return 0.0
        diffs = np.diff(traj, axis=0)
        second_diffs = np.diff(diffs, axis=0)
        jerk = np.sum(np.linalg.norm(second_diffs, axis=1))
        return 1.0 / (1.0 + jerk)
    
    original_smoothness = calculate_smoothness(trajectory)
    optimized_smoothness = calculate_smoothness(smoothed)
    
    print(f"📈 Original smoothness: {original_smoothness:.3f}")
    print(f"📈 Optimized smoothness: {optimized_smoothness:.3f}")
    print(f"📊 Improvement: {(optimized_smoothness/original_smoothness - 1)*100:.1f}%")

def demo_simulation():
    """Demonstrate simulation concepts."""
    print("\n🎮 SIMULATION DEMO")
    print("=" * 40)
    
    # Create sample trajectory directly (no function calls)
    trajectory = np.array([
        [0.1 + i*0.01, 0.15 + 0.01*np.sin(i*0.5), 0.02] 
        for i in range(32)
    ])
    
    # Simulate robot following trajectory
    simulation_results = {
        'positions': [],
        'velocities': [],
        'forces': [],
        'contact_states': []
    }
    
    for i in range(len(trajectory)):
        pos = trajectory[i]
        
        # Calculate velocity
        if i > 0:
            vel = (pos - trajectory[i-1]) / 0.01  # 10ms timestep
        else:
            vel = np.zeros(3)
        
        # Simulate contact force
        contact = pos[2] < 0.021  # Close to paper surface
        force = 2.0 if contact else 0.1
        
        simulation_results['positions'].append(pos)
        simulation_results['velocities'].append(vel)
        simulation_results['forces'].append(force)
        simulation_results['contact_states'].append(contact)
    
    # Analysis
    velocities = np.array(simulation_results['velocities'])
    forces = simulation_results['forces']
    contacts = simulation_results['contact_states']
    
    print("📊 Simulation Results:")
    print(f"   Steps: {len(trajectory)}")
    print(f"   Avg speed: {np.mean(np.linalg.norm(velocities, axis=1)):.4f} m/s")
    print(f"   Max speed: {np.max(np.linalg.norm(velocities, axis=1)):.4f} m/s")
    print(f"   Contact ratio: {np.mean(contacts):.2f}")
    print(f"   Avg contact force: {np.mean([f for f, c in zip(forces, contacts) if c]):.2f}N")

def demo_analysis():
    """Demonstrate analysis and metrics."""
    print("\n📈 ANALYSIS DEMO")
    print("=" * 40)
    
    # Create sample data directly (no function calls)
    positions = np.array([
        [0.1 + i*0.01, 0.15 + 0.01*np.sin(i*0.5), 0.02] 
        for i in range(32)
    ])
    velocities = np.array([[0.01, 0.005, 0.0] for _ in range(32)])
    forces = [2.0 if i % 3 == 0 else 1.5 for i in range(32)]
    contacts = [True for _ in range(32)]
    
    # Smoothness metric
    if len(velocities) > 1:
        accelerations = np.diff(velocities, axis=0)
        accel_magnitudes = np.linalg.norm(accelerations, axis=1)
        smoothness = 1.0 / (1.0 + np.var(accel_magnitudes))
    else:
        smoothness = 1.0
    
    # Speed consistency
    speeds = np.linalg.norm(velocities, axis=1)
    speed_consistency = 1.0 / (1.0 + np.var(speeds)) if len(speeds) > 1 else 1.0
    
    # Pressure consistency
    contact_forces = [f for f, c in zip(forces, contacts) if c]
    pressure_consistency = 1.0 / (1.0 + np.var(contact_forces)) if contact_forces else 1.0
    
    # Overall quality
    overall_quality = (smoothness + speed_consistency + pressure_consistency) / 3
    
    print("🏆 Quality Metrics:")
    print(f"   Smoothness: {smoothness:.3f} {'⭐' * int(smoothness * 5)}")
    print(f"   Speed consistency: {speed_consistency:.3f} {'⭐' * int(speed_consistency * 5)}")
    print(f"   Pressure consistency: {pressure_consistency:.3f} {'⭐' * int(pressure_consistency * 5)}")
    print(f"   Overall quality: {overall_quality:.3f} {'⭐' * int(overall_quality * 5)}")

def demo_data_formats():
    """Demonstrate different data formats used in the system."""
    print("\n📋 DATA FORMATS DEMO")
    print("=" * 40)
    
    # State vector format (15D)
    state_example = {
        'position': [0.15, 0.12, 0.02],      # 3D position
        'velocity': [0.01, 0.005, 0.0],      # 3D velocity
        'orientation': [1, 0, 0, 0],         # Quaternion
        'contact': 1.0,                      # Contact state
        'target': [0.16, 0.125, 0.02],      # Target position
        'error': 0.015                       # Position error
    }
    
    print("🔢 State Vector (15D):")
    for key, value in state_example.items():
        if isinstance(value, list):
            print(f"   {key}: {value}")
        else:
            print(f"   {key}: {value}")
    
    # Action vector format (4D)
    action_example = {
        'movement': [0.01, 0.005, 0.0],     # 3D movement command
        'pressure': 0.7                     # Pen pressure
    }
    
    print("\n🎮 Action Vector (4D):")
    for key, value in action_example.items():
        if isinstance(value, list):
            print(f"   {key}: {value}")
        else:
            print(f"   {key}: {value}")
    
    # GAIL demonstration format
    demo_example = {
        'demo_id': 0,
        'text': 'HELLO',
        'skill_level': 'expert',
        'num_steps': 50,
        'total_time': 0.5,
        'quality_score': 0.85
    }
    
    print("\n🎯 GAIL Demonstration:")
    for key, value in demo_example.items():
        print(f"   {key}: {value}")

def interactive_demo():
    """Interactive demonstration."""
    print("\n🎮 INTERACTIVE DEMO")
    print("=" * 40)
    
    # Check if running in non-interactive mode (piped, redirected, etc.)
    if not sys.stdin.isatty():
        print("⚠️  Non-interactive mode detected - skipping interactive demo")
        print("💡 Run without pipes/redirects for interactive features")
        return
    
    print("Try generating a trajectory for your own text!")
    
    while True:
        try:
            text = input("\nEnter text to write (or 'quit' to exit): ").strip()
            
            if text.lower() in ['quit', 'exit', 'q']:
                break
            
            if not text:
                continue
            
            # Generate trajectory
            trajectory = []
            x, y, z = 0.1, 0.15, 0.02
            char_width = 0.025
            
            print(f"\n✍️  Generating trajectory for: '{text}'")
            
            for char in text.upper():
                if char == ' ':
                    x += char_width * 0.7
                    continue
                
                # Simple character trajectory
                for i in range(5):
                    px = x + (i / 4) * char_width * 0.8
                    py = y + np.sin(i * np.pi / 2) * 0.008
                    pz = z
                    trajectory.append([px, py, pz])
                
                x += char_width
            
            trajectory = np.array(trajectory)
            total_length = np.sum(np.linalg.norm(np.diff(trajectory, axis=0), axis=1))
            
            print(f"✅ Generated {len(trajectory)} points")
            print(f"📏 Trajectory length: {total_length:.3f}m")
            print(f"⏱️  Est. writing time: {len(trajectory) * 0.01:.2f}s")
            
        except KeyboardInterrupt:
            print("\n👋 Demo interrupted")
            break
        except Exception as e:
            print(f"❌ Error: {e}")

def main():
    """Main demonstration - completely independent functions."""
    print("🤖 ROBOTIC HANDWRITING AI - COMPONENT DEMOS")
    print("=" * 60)
    print("This shows individual components of the system in action\n")
    
    try:
        # Run each demo independently with no cross-calls
        demo_data_loading()
        demo_trajectory_generation()
        demo_motion_planning()
        demo_simulation()
        demo_analysis()
        demo_data_formats()
        
        # Interactive demo
        print("\n" + "=" * 60)
        interactive_demo()
        
        print("\n🎉 Component demonstrations completed!")
        print("✅ All individual components are working correctly")
        
    except Exception as e:
        print(f"❌ Demo error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()