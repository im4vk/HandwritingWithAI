#!/usr/bin/env python3
"""
Simple Robotic Handwriting AI Demo
=================================

A straightforward demonstration of the robotic handwriting system
using working sample data and simplified components.
"""

import json
import numpy as np
from pathlib import Path
import time

def load_working_data():
    """Load the working sample data."""
    print("📊 Loading Sample Data...")
    
    # Try different data files
    data_files = [
        "data/datasets/test_samples.json",
        "data/datasets/training_samples.json", 
        "data/training/gail_train/demonstrations.json"
    ]
    
    for data_file in data_files:
        try:
            with open(data_file, 'r') as f:
                data = json.load(f)
            print(f"✅ Loaded {len(data)} samples from {data_file}")
            return data
        except Exception as e:
            continue
    
    print("❌ Could not load sample data, generating demo data...")
    return generate_demo_data()

def generate_demo_data():
    """Generate simple demo data."""
    samples = []
    texts = ["HELLO", "WORLD", "AI", "ROBOT", "DEMO"]
    
    for i, text in enumerate(texts):
        trajectory = []
        x, y, z = 0.1, 0.15, 0.02
        
        # Generate simple trajectory
        for char in text:
            for j in range(8):
                pos_x = x + j * 0.003
                pos_y = y + np.sin(j * 0.5) * 0.005
                pos_z = z
                trajectory.append([pos_x, pos_y, pos_z])
            x += 0.025
        
        sample = {
            'sample_id': i,
            'sentence': text,
            'trajectory': trajectory,
            'contact_states': [True] * len(trajectory),
            'metadata': {
                'num_points': len(trajectory),
                'writing_time': len(trajectory) * 0.01
            }
        }
        samples.append(sample)
    
    return samples

def demonstrate_trajectory_analysis(samples):
    """Analyze trajectory characteristics."""
    print("\n✍️  Trajectory Analysis...")
    
    for sample in samples[:3]:  # Analyze first 3 samples
        trajectory = np.array(sample['trajectory'])
        text = sample['sentence']
        
        # Calculate metrics
        if len(trajectory) > 1:
            distances = np.linalg.norm(np.diff(trajectory, axis=0), axis=1)
            total_distance = np.sum(distances)
            avg_speed = total_distance / sample['metadata']['writing_time']
            
            print(f"📝 '{text}':")
            print(f"   Points: {len(trajectory)}")
            print(f"   Distance: {total_distance:.3f}m")
            print(f"   Time: {sample['metadata']['writing_time']:.2f}s")
            print(f"   Avg Speed: {avg_speed:.4f}m/s")
            print(f"   Bounds: X={trajectory[:, 0].min():.3f}-{trajectory[:, 0].max():.3f}m")

def demonstrate_motion_planning(sample):
    """Demonstrate motion planning concepts."""
    print("\n⚙️  Motion Planning Demo...")
    
    trajectory = np.array(sample['trajectory'])
    print(f"Planning motion for: '{sample['sentence']}'")
    
    # Simple trajectory smoothing
    def smooth_trajectory(traj, alpha=0.3):
        smoothed = traj.copy()
        for i in range(1, len(traj) - 1):
            for dim in range(3):
                smoothed[i, dim] = (alpha * traj[i, dim] + 
                                  (1-alpha) * (traj[i-1, dim] + traj[i+1, dim]) / 2)
        return smoothed
    
    smoothed = smooth_trajectory(trajectory)
    
    # Calculate improvement
    def calculate_jerk(traj):
        if len(traj) < 3:
            return 0.0
        velocities = np.diff(traj, axis=0)
        accelerations = np.diff(velocities, axis=0)
        return np.mean(np.linalg.norm(accelerations, axis=1))
    
    original_jerk = calculate_jerk(trajectory)
    smoothed_jerk = calculate_jerk(smoothed)
    
    print(f"✅ Motion planning results:")
    print(f"   Original jerk: {original_jerk:.6f}")
    print(f"   Smoothed jerk: {smoothed_jerk:.6f}")
    print(f"   Improvement: {((original_jerk - smoothed_jerk) / original_jerk * 100):.1f}%")
    
    return smoothed

def demonstrate_robot_simulation(trajectory, text):
    """Simulate robot execution."""
    print("\n🎮 Robot Simulation...")
    
    print(f"Simulating robot writing: '{text}'")
    
    # Simulate robot states
    robot_states = []
    rewards = []
    
    for i, pos in enumerate(trajectory):
        # Mock robot state
        state = {
            'position': pos.tolist(),
            'velocity': [0.01, 0.005, 0.0] if i > 0 else [0, 0, 0],
            'in_contact': pos[2] < 0.021,
            'pressure': 0.7 if pos[2] < 0.021 else 0.1,
            'timestamp': i * 0.01
        }
        
        # Simple reward calculation
        reward = 1.0  # Base reward
        if i > 0:
            movement = np.linalg.norm(pos - trajectory[i-1])
            if movement < 0.01:  # Smooth movement
                reward += 0.5
            if state['in_contact']:  # Good contact
                reward += 0.3
        
        robot_states.append(state)
        rewards.append(reward)
    
    # Calculate simulation results
    total_reward = sum(rewards)
    avg_reward = total_reward / len(rewards)
    contact_ratio = sum(1 for state in robot_states if state['in_contact']) / len(robot_states)
    
    print(f"✅ Simulation completed:")
    print(f"   Steps: {len(robot_states)}")
    print(f"   Total reward: {total_reward:.2f}")
    print(f"   Average reward: {avg_reward:.3f}")
    print(f"   Contact ratio: {contact_ratio:.2f}")
    print(f"   Duration: {len(robot_states) * 0.01:.2f}s")
    
    return robot_states, rewards

def demonstrate_quality_analysis(robot_states, rewards, text):
    """Analyze handwriting quality."""
    print("\n📈 Quality Analysis...")
    
    # Extract data for analysis
    positions = [state['position'] for state in robot_states]
    pressures = [state['pressure'] for state in robot_states if state['in_contact']]
    
    # Calculate quality metrics
    smoothness = 1.0 / (1.0 + np.var(rewards))  # Reward consistency as smoothness proxy
    pressure_consistency = 1.0 / (1.0 + np.var(pressures)) if pressures else 0.5
    
    # Speed analysis
    if len(positions) > 1:
        speeds = []
        for i in range(1, len(positions)):
            dist = np.linalg.norm(np.array(positions[i]) - np.array(positions[i-1]))
            speed = dist / 0.01  # 10ms timestep
            speeds.append(speed)
        speed_consistency = 1.0 / (1.0 + np.var(speeds))
    else:
        speed_consistency = 1.0
    
    # Overall quality
    overall_quality = (smoothness + pressure_consistency + speed_consistency) / 3
    
    print(f"🏆 Quality Assessment for '{text}':")
    print(f"   Smoothness: {smoothness:.3f} {'⭐' * int(smoothness * 5)}")
    print(f"   Pressure consistency: {pressure_consistency:.3f} {'⭐' * int(pressure_consistency * 5)}")
    print(f"   Speed consistency: {speed_consistency:.3f} {'⭐' * int(speed_consistency * 5)}")
    print(f"   Overall quality: {overall_quality:.3f} {'⭐' * int(overall_quality * 5)}")
    
    return {
        'smoothness': smoothness,
        'pressure_consistency': pressure_consistency,
        'speed_consistency': speed_consistency,
        'overall_quality': overall_quality
    }

def create_ascii_visualization(trajectory, text):
    """Create simple ASCII visualization."""
    print(f"\n🎨 Visualization for '{text}':")
    print("=" * 50)
    
    # Extract X and Y coordinates
    x_coords = trajectory[:, 0]
    y_coords = trajectory[:, 1]
    
    # Normalize to grid
    width, height = 40, 10
    
    if len(set(x_coords)) > 1 and len(set(y_coords)) > 1:
        x_min, x_max = x_coords.min(), x_coords.max()
        y_min, y_max = y_coords.min(), y_coords.max()
        
        grid = [[' ' for _ in range(width)] for _ in range(height)]
        
        for x, y in zip(x_coords, y_coords):
            x_grid = int((x - x_min) / (x_max - x_min) * (width - 1))
            y_grid = height - 1 - int((y - y_min) / (y_max - y_min) * (height - 1))
            
            if 0 <= x_grid < width and 0 <= y_grid < height:
                grid[y_grid][x_grid] = '●'
        
        for row in grid:
            print(''.join(row))
    else:
        print("●" * len(text) * 5)  # Simple representation
    
    print("=" * 50)

def run_complete_demo():
    """Run the complete demonstration."""
    print("🤖 ROBOTIC HANDWRITING AI - SIMPLE DEMO")
    print("=" * 60)
    print("Demonstrating the complete pipeline with working components\n")
    
    # Step 1: Load data
    samples = load_working_data()
    if not samples:
        print("❌ No data available for demo")
        return
    
    # Step 2: Analyze trajectories
    demonstrate_trajectory_analysis(samples)
    
    # Step 3: Motion planning demo
    sample = samples[0]  # Use first sample
    optimized_trajectory = demonstrate_motion_planning(sample)
    
    # Step 4: Robot simulation
    robot_states, rewards = demonstrate_robot_simulation(
        optimized_trajectory, sample['sentence']
    )
    
    # Step 5: Quality analysis
    quality_metrics = demonstrate_quality_analysis(
        robot_states, rewards, sample['sentence']
    )
    
    # Step 6: Visualization
    create_ascii_visualization(optimized_trajectory, sample['sentence'])
    
    # Step 7: Summary
    print("\n🎉 Demo Summary:")
    print("=" * 30)
    print(f"✅ Text processed: '{sample['sentence']}'")
    print(f"✅ Trajectory points: {len(optimized_trajectory)}")
    print(f"✅ Simulation steps: {len(robot_states)}")
    print(f"✅ Overall quality: {quality_metrics['overall_quality']:.3f}")
    print(f"✅ Total reward: {sum(rewards):.2f}")
    
    print("\n🔧 System Components Demonstrated:")
    print("   📊 Data Loading & Processing")
    print("   ✍️  Trajectory Generation")
    print("   ⚙️  Motion Planning & Optimization")
    print("   🎮 Robot Simulation")
    print("   📈 Quality Analysis")
    print("   🎨 Visualization")
    
    return {
        'sample': sample,
        'trajectory': optimized_trajectory,
        'simulation': robot_states,
        'quality': quality_metrics
    }

def interactive_text_demo():
    """Interactive demo for custom text."""
    print("\n🎮 INTERACTIVE DEMO")
    print("=" * 30)
    print("Try your own text! (Enter 'quit' to exit)")
    
    while True:
        try:
            text = input("\nEnter text to simulate: ").strip()
            
            if text.lower() in ['quit', 'exit', 'q']:
                break
            
            if not text:
                continue
            
            print(f"\n✍️  Processing: '{text}'")
            
            # Generate trajectory
            trajectory = []
            x, y, z = 0.1, 0.15, 0.02
            
            for char in text.upper():
                if char == ' ':
                    x += 0.02
                    continue
                
                for i in range(6):
                    px = x + i * 0.004
                    py = y + np.sin(i * np.pi / 3) * 0.008
                    pz = z
                    trajectory.append([px, py, pz])
                
                x += 0.025
            
            trajectory = np.array(trajectory)
            
            # Quick analysis
            if len(trajectory) > 1:
                distances = np.linalg.norm(np.diff(trajectory, axis=0), axis=1)
                total_distance = np.sum(distances)
                
                print(f"✅ Generated trajectory:")
                print(f"   Points: {len(trajectory)}")
                print(f"   Distance: {total_distance:.3f}m")
                print(f"   Est. time: {len(trajectory) * 0.01:.2f}s")
                
                # Simple visualization
                create_ascii_visualization(trajectory, text)
            
        except KeyboardInterrupt:
            print("\n👋 Demo interrupted")
            break
        except Exception as e:
            print(f"❌ Error: {e}")

def main():
    """Main demo function."""
    try:
        # Run complete demo
        results = run_complete_demo()
        
        # Interactive demo
        interactive_text_demo()
        
        print("\n🎉 All demonstrations completed successfully!")
        
    except Exception as e:
        print(f"❌ Demo error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()