#!/usr/bin/env python3
"""
Basic GAIL Training Data Generator (No Dependencies)
===================================================

Simple GAIL data generator without external dependencies.
"""

import json
import random
from pathlib import Path


def create_state_action_pairs(trajectory, text):
    """Create state-action pairs from trajectory."""
    states = []
    actions = []
    
    for i in range(len(trajectory) - 1):
        current_pos = trajectory[i]
        next_pos = trajectory[i + 1]
        
        # Calculate velocity (simplified)
        if i > 0:
            prev_pos = trajectory[i - 1]
            velocity = [
                (current_pos[0] - prev_pos[0]) / 0.001,
                (current_pos[1] - prev_pos[1]) / 0.001,
                (current_pos[2] - prev_pos[2]) / 0.001
            ]
        else:
            velocity = [0.0, 0.0, 0.0]
        
        # Mock orientation (quaternion)
        orientation = [1.0, 0.0, 0.0, 0.0]
        
        # Contact state
        is_in_contact = 1.0 if current_pos[2] < 0.021 else 0.0
        
        # Target position
        target_pos = next_pos
        
        # Position error
        position_error = ((current_pos[0] - target_pos[0])**2 + 
                         (current_pos[1] - target_pos[1])**2 + 
                         (current_pos[2] - target_pos[2])**2)**0.5
        
        # State vector (15D)
        state = (current_pos + velocity + orientation + 
                [is_in_contact] + target_pos + [position_error])
        
        # Action vector (4D: dx, dy, dz, pressure)
        action = [
            next_pos[0] - current_pos[0],  # dx
            next_pos[1] - current_pos[1],  # dy
            next_pos[2] - current_pos[2],  # dz
            0.7 if is_in_contact > 0.5 else 0.1  # pressure
        ]
        
        states.append(state)
        actions.append(action)
    
    return states, actions


def generate_gail_demonstration(text, demo_id, skill_level="expert"):
    """Generate a GAIL demonstration."""
    # Create simple trajectory
    trajectory = []
    x, y, z = 0.1, 0.1, 0.02
    char_width = 0.015
    char_height = 0.02
    
    # Skill-based noise
    noise_levels = {"expert": 0.0005, "intermediate": 0.001, "novice": 0.002}
    noise = noise_levels.get(skill_level, 0.001)
    
    for char in text:
        if char == ' ':
            x += char_width * 0.8
        else:
            # Character trajectory
            for step in range(5):  # 5 points per character
                pos_x = x + (step / 4) * char_width
                pos_y = y + random.uniform(-noise, noise)
                pos_z = z + random.uniform(-noise*0.1, noise*0.1)
                
                trajectory.append([pos_x, pos_y, pos_z])
            
            x += char_width * 1.2
    
    # Convert to state-action pairs
    states, actions = create_state_action_pairs(trajectory, text)
    
    return {
        "demo_id": demo_id,
        "text": text,
        "skill_level": skill_level,
        "states": states,
        "actions": actions,
        "trajectory_points": trajectory,
        "metadata": {
            "num_steps": len(states),
            "total_time": len(states) * 0.001,
            "trajectory_length": len(trajectory) * 0.015
        }
    }


def main():
    """Generate basic GAIL training data."""
    print("Generating basic GAIL training data...")
    
    # Create output directory
    training_dir = Path("training")
    training_dir.mkdir(exist_ok=True)
    
    # Load sample texts
    with open('sample_texts.json', 'r') as f:
        sample_texts = json.load(f)
    
    # Flatten all texts
    all_texts = []
    for category, texts in sample_texts.items():
        all_texts.extend(texts[:5])  # Take first 5 from each category
    
    # Generate demonstrations
    demonstrations = []
    skill_levels = ["expert"] * 35 + ["intermediate"] * 10 + ["novice"] * 5
    
    for i in range(50):
        text = random.choice(all_texts)
        skill_level = skill_levels[i % len(skill_levels)]
        
        demo = generate_gail_demonstration(text, i, skill_level)
        demonstrations.append(demo)
    
    # Split into train/validation
    random.shuffle(demonstrations)
    split_idx = int(len(demonstrations) * 0.8)
    
    train_demos = demonstrations[:split_idx]
    val_demos = demonstrations[split_idx:]
    
    # Save training data
    train_dir = training_dir / "gail_train"
    train_dir.mkdir(exist_ok=True)
    
    with open(train_dir / "demonstrations.json", 'w') as f:
        json.dump(train_demos, f, indent=2)
    
    # Save validation data
    val_dir = training_dir / "gail_validation"
    val_dir.mkdir(exist_ok=True)
    
    with open(val_dir / "demonstrations.json", 'w') as f:
        json.dump(val_demos, f, indent=2)
    
    # Save complete dataset
    complete_dir = training_dir / "gail_complete"
    complete_dir.mkdir(exist_ok=True)
    
    with open(complete_dir / "demonstrations.json", 'w') as f:
        json.dump(demonstrations, f, indent=2)
    
    # Create statistics
    stats = {
        "num_demonstrations": len(demonstrations),
        "training_demos": len(train_demos),
        "validation_demos": len(val_demos),
        "state_dim": 15,
        "action_dim": 4,
        "skill_distribution": {
            "expert": sum(1 for d in demonstrations if d["skill_level"] == "expert"),
            "intermediate": sum(1 for d in demonstrations if d["skill_level"] == "intermediate"),
            "novice": sum(1 for d in demonstrations if d["skill_level"] == "novice")
        }
    }
    
    with open(training_dir / "statistics.json", 'w') as f:
        json.dump(stats, f, indent=2)
    
    print(f"Generated {len(demonstrations)} GAIL demonstrations")
    print(f"Training: {len(train_demos)}, Validation: {len(val_demos)}")
    print(f"Skill distribution: {stats['skill_distribution']}")
    print("GAIL training data generation completed!")


if __name__ == "__main__":
    main()