#!/usr/bin/env python3

import sys
import numpy as np
import math
from pathlib import Path

# Add the project root to path
sys.path.insert(0, str(Path(__file__).parent))

from src.ai_models.gail_model import HandwritingGAIL
from src.robot_models.virtual_robot import VirtualRobotArm
from src.simulation.handwriting_environment import HandwritingEnvironment, EnvironmentConfig

def create_clear_letter_trajectory(letter, start_pos, scale):
    """Create CLEAR, RECOGNIZABLE letter trajectories."""
    points = []
    x_start, y_start, z = start_pos
    
    if letter == 'A':
        points = [
            [x_start, y_start, z],
            [x_start - scale*0.4, y_start + scale*1.0, z],
            [x_start, y_start + scale*1.2, z],
            [x_start + scale*0.4, y_start + scale*1.0, z],
            [x_start + scale*0.4, y_start, z],
            [x_start - scale*0.2, y_start + scale*0.6, z],
            [x_start + scale*0.2, y_start + scale*0.6, z]
        ]
    elif letter == 'B':
        points = [
            [x_start, y_start, z],
            [x_start, y_start + scale*1.2, z],
            [x_start + scale*0.4, y_start + scale*1.2, z],
            [x_start + scale*0.5, y_start + scale*1.0, z],
            [x_start + scale*0.4, y_start + scale*0.8, z],
            [x_start, y_start + scale*0.6, z],
            [x_start + scale*0.5, y_start + scale*0.6, z],
            [x_start + scale*0.5, y_start + scale*0.2, z],
            [x_start, y_start, z]
        ]
    elif letter == 'C':
        for i in range(15):
            angle = math.pi * 0.2 + i * (math.pi * 1.6) / 14
            x = x_start + scale * 0.5 * math.cos(angle)
            y = y_start + scale * 0.6 + scale * 0.5 * math.sin(angle)
            points.append([x, y, z])
    elif letter == 'O':
        for i in range(20):
            angle = i * 2 * math.pi / 19
            x = x_start + scale * 0.4 * math.cos(angle)
            y = y_start + scale * 0.6 + scale * 0.6 * math.sin(angle)
            points.append([x, y, z])
        points.append(points[0])
    elif letter == 'H':
        points = [
            [x_start - scale*0.3, y_start, z],
            [x_start - scale*0.3, y_start + scale*1.2, z],
            [x_start - scale*0.3, y_start + scale*0.6, z],
            [x_start + scale*0.3, y_start + scale*0.6, z],
            [x_start + scale*0.3, y_start + scale*1.2, z],
            [x_start + scale*0.3, y_start, z]
        ]
    elif letter == 'I':
        points = [
            [x_start, y_start, z],
            [x_start, y_start + scale*1.2, z]
        ]
    elif letter == 'L':
        points = [
            [x_start, y_start + scale*1.2, z],
            [x_start, y_start, z],
            [x_start + scale*0.5, y_start, z]
        ]
    elif letter == 'E':
        points = [
            [x_start, y_start, z],
            [x_start, y_start + scale*1.2, z],
            [x_start + scale*0.5, y_start + scale*1.2, z],
            [x_start, y_start + scale*1.2, z],
            [x_start, y_start + scale*0.6, z],
            [x_start + scale*0.4, y_start + scale*0.6, z],
            [x_start, y_start + scale*0.6, z],
            [x_start, y_start, z],
            [x_start + scale*0.5, y_start, z]
        ]
    elif letter == 'T':
        points = [
            [x_start - scale*0.4, y_start + scale*1.2, z],
            [x_start + scale*0.4, y_start + scale*1.2, z],
            [x_start, y_start + scale*1.2, z],
            [x_start, y_start, z]
        ]
    elif letter == 'S':
        # Top curve
        for i in range(8):
            angle = math.pi + i * math.pi / 7
            x = x_start + scale * 0.3 * math.cos(angle)
            y = y_start + scale * 0.9 + scale * 0.3 * math.sin(angle)
            points.append([x, y, z])
        # Bottom curve
        for i in range(8):
            angle = i * math.pi / 7
            x = x_start + scale * 0.3 * math.cos(angle)
            y = y_start + scale * 0.3 + scale * 0.3 * math.sin(angle)
            points.append([x, y, z])
    else:
        # Default circle
        for i in range(12):
            angle = i * 2 * math.pi / 11
            x = x_start + scale * 0.3 * math.cos(angle)
            y = y_start + scale * 0.6 + scale * 0.4 * math.sin(angle)
            points.append([x, y, z])
    
    return points

def show_ascii_trajectory(points, text):
    """Show trajectory as ASCII art."""
    if not points:
        return
    
    points = np.array(points)
    x_coords = points[:, 0]
    y_coords = points[:, 1]
    
    x_min, x_max = x_coords.min(), x_coords.max()
    y_min, y_max = y_coords.min(), y_coords.max()
    
    # Create ASCII grid
    width, height = 80, 15
    grid = [[' ' for _ in range(width)] for _ in range(height)]
    
    for x, y in zip(x_coords, y_coords):
        if x_max > x_min:
            grid_x = int((x - x_min) / (x_max - x_min) * (width - 1))
        else:
            grid_x = width // 2
            
        if y_max > y_min:
            grid_y = int((y_max - y) / (y_max - y_min) * (height - 1))
        else:
            grid_y = height // 2
            
        grid_x = max(0, min(width - 1, grid_x))
        grid_y = max(0, min(height - 1, grid_y))
        grid[grid_y][grid_x] = '●'
    
    print(f"\n🎨 Clear Handwriting for '{text}':")
    print("=" * 80)
    for row in grid:
        print(''.join(row))
    print("=" * 80)
    print(f"Bounds: X={x_min:.3f}-{x_max:.3f}m, Y={y_min:.3f}-{y_max:.3f}m")

def generate_word_trajectory(text):
    """Generate clear trajectory for a word."""
    all_points = []
    current_x = 0.1
    scale = 0.02
    letter_spacing = 0.025
    
    for letter in text.upper():
        if letter == ' ':
            current_x += letter_spacing * 2
            continue
        elif not letter.isalpha():
            continue
            
        start_pos = [current_x, 0.15, 0.02]
        letter_trajectory = create_clear_letter_trajectory(letter, start_pos, scale)
        
        if letter_trajectory:
            all_points.extend(letter_trajectory)
            
            # Calculate letter width and update position
            if letter_trajectory:
                letter_points = np.array(letter_trajectory)
                letter_width = letter_points[:, 0].max() - letter_points[:, 0].min()
                current_x += max(letter_width, scale * 0.8) + letter_spacing
    
    return all_points

def run_clear_demo():
    """Run demo with clear letter generation."""
    print("🤖 Clear Letter Handwriting Demo")
    print("=" * 50)
    
    # Test individual letters
    print("\n📝 Individual Letter Tests:")
    test_letters = ['A', 'B', 'C', 'O', 'H', 'I', 'L', 'E']
    for letter in test_letters:
        trajectory = create_clear_letter_trajectory(letter, [0.1, 0.15, 0.02], 0.02)
        print(f"   ✅ {letter}: {len(trajectory)} points")
    
    # Test words
    print("\n📝 Word Generation Tests:")
    test_words = ["HELLO", "ABC", "TEST", "CLEAR"]
    
    for word in test_words:
        print(f"\n✍️  Generating: '{word}'")
        trajectory = generate_word_trajectory(word)
        print(f"   ✅ Generated {len(trajectory)} points")
        show_ascii_trajectory(trajectory, word)
    
    # Interactive mode
    print("\n🎮 Interactive Mode:")
    print("Enter text to generate clear handwriting (or 'quit' to exit):")
    
    while True:
        try:
            user_input = input("\nEnter text: ").strip()
            if user_input.lower() in ['quit', 'exit', 'q']:
                break
            
            if user_input:
                trajectory = generate_word_trajectory(user_input)
                print(f"✅ Generated {len(trajectory)} points for '{user_input}'")
                show_ascii_trajectory(trajectory, user_input)
        except KeyboardInterrupt:
            break
    
    print("\n🎉 Clear letter demo completed!")

if __name__ == "__main__":
    run_clear_demo()