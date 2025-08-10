#!/usr/bin/env python3

import sys
import numpy as np
import math
from pathlib import Path

# Add the project root to path
sys.path.insert(0, str(Path(__file__).parent))

def create_clear_letter_trajectory(letter, start_pos, scale):
    """Create CLEAR, RECOGNIZABLE letter trajectories."""
    points = []
    x_start, y_start, z = start_pos
    
    if letter == 'A':
        # Clear triangle with crossbar
        points = [
            [x_start, y_start, z],
            [x_start - scale*0.4, y_start + scale*1.0, z],  # Left diagonal up
            [x_start, y_start + scale*1.2, z],  # Peak
            [x_start + scale*0.4, y_start + scale*1.0, z],  # Right diagonal down
            [x_start + scale*0.4, y_start, z],  # Right base
            [x_start - scale*0.2, y_start + scale*0.6, z],  # Move to crossbar start
            [x_start + scale*0.2, y_start + scale*0.6, z]   # Crossbar
        ]
    
    elif letter == 'B':
        # Clear B with two bumps
        points = [
            [x_start, y_start, z],
            [x_start, y_start + scale*1.2, z],  # Vertical up
            [x_start + scale*0.4, y_start + scale*1.2, z],  # Top horizontal
            [x_start + scale*0.5, y_start + scale*1.0, z],  # Top curve
            [x_start + scale*0.4, y_start + scale*0.8, z],  # Back to middle
            [x_start, y_start + scale*0.6, z],  # Middle point
            [x_start + scale*0.5, y_start + scale*0.6, z],  # Bottom curve start
            [x_start + scale*0.5, y_start + scale*0.2, z],  # Bottom curve
            [x_start, y_start, z]  # Back to base
        ]
    
    elif letter == 'C':
        # Clear C arc
        for i in range(15):
            angle = math.pi * 0.2 + i * (math.pi * 1.6) / 14  # 3/4 circle
            x = x_start + scale * 0.5 * math.cos(angle)
            y = y_start + scale * 0.6 + scale * 0.5 * math.sin(angle)
            points.append([x, y, z])
    
    elif letter == 'O':
        # Clear O (complete circle)
        for i in range(20):
            angle = i * 2 * math.pi / 19
            x = x_start + scale * 0.4 * math.cos(angle)
            y = y_start + scale * 0.6 + scale * 0.6 * math.sin(angle)
            points.append([x, y, z])
        points.append(points[0])  # Close the circle
        
    else:
        # Default circle for unknown letters
        for i in range(12):
            angle = i * 2 * math.pi / 11
            x = x_start + scale * 0.3 * math.cos(angle)
            y = y_start + scale * 0.6 + scale * 0.4 * math.sin(angle)
            points.append([x, y, z])
    
    return points

def show_ascii_trajectory(points, letter):
    """Show trajectory as ASCII art."""
    if not points:
        return
    
    points = np.array(points)
    x_coords = points[:, 0]
    y_coords = points[:, 1]
    
    x_min, x_max = x_coords.min(), x_coords.max()
    y_min, y_max = y_coords.min(), y_coords.max()
    
    # Create ASCII grid
    width, height = 40, 15
    grid = [[' ' for _ in range(width)] for _ in range(height)]
    
    for x, y in zip(x_coords, y_coords):
        # Map to grid coordinates
        if x_max > x_min:
            grid_x = int((x - x_min) / (x_max - x_min) * (width - 1))
        else:
            grid_x = width // 2
            
        if y_max > y_min:
            grid_y = int((y_max - y) / (y_max - y_min) * (height - 1))  # Flip Y
        else:
            grid_y = height // 2
            
        grid_x = max(0, min(width - 1, grid_x))
        grid_y = max(0, min(height - 1, grid_y))
        grid[grid_y][grid_x] = '●'
    
    print(f"\n🎨 Clear Trajectory for '{letter}':")
    print("=" * 50)
    for row in grid:
        print(''.join(row))
    print("=" * 50)

def test_clear_letters():
    """Test the clear letter generation."""
    print("🧪 Testing Clear Letter Trajectory Generation")
    print("=" * 60)
    
    letters = ['A', 'B', 'C', 'O']
    scale = 0.02
    
    for letter in letters:
        start_pos = [0.1, 0.15, 0.02]
        trajectory = create_clear_letter_trajectory(letter, start_pos, scale)
        
        print(f"\n✍️  Letter '{letter}': {len(trajectory)} points")
        show_ascii_trajectory(trajectory, letter)
        
        # Show point details for first few points
        if len(trajectory) >= 3:
            print(f"   📍 First 3 points:")
            for i, point in enumerate(trajectory[:3]):
                print(f"      {i+1}: ({point[0]:.4f}, {point[1]:.4f}, {point[2]:.4f})")

if __name__ == "__main__":
    test_clear_letters()