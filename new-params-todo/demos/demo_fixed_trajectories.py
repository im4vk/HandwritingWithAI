#!/usr/bin/env python3
"""
Fixed Trajectory Demo - Proper Letter Shapes
==========================================

This demo fixes the trajectory generation to create actual letter shapes
that look like real handwriting.
"""

import numpy as np
import json
from pathlib import Path

class ProperLetterGenerator:
    """Generate proper letter trajectories that look like actual handwriting."""
    
    def __init__(self):
        """Initialize the letter generator."""
        self.char_width = 0.02
        self.char_height = 0.03
        self.stroke_density = 15  # Points per stroke
        
    def create_letter_trajectory(self, letter: str, start_x: float, start_y: float, start_z: float):
        """Create proper trajectory for a single letter."""
        letter = letter.upper()
        trajectories = []
        
        if letter == 'A':
            # Letter A: Two diagonal lines and a horizontal crossbar
            trajectories.extend([
                # Left diagonal line (bottom to top)
                self.create_line([start_x, start_y], [start_x + self.char_width/2, start_y + self.char_height]),
                # Right diagonal line (top to bottom)  
                self.create_line([start_x + self.char_width/2, start_y + self.char_height], [start_x + self.char_width, start_y]),
                # Crossbar (left to right at middle)
                self.create_line([start_x + self.char_width*0.25, start_y + self.char_height*0.4], 
                               [start_x + self.char_width*0.75, start_y + self.char_height*0.4])
            ])
            
        elif letter == 'B':
            # Letter B: Vertical line + two bumps
            trajectories.extend([
                # Vertical line
                self.create_line([start_x, start_y], [start_x, start_y + self.char_height]),
                # Top bump
                self.create_arc([start_x, start_y + self.char_height], [start_x + self.char_width*0.8, start_y + self.char_height*0.5], "right"),
                # Bottom bump
                self.create_arc([start_x, start_y + self.char_height*0.5], [start_x + self.char_width*0.8, start_y], "right")
            ])
            
        elif letter == 'C':
            # Letter C: Arc from top to bottom
            trajectories.append(
                self.create_arc([start_x + self.char_width*0.8, start_y + self.char_height*0.8], 
                              [start_x + self.char_width*0.8, start_y + self.char_height*0.2], "left")
            )
            
        elif letter == 'D':
            # Letter D: Vertical line + arc
            trajectories.extend([
                self.create_line([start_x, start_y], [start_x, start_y + self.char_height]),
                self.create_arc([start_x, start_y + self.char_height], [start_x, start_y], "right")
            ])
            
        elif letter == 'E':
            # Letter E: Vertical line + three horizontal lines
            trajectories.extend([
                self.create_line([start_x, start_y], [start_x, start_y + self.char_height]),
                self.create_line([start_x, start_y + self.char_height], [start_x + self.char_width*0.8, start_y + self.char_height]),
                self.create_line([start_x, start_y + self.char_height*0.5], [start_x + self.char_width*0.6, start_y + self.char_height*0.5]),
                self.create_line([start_x, start_y], [start_x + self.char_width*0.8, start_y])
            ])
            
        elif letter == 'F':
            # Letter F: Vertical line + two horizontal lines
            trajectories.extend([
                self.create_line([start_x, start_y], [start_x, start_y + self.char_height]),
                self.create_line([start_x, start_y + self.char_height], [start_x + self.char_width*0.8, start_y + self.char_height]),
                self.create_line([start_x, start_y + self.char_height*0.5], [start_x + self.char_width*0.6, start_y + self.char_height*0.5])
            ])
            
        elif letter == 'G':
            # Letter G: C shape + horizontal line
            trajectories.extend([
                self.create_arc([start_x + self.char_width*0.8, start_y + self.char_height*0.8], 
                              [start_x + self.char_width*0.8, start_y + self.char_height*0.2], "left"),
                self.create_line([start_x + self.char_width*0.5, start_y + self.char_height*0.3], 
                               [start_x + self.char_width*0.8, start_y + self.char_height*0.3])
            ])
            
        elif letter == 'H':
            # Letter H: Two vertical lines + horizontal crossbar
            trajectories.extend([
                self.create_line([start_x, start_y], [start_x, start_y + self.char_height]),
                self.create_line([start_x + self.char_width, start_y], [start_x + self.char_width, start_y + self.char_height]),
                self.create_line([start_x, start_y + self.char_height*0.5], [start_x + self.char_width, start_y + self.char_height*0.5])
            ])
            
        elif letter == 'I':
            # Letter I: Vertical line with serifs
            trajectories.extend([
                self.create_line([start_x + self.char_width*0.2, start_y + self.char_height], [start_x + self.char_width*0.8, start_y + self.char_height]),
                self.create_line([start_x + self.char_width*0.5, start_y + self.char_height], [start_x + self.char_width*0.5, start_y]),
                self.create_line([start_x + self.char_width*0.2, start_y], [start_x + self.char_width*0.8, start_y])
            ])
            
        elif letter == 'L':
            # Letter L: Vertical line + horizontal line
            trajectories.extend([
                self.create_line([start_x, start_y + self.char_height], [start_x, start_y]),
                self.create_line([start_x, start_y], [start_x + self.char_width*0.8, start_y])
            ])
            
        elif letter == 'O':
            # Letter O: Circle/oval
            trajectories.append(
                self.create_oval([start_x + self.char_width*0.5, start_y + self.char_height*0.5], 
                               self.char_width*0.4, self.char_height*0.4)
            )
            
        elif letter == 'R':
            # Letter R: Vertical line + bump + diagonal line
            trajectories.extend([
                self.create_line([start_x, start_y], [start_x, start_y + self.char_height]),
                self.create_arc([start_x, start_y + self.char_height], [start_x, start_y + self.char_height*0.5], "right"),
                self.create_line([start_x + self.char_width*0.5, start_y + self.char_height*0.5], 
                               [start_x + self.char_width, start_y])
            ])
            
        elif letter == 'S':
            # Letter S: S-curve
            trajectories.append(
                self.create_s_curve([start_x, start_y + self.char_height*0.2], 
                                  [start_x + self.char_width, start_y + self.char_height*0.8])
            )
            
        elif letter == 'T':
            # Letter T: Horizontal line + vertical line
            trajectories.extend([
                self.create_line([start_x, start_y + self.char_height], [start_x + self.char_width, start_y + self.char_height]),
                self.create_line([start_x + self.char_width*0.5, start_y + self.char_height], [start_x + self.char_width*0.5, start_y])
            ])
            
        elif letter == 'U':
            # Letter U: U-shape
            trajectories.append(
                self.create_u_shape([start_x, start_y + self.char_height], [start_x + self.char_width, start_y + self.char_height])
            )
            
        elif letter == 'V':
            # Letter V: Two diagonal lines meeting at bottom
            trajectories.extend([
                self.create_line([start_x, start_y + self.char_height], [start_x + self.char_width*0.5, start_y]),
                self.create_line([start_x + self.char_width*0.5, start_y], [start_x + self.char_width, start_y + self.char_height])
            ])
            
        elif letter == 'N':
            # Letter N: Two vertical lines + diagonal
            trajectories.extend([
                self.create_line([start_x, start_y], [start_x, start_y + self.char_height]),
                self.create_line([start_x, start_y], [start_x + self.char_width, start_y + self.char_height]),
                self.create_line([start_x + self.char_width, start_y + self.char_height], [start_x + self.char_width, start_y])
            ])
            
        else:
            # Default: simple vertical line for unknown letters
            trajectories.append(
                self.create_line([start_x + self.char_width*0.5, start_y], [start_x + self.char_width*0.5, start_y + self.char_height])
            )
        
        # Convert to 3D coordinates
        trajectory_3d = []
        for stroke in trajectories:
            for point in stroke:
                trajectory_3d.append([point[0], point[1], start_z])
        
        return trajectory_3d
    
    def create_line(self, start, end):
        """Create a straight line between two points."""
        points = []
        for i in range(self.stroke_density):
            t = i / (self.stroke_density - 1)
            x = start[0] + t * (end[0] - start[0])
            y = start[1] + t * (end[1] - start[1])
            points.append([x, y])
        return points
    
    def create_arc(self, start, end, direction="right"):
        """Create an arc between two points."""
        points = []
        center_x = (start[0] + end[0]) / 2
        center_y = (start[1] + end[1]) / 2
        
        # Calculate radius and angles
        radius = max(abs(end[0] - start[0]), abs(end[1] - start[1])) / 2
        
        if direction == "right":
            # Arc curves to the right
            for i in range(self.stroke_density):
                t = i / (self.stroke_density - 1)
                angle = np.pi * t  # Half circle
                x = center_x + radius * np.cos(angle)
                y = center_y + radius * np.sin(angle) * 0.5
                points.append([x, y])
        else:
            # Arc curves to the left
            for i in range(self.stroke_density):
                t = i / (self.stroke_density - 1)
                angle = np.pi + np.pi * t  # Other half circle
                x = center_x + radius * np.cos(angle)
                y = center_y + radius * np.sin(angle) * 0.5
                points.append([x, y])
        
        return points
    
    def create_oval(self, center, width, height):
        """Create an oval/circle."""
        points = []
        for i in range(self.stroke_density * 2):  # Full circle needs more points
            t = 2 * np.pi * i / (self.stroke_density * 2)
            x = center[0] + width * np.cos(t)
            y = center[1] + height * np.sin(t)
            points.append([x, y])
        return points
    
    def create_s_curve(self, start, end):
        """Create an S-shaped curve."""
        points = []
        for i in range(self.stroke_density):
            t = i / (self.stroke_density - 1)
            # S-curve using sine function
            x = start[0] + t * (end[0] - start[0])
            y_progress = start[1] + t * (end[1] - start[1])
            y_curve = np.sin(t * np.pi * 2) * self.char_width * 0.3
            y = y_progress + y_curve
            points.append([x, y])
        return points
    
    def create_u_shape(self, start, end):
        """Create a U-shaped curve."""
        points = []
        for i in range(self.stroke_density):
            t = i / (self.stroke_density - 1)
            # U-curve
            x = start[0] + t * (end[0] - start[0])
            y_bottom = min(start[1], end[1]) - self.char_height * 0.8
            y = start[1] - (4 * t * (1 - t)) * (start[1] - y_bottom)
            points.append([x, y])
        return points
    
    def generate_word_trajectory(self, word: str, start_x=0.1, start_y=0.15, start_z=0.02):
        """Generate trajectory for a complete word."""
        trajectory = []
        current_x = start_x
        
        for char in word.upper():
            if char == ' ':
                current_x += self.char_width * 0.8
            else:
                char_trajectory = self.create_letter_trajectory(char, current_x, start_y, start_z)
                trajectory.extend(char_trajectory)
                current_x += self.char_width * 1.2
        
        return trajectory

def create_proper_visualization(trajectory, text, width=50, height=15):
    """Create a proper ASCII visualization of the trajectory."""
    if len(trajectory) == 0:
        return "No trajectory data"
    
    # Convert to numpy array
    traj = np.array(trajectory)
    
    # Get X and Y bounds
    x_coords = traj[:, 0]
    y_coords = traj[:, 1]
    
    x_min, x_max = x_coords.min(), x_coords.max()
    y_min, y_max = y_coords.min(), y_coords.max()
    
    # Create grid
    grid = [[' ' for _ in range(width)] for _ in range(height)]
    
    # Map trajectory points to grid
    if x_max > x_min and y_max > y_min:
        for point in traj:
            x_grid = int((point[0] - x_min) / (x_max - x_min) * (width - 1))
            y_grid = height - 1 - int((point[1] - y_min) / (y_max - y_min) * (height - 1))
            
            if 0 <= x_grid < width and 0 <= y_grid < height:
                grid[y_grid][x_grid] = '●'
    
    # Convert grid to string
    result = f"🎨 Trajectory for '{text}':\n"
    result += "=" * (width + 10) + "\n"
    for row in grid:
        result += ''.join(row) + "\n"
    result += "=" * (width + 10) + "\n"
    result += f"Bounds: X={x_min:.3f}-{x_max:.3f}m, Y={y_min:.3f}-{y_max:.3f}m\n"
    
    return result

def test_individual_letters():
    """Test individual letter generation."""
    print("🔤 TESTING INDIVIDUAL LETTERS")
    print("=" * 50)
    
    generator = ProperLetterGenerator()
    test_letters = ['A', 'B', 'H', 'I', 'O', 'S']
    
    for letter in test_letters:
        trajectory = generator.generate_word_trajectory(letter)
        print(f"\n📝 Letter '{letter}':")
        print(f"   Points: {len(trajectory)}")
        
        # Quick analysis
        if len(trajectory) > 1:
            traj_array = np.array(trajectory)
            x_range = traj_array[:, 0].max() - traj_array[:, 0].min()
            y_range = traj_array[:, 1].max() - traj_array[:, 1].min()
            print(f"   X range: {x_range:.3f}m")
            print(f"   Y range: {y_range:.3f}m")
            
            # Show visualization
            viz = create_proper_visualization(trajectory, letter, width=30, height=10)
            print(viz)

def test_word_generation():
    """Test word generation with proper letters."""
    print("\n📝 TESTING WORD GENERATION")
    print("=" * 50)
    
    generator = ProperLetterGenerator()
    test_words = ['HI', 'HELLO', 'AI', 'ROBOT']
    
    for word in test_words:
        print(f"\n✍️  Generating '{word}':")
        trajectory = generator.generate_word_trajectory(word)
        
        # Analysis
        traj_array = np.array(trajectory)
        total_distance = 0
        if len(trajectory) > 1:
            for i in range(len(trajectory) - 1):
                dist = np.linalg.norm(traj_array[i+1] - traj_array[i])
                total_distance += dist
        
        print(f"   Points: {len(trajectory)}")
        print(f"   Distance: {total_distance:.3f}m")
        print(f"   Bounds: X={traj_array[:, 0].min():.3f}-{traj_array[:, 0].max():.3f}m")
        
        # Show visualization
        viz = create_proper_visualization(trajectory, word, width=60, height=12)
        print(viz)

def interactive_proper_demo():
    """Interactive demo with proper letter generation."""
    print("\n🎮 INTERACTIVE DEMO - PROPER LETTERS")
    print("=" * 50)
    print("Try typing words with proper letter shapes!")
    
    generator = ProperLetterGenerator()
    
    while True:
        try:
            text = input("\nEnter text (or 'quit' to exit): ").strip()
            
            if text.lower() in ['quit', 'exit', 'q']:
                break
            
            if not text:
                continue
            
            print(f"\n✍️  Generating proper trajectory for: '{text.upper()}'")
            
            trajectory = generator.generate_word_trajectory(text)
            
            if len(trajectory) > 1:
                traj_array = np.array(trajectory)
                total_distance = 0
                for i in range(len(trajectory) - 1):
                    dist = np.linalg.norm(traj_array[i+1] - traj_array[i])
                    total_distance += dist
                
                print(f"✅ Generated trajectory:")
                print(f"   Points: {len(trajectory)}")
                print(f"   Distance: {total_distance:.3f}m")
                print(f"   Est. time: {len(trajectory) * 0.01:.2f}s")
                
                # Show proper visualization
                viz = create_proper_visualization(trajectory, text.upper(), width=80, height=15)
                print(viz)
                
                # Save trajectory for inspection
                output_file = f"results/proper_trajectory_{text.lower()}.json"
                Path("results").mkdir(exist_ok=True)
                with open(output_file, 'w') as f:
                    json.dump(trajectory, f, indent=2)
                print(f"💾 Trajectory saved to {output_file}")
            
        except KeyboardInterrupt:
            print("\n👋 Demo interrupted")
            break
        except Exception as e:
            print(f"❌ Error: {e}")

def main():
    """Main function to test proper trajectory generation."""
    print("🤖 FIXED TRAJECTORY DEMO - PROPER LETTER SHAPES")
    print("=" * 70)
    print("This demo generates actual letter-shaped trajectories\n")
    
    try:
        # Test individual letters
        test_individual_letters()
        
        # Test word generation
        test_word_generation()
        
        # Interactive demo
        interactive_proper_demo()
        
        print("\n🎉 Proper trajectory demo completed!")
        print("✅ Now generating actual letter shapes instead of sine waves!")
        
    except Exception as e:
        print(f"❌ Demo error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()