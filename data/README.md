# Robotic Handwriting Data
This directory contains sample datasets, training data, and utilities for the robotic handwriting AI system.

## Directory Structure
```
data/
├── README.md                      # This file
├── generate_sample_data.py        # Main data generation script
├── generate_gail_training_data.py # GAIL-specific training data generator
├── sample_texts.json             # Collection of text samples for generation
├── datasets/                     # Generated synthetic datasets
├── trajectories/                 # Individual trajectory files
├── training/                     # Training data for machine learning models
└── samples/                      # Sample data files and examples
```

## Data Generation Scripts

### 1. `generate_sample_data.py`
Main script for generating synthetic handwriting datasets and benchmark trajectories.

**Features:**
- Synthetic handwriting trajectory generation
- Benchmark trajectories (lines, circles, figure-8, spirals)
- Character stroke definitions
- Noise modeling for realistic trajectories
- Multiple output formats (JSON, CSV, NPZ)

**Usage:**
```bash
cd data
python generate_sample_data.py
```

**Generated Data:**
- `synthetic_handwriting.json` - 50 synthetic handwriting samples
- `gail_training_data.json` - 200 training samples for GAIL
- `test_data.json` - 30 test samples
- `benchmarks.json` - Benchmark trajectories
- Individual trajectory files in NPZ format
- Visualization plots

### 2. `generate_gail_training_data.py`
Specialized script for generating expert demonstration data for GAIL training.

**Features:**
- Expert demonstration trajectories
- State-action pair generation
- Multiple skill levels (novice, intermediate, expert)
- Reward calculation for reference
- Training/validation split

**Usage:**
```bash
cd data
python generate_gail_training_data.py
```

**Generated Data:**
- `gail_train/` - Training demonstrations
- `gail_validation/` - Validation demonstrations  
- `gail_complete/` - Complete dataset
- Expert demonstrations in multiple formats

### 3. `sample_texts.json`
Collection of text samples organized by category:

**Categories:**
- `simple_words` - Basic words for testing
- `pangrams` - Sentences containing all letters
- `technical_terms` - AI/robotics terminology
- `sentences` - Complete sentences
- `alphabets` - Letter and number sequences
- `mathematical_expressions` - Math formulas
- `names` - Common names
- `quotes` - Inspirational quotes

## Data Formats

### Trajectory Data Format
Each trajectory sample contains:
```json
{
  "sample_id": 0,
  "sentence": "Hello World",
  "trajectory": [[x1, y1, z1], [x2, y2, z2], ...],
  "contact_states": [true, true, false, ...],
  "word_boundaries": [
    {"word": "Hello", "start_idx": 0, "end_idx": 45},
    {"word": "World", "start_idx": 46, "end_idx": 89}
  ],
  "start_position": [0.1, 0.1, 0.02],
  "noise_level": 0.001,
  "metadata": {
    "num_points": 90,
    "num_words": 2,
    "writing_time": 0.9,
    "paper_size": [0.21, 0.297]
  }
}
```

### GAIL Training Data Format
GAIL demonstrations contain:
```python
{
  "demo_id": 0,
  "text": "Hello",
  "skill_level": "expert",
  "states": numpy.array,      # Shape: (n_steps, state_dim)
  "actions": numpy.array,     # Shape: (n_steps, action_dim)
  "rewards": numpy.array,     # Shape: (n_steps,)
  "trajectory_points": numpy.array,  # Shape: (n_points, 3)
  "metadata": {
    "num_steps": 150,
    "total_time": 0.15,
    "skill_params": {...},
    "trajectory_length": 0.025
  }
}
```

### State Vector (15D)
- Position (3D): [x, y, z] coordinates in meters
- Velocity (3D): [vx, vy, vz] in m/s
- Orientation (4D): Quaternion [w, x, y, z]
- Contact (1D): Binary contact state
- Target (3D): Target position [x, y, z]
- Error (1D): Position error magnitude

### Action Vector (4D)
- Movement (3D): [dx, dy, dz] relative displacement
- Pressure (1D): Normalized pressure value [0, 1]

## Benchmark Trajectories

### Available Benchmarks
1. **Line** - Simple horizontal line (easy)
2. **Circle** - Perfect circle trajectory (medium)
3. **Figure-8** - Figure-8 pattern (hard)
4. **Spiral** - Expanding spiral (hard)

Each benchmark includes:
- Trajectory points
- Contact states
- Difficulty level
- Description

## Data Statistics

### Synthetic Dataset
- 50 handwriting samples
- Average 150 trajectory points per sample
- Noise levels: 0.0005 - 0.002 meters
- Characters: A-Z, 0-9, basic punctuation

### GAIL Training Dataset
- 400 training demonstrations
- 100 validation demonstrations  
- Skill distribution: 70% expert, 25% intermediate, 5% novice
- State dimension: 15
- Action dimension: 4

## Usage Examples

### Loading Trajectory Data
```python
import json
import numpy as np

# Load JSON dataset
with open('datasets/synthetic_handwriting.json', 'r') as f:
    data = json.load(f)

# Load individual trajectory
trajectory_data = np.load('datasets/trajectories/synthetic_handwriting/sample_001.npz')
trajectory = trajectory_data['trajectory']
contacts = trajectory_data['contact_states']
```

### Loading GAIL Data
```python
import pickle
import numpy as np

# Load complete demonstrations
with open('training/gail_train/demonstrations.pkl', 'rb') as f:
    demonstrations = pickle.load(f)

# Load combined arrays
data = np.load('training/gail_train/expert_demonstrations.npz')
states = data['states']
actions = data['actions']
```

### Visualization
```python
from generate_sample_data import HandwritingDataGenerator
import matplotlib.pyplot as plt

generator = HandwritingDataGenerator()

# Load and visualize a sample
with open('datasets/synthetic_handwriting.json', 'r') as f:
    samples = json.load(f)

sample = samples[0]
generator.visualize_sample(sample)
```

## Extending the Data

### Adding New Characters
Modify the `define_character_strokes()` method in `generate_sample_data.py`:
```python
def define_character_strokes(self):
    strokes = {
        'A': [[(0, 0), (0.5, 1), (1, 0)], [(0.3, 0.4), (0.7, 0.4)]],
        # Add your character here
        'NEW_CHAR': [[(x1, y1), (x2, y2), ...], ...]
    }
    return strokes
```

### Adding New Text Categories
Edit `sample_texts.json`:
```json
{
  "existing_category": [...],
  "new_category": [
    "text sample 1",
    "text sample 2"
  ]
}
```

### Custom Skill Parameters
Modify skill parameters in `generate_gail_training_data.py`:
```python
skill_params = {
    "custom_level": {
        "noise": 0.0015,
        "smoothness": 0.8,
        "speed": 0.95
    }
}
```

## Data Quality Notes

- **Coordinate System**: X-axis (left-right), Y-axis (bottom-top), Z-axis (up from paper)
- **Units**: All positions in meters, time in seconds
- **Paper Reference**: A4 size (0.21m × 0.297m) with origin at bottom-left
- **Contact Detection**: Z-coordinate threshold of 0.001m above paper surface
- **Noise Model**: Gaussian noise with position-dependent variance

## File Formats

- **JSON**: Human-readable, good for inspection and small datasets
- **NPZ**: Compressed NumPy format, efficient for large arrays
- **PKL**: Python pickle format, preserves object structure
- **CSV**: Summary statistics and metadata

## Dependencies

Required Python packages:
- numpy
- matplotlib
- json (built-in)
- pickle (built-in)
- pathlib (built-in)
- csv (built-in)

## Contributing

To add new data generation capabilities:
1. Create a new generator class inheriting from base functionality
2. Implement required methods for your data type
3. Add appropriate tests and documentation
4. Update this README with usage examples

## License

This data generation code is part of the Robotic Handwriting AI project and follows the same licensing terms.