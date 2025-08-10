"""
Robotic Handwriting AI System
===========================

A comprehensive system for generating human-like handwriting using
AI-powered virtual robots with biomechanically-inspired motion models.

Modules:
--------
- robot_models: Virtual robot definitions and kinematics
- ai_models: Neural networks and learning algorithms  
- trajectory_generation: Motion planning and path generation
- motion_planning: Inverse kinematics and dynamics
- data_processing: Data handling and preprocessing utilities
- simulation: Physics simulation and environment setup
- visualization: Rendering and real-time display

Author: Your Name
License: MIT
"""

__version__ = "1.0.0"
__author__ = "Your Name"
__email__ = "your.email@example.com"

# Core imports
from . import robot_models
from . import ai_models  
from . import trajectory_generation
from . import motion_planning
from . import data_processing
from . import simulation
from . import visualization

# Main classes
from .robot_models.virtual_robot import VirtualRobotArm
from .ai_models.gail_model import HandwritingGAIL
from .trajectory_generation.sigma_lognormal import SigmaLognormalGenerator
from .simulation.handwriting_environment import HandwritingEnvironment

__all__ = [
    'robot_models',
    'ai_models', 
    'trajectory_generation',
    'motion_planning',
    'data_processing',
    'simulation',
    'visualization',
    'VirtualRobotArm',
    'HandwritingGAIL',
    'SigmaLognormalGenerator',
    'HandwritingEnvironment'
]