"""
AI Models Package
================

Neural networks and learning algorithms for robotic handwriting generation.

Classes:
--------
- HandwritingGAIL: Generative Adversarial Imitation Learning for handwriting
- PhysicsInformedNN: Physics-Informed Neural Network for motion dynamics
- TrajectoryPredictor: Neural network for trajectory prediction
- StyleEncoder: Encoder for handwriting style features
"""

from .gail_model import HandwritingGAIL, GAILPolicy, GAILDiscriminator
from .pinn_model import PhysicsInformedNN, DynamicsConstraints
from .trajectory_predictor import TrajectoryPredictor
from .style_encoder import StyleEncoder
from .base_model import BaseNeuralNetwork
from .utils import ModelUtils, TrainingUtils

__all__ = [
    'HandwritingGAIL',
    'GAILPolicy', 
    'GAILDiscriminator',
    'PhysicsInformedNN',
    'DynamicsConstraints',
    'TrajectoryPredictor',
    'StyleEncoder',
    'BaseNeuralNetwork',
    'ModelUtils',
    'TrainingUtils'
]