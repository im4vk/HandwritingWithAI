"""
Trajectory Generation Package
============================

Motion planning and trajectory generation algorithms for robotic handwriting.
Implements biomechanically-inspired models for generating human-like handwriting motions.

Classes:
--------
- SigmaLognormalGenerator: Sigma-lognormal velocity model for handwriting
- BiomechanicalModel: Human motor control biomechanical models
- TrajectoryGenerator: Main trajectory generation interface
- BezierCurveGenerator: Smooth curve generation using Bezier curves
- MovementPrimitive: Basic movement primitive for handwriting strokes
"""

from .sigma_lognormal import SigmaLognormalGenerator, LognormalParameter
from .biomechanical_models import BiomechanicalModel, MusculoskeletalModel, NeuralControlModel
from .trajectory_generator import TrajectoryGenerator, HandwritingPath
from .bezier_curves import BezierCurveGenerator, ControlPoint
from .movement_primitives import MovementPrimitive, StrokePrimitive
from .utils import TrajectoryUtils, VelocityProfile, AccelerationProfile

__all__ = [
    'SigmaLognormalGenerator',
    'LognormalParameter',
    'BiomechanicalModel',
    'MusculoskeletalModel', 
    'NeuralControlModel',
    'TrajectoryGenerator',
    'HandwritingPath',
    'BezierCurveGenerator',
    'ControlPoint',
    'MovementPrimitive',
    'StrokePrimitive',
    'TrajectoryUtils',
    'VelocityProfile',
    'AccelerationProfile'
]