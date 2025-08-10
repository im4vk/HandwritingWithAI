"""
Simulation package for robotic handwriting environments.

This package provides physics simulation environments for testing and training
robotic handwriting systems using MuJoCo and PyBullet engines.
"""

from .base_environment import BaseEnvironment
from .handwriting_environment import HandwritingEnvironment
from .physics_engines import MuJoCoEngine, PyBulletEngine
from .environment_config import EnvironmentConfig
from .utils import (
    setup_simulation,
    reset_simulation,
    step_simulation,
    get_simulation_state,
    set_simulation_state
)

__all__ = [
    'BaseEnvironment',
    'HandwritingEnvironment', 
    'MuJoCoEngine',
    'PyBulletEngine',
    'EnvironmentConfig',
    'setup_simulation',
    'reset_simulation',
    'step_simulation',
    'get_simulation_state',
    'set_simulation_state'
]

__version__ = "1.0.0"