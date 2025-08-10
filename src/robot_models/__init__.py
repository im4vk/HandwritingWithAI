"""
Robot Models Package
===================

Virtual robot definitions including kinematics, dynamics, and control interfaces.

Classes:
--------
- VirtualRobotArm: Main 7-DOF robot arm simulation
- DexterousHand: Multi-finger hand model
- PenGripper: Specialized pen holding end-effector
- KinematicChain: Forward/inverse kinematics solver
"""

from .virtual_robot import VirtualRobotArm
from .dexterous_hand import DexterousHand  
from .pen_gripper import PenGripper
from .kinematics import KinematicChain

__all__ = [
    'VirtualRobotArm',
    'DexterousHand',
    'PenGripper', 
    'KinematicChain'
]