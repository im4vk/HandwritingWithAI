"""
Physics engine implementations for robotic handwriting simulation.

This module provides integration with MuJoCo and PyBullet physics engines,
offering a unified interface for robotic simulation.
"""

import numpy as np
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, List, Tuple
import logging

logger = logging.getLogger(__name__)


class PhysicsEngine(ABC):
    """
    Abstract base class for physics engines.
    
    Provides a common interface for different physics engines used in
    robotic handwriting simulation.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the physics engine.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.timestep = config.get('timestep', 0.001)
        self.gravity = config.get('gravity', [0, 0, -9.81])
        self.is_initialized = False
        
    @abstractmethod
    def initialize(self) -> bool:
        """Initialize the physics engine."""
        pass
    
    @abstractmethod
    def load_robot_model(self, model_path: str) -> bool:
        """Load robot model from file."""
        pass
    
    @abstractmethod
    def step_simulation(self):
        """Step the physics simulation forward."""
        pass
    
    @abstractmethod
    def reset(self):
        """Reset the simulation state."""
        pass
    
    @abstractmethod
    def close(self):
        """Close and cleanup the physics engine."""
        pass
    
    @abstractmethod
    def get_state(self) -> Dict[str, Any]:
        """Get current simulation state."""
        pass
    
    @abstractmethod
    def set_state(self, state: Dict[str, Any]) -> bool:
        """Set simulation state."""
        pass


class MuJoCoEngine(PhysicsEngine):
    """
    MuJoCo physics engine implementation for robotic handwriting.
    
    Provides high-fidelity physics simulation with contact dynamics
    suitable for precise robotic control tasks.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize MuJoCo physics engine."""
        super().__init__(config)
        
        self.model = None
        self.data = None
        self.viewer = None
        
        # Robot and environment object IDs
        self.robot_body_id = None
        self.pen_body_id = None
        self.paper_body_id = None
        
        # Visualization settings
        self.enable_visualization = config.get('enable_visualization', False)
        self.camera_config = config.get('camera_config', {})
        
        # Contact detection settings
        self.contact_threshold = config.get('contact_threshold', 1e-4)
        
    def initialize(self) -> bool:
        """
        Initialize MuJoCo physics engine.
        
        Returns:
            bool: True if initialization successful
        """
        try:
            import mujoco as mj
            self.mj = mj
            
            # Create default model if none specified
            model_path = self.config.get('model_path', self.create_default_model())
            
            # Load model
            self.model = mj.MjModel.from_xml_path(model_path)
            self.data = mj.MjData(self.model)
            
            # Set simulation parameters
            self.model.opt.timestep = self.timestep
            self.model.opt.gravity = self.gravity
            
            # Initialize visualization if enabled
            if self.enable_visualization:
                self.setup_visualization()
            
            self.is_initialized = True
            logger.info("MuJoCo engine initialized successfully")
            return True
            
        except ImportError:
            logger.error("MuJoCo not installed. Please install mujoco package.")
            return False
        except Exception as e:
            logger.error(f"Failed to initialize MuJoCo engine: {e}")
            return False
    
    def create_default_model(self) -> str:
        """
        Create a default MuJoCo model XML for handwriting simulation.
        
        Returns:
            str: Path to the created model file
        """
        xml_content = '''
        <mujoco model="handwriting_robot">
          <option timestep="0.001" gravity="0 0 -9.81"/>
          
          <asset>
            <texture name="grid" type="2d" builtin="checker" rgb1=".1 .2 .3" rgb2=".2 .3 .4" width="300" height="300"/>
            <material name="grid" texture="grid" texrepeat="1 1" reflectance=".2"/>
          </asset>
          
          <worldbody>
            <!-- Ground plane -->
            <geom name="ground" type="plane" size="2 2 .1" material="grid"/>
            
            <!-- Paper surface -->
            <body name="paper" pos="0.5 0 0.01">
              <geom name="paper_geom" type="box" size="0.105 0.148 0.001" rgba="1 1 1 1"/>
            </body>
            
            <!-- Robot base -->
            <body name="robot_base" pos="0 0 0.1">
              <geom name="base_geom" type="cylinder" size="0.05 0.05" rgba="0.5 0.5 0.5 1"/>
              
              <!-- Robot arm link 1 -->
              <body name="link1" pos="0 0 0.05">
                <joint name="joint1" type="hinge" axis="0 0 1" range="-180 180"/>
                <geom name="link1_geom" type="cylinder" size="0.02 0.2" rgba="0.7 0.3 0.3 1"/>
                
                <!-- Robot arm link 2 -->
                <body name="link2" pos="0 0 0.2">
                  <joint name="joint2" type="hinge" axis="1 0 0" range="-90 90"/>
                  <geom name="link2_geom" type="cylinder" size="0.015 0.15" rgba="0.3 0.7 0.3 1"/>
                  
                  <!-- Pen end-effector -->
                  <body name="pen" pos="0 0 0.15">
                    <joint name="pen_joint" type="free"/>
                    <geom name="pen_geom" type="cylinder" size="0.002 0.075" rgba="0 0 1 1"/>
                  </body>
                </body>
              </body>
            </body>
          </worldbody>
          
          <actuator>
            <motor name="motor1" joint="joint1" gear="1"/>
            <motor name="motor2" joint="joint2" gear="1"/>
          </actuator>
        </mujoco>
        '''
        
        # Save to temporary file
        import tempfile
        import os
        
        model_dir = os.path.join(os.getcwd(), 'robotic-handwriting-ai', 'models')
        os.makedirs(model_dir, exist_ok=True)
        
        model_path = os.path.join(model_dir, 'default_handwriting_robot.xml')
        with open(model_path, 'w') as f:
            f.write(xml_content)
        
        return model_path
    
    def load_robot_model(self, model_path: str) -> bool:
        """
        Load robot model from XML file.
        
        Args:
            model_path: Path to MuJoCo XML model file
            
        Returns:
            bool: True if model loaded successfully
        """
        try:
            self.model = self.mj.MjModel.from_xml_path(model_path)
            self.data = self.mj.MjData(self.model)
            
            # Find important body IDs
            self.robot_body_id = self.mj.mj_name2id(self.model, self.mj.mjtObj.mjOBJ_BODY, "robot_base")
            self.pen_body_id = self.mj.mj_name2id(self.model, self.mj.mjtObj.mjOBJ_BODY, "pen")
            self.paper_body_id = self.mj.mj_name2id(self.model, self.mj.mjtObj.mjOBJ_BODY, "paper")
            
            logger.info(f"Robot model loaded: {model_path}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to load robot model: {e}")
            return False
    
    def setup_visualization(self):
        """Setup MuJoCo visualization."""
        try:
            import mujoco.viewer as viewer
            self.viewer = viewer.launch_passive(self.model, self.data)
            logger.info("MuJoCo viewer initialized")
        except Exception as e:
            logger.warning(f"Failed to setup visualization: {e}")
            self.enable_visualization = False
    
    def step_simulation(self):
        """Step MuJoCo simulation forward."""
        if not self.is_initialized:
            return
        
        self.mj.mj_step(self.model, self.data)
        
        # Update viewer if enabled
        if self.enable_visualization and self.viewer is not None:
            self.viewer.sync()
    
    def reset(self):
        """Reset MuJoCo simulation to initial state."""
        if not self.is_initialized:
            return
        
        self.mj.mj_resetData(self.model, self.data)
        self.mj.mj_forward(self.model, self.data)
    
    def get_pen_position(self) -> np.ndarray:
        """Get current pen position."""
        if self.pen_body_id is not None:
            return self.data.xpos[self.pen_body_id].copy()
        return np.zeros(3)
    
    def get_pen_velocity(self) -> np.ndarray:
        """Get current pen velocity."""
        if self.pen_body_id is not None:
            body_id = self.pen_body_id
            # Get linear velocity from body velocity
            return self.data.cvel[body_id, 3:6].copy()
        return np.zeros(3)
    
    def get_pen_orientation(self) -> np.ndarray:
        """Get current pen orientation (quaternion)."""
        if self.pen_body_id is not None:
            return self.data.xquat[self.pen_body_id].copy()
        return np.array([1, 0, 0, 0])  # Identity quaternion
    
    def set_pen_position(self, position: np.ndarray):
        """Set pen position."""
        if self.pen_body_id is not None:
            self.data.xpos[self.pen_body_id] = position
            self.mj.mj_forward(self.model, self.data)
    
    def move_pen_relative(self, movement: np.ndarray, pressure: float):
        """
        Move pen by relative amount with specified pressure.
        
        Args:
            movement: [dx, dy, dz] relative movement
            pressure: Pressure force to apply (0-1)
        """
        if self.pen_body_id is not None:
            current_pos = self.get_pen_position()
            new_pos = current_pos + movement
            self.set_pen_position(new_pos)
            
            # Apply pressure force (simplified)
            if pressure > 0:
                force = np.array([0, 0, -pressure * 10.0])  # Downward force
                self.data.xfrc_applied[self.pen_body_id, :3] = force
    
    def is_pen_in_contact(self) -> bool:
        """Check if pen is in contact with paper."""
        if self.pen_body_id is None or self.paper_body_id is None:
            return False
        
        # Check contact forces
        for i in range(self.data.ncon):
            contact = self.data.contact[i]
            geom1 = contact.geom1
            geom2 = contact.geom2
            
            # Get body IDs from geometry IDs
            body1 = self.model.geom_bodyid[geom1]
            body2 = self.model.geom_bodyid[geom2]
            
            if (body1 == self.pen_body_id and body2 == self.paper_body_id) or \
               (body1 == self.paper_body_id and body2 == self.pen_body_id):
                if contact.dist < self.contact_threshold:
                    return True
        
        return False
    
    def get_contact_force(self) -> float:
        """Get contact force magnitude."""
        if not self.is_pen_in_contact():
            return 0.0
        
        # Find contact and return force magnitude
        for i in range(self.data.ncon):
            contact = self.data.contact[i]
            geom1 = contact.geom1
            geom2 = contact.geom2
            
            body1 = self.model.geom_bodyid[geom1]
            body2 = self.model.geom_bodyid[geom2]
            
            if (body1 == self.pen_body_id and body2 == self.paper_body_id) or \
               (body1 == self.paper_body_id and body2 == self.pen_body_id):
                # Get contact force
                c_array = np.zeros(6, dtype=np.float64)
                self.mj.mj_contactForce(self.model, self.data, i, c_array)
                return np.linalg.norm(c_array[:3])
        
        return 0.0
    
    def create_paper_surface(self, config: Dict[str, Any]):
        """Create paper surface (already exists in default model)."""
        pass  # Paper is part of the default model
    
    def setup_pen_end_effector(self, config: Dict[str, Any]):
        """Setup pen end-effector properties (already exists in default model)."""
        pass  # Pen is part of the default model
    
    def get_state(self) -> Dict[str, Any]:
        """Get current MuJoCo simulation state."""
        return {
            'qpos': self.data.qpos.copy(),
            'qvel': self.data.qvel.copy(),
            'time': self.data.time
        }
    
    def set_state(self, state: Dict[str, Any]) -> bool:
        """Set MuJoCo simulation state."""
        try:
            self.data.qpos[:] = state['qpos']
            self.data.qvel[:] = state['qvel']
            self.data.time = state['time']
            self.mj.mj_forward(self.model, self.data)
            return True
        except Exception as e:
            logger.error(f"Failed to set MuJoCo state: {e}")
            return False
    
    def close(self):
        """Close MuJoCo engine."""
        if self.viewer is not None:
            self.viewer.close()
            self.viewer = None
        
        self.is_initialized = False
        logger.info("MuJoCo engine closed")


class PyBulletEngine(PhysicsEngine):
    """
    PyBullet physics engine implementation for robotic handwriting.
    
    Provides an alternative physics simulation backend with good
    performance and visualization capabilities.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize PyBullet physics engine."""
        super().__init__(config)
        
        self.client_id = None
        self.robot_id = None
        self.pen_id = None
        self.paper_id = None
        
        # Visualization settings
        self.enable_visualization = config.get('enable_visualization', False)
        
    def initialize(self) -> bool:
        """
        Initialize PyBullet physics engine.
        
        Returns:
            bool: True if initialization successful
        """
        try:
            import pybullet as p
            self.p = p
            
            # Connect to physics server
            if self.enable_visualization:
                self.client_id = p.connect(p.GUI)
            else:
                self.client_id = p.connect(p.DIRECT)
            
            # Set gravity and timestep
            p.setGravity(*self.gravity)
            p.setTimeStep(self.timestep)
            
            # Load ground plane
            p.loadURDF("plane.urdf")
            
            self.is_initialized = True
            logger.info("PyBullet engine initialized successfully")
            return True
            
        except ImportError:
            logger.error("PyBullet not installed. Please install pybullet package.")
            return False
        except Exception as e:
            logger.error(f"Failed to initialize PyBullet engine: {e}")
            return False
    
    def load_robot_model(self, model_path: str) -> bool:
        """
        Load robot model from URDF file.
        
        Args:
            model_path: Path to URDF model file
            
        Returns:
            bool: True if model loaded successfully
        """
        try:
            # For now, create a simple robot programmatically
            self.create_simple_robot()
            logger.info("Simple robot created in PyBullet")
            return True
            
        except Exception as e:
            logger.error(f"Failed to load robot model: {e}")
            return False
    
    def create_simple_robot(self):
        """Create a simple 2-DOF robot arm for handwriting."""
        # Create robot base
        base_collision = self.p.createCollisionShape(self.p.GEOM_CYLINDER, radius=0.05, height=0.1)
        base_visual = self.p.createVisualShape(self.p.GEOM_CYLINDER, radius=0.05, length=0.1, rgbaColor=[0.5, 0.5, 0.5, 1])
        
        # Create paper surface
        paper_collision = self.p.createCollisionShape(self.p.GEOM_BOX, halfExtents=[0.105, 0.148, 0.001])
        paper_visual = self.p.createVisualShape(self.p.GEOM_BOX, halfExtents=[0.105, 0.148, 0.001], rgbaColor=[1, 1, 1, 1])
        self.paper_id = self.p.createMultiBody(baseMass=0, baseCollisionShapeIndex=paper_collision, 
                                               baseVisualShapeIndex=paper_visual, basePosition=[0.5, 0, 0.01])
        
        # Create a simple pen object
        pen_collision = self.p.createCollisionShape(self.p.GEOM_CYLINDER, radius=0.002, height=0.15)
        pen_visual = self.p.createVisualShape(self.p.GEOM_CYLINDER, radius=0.002, length=0.15, rgbaColor=[0, 0, 1, 1])
        self.pen_id = self.p.createMultiBody(baseMass=0.02, baseCollisionShapeIndex=pen_collision,
                                             baseVisualShapeIndex=pen_visual, basePosition=[0.4, 0, 0.1])
    
    def step_simulation(self):
        """Step PyBullet simulation forward."""
        if not self.is_initialized:
            return
        
        self.p.stepSimulation()
    
    def reset(self):
        """Reset PyBullet simulation."""
        if not self.is_initialized:
            return
        
        # Reset pen position
        if self.pen_id is not None:
            self.p.resetBasePositionAndOrientation(self.pen_id, [0.4, 0, 0.1], [0, 0, 0, 1])
    
    def get_pen_position(self) -> np.ndarray:
        """Get current pen position."""
        if self.pen_id is not None:
            pos, _ = self.p.getBasePositionAndOrientation(self.pen_id)
            return np.array(pos)
        return np.zeros(3)
    
    def get_pen_velocity(self) -> np.ndarray:
        """Get current pen velocity."""
        if self.pen_id is not None:
            vel, _ = self.p.getBaseVelocity(self.pen_id)
            return np.array(vel)
        return np.zeros(3)
    
    def get_pen_orientation(self) -> np.ndarray:
        """Get current pen orientation (quaternion)."""
        if self.pen_id is not None:
            _, orn = self.p.getBasePositionAndOrientation(self.pen_id)
            return np.array(orn)
        return np.array([1, 0, 0, 0])
    
    def set_pen_position(self, position: np.ndarray):
        """Set pen position."""
        if self.pen_id is not None:
            current_orn = self.get_pen_orientation()
            self.p.resetBasePositionAndOrientation(self.pen_id, position, current_orn)
    
    def move_pen_relative(self, movement: np.ndarray, pressure: float):
        """Move pen by relative amount with specified pressure."""
        current_pos = self.get_pen_position()
        new_pos = current_pos + movement
        self.set_pen_position(new_pos)
        
        # Apply pressure force
        if pressure > 0:
            force = np.array([0, 0, -pressure * 10.0])
            self.p.applyExternalForce(self.pen_id, -1, force, [0, 0, 0], self.p.WORLD_FRAME)
    
    def is_pen_in_contact(self) -> bool:
        """Check if pen is in contact with paper."""
        if self.pen_id is None or self.paper_id is None:
            return False
        
        contacts = self.p.getContactPoints(self.pen_id, self.paper_id)
        return len(contacts) > 0
    
    def get_contact_force(self) -> float:
        """Get contact force magnitude."""
        if not self.is_pen_in_contact():
            return 0.0
        
        contacts = self.p.getContactPoints(self.pen_id, self.paper_id)
        if contacts:
            return abs(contacts[0][9])  # Normal force
        return 0.0
    
    def create_paper_surface(self, config: Dict[str, Any]):
        """Create paper surface (already created in create_simple_robot)."""
        pass
    
    def setup_pen_end_effector(self, config: Dict[str, Any]):
        """Setup pen end-effector (already created in create_simple_robot)."""
        pass
    
    def get_state(self) -> Dict[str, Any]:
        """Get current PyBullet simulation state."""
        state = {}
        if self.pen_id is not None:
            pos, orn = self.p.getBasePositionAndOrientation(self.pen_id)
            vel, ang_vel = self.p.getBaseVelocity(self.pen_id)
            state['pen_position'] = pos
            state['pen_orientation'] = orn
            state['pen_velocity'] = vel
            state['pen_angular_velocity'] = ang_vel
        return state
    
    def set_state(self, state: Dict[str, Any]) -> bool:
        """Set PyBullet simulation state."""
        try:
            if self.pen_id is not None and 'pen_position' in state:
                self.p.resetBasePositionAndOrientation(
                    self.pen_id, 
                    state['pen_position'], 
                    state['pen_orientation']
                )
                if 'pen_velocity' in state:
                    self.p.resetBaseVelocity(
                        self.pen_id,
                        state['pen_velocity'],
                        state['pen_angular_velocity']
                    )
            return True
        except Exception as e:
            logger.error(f"Failed to set PyBullet state: {e}")
            return False
    
    def close(self):
        """Close PyBullet engine."""
        if self.client_id is not None:
            self.p.disconnect(self.client_id)
            self.client_id = None
        
        self.is_initialized = False
        logger.info("PyBullet engine closed")


class EnhancedMockEngine(PhysicsEngine):
    """
    Enhanced mock physics engine that provides realistic simulation behavior
    without requiring external physics engine dependencies.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize enhanced mock physics engine."""
        super().__init__(config)
        
        # Simulation parameters
        self.gravity = config.get('gravity', [0, 0, -9.81])
        self.timestep = config.get('timestep', 0.001)
        self.damping = config.get('damping', 0.1)
        self.friction = config.get('friction', 0.3)
        
        # Robot state
        self.robot_state = {
            'joint_positions': np.zeros(7),
            'joint_velocities': np.zeros(7),
            'joint_forces': np.zeros(7),
            'pen_position': np.array([0.5, 0.0, 0.02]),
            'pen_velocity': np.zeros(3),
            'pen_force': 0.0,
            'contact_state': False
        }
        
        # Contact model parameters
        self.contact_stiffness = 1000.0  # N/m
        self.contact_damping = 50.0      # N*s/m
        self.paper_height = 0.001        # Paper thickness
        
        # Physics simulation state
        self.time = 0.0
        self.step_count = 0
        
        logger.info("Enhanced mock physics engine initialized")
    
    def initialize(self) -> bool:
        """Initialize the mock physics simulation."""
        try:
            # Reset all states
            self.robot_state['joint_positions'] = np.zeros(7)
            self.robot_state['joint_velocities'] = np.zeros(7)
            self.robot_state['joint_forces'] = np.zeros(7)
            self.robot_state['pen_position'] = np.array([0.5, 0.0, 0.02])
            self.robot_state['pen_velocity'] = np.zeros(3)
            self.robot_state['pen_force'] = 0.0
            self.robot_state['contact_state'] = False
            
            self.time = 0.0
            self.step_count = 0
            self.is_initialized = True
            
            logger.info("✅ Enhanced mock physics engine initialized successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize enhanced mock engine: {e}")
            return False
    
    def step_simulation(self):
        """Step the mock simulation forward with realistic physics."""
        if not self.is_initialized:
            return
        
        # Update time
        self.time += self.timestep
        self.step_count += 1
        
        # Apply realistic joint dynamics with damping
        for i in range(7):
            # Simple PD controller simulation
            target_vel = self.robot_state['joint_velocities'][i]
            actual_vel = target_vel * (1 - self.damping * self.timestep)
            
            # Apply velocity to position
            self.robot_state['joint_positions'][i] += actual_vel * self.timestep
            self.robot_state['joint_velocities'][i] = actual_vel
            
            # Add small noise for realism
            noise = np.random.normal(0, 0.001)
            self.robot_state['joint_positions'][i] += noise
        
        # Update pen position based on joint kinematics (simplified)
        # This simulates forward kinematics calculation
        base_pos = np.array([0.5, 0.0, 0.0])
        arm_reach = 0.6  # Total arm reach
        
        # Simplified kinematics based on first few joints
        q1, q2, q3 = self.robot_state['joint_positions'][:3]
        
        pen_x = base_pos[0] + arm_reach * np.cos(q1) * np.cos(q2)
        pen_y = base_pos[1] + arm_reach * np.sin(q1) * np.cos(q2)
        pen_z = base_pos[2] + 0.5 + arm_reach * np.sin(q2)  # Base height + arm
        
        # Update pen position with smoothing
        target_pos = np.array([pen_x, pen_y, pen_z])
        self.robot_state['pen_position'] = (
            0.9 * self.robot_state['pen_position'] + 0.1 * target_pos
        )
        
        # Calculate pen velocity
        if hasattr(self, '_prev_pen_pos'):
            self.robot_state['pen_velocity'] = (
                self.robot_state['pen_position'] - self._prev_pen_pos
            ) / self.timestep
        self._prev_pen_pos = self.robot_state['pen_position'].copy()
        
        # Contact detection and force calculation
        pen_height = self.robot_state['pen_position'][2]
        paper_surface = self.paper_height
        
        if pen_height <= paper_surface + 0.002:  # 2mm contact threshold
            self.robot_state['contact_state'] = True
            
            # Calculate contact force based on penetration depth
            penetration = max(0, paper_surface - pen_height)
            contact_force = self.contact_stiffness * penetration
            
            # Add velocity damping
            pen_vel_z = self.robot_state['pen_velocity'][2]
            damping_force = -self.contact_damping * pen_vel_z
            
            self.robot_state['pen_force'] = contact_force + damping_force
            
            # Prevent pen from going below paper
            if pen_height < paper_surface:
                self.robot_state['pen_position'][2] = paper_surface
                self.robot_state['pen_velocity'][2] = 0
        else:
            self.robot_state['contact_state'] = False
            self.robot_state['pen_force'] = 0.0
        
        # Add realistic noise to forces
        if self.robot_state['contact_state']:
            force_noise = np.random.normal(0, 0.1)
            self.robot_state['pen_force'] += force_noise
    
    def set_joint_positions(self, positions: np.ndarray):
        """Set robot joint positions."""
        if len(positions) == 7:
            self.robot_state['joint_positions'] = positions.copy()
    
    def set_joint_velocities(self, velocities: np.ndarray):
        """Set robot joint velocities."""
        if len(velocities) == 7:
            self.robot_state['joint_velocities'] = velocities.copy()
    
    def get_state(self) -> Dict[str, Any]:
        """Get current simulation state with realistic values."""
        return {
            'robot_joint_positions': self.robot_state['joint_positions'].copy(),
            'robot_joint_velocities': self.robot_state['joint_velocities'].copy(),
            'robot_joint_forces': self.robot_state['joint_forces'].copy(),
            'pen_position': self.robot_state['pen_position'].copy(),
            'pen_velocity': self.robot_state['pen_velocity'].copy(),
            'pen_force': self.robot_state['pen_force'],
            'contact_state': self.robot_state['contact_state'],
            'simulation_time': self.time,
            'step_count': self.step_count,
            # Add realistic physics metrics
            'contact_normal': np.array([0, 0, 1]) if self.robot_state['contact_state'] else np.zeros(3),
            'friction_force': self.friction * self.robot_state['pen_force'] if self.robot_state['contact_state'] else 0.0,
            'kinetic_energy': 0.5 * np.sum(self.robot_state['joint_velocities']**2),
            'potential_energy': 9.81 * self.robot_state['pen_position'][2] * 0.1  # Assume 0.1kg pen
        }
    
    def set_state(self, state: Dict[str, Any]) -> bool:
        """Set simulation state."""
        try:
            if 'robot_joint_positions' in state:
                self.robot_state['joint_positions'] = np.array(state['robot_joint_positions'])
            if 'robot_joint_velocities' in state:
                self.robot_state['joint_velocities'] = np.array(state['robot_joint_velocities'])
            if 'pen_position' in state:
                self.robot_state['pen_position'] = np.array(state['pen_position'])
            return True
        except Exception as e:
            logger.error(f"Failed to set enhanced mock state: {e}")
            return False
    
    def load_robot_model(self, model_path: str = None) -> bool:
        """Load robot model (mock implementation)."""
        logger.info("✅ Enhanced mock robot model loaded successfully")
        return True
    
    def create_paper_surface(self, *args, **kwargs):
        """Create paper surface (mock implementation)."""
        logger.info("✅ Enhanced mock paper surface created")
        return True
    
    def setup_pen_end_effector(self, *args, **kwargs):
        """Setup pen end effector (mock implementation)."""
        logger.info("✅ Enhanced mock pen end effector setup")
        return True
    
    def set_pen_position(self, position: np.ndarray):
        """Set pen position in simulation."""
        if len(position) >= 3:
            self.robot_state['pen_position'] = np.array(position[:3])
            logger.debug(f"Enhanced mock pen position set to {position[:3]}")
    
    def get_pen_position(self) -> np.ndarray:
        """Get current pen position."""
        return self.robot_state['pen_position'].copy()
    
    def set_pen_force(self, force: float):
        """Set pen contact force."""
        self.robot_state['pen_force'] = float(force)
        logger.debug(f"Enhanced mock pen force set to {force}")
    
    def get_contact_state(self) -> bool:
        """Get current contact state."""
        return self.robot_state['contact_state']
    
    def get_pen_velocity(self) -> np.ndarray:
        """Get current pen velocity."""
        return self.robot_state['pen_velocity'].copy()
    
    def get_pen_force(self) -> float:
        """Get current pen contact force."""
        return self.robot_state['pen_force']
    
    def get_pen_orientation(self) -> np.ndarray:
        """Get current pen orientation (quaternion)."""
        # Default orientation (pointing down)
        return np.array([0, 0, 0, 1])  # [x, y, z, w] quaternion
    
    def is_pen_in_contact(self) -> bool:
        """Check if pen is in contact with paper."""
        return self.robot_state['contact_state']
    
    def move_pen_relative(self, movement: np.ndarray, pressure: float):
        """Move pen relative to current position with specified pressure."""
        if len(movement) >= 3:
            current_pos = self.robot_state['pen_position']
            new_position = current_pos + np.array(movement[:3])
            self.set_pen_position(new_position)
            self.set_pen_force(pressure)
            
            # Update pen velocity based on movement
            self.robot_state['pen_velocity'] = np.array(movement[:3]) / self.timestep
            
            # Update contact state based on Z position and pressure
            paper_z = 0.01  # Paper height
            pen_z = new_position[2]
            if pen_z <= paper_z + 0.001 and pressure > 0.1:  # Close to paper with pressure
                self.robot_state['contact_state'] = True
            else:
                self.robot_state['contact_state'] = False
                
            logger.debug(f"Enhanced mock pen moved by {movement[:3]} with pressure {pressure}")
    
    def get_contact_force(self) -> float:
        """Get current contact force."""
        return self.robot_state['pen_force']
    
    def reset(self):
        """Reset the simulation to initial state."""
        self.robot_state['joint_positions'] = np.zeros(7)
        self.robot_state['joint_velocities'] = np.zeros(7)
        self.robot_state['joint_forces'] = np.zeros(7)
        self.robot_state['pen_position'] = np.array([0.5, 0.0, 0.02])
        self.robot_state['pen_velocity'] = np.zeros(3)
        self.robot_state['pen_force'] = 0.0
        self.robot_state['contact_state'] = False
        self.time = 0.0
        self.step_count = 0
        logger.info("Enhanced mock simulation reset")
    
    def close(self):
        """Close the mock engine."""
        self.is_initialized = False
        logger.info("Enhanced mock physics engine closed")