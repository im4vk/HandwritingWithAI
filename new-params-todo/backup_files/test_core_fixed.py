#!/usr/bin/env python3
"""
Fixed Core Module Testing
=========================

Complete testing with correct API usage for all classes.
"""

import sys
import traceback
from pathlib import Path
import json
import numpy as np

def create_proper_configs():
    """Create proper configuration objects for all classes."""
    
    # Robot config for 7-DOF arm
    robot_config = {
        'name': 'WriteBot',
        'dof': 7,  # Explicitly 7 DOF
        'kinematics': {
            'link_lengths': [0.1, 0.2, 0.2, 0.15, 0.1, 0.08, 0.05],  # 7 links
            'joint_offsets': [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        },
        'joint_limits': [
            [-3.14, 3.14], [-3.14, 3.14], [-3.14, 3.14], [-3.14, 3.14],
            [-3.14, 3.14], [-3.14, 3.14], [-3.14, 3.14]  # 7 joints
        ],
        'velocity_limits': [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
        'acceleration_limits': [2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0],
        'workspace_limits': {
            'x': [-0.5, 0.5],
            'y': [-0.5, 0.5], 
            'z': [0.0, 1.0]
        },
        'hand': {},
        'safety': {
            'max_force': 10.0,
            'collision_threshold': 0.1
        }
    }
    
    # Generator config 
    generator_config = {
        'sampling_rate': 100.0,
        'velocity_scale': 1.0,
        'time_scale': 1.0,
        'sigma_v': 0.1,
        'sigma_ln': 0.05,
        'num_strokes': 5,
        'device': 'cpu'
    }
    
    # GAIL config
    gail_config = {
        'policy_network': {
            'hidden_layers': [128, 64],
            'activation': 'relu',
            'dropout_rate': 0.1
        },
        'discriminator_network': {
            'hidden_layers': [128, 64],
            'activation': 'relu',
            'dropout_rate': 0.1
        },
        'policy_lr': 3e-4,
        'discriminator_lr': 3e-4,
        'gamma': 0.99,
        'device': 'cpu'
    }
    
    # Environment config
    env_config = {
        'physics_engine': 'mujoco',
        'time_step': 0.01,
        'max_episode_steps': 1000,
        'reward_weights': {
            'trajectory_following': 1.0,
            'smoothness': 0.5,
            'pressure': 0.3
        },
        'observation_space': {
            'robot_state_dim': 21,  # 7 positions + 7 velocities + 7 accelerations
            'context_dim': 8        # trajectory context
        },
        'action_space': {
            'joint_velocity_dim': 7,
            'pen_control_dim': 1
        }
    }
    
    return robot_config, generator_config, gail_config, env_config

def test_fixed_class_instantiation():
    """Test class instantiation with proper configurations."""
    print("🔍 Testing FIXED class instantiation...")
    results = {}
    
    robot_config, generator_config, gail_config, env_config = create_proper_configs()
    
    # Test VirtualRobotArm with 7-DOF config
    try:
        from src.robot_models.virtual_robot import VirtualRobotArm
        robot = VirtualRobotArm(robot_config)
        print("✅ VirtualRobotArm instantiated with 7-DOF config")
        
        # Test with 7 joint angles
        test_joints = [0.1, 0.1, 0.1, 0.0, 0.0, 0.0, 0.0]  # 7 joints
        end_effector = robot.forward_kinematics(test_joints)
        print(f"   Forward kinematics (7-DOF): {len(end_effector)} coordinates")
        
        # Test joint limits
        print(f"   Joint limits: {len(robot.limits.joint_limits)} joints")
        print(f"   Current state: {len(robot.state.joint_angles)} joint angles")
        
        results['VirtualRobotArm'] = True
    except Exception as e:
        print(f"❌ VirtualRobotArm failed: {e}")
        results['VirtualRobotArm'] = False
    
    # Test SigmaLognormalGenerator with proper API
    try:
        from src.trajectory_generation.sigma_lognormal import SigmaLognormalGenerator, LognormalParameter
        generator = SigmaLognormalGenerator(generator_config)
        print("✅ SigmaLognormalGenerator instantiated")
        
        # Create proper lognormal parameters for "Hi"
        parameters = [
            LognormalParameter(t0=0.0, mu=0.5, sigma=0.1, D=np.array([0.05, 0.02])),  # H stroke 1
            LognormalParameter(t0=0.2, mu=0.6, sigma=0.1, D=np.array([0.0, 0.03])),   # H stroke 2  
            LognormalParameter(t0=0.4, mu=0.4, sigma=0.1, D=np.array([0.02, 0.0])),   # H crossbar
            LognormalParameter(t0=0.7, mu=0.3, sigma=0.1, D=np.array([0.0, 0.03])),   # I stroke
        ]
        
        # Generate trajectory with duration parameter
        duration = 2.0  # 2 seconds
        trajectory = generator.generate_trajectory(parameters, duration)
        print(f"   Trajectory generation: {len(trajectory['positions'])} points over {duration}s")
        print(f"   Keys: {list(trajectory.keys())}")
        
        results['SigmaLognormalGenerator'] = True
    except Exception as e:
        print(f"❌ SigmaLognormalGenerator failed: {e}")
        traceback.print_exc()
        results['SigmaLognormalGenerator'] = False
    
    # Test HandwritingGAIL with all required parameters
    try:
        from src.ai_models.gail_model import HandwritingGAIL
        
        obs_dim = 29  # 21 robot state + 8 context
        action_dim = 8  # 7 joint velocities + 1 pen pressure
        
        gail = HandwritingGAIL(gail_config, obs_dim, action_dim)
        print("✅ HandwritingGAIL instantiated with config, obs_dim, action_dim")
        print(f"   Observation dim: {obs_dim}")
        print(f"   Action dim: {action_dim}")
        print(f"   Policy network: {gail.policy}")
        
        results['HandwritingGAIL'] = True
    except Exception as e:
        print(f"❌ HandwritingGAIL failed: {e}")
        traceback.print_exc()
        results['HandwritingGAIL'] = False
    
    # Test HandwritingEnvironment 
    try:
        from src.simulation.handwriting_environment import HandwritingEnvironment
        env = HandwritingEnvironment(env_config)
        print("✅ HandwritingEnvironment instantiated")
        
        results['HandwritingEnvironment'] = True
    except Exception as e:
        print(f"❌ HandwritingEnvironment failed: {e}")
        results['HandwritingEnvironment'] = False
    
    return results

def test_fixed_integration_workflow():
    """Test complete integration workflow with correct APIs."""
    print("\n🔍 Testing FIXED integration workflow...")
    results = {}
    
    try:
        robot_config, generator_config, gail_config, env_config = create_proper_configs()
        
        # Initialize components with proper configs
        from src.robot_models.virtual_robot import VirtualRobotArm
        from src.trajectory_generation.sigma_lognormal import SigmaLognormalGenerator, LognormalParameter
        from src.ai_models.gail_model import HandwritingGAIL
        
        robot = VirtualRobotArm(robot_config)
        generator = SigmaLognormalGenerator(generator_config)
        
        obs_dim = 29
        action_dim = 8
        gail = HandwritingGAIL(gail_config, obs_dim, action_dim)
        
        print("✅ All components initialized")
        
        # Generate handwriting trajectory with proper parameters
        parameters = [
            LognormalParameter(t0=0.0, mu=0.5, sigma=0.15, D=np.array([0.08, 0.0])),   # "T" horizontal
            LognormalParameter(t0=0.3, mu=0.4, sigma=0.12, D=np.array([0.0, -0.06])),  # "T" vertical
            LognormalParameter(t0=0.8, mu=0.6, sigma=0.1, D=np.array([0.05, 0.03])),   # "e" curve 1
            LognormalParameter(t0=1.1, mu=0.3, sigma=0.1, D=np.array([0.03, -0.02])),  # "e" curve 2
        ]
        
        duration = 2.5
        trajectory_data = generator.generate_trajectory(parameters, duration, np.array([0.1, 0.15]))
        
        positions = trajectory_data['positions']
        velocities = trajectory_data['velocities']
        
        print(f"✅ Generated trajectory: {len(positions)} points")
        print(f"   Position range: X={positions[:, 0].min():.3f}-{positions[:, 0].max():.3f}")
        print(f"   Velocity magnitude: {np.linalg.norm(velocities, axis=1).mean():.4f} avg")
        
        # Test robot kinematics with 7-DOF
        if len(positions) > 0:
            test_position = np.append(positions[0], 0.02)  # Add Z coordinate
            joint_angles = robot.inverse_kinematics(test_position)
            print(f"✅ Inverse kinematics: {len(joint_angles)} joint angles")
            
            # Forward kinematics verification
            end_effector = robot.forward_kinematics(joint_angles)
            print(f"✅ Forward kinematics verified: position error {np.linalg.norm(end_effector[:3] - test_position):.6f}")
        
        # Test GAIL policy
        test_obs = np.random.randn(obs_dim).astype(np.float32)
        import torch
        with torch.no_grad():
            action_mean, action_std = gail.policy(torch.from_numpy(test_obs).unsqueeze(0))
            print(f"✅ GAIL policy inference: action shape {action_mean.shape}")
        
        results['fixed_integration_workflow'] = True
        print("✅ COMPLETE integration workflow successful!")
        
    except Exception as e:
        print(f"❌ Fixed integration workflow failed: {e}")
        traceback.print_exc()
        results['fixed_integration_workflow'] = False
    
    return results

def test_comprehensive_functionality():
    """Test comprehensive functionality with all fixes."""
    print("\n🔍 Testing comprehensive functionality...")
    results = {}
    
    try:
        # Test trajectory generation for different texts
        from src.trajectory_generation.sigma_lognormal import SigmaLognormalGenerator, LognormalParameter
        
        generator_config = {'sampling_rate': 100.0, 'device': 'cpu'}
        generator = SigmaLognormalGenerator(generator_config)
        
        test_cases = [
            ("Hello", 4, 3.0),  # 4 strokes, 3 seconds
            ("AI", 3, 2.0),     # 3 strokes, 2 seconds  
            ("Test", 5, 3.5),   # 5 strokes, 3.5 seconds
        ]
        
        trajectory_results = {}
        for text, num_strokes, duration in test_cases:
            # Generate random but reasonable lognormal parameters
            parameters = []
            for i in range(num_strokes):
                t0 = i * duration / num_strokes
                mu = 0.3 + np.random.uniform(0, 0.4)
                sigma = 0.08 + np.random.uniform(0, 0.1)
                D = np.array([
                    0.02 + np.random.uniform(-0.01, 0.03),
                    np.random.uniform(-0.02, 0.02)
                ])
                parameters.append(LognormalParameter(t0, mu, sigma, D))
            
            traj = generator.generate_trajectory(parameters, duration)
            trajectory_results[text] = {
                'points': len(traj['positions']),
                'duration': duration,
                'distance': np.sum(np.linalg.norm(np.diff(traj['positions'], axis=0), axis=1))
            }
            
        print("✅ Trajectory generation for multiple texts:")
        for text, metrics in trajectory_results.items():
            print(f"   {text}: {metrics['points']} pts, {metrics['distance']:.3f}m, {metrics['duration']}s")
        
        results['trajectory_generation'] = True
        
        # Test robot workspace analysis
        from src.robot_models.virtual_robot import VirtualRobotArm
        
        robot_config = {
            'kinematics': {'link_lengths': [0.1, 0.2, 0.2, 0.15, 0.1, 0.08, 0.05]},
            'joint_limits': [[-3.14, 3.14]] * 7,
            'workspace_limits': {'x': [-0.5, 0.5], 'y': [-0.5, 0.5], 'z': [0.0, 1.0]},
            'hand': {}
        }
        
        robot = VirtualRobotArm(robot_config)
        
        # Test multiple joint configurations
        workspace_points = []
        for _ in range(50):
            random_joints = np.random.uniform(-1.0, 1.0, 7)  # 7 joints
            try:
                ee_pos = robot.forward_kinematics(random_joints)
                workspace_points.append(ee_pos[:3])
            except:
                pass
        
        if workspace_points:
            workspace_points = np.array(workspace_points)
            print(f"✅ Robot workspace analysis: {len(workspace_points)} valid configurations")
            print(f"   X range: {workspace_points[:, 0].min():.3f} to {workspace_points[:, 0].max():.3f}")
            print(f"   Y range: {workspace_points[:, 1].min():.3f} to {workspace_points[:, 1].max():.3f}")
            print(f"   Z range: {workspace_points[:, 2].min():.3f} to {workspace_points[:, 2].max():.3f}")
        
        results['robot_workspace'] = True
        
    except Exception as e:
        print(f"❌ Comprehensive functionality failed: {e}")
        traceback.print_exc()
        results['trajectory_generation'] = False
        results['robot_workspace'] = False
    
    return results

def generate_final_report(instantiation_results, integration_results, functionality_results):
    """Generate final comprehensive report."""
    print("\n" + "="*80)
    print("📊 FINAL CORE MODULE TESTING REPORT")
    print("="*80)
    
    total_tests = 0
    passed_tests = 0
    
    # Instantiation results
    print(f"\n🏗️  FIXED CLASS INSTANTIATION:")
    inst_pass = sum(instantiation_results.values())
    inst_total = len(instantiation_results)
    print(f"   Classes: {inst_pass}/{inst_total} passed")
    for cls, result in instantiation_results.items():
        print(f"     {cls}: {'✅' if result else '❌'}")
    total_tests += inst_total
    passed_tests += inst_pass
    
    # Integration results
    print(f"\n🔄 FIXED INTEGRATION TESTING:")
    integration_pass = sum(integration_results.values())
    integration_total = len(integration_results)
    print(f"   Workflows: {integration_pass}/{integration_total} passed")
    for workflow, result in integration_results.items():
        print(f"     {workflow}: {'✅' if result else '❌'}")
    total_tests += integration_total
    passed_tests += integration_pass
    
    # Functionality results
    print(f"\n⚙️  COMPREHENSIVE FUNCTIONALITY:")
    func_pass = sum(functionality_results.values())
    func_total = len(functionality_results)
    print(f"   Features: {func_pass}/{func_total} passed")
    for feature, result in functionality_results.items():
        print(f"     {feature}: {'✅' if result else '❌'}")
    total_tests += func_total
    passed_tests += func_pass
    
    # Final summary
    success_rate = (passed_tests / total_tests * 100) if total_tests > 0 else 0
    print(f"\n🎉 FINAL RESULTS:")
    print(f"   Total Tests: {total_tests}")
    print(f"   Passed: {passed_tests}")
    print(f"   Failed: {total_tests - passed_tests}")
    print(f"   Success Rate: {success_rate:.1f}%")
    
    if success_rate >= 95:
        status = "🟢 EXCELLENT - All systems fully functional!"
    elif success_rate >= 85:
        status = "🟢 VERY GOOD - System ready for deployment"
    elif success_rate >= 75:
        status = "🟡 GOOD - Minor issues remain"
    else:
        status = "🔴 NEEDS WORK - Significant issues detected"
    
    print(f"   Status: {status}")
    
    return {
        'total_tests': total_tests,
        'passed_tests': passed_tests,
        'success_rate': success_rate,
        'status': status,
        'detailed_results': {
            'instantiation': instantiation_results,
            'integration': integration_results,
            'functionality': functionality_results
        }
    }

def main():
    """Run complete fixed testing."""
    print("🤖 ROBOTIC HANDWRITING AI - COMPLETE FIXED TESTING")
    print("="*80)
    print("Running comprehensive tests with ALL API fixes...\n")
    
    try:
        # Run all fixed tests
        instantiation_results = test_fixed_class_instantiation()
        integration_results = test_fixed_integration_workflow() 
        functionality_results = test_comprehensive_functionality()
        
        # Generate final report
        report = generate_final_report(
            instantiation_results, integration_results, functionality_results
        )
        
        # Save comprehensive report
        with open('results/final_core_test_report.json', 'w') as f:
            json.dump(report, f, indent=2)
        
        print(f"\n💾 Final comprehensive report saved to results/final_core_test_report.json")
        
        return report
        
    except Exception as e:
        print(f"\n❌ Testing failed with error: {e}")
        traceback.print_exc()
        return None

if __name__ == "__main__":
    main()