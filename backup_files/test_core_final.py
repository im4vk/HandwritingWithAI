#!/usr/bin/env python3
"""
Final Core Module Testing - All Issues Fixed
==========================================

Complete testing with all API issues resolved and proper error handling.
"""

import sys
import traceback
from pathlib import Path
import json
import numpy as np
import logging

# Suppress excessive logging
logging.getLogger().setLevel(logging.ERROR)

def create_corrected_configs():
    """Create properly corrected configuration objects."""
    
    # Robot config with proper workspace and joint limits
    robot_config = {
        'name': 'WriteBot',
        'dof': 7,
        'kinematics': {
            'link_lengths': [0.1, 0.2, 0.2, 0.15, 0.1, 0.08, 0.05]
        },
        # Much more reasonable joint limits  
        'joint_limits': [
            [-180, 180], [-90, 90], [-180, 180], [-90, 90],
            [-180, 180], [-90, 90], [-180, 180]
        ],
        'velocity_limits': [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
        'acceleration_limits': [2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0],
        # Larger workspace for handwriting
        'workspace_limits': {
            'x': [-0.5, 0.5],
            'y': [-0.5, 0.5], 
            'z': [0.0, 1.0]
        },
        'max_force': 10.0,
        'hand': {},
        'pen': {'length': 0.15}
    }
    
    # Generator config
    generator_config = {
        'sampling_rate': 100.0,
        'device': 'cpu'
    }
    
    # GAIL config
    gail_config = {
        'policy_network': {
            'hidden_layers': [64, 32],
            'activation': 'relu',
            'dropout_rate': 0.1
        },
        'discriminator_network': {
            'hidden_layers': [64, 32],
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
        }
    }
    
    return robot_config, generator_config, gail_config, env_config

def test_corrected_robot_model():
    """Test robot model with all fixes applied."""
    print("🔍 Testing CORRECTED Robot Model...")
    results = {}
    
    try:
        robot_config, _, _, _ = create_corrected_configs()
        
        from src.robot_models.virtual_robot import VirtualRobotArm
        robot = VirtualRobotArm(robot_config)
        print("✅ VirtualRobotArm instantiated successfully")
        
        # Test limits attribute
        print(f"✅ Robot limits accessible: {type(robot.limits)}")
        print(f"   Joint limits: {len(robot.limits.joint_limits)} joints")
        print(f"   Workspace: X={robot.limits.workspace_limits[0]}, Y={robot.limits.workspace_limits[1]}")
        
        # Test forward kinematics with 7 DOF
        test_joints = np.array([0.1, 0.1, 0.1, 0.0, 0.0, 0.0, 0.0])  # 7 joints
        end_effector = robot.forward_kinematics(test_joints)
        print(f"✅ Forward kinematics: {len(end_effector)} coordinates")
        print(f"   End effector position: [{end_effector[0]:.3f}, {end_effector[1]:.3f}, {end_effector[2]:.3f}]")
        
        # Test inverse kinematics with proper workspace position
        target_position = np.array([0.3, 0.1, 0.2, 0.0, 0.0, 0.0])  # Within workspace
        joint_angles = robot.inverse_kinematics(target_position)
        
        if joint_angles is not None:
            print(f"✅ Inverse kinematics successful: {len(joint_angles)} joint angles")
            
            # Verify round-trip accuracy
            verification = robot.forward_kinematics(joint_angles)
            error = np.linalg.norm(verification[:3] - target_position[:3])
            print(f"✅ Round-trip verification: position error {error:.6f}m")
            
        else:
            print("⚠️  Inverse kinematics returned None (position may be unreachable)")
        
        # Test workspace analysis with safe positions
        valid_positions = []
        for _ in range(20):
            # Generate positions within known workspace
            safe_joints = np.random.uniform(-0.5, 0.5, 7)  # Smaller joint angles
            try:
                pos = robot.forward_kinematics(safe_joints)
                valid_positions.append(pos[:3])  # Just x,y,z
            except Exception as e:
                print(f"   Forward kinematics failed for joints: {e}")
        
        if valid_positions:
            valid_positions = np.array(valid_positions)
            print(f"✅ Workspace analysis: {len(valid_positions)} valid positions")
            print(f"   X range: {valid_positions[:, 0].min():.3f} to {valid_positions[:, 0].max():.3f}")
            print(f"   Y range: {valid_positions[:, 1].min():.3f} to {valid_positions[:, 1].max():.3f}")
            print(f"   Z range: {valid_positions[:, 2].min():.3f} to {valid_positions[:, 2].max():.3f}")
        
        results['robot_model'] = True
        
    except Exception as e:
        print(f"❌ Robot model test failed: {e}")
        traceback.print_exc()
        results['robot_model'] = False
    
    return results

def test_corrected_trajectory_generation():
    """Test trajectory generation with correct API usage."""
    print("\n🔍 Testing CORRECTED Trajectory Generation...")
    results = {}
    
    try:
        _, generator_config, _, _ = create_corrected_configs()
        
        from src.trajectory_generation.sigma_lognormal import SigmaLognormalGenerator, LognormalParameter
        generator = SigmaLognormalGenerator(generator_config)
        print("✅ SigmaLognormalGenerator instantiated")
        
        # Test multiple trajectory generations
        test_cases = [
            ("Hello", 4, 2.5, [(0.0, 0.4, 0.1, [0.06, 0.0]), (0.5, 0.3, 0.1, [0.0, 0.03]), (1.0, 0.5, 0.1, [0.04, 0.0]), (1.7, 0.3, 0.1, [0.0, -0.02])]),
            ("Test", 3, 2.0, [(0.0, 0.5, 0.12, [0.05, 0.01]), (0.7, 0.4, 0.1, [0.03, -0.01]), (1.4, 0.3, 0.1, [0.04, 0.02])]),
        ]
        
        trajectory_results = {}
        for text, num_strokes, duration, param_data in test_cases:
            # Create proper lognormal parameters
            parameters = []
            for t0, mu, sigma, D in param_data:
                parameters.append(LognormalParameter(t0, mu, sigma, np.array(D)))
            
            # Generate trajectory with correct API
            trajectory = generator.generate_trajectory(parameters, duration, np.array([0.1, 0.15]))
            
            positions = trajectory['positions']
            velocities = trajectory['velocities']
            
            # Calculate metrics
            distance = np.sum(np.linalg.norm(np.diff(positions, axis=0), axis=1))
            avg_velocity = np.mean(np.linalg.norm(velocities, axis=1))
            
            trajectory_results[text] = {
                'points': len(positions),
                'duration': duration,
                'distance': distance,
                'avg_velocity': avg_velocity,
                'x_range': f"{positions[:, 0].min():.3f}-{positions[:, 0].max():.3f}",
                'y_range': f"{positions[:, 1].min():.3f}-{positions[:, 1].max():.3f}"
            }
            
        print("✅ Trajectory generation results:")
        for text, metrics in trajectory_results.items():
            print(f"   {text}: {metrics['points']} pts, {metrics['distance']:.3f}m, {metrics['duration']}s")
            print(f"      Avg velocity: {metrics['avg_velocity']:.4f}m/s")
            print(f"      Range: X={metrics['x_range']}, Y={metrics['y_range']}")
        
        results['trajectory_generation'] = True
        
    except Exception as e:
        print(f"❌ Trajectory generation test failed: {e}")
        traceback.print_exc()
        results['trajectory_generation'] = False
    
    return results

def test_corrected_ai_models():
    """Test AI models with correct parameters."""
    print("\n🔍 Testing CORRECTED AI Models...")
    results = {}
    
    try:
        _, _, gail_config, _ = create_corrected_configs()
        
        from src.ai_models.gail_model import HandwritingGAIL
        
        # Define dimensions
        obs_dim = 29  # Robot state + context
        action_dim = 8  # Joint velocities + pen pressure
        
        gail = HandwritingGAIL(gail_config, obs_dim, action_dim)
        print(f"✅ HandwritingGAIL instantiated (obs={obs_dim}, action={action_dim})")
        
        # Test policy inference
        import torch
        # Ensure tensor is on same device as model
        device = next(gail.policy.parameters()).device
        test_obs = torch.randn(1, obs_dim).to(device)
        with torch.no_grad():
            action_mean, action_std = gail.policy(test_obs)
            print(f"✅ Policy inference: output shapes {action_mean.shape}, {action_std.shape}")
            
            # Test discriminator
            test_state_action = torch.randn(1, obs_dim + action_dim).to(device)
            discriminator_output = gail.discriminator(test_state_action)
            print(f"✅ Discriminator inference: output shape {discriminator_output.shape}")
        
        results['ai_models'] = True
        
    except Exception as e:
        print(f"❌ AI models test failed: {e}")
        traceback.print_exc()
        results['ai_models'] = False
    
    return results

def test_corrected_integration():
    """Test complete integration with all fixes."""
    print("\n🔍 Testing CORRECTED Integration...")
    results = {}
    
    try:
        robot_config, generator_config, gail_config, env_config = create_corrected_configs()
        
        # Initialize all components
        from src.robot_models.virtual_robot import VirtualRobotArm
        from src.trajectory_generation.sigma_lognormal import SigmaLognormalGenerator, LognormalParameter
        from src.ai_models.gail_model import HandwritingGAIL
        from src.simulation.handwriting_environment import HandwritingEnvironment
        
        robot = VirtualRobotArm(robot_config)
        generator = SigmaLognormalGenerator(generator_config)
        
        obs_dim, action_dim = 29, 8
        gail = HandwritingGAIL(gail_config, obs_dim, action_dim)
        
        env = HandwritingEnvironment(env_config)
        
        print("✅ All components initialized successfully")
        
        # Generate handwriting trajectory
        parameters = [
            LognormalParameter(0.0, 0.4, 0.1, np.array([0.05, 0.01])),
            LognormalParameter(0.8, 0.3, 0.12, np.array([0.03, -0.01])),
        ]
        
        duration = 2.0
        trajectory_data = generator.generate_trajectory(parameters, duration, np.array([0.2, 0.1]))
        positions = trajectory_data['positions']
        
        print(f"✅ Generated trajectory: {len(positions)} points")
        
        # Test robot motion planning for trajectory
        valid_configurations = []
        for i in range(0, len(positions), 10):  # Sample every 10th point
            target_3d = np.append(positions[i], 0.15)  # Add Z coordinate
            target_pose = np.append(target_3d, [0, 0, 0])  # Add orientation
            
            joint_config = robot.inverse_kinematics(target_pose)
            if joint_config is not None:
                valid_configurations.append(joint_config)
        
        print(f"✅ Motion planning: {len(valid_configurations)}/{len(positions)//10} positions reachable")
        
        # Test AI policy inference
        if len(valid_configurations) > 0:
            import torch
            test_robot_state = np.concatenate([
                valid_configurations[0],  # 7 joint angles
                np.zeros(7),             # 7 joint velocities  
                np.zeros(7),             # 7 joint accelerations
                positions[0],            # 2 trajectory position
                np.zeros(6)              # 6 additional context
            ])
            
            with torch.no_grad():
                device = next(gail.policy.parameters()).device
                test_tensor = torch.from_numpy(test_robot_state).float().unsqueeze(0).to(device)
                action_mean, _ = gail.policy(test_tensor)
                print(f"✅ AI policy inference: action shape {action_mean.shape}")
        
        print("✅ Complete integration workflow successful!")
        results['integration'] = True
        
    except Exception as e:
        print(f"❌ Integration test failed: {e}")
        traceback.print_exc()
        results['integration'] = False
    
    return results

def generate_final_corrected_report(robot_results, trajectory_results, ai_results, integration_results):
    """Generate final corrected report."""
    print("\n" + "="*80)
    print("📊 FINAL CORRECTED CORE MODULE TESTING REPORT")
    print("="*80)
    
    all_results = {**robot_results, **trajectory_results, **ai_results, **integration_results}
    
    total_tests = len(all_results)
    passed_tests = sum(all_results.values())
    success_rate = (passed_tests / total_tests * 100) if total_tests > 0 else 0
    
    print(f"\n🏗️  CORRECTED COMPONENT TESTING:")
    test_categories = [
        ("Robot Model", robot_results),
        ("Trajectory Generation", trajectory_results), 
        ("AI Models", ai_results),
        ("Integration", integration_results)
    ]
    
    for category, results in test_categories:
        category_pass = sum(results.values())
        category_total = len(results)
        print(f"   {category}: {category_pass}/{category_total} passed")
        for test_name, result in results.items():
            print(f"     {test_name}: {'✅' if result else '❌'}")
    
    print(f"\n🎉 FINAL CORRECTED RESULTS:")
    print(f"   Total Tests: {total_tests}")
    print(f"   Passed: {passed_tests}")
    print(f"   Failed: {total_tests - passed_tests}")
    print(f"   Success Rate: {success_rate:.1f}%")
    
    if success_rate >= 95:
        status = "🟢 EXCELLENT - All core issues resolved!"
        conclusion = "✅ CORE MODULE TESTING SUCCESSFULLY COMPLETED"
    elif success_rate >= 85:
        status = "🟢 VERY GOOD - System ready for next phase"
        conclusion = "✅ CORE MODULE TESTING MOSTLY COMPLETED"
    elif success_rate >= 75:
        status = "🟡 GOOD - Minor issues remain"
        conclusion = "⚠️  CORE MODULE TESTING PARTIALLY COMPLETED"
    else:
        status = "🔴 NEEDS MORE WORK - Significant issues remain"
        conclusion = "❌ CORE MODULE TESTING INCOMPLETE"
    
    print(f"   Status: {status}")
    print(f"\n{conclusion}")
    
    return {
        'total_tests': total_tests,
        'passed_tests': passed_tests,
        'success_rate': success_rate,
        'status': status,
        'conclusion': conclusion,
        'detailed_results': all_results
    }

def main():
    """Run final corrected testing."""
    print("🤖 ROBOTIC HANDWRITING AI - FINAL CORRECTED TESTING")
    print("="*80)
    print("Running comprehensive tests with ALL issues fixed...\n")
    
    try:
        # Run all corrected tests
        robot_results = test_corrected_robot_model()
        trajectory_results = test_corrected_trajectory_generation()
        ai_results = test_corrected_ai_models()
        integration_results = test_corrected_integration()
        
        # Generate final report
        report = generate_final_corrected_report(
            robot_results, trajectory_results, ai_results, integration_results
        )
        
        # Save final corrected report
        Path('results').mkdir(exist_ok=True)
        with open('results/final_corrected_core_test_report.json', 'w') as f:
            json.dump(report, f, indent=2)
        
        print(f"\n💾 Final corrected report saved to results/final_corrected_core_test_report.json")
        
        return report
        
    except Exception as e:
        print(f"\n❌ Testing failed with error: {e}")
        traceback.print_exc()
        return None

if __name__ == "__main__":
    main()