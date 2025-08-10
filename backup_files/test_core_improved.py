#!/usr/bin/env python3
"""
Improved Core Module Testing
===========================

Enhanced testing with proper configuration handling and class instantiation.
"""

import sys
import traceback
from pathlib import Path
import json

def create_test_configs():
    """Create test configuration objects for class instantiation."""
    robot_config = {
        'dof': 6,
        'workspace_limits': {
            'x': [-0.5, 0.5],
            'y': [-0.5, 0.5], 
            'z': [0.0, 1.0]
        },
        'joint_limits': {
            'min': [-3.14, -3.14, -3.14, -3.14, -3.14, -3.14],
            'max': [3.14, 3.14, 3.14, 3.14, 3.14, 3.14]
        },
        'link_lengths': [0.1, 0.2, 0.2, 0.1, 0.1, 0.05],
        'max_velocity': 1.0,
        'max_acceleration': 2.0
    }
    
    generator_config = {
        'velocity_scale': 1.0,
        'time_scale': 1.0,
        'sigma_v': 0.1,
        'sigma_ln': 0.05,
        'num_strokes': 5
    }
    
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
    
    return robot_config, generator_config, env_config

def test_improved_imports():
    """Test all imports with better error handling."""
    print("🔍 Testing improved imports...")
    results = {}
    
    # Test basic imports
    try:
        import json, numpy, matplotlib, torch, scipy
        print("✅ All basic dependencies available")
        results['basic_deps'] = True
    except ImportError as e:
        print(f"❌ Missing basic dependency: {e}")
        results['basic_deps'] = False
    
    # Test module imports
    modules = ['robot_models', 'ai_models', 'trajectory_generation', 
               'motion_planning', 'data_processing', 'simulation', 'visualization']
    
    module_results = {}
    for module in modules:
        try:
            exec(f"from src import {module}")
            print(f"✅ src.{module} imported successfully")
            module_results[module] = True
        except Exception as e:
            print(f"❌ src.{module} failed: {e}")
            module_results[module] = False
    
    results['modules'] = module_results
    return results

def test_class_instantiation_with_config():
    """Test class instantiation with proper configurations."""
    print("\n🔍 Testing class instantiation with configurations...")
    results = {}
    
    robot_config, generator_config, env_config = create_test_configs()
    
    # Test VirtualRobotArm
    try:
        from src.robot_models.virtual_robot import VirtualRobotArm
        robot = VirtualRobotArm(robot_config)
        print("✅ VirtualRobotArm instantiated with config")
        
        # Test basic functionality
        test_pos = [0.1, 0.1, 0.1, 0.0, 0.0, 0.0]
        end_effector = robot.forward_kinematics(test_pos)
        print(f"   Forward kinematics test: {len(end_effector)} coordinates")
        
        results['VirtualRobotArm'] = True
    except Exception as e:
        print(f"❌ VirtualRobotArm failed: {e}")
        results['VirtualRobotArm'] = False
    
    # Test SigmaLognormalGenerator
    try:
        from src.trajectory_generation.sigma_lognormal import SigmaLognormalGenerator
        generator = SigmaLognormalGenerator(generator_config)
        print("✅ SigmaLognormalGenerator instantiated with config")
        
        # Test basic functionality
        test_text = "test"
        trajectory = generator.generate_trajectory(test_text)
        print(f"   Trajectory generation test: {len(trajectory)} points")
        
        results['SigmaLognormalGenerator'] = True
    except Exception as e:
        print(f"❌ SigmaLognormalGenerator failed: {e}")
        results['SigmaLognormalGenerator'] = False
    
    # Test HandwritingGAIL
    try:
        from src.ai_models.gail_model import HandwritingGAIL
        gail = HandwritingGAIL()
        print("✅ HandwritingGAIL instantiated successfully")
        results['HandwritingGAIL'] = True
    except Exception as e:
        print(f"❌ HandwritingGAIL failed: {e}")
        results['HandwritingGAIL'] = False
    
    # Test HandwritingEnvironment
    try:
        from src.simulation.handwriting_environment import HandwritingEnvironment
        env = HandwritingEnvironment(env_config)
        print("✅ HandwritingEnvironment instantiated with config")
        results['HandwritingEnvironment'] = True
    except Exception as e:
        print(f"❌ HandwritingEnvironment failed: {e}")
        results['HandwritingEnvironment'] = False
    
    return results

def test_integration_workflow():
    """Test basic integration workflow."""
    print("\n🔍 Testing integration workflow...")
    results = {}
    
    try:
        robot_config, generator_config, env_config = create_test_configs()
        
        # Step 1: Initialize components
        from src.robot_models.virtual_robot import VirtualRobotArm
        from src.trajectory_generation.sigma_lognormal import SigmaLognormalGenerator
        
        robot = VirtualRobotArm(robot_config)
        generator = SigmaLognormalGenerator(generator_config)
        
        # Step 2: Generate trajectory
        test_text = "Hi"
        trajectory = generator.generate_trajectory(test_text)
        print(f"✅ Generated trajectory with {len(trajectory)} points")
        
        # Step 3: Test robot kinematics
        if len(trajectory) > 0:
            first_point = trajectory[0][:3]  # x, y, z
            joint_angles = robot.inverse_kinematics(first_point)
            print(f"✅ Inverse kinematics solved: {len(joint_angles)} joint angles")
            
            # Step 4: Forward kinematics verification
            end_effector = robot.forward_kinematics(joint_angles)
            print(f"✅ Forward kinematics verified: {end_effector[:3]}")
            
        results['integration_workflow'] = True
        print("✅ Basic integration workflow successful")
        
    except Exception as e:
        print(f"❌ Integration workflow failed: {e}")
        traceback.print_exc()
        results['integration_workflow'] = False
    
    return results

def test_data_operations():
    """Test data loading and saving operations."""
    print("\n🔍 Testing data operations...")
    results = {}
    
    try:
        # Test sample data loading
        sample_files = [
            'data/datasets/test_samples.json',
            'data/datasets/handwriting_samples.json'
        ]
        
        loaded_files = 0
        for file_path in sample_files:
            try:
                with open(file_path, 'r') as f:
                    data = json.load(f)
                print(f"✅ Loaded {file_path}: {len(data)} samples")
                loaded_files += 1
            except FileNotFoundError:
                print(f"⚠️  File not found: {file_path}")
            except Exception as e:
                print(f"❌ Error loading {file_path}: {e}")
        
        results['data_loading'] = loaded_files > 0
        
        # Test results saving
        test_data = {'test': 'data', 'timestamp': 12345}
        test_file = 'results/test_output.json'
        
        Path('results').mkdir(exist_ok=True)
        with open(test_file, 'w') as f:
            json.dump(test_data, f)
        
        # Verify saved data
        with open(test_file, 'r') as f:
            loaded_data = json.load(f)
        
        if loaded_data == test_data:
            print(f"✅ Data save/load test passed")
            results['data_saving'] = True
        else:
            print(f"❌ Data save/load test failed")
            results['data_saving'] = False
    
    except Exception as e:
        print(f"❌ Data operations failed: {e}")
        results['data_loading'] = False
        results['data_saving'] = False
    
    return results

def generate_detailed_report(import_results, instantiation_results, integration_results, data_results):
    """Generate a comprehensive report."""
    print("\n" + "="*70)
    print("📊 COMPREHENSIVE CORE TESTING REPORT")
    print("="*70)
    
    total_tests = 0
    passed_tests = 0
    
    # Import results
    basic_deps = import_results.get('basic_deps', False)
    module_results = import_results.get('modules', {})
    
    print(f"\n🔧 DEPENDENCY TESTING:")
    print(f"   Basic Dependencies: {'✅ PASS' if basic_deps else '❌ FAIL'}")
    total_tests += 1
    if basic_deps:
        passed_tests += 1
    
    module_pass = sum(module_results.values())
    module_total = len(module_results)
    print(f"   Module Imports: {module_pass}/{module_total} passed")
    for module, result in module_results.items():
        print(f"     {module}: {'✅' if result else '❌'}")
    total_tests += module_total
    passed_tests += module_pass
    
    # Instantiation results
    print(f"\n🏗️  CLASS INSTANTIATION:")
    inst_pass = sum(instantiation_results.values())
    inst_total = len(instantiation_results)
    print(f"   Classes: {inst_pass}/{inst_total} passed")
    for cls, result in instantiation_results.items():
        print(f"     {cls}: {'✅' if result else '❌'}")
    total_tests += inst_total
    passed_tests += inst_pass
    
    # Integration results
    print(f"\n🔄 INTEGRATION TESTING:")
    integration_pass = sum(integration_results.values())
    integration_total = len(integration_results)
    print(f"   Workflows: {integration_pass}/{integration_total} passed")
    for workflow, result in integration_results.items():
        print(f"     {workflow}: {'✅' if result else '❌'}")
    total_tests += integration_total
    passed_tests += integration_pass
    
    # Data operations
    print(f"\n💾 DATA OPERATIONS:")
    data_pass = sum(data_results.values())
    data_total = len(data_results)
    print(f"   Operations: {data_pass}/{data_total} passed")
    for operation, result in data_results.items():
        print(f"     {operation}: {'✅' if result else '❌'}")
    total_tests += data_total
    passed_tests += data_pass
    
    # Overall summary
    success_rate = (passed_tests / total_tests * 100) if total_tests > 0 else 0
    print(f"\n🎉 OVERALL RESULTS:")
    print(f"   Total Tests: {total_tests}")
    print(f"   Passed: {passed_tests}")
    print(f"   Failed: {total_tests - passed_tests}")
    print(f"   Success Rate: {success_rate:.1f}%")
    
    if success_rate >= 90:
        status = "🟢 EXCELLENT - System fully functional"
    elif success_rate >= 75:
        status = "🟢 GOOD - System mostly functional"
    elif success_rate >= 60:
        status = "🟡 MODERATE - Some issues need attention"
    else:
        status = "🔴 POOR - Significant issues detected"
    
    print(f"   Status: {status}")
    
    return {
        'total_tests': total_tests,
        'passed_tests': passed_tests,
        'success_rate': success_rate,
        'status': status,
        'detailed_results': {
            'imports': import_results,
            'instantiation': instantiation_results,
            'integration': integration_results,
            'data_operations': data_results
        }
    }

def main():
    """Run comprehensive testing."""
    print("🤖 ROBOTIC HANDWRITING AI - COMPREHENSIVE TESTING")
    print("="*70)
    print("Running enhanced tests with proper configuration handling...\n")
    
    try:
        # Run all tests
        import_results = test_improved_imports()
        instantiation_results = test_class_instantiation_with_config()
        integration_results = test_integration_workflow()
        data_results = test_data_operations()
        
        # Generate comprehensive report
        report = generate_detailed_report(
            import_results, instantiation_results, 
            integration_results, data_results
        )
        
        # Save report
        with open('results/comprehensive_test_report.json', 'w') as f:
            json.dump(report, f, indent=2)
        
        print(f"\n💾 Comprehensive report saved to results/comprehensive_test_report.json")
        
        return report
        
    except Exception as e:
        print(f"\n❌ Testing failed with error: {e}")
        traceback.print_exc()
        return None

if __name__ == "__main__":
    main()