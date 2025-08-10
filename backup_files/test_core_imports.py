#!/usr/bin/env python3
"""
Test Core Module Imports
========================

Test script to verify that all core modules and classes can be imported
without errors and basic functionality works.
"""

import sys
import traceback
from pathlib import Path

def test_basic_imports():
    """Test basic Python standard library imports."""
    print("🔍 Testing basic imports...")
    try:
        import json, numpy, matplotlib
        print("✅ Basic imports successful (json, numpy, matplotlib)")
        return True
    except ImportError as e:
        print(f"❌ Basic import failed: {e}")
        return False

def test_src_module_imports():
    """Test importing src modules."""
    print("\n🔍 Testing src module imports...")
    results = {}
    
    modules_to_test = [
        'robot_models',
        'ai_models', 
        'trajectory_generation',
        'motion_planning',
        'data_processing',
        'simulation',
        'visualization'
    ]
    
    for module_name in modules_to_test:
        try:
            module = __import__(f'src.{module_name}', fromlist=[''])
            print(f"✅ Successfully imported src.{module_name}")
            results[module_name] = True
        except ImportError as e:
            print(f"❌ Failed to import src.{module_name}: {e}")
            results[module_name] = False
        except Exception as e:
            print(f"❌ Error importing src.{module_name}: {e}")
            results[module_name] = False
    
    return results

def test_main_class_imports():
    """Test importing main classes."""
    print("\n🔍 Testing main class imports...")
    results = {}
    
    classes_to_test = [
        ('src.robot_models.virtual_robot', 'VirtualRobotArm'),
        ('src.ai_models.gail_model', 'HandwritingGAIL'),
        ('src.trajectory_generation.sigma_lognormal', 'SigmaLognormalGenerator'),
        ('src.simulation.handwriting_environment', 'HandwritingEnvironment')
    ]
    
    for module_path, class_name in classes_to_test:
        try:
            module = __import__(module_path, fromlist=[class_name])
            cls = getattr(module, class_name)
            print(f"✅ Successfully imported {class_name} from {module_path}")
            results[class_name] = True
        except ImportError as e:
            print(f"❌ Failed to import {class_name}: {e}")
            results[class_name] = False
        except AttributeError as e:
            print(f"❌ Class {class_name} not found in module: {e}")
            results[class_name] = False
        except Exception as e:
            print(f"❌ Error importing {class_name}: {e}")
            results[class_name] = False
    
    return results

def test_class_instantiation():
    """Test basic class instantiation."""
    print("\n🔍 Testing class instantiation...")
    results = {}
    
    # Test VirtualRobotArm
    try:
        from src.robot_models.virtual_robot import VirtualRobotArm
        robot = VirtualRobotArm()
        print("✅ VirtualRobotArm instantiated successfully")
        results['VirtualRobotArm'] = True
    except Exception as e:
        print(f"❌ VirtualRobotArm instantiation failed: {e}")
        results['VirtualRobotArm'] = False
    
    # Test SigmaLognormalGenerator
    try:
        from src.trajectory_generation.sigma_lognormal import SigmaLognormalGenerator
        generator = SigmaLognormalGenerator()
        print("✅ SigmaLognormalGenerator instantiated successfully")
        results['SigmaLognormalGenerator'] = True
    except Exception as e:
        print(f"❌ SigmaLognormalGenerator instantiation failed: {e}")
        results['SigmaLognormalGenerator'] = False
    
    # Test HandwritingEnvironment
    try:
        from src.simulation.handwriting_environment import HandwritingEnvironment
        env = HandwritingEnvironment()
        print("✅ HandwritingEnvironment instantiated successfully")
        results['HandwritingEnvironment'] = True
    except Exception as e:
        print(f"❌ HandwritingEnvironment instantiation failed: {e}")
        results['HandwritingEnvironment'] = False
    
    return results

def test_directory_structure():
    """Test directory structure completeness."""
    print("\n🔍 Testing directory structure...")
    
    required_dirs = [
        'src/robot_models',
        'src/ai_models',
        'src/trajectory_generation', 
        'src/motion_planning',
        'src/data_processing',
        'src/simulation',
        'src/visualization',
        'data',
        'config',
        'results',
        'tests'
    ]
    
    results = {}
    for dir_path in required_dirs:
        full_path = Path(dir_path)
        if full_path.exists():
            print(f"✅ Directory exists: {dir_path}")
            results[dir_path] = True
        else:
            print(f"❌ Directory missing: {dir_path}")
            results[dir_path] = False
    
    return results

def generate_test_report(basic_imports, module_imports, class_imports, instantiation, directory_structure):
    """Generate comprehensive test report."""
    print("\n" + "="*60)
    print("📊 CORE MODULE TESTING REPORT")
    print("="*60)
    
    total_tests = 0
    passed_tests = 0
    
    # Basic imports
    print(f"\n🔧 Basic Imports: {'✅ PASS' if basic_imports else '❌ FAIL'}")
    total_tests += 1
    if basic_imports:
        passed_tests += 1
    
    # Module imports
    module_pass = sum(module_imports.values())
    module_total = len(module_imports)
    print(f"\n📦 Module Imports: {module_pass}/{module_total} passed")
    for module, result in module_imports.items():
        print(f"   {module}: {'✅' if result else '❌'}")
    total_tests += module_total
    passed_tests += module_pass
    
    # Class imports
    class_pass = sum(class_imports.values())
    class_total = len(class_imports)
    print(f"\n🏗️  Class Imports: {class_pass}/{class_total} passed")
    for cls, result in class_imports.items():
        print(f"   {cls}: {'✅' if result else '❌'}")
    total_tests += class_total
    passed_tests += class_pass
    
    # Instantiation
    inst_pass = sum(instantiation.values())
    inst_total = len(instantiation)
    print(f"\n🎯 Class Instantiation: {inst_pass}/{inst_total} passed")
    for cls, result in instantiation.items():
        print(f"   {cls}: {'✅' if result else '❌'}")
    total_tests += inst_total
    passed_tests += inst_pass
    
    # Directory structure
    dir_pass = sum(directory_structure.values())
    dir_total = len(directory_structure)
    print(f"\n📁 Directory Structure: {dir_pass}/{dir_total} complete")
    for dir_path, result in directory_structure.items():
        print(f"   {dir_path}: {'✅' if result else '❌'}")
    total_tests += dir_total
    passed_tests += dir_pass
    
    # Summary
    success_rate = (passed_tests / total_tests * 100) if total_tests > 0 else 0
    print(f"\n🎉 OVERALL RESULTS:")
    print(f"   Total Tests: {total_tests}")
    print(f"   Passed: {passed_tests}")
    print(f"   Failed: {total_tests - passed_tests}")
    print(f"   Success Rate: {success_rate:.1f}%")
    
    if success_rate >= 80:
        print("   Status: 🟢 GOOD - System mostly functional")
    elif success_rate >= 60:
        print("   Status: 🟡 MODERATE - Some issues need fixing")
    else:
        print("   Status: 🔴 POOR - Significant issues detected")
    
    return {
        'total_tests': total_tests,
        'passed_tests': passed_tests, 
        'success_rate': success_rate,
        'detailed_results': {
            'basic_imports': basic_imports,
            'module_imports': module_imports,
            'class_imports': class_imports,
            'instantiation': instantiation,
            'directory_structure': directory_structure
        }
    }

def main():
    """Run all core module tests."""
    print("🤖 ROBOTIC HANDWRITING AI - CORE MODULE TESTING")
    print("="*60)
    print("Testing all core modules and functionality...\n")
    
    try:
        # Run all tests
        basic_imports = test_basic_imports()
        module_imports = test_src_module_imports()
        class_imports = test_main_class_imports()
        instantiation = test_class_instantiation()
        directory_structure = test_directory_structure()
        
        # Generate report
        report = generate_test_report(
            basic_imports, module_imports, class_imports, 
            instantiation, directory_structure
        )
        
        # Save report
        import json
        with open('results/core_module_test_report.json', 'w') as f:
            json.dump(report, f, indent=2)
        
        print(f"\n💾 Test report saved to results/core_module_test_report.json")
        
        return report
        
    except Exception as e:
        print(f"\n❌ Testing failed with error: {e}")
        traceback.print_exc()
        return None

if __name__ == "__main__":
    main()