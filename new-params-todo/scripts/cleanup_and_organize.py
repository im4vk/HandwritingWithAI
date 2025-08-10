#!/usr/bin/env python3
"""
Robotic Handwriting AI - Project Cleanup and Organization
=========================================================

This script cleans up the project structure by:
1. Removing empty directories
2. Consolidating duplicate directory structures  
3. Removing temporary and backup files
4. Organizing files into logical locations
5. Creating a clean, professional project structure
"""

import os
import shutil
import subprocess
from pathlib import Path
import json

def print_header(title):
    """Print a formatted header."""
    print(f"\n{'='*60}")
    print(f"🧹 {title}")
    print('='*60)

def print_action(action, item=""):
    """Print a formatted action."""
    print(f"   {action} {item}")

def remove_empty_directories():
    """Remove all empty directories."""
    print_header("REMOVING EMPTY DIRECTORIES")
    
    # Get list of empty directories
    result = subprocess.run(['find', '.', '-type', 'd', '-empty'], 
                          capture_output=True, text=True)
    empty_dirs = [d for d in result.stdout.strip().split('\n') if d and d != '.']
    
    removed_count = 0
    for empty_dir in empty_dirs:
        try:
            if os.path.exists(empty_dir):
                os.rmdir(empty_dir)
                print_action("✅ Removed empty directory:", empty_dir)
                removed_count += 1
        except OSError as e:
            print_action("⚠️  Could not remove:", f"{empty_dir} ({e})")
    
    print_action(f"📊 Total empty directories removed: {removed_count}")

def consolidate_nested_structure():
    """Fix the duplicated robotic-handwriting-ai/robotic-handwriting-ai structure."""
    print_header("CONSOLIDATING NESTED STRUCTURE")
    
    nested_path = "robotic-handwriting-ai"
    if os.path.exists(nested_path):
        nested_models = os.path.join(nested_path, "models")
        if os.path.exists(nested_models):
            # Move the XML file to the correct models directory
            xml_file = os.path.join(nested_models, "default_handwriting_robot.xml")
            if os.path.exists(xml_file):
                target_dir = "models"
                os.makedirs(target_dir, exist_ok=True)
                target_file = os.path.join(target_dir, "default_handwriting_robot.xml")
                shutil.move(xml_file, target_file)
                print_action("✅ Moved robot model file:", f"{xml_file} → {target_file}")
        
        # Remove the nested directory structure
        try:
            shutil.rmtree(nested_path)
            print_action("✅ Removed nested directory:", nested_path)
        except Exception as e:
            print_action("⚠️  Could not remove nested directory:", str(e))

def remove_temporary_files():
    """Remove temporary, backup, and test files."""
    print_header("REMOVING TEMPORARY AND BACKUP FILES")
    
    # Files to remove (temporary and backup files)
    temp_files = [
        "demo_components_fixed.py",
        "demo_components_old_backup.py", 
        "final_complete_fix.py",
        "test_core_final.py",
        "test_core_fixed.py",
        "test_core_improved.py",
        "test_core_imports.py",
        "fix_data_files.py"
    ]
    
    removed_count = 0
    for temp_file in temp_files:
        if os.path.exists(temp_file):
            os.remove(temp_file)
            print_action("✅ Removed temporary file:", temp_file)
            removed_count += 1
    
    print_action(f"📊 Total temporary files removed: {removed_count}")

def organize_demo_files():
    """Organize demo files into a dedicated directory."""
    print_header("ORGANIZING DEMO FILES")
    
    # Create demos directory
    demos_dir = "demos"
    os.makedirs(demos_dir, exist_ok=True)
    
    demo_files = [
        "demo_working.py",
        "demo_end_to_end.py", 
        "demo_components.py",
        "demo_simple.py",
        "demo_fixed_trajectories.py"
    ]
    
    moved_count = 0
    for demo_file in demo_files:
        if os.path.exists(demo_file):
            target = os.path.join(demos_dir, demo_file)
            shutil.move(demo_file, target)
            print_action("✅ Moved demo file:", f"{demo_file} → {target}")
            moved_count += 1
    
    print_action(f"📊 Total demo files organized: {moved_count}")

def organize_test_files():
    """Organize test files into the tests directory."""
    print_header("ORGANIZING TEST FILES")
    
    # Ensure tests directory exists
    tests_dir = "tests"
    os.makedirs(tests_dir, exist_ok=True)
    
    test_files = [
        "test_visualization_system.py",
        "comprehensive_demo_test.py"
    ]
    
    moved_count = 0
    for test_file in test_files:
        if os.path.exists(test_file):
            target = os.path.join(tests_dir, test_file)
            shutil.move(test_file, target)
            print_action("✅ Moved test file:", f"{test_file} → {target}")
            moved_count += 1
    
    print_action(f"📊 Total test files organized: {moved_count}")

def organize_documentation():
    """Organize documentation files."""
    print_header("ORGANIZING DOCUMENTATION")
    
    # Create docs directory if it doesn't exist
    docs_dir = "docs"
    os.makedirs(docs_dir, exist_ok=True)
    
    doc_files = [
        ("SUCCESS_SUMMARY.md", "SUCCESS_SUMMARY.md"),
        ("SYSTEM_OVERVIEW.md", "SYSTEM_OVERVIEW.md"),
    ]
    
    moved_count = 0
    for doc_file, target_name in doc_files:
        if os.path.exists(doc_file):
            target = os.path.join(docs_dir, target_name)
            shutil.move(doc_file, target)
            print_action("✅ Moved documentation:", f"{doc_file} → {target}")
            moved_count += 1
    
    print_action(f"📊 Total documentation files organized: {moved_count}")

def create_directory_structure():
    """Ensure proper directory structure exists."""
    print_header("CREATING PROPER DIRECTORY STRUCTURE")
    
    # Core directories that should exist
    core_dirs = [
        "src",
        "tests", 
        "demos",
        "docs",
        "config",
        "data/datasets",
        "data/samples", 
        "results",
        "models",
        "scripts"
    ]
    
    created_count = 0
    for dir_path in core_dirs:
        if not os.path.exists(dir_path):
            os.makedirs(dir_path, exist_ok=True)
            print_action("✅ Created directory:", dir_path)
            created_count += 1
        else:
            print_action("✓ Directory exists:", dir_path)
    
    print_action(f"📊 Total directories created: {created_count}")

def create_directory_readme_files():
    """Create README files for key directories to explain their purpose."""
    print_header("CREATING DIRECTORY README FILES")
    
    readme_content = {
        "demos/README.md": """# Demos Directory

This directory contains demonstration scripts showcasing different aspects of the robotic handwriting AI system:

- `demo_end_to_end.py` - Complete end-to-end pipeline demonstration
- `demo_working.py` - Working demo with proper letter shapes  
- `demo_components.py` - Individual component testing
- `demo_simple.py` - Simple pipeline demonstration
- `demo_fixed_trajectories.py` - Fixed trajectory generation demo

Run any demo script to see the system in action:
```bash
python3 demo_end_to_end.py
```
""",
        
        "tests/README.md": """# Tests Directory

This directory contains test scripts for verifying system functionality:

- `test_visualization_system.py` - Comprehensive visualization system testing
- `comprehensive_demo_test.py` - All demos testing suite

Run tests to verify system integrity:
```bash
python3 test_visualization_system.py
python3 comprehensive_demo_test.py
```
""",

        "docs/README.md": """# Documentation Directory

This directory contains project documentation:

- `SUCCESS_SUMMARY.md` - Summary of completed features and achievements
- `SYSTEM_OVERVIEW.md` - Technical overview of the system architecture
- API documentation and user guides

See the main README.md in the project root for getting started information.
""",

        "scripts/README.md": """# Scripts Directory

This directory is for utility scripts and tools:

- Data generation scripts
- Model training scripts  
- Deployment scripts
- Maintenance utilities

Place any automation or utility scripts here.
""",

        "models/README.md": """# Models Directory

This directory contains AI models and robot configurations:

- `default_handwriting_robot.xml` - Default robot model configuration
- Trained AI models (GAIL, PINN, etc.)
- Model checkpoints and exports

Organize models by type and training date.
"""
    }
    
    created_count = 0
    for file_path, content in readme_content.items():
        if not os.path.exists(file_path):
            os.makedirs(os.path.dirname(file_path), exist_ok=True)
            with open(file_path, 'w') as f:
                f.write(content)
            print_action("✅ Created README:", file_path)
            created_count += 1
    
    print_action(f"📊 Total README files created: {created_count}")

def update_main_readme():
    """Update the main README.md with the new structure."""
    print_header("UPDATING MAIN README")
    
    readme_addition = """
## 📁 Project Structure

```
robotic-handwriting-ai/
├── 📁 src/                    # Source code modules
│   ├── ai_models/            # AI models (GAIL, PINN, etc.)
│   ├── motion_planning/      # Motion planning algorithms  
│   ├── simulation/           # Physics simulation
│   ├── visualization/        # Rendering and plotting
│   ├── trajectory_generation/ # Trajectory generation
│   ├── data_processing/      # Data utilities
│   └── robot_models/         # Robot kinematics
├── 📁 demos/                 # Demonstration scripts
├── 📁 tests/                 # Test scripts
├── 📁 docs/                  # Documentation
├── 📁 config/                # Configuration files
├── 📁 data/                  # Datasets and samples
├── 📁 results/               # Output results
├── 📁 models/                # AI models and robot configs
├── 📁 scripts/               # Utility scripts
├── 📄 README.md              # This file
├── 📄 requirements.txt       # Dependencies
└── 📄 requirements_core.txt  # Core dependencies
```

## 🚀 Quick Start

1. **Run End-to-End Demo**:
   ```bash
   python3 demos/demo_end_to_end.py
   ```

2. **Test System Components**:
   ```bash
   python3 tests/comprehensive_demo_test.py
   ```

3. **Test Visualization System**:
   ```bash
   python3 tests/test_visualization_system.py
   ```
"""
    
    # Read current README
    readme_path = "README.md"
    if os.path.exists(readme_path):
        with open(readme_path, 'r') as f:
            content = f.read()
        
        # Check if structure section already exists
        if "## 📁 Project Structure" not in content:
            # Add the structure section before the last line
            lines = content.split('\n')
            # Find a good insertion point (before Contributing, License, etc.)
            insert_idx = len(lines)
            for i, line in enumerate(lines):
                if any(keyword in line.lower() for keyword in ['contributing', 'license', 'acknowledgments']):
                    insert_idx = i
                    break
            
            lines.insert(insert_idx, readme_addition)
            
            with open(readme_path, 'w') as f:
                f.write('\n'.join(lines))
            
            print_action("✅ Updated main README with project structure")
        else:
            print_action("✓ Main README already has project structure")

def generate_cleanup_summary():
    """Generate a summary of the cleanup process."""
    print_header("CLEANUP SUMMARY")
    
    # Count files and directories
    total_files = sum(len(files) for _, _, files in os.walk('.'))
    total_dirs = sum(len(dirs) for _, dirs, _ in os.walk('.'))
    
    print_action(f"📊 Project Statistics:")
    print_action(f"   📁 Total directories: {total_dirs}")
    print_action(f"   📄 Total files: {total_files}")
    print_action(f"   🎯 Structure: Clean and organized")
    print_action(f"   ✅ Status: Ready for production")

def main():
    """Main cleanup and organization process."""
    print("🧹 ROBOTIC HANDWRITING AI - PROJECT CLEANUP & ORGANIZATION")
    print("=" * 70)
    print("This script will clean up and organize the project structure...")
    
    # Execute cleanup steps in order
    remove_temporary_files()
    consolidate_nested_structure()
    organize_demo_files()
    organize_test_files()
    organize_documentation()
    create_directory_structure()
    remove_empty_directories()  # Run this after creating structure
    create_directory_readme_files()
    update_main_readme()
    generate_cleanup_summary()
    
    print_header("🎉 CLEANUP COMPLETED SUCCESSFULLY!")
    print("Your robotic handwriting AI project is now clean and organized!")
    print("\n🚀 Next steps:")
    print("   1. Run: python3 demos/demo_end_to_end.py")
    print("   2. Test: python3 tests/comprehensive_demo_test.py")
    print("   3. Explore: Check the docs/ directory for documentation")

if __name__ == "__main__":
    main()