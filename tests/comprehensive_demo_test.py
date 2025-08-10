#!/usr/bin/env python3
"""
Comprehensive Demo Test - All Fixes Applied
==========================================

Test all demos with fixes to ensure 100% functionality.
"""

import subprocess
import sys
import os
import json
from pathlib import Path

def test_demo(demo_name, description):
    """Test a single demo script."""
    print(f"\n🎮 Testing {demo_name}...")
    print(f"   {description}")
    
    try:
        # Add environment variable to disable interactive mode
        env = os.environ.copy()
        env['PYTHONIOENCODING'] = 'utf-8'
        env['CI'] = 'true'  # Signal non-interactive environment
        
        result = subprocess.run([sys.executable, demo_name], 
                              capture_output=True, text=True, timeout=30,
                              env=env, input='quit\n')
        
        if result.returncode == 0:
            # Check for warning indicators
            output = result.stdout
            has_warnings = ("⚠️" in output or "❌" in output or 
                          "warning" in output.lower() or "error" in output.lower())
            
            if has_warnings:
                print(f"   🟡 PARTIAL - Works with warnings")
                return "partial"
            else:
                print(f"   ✅ PERFECT - Works flawlessly")
                return "perfect"
        else:
            print(f"   ❌ FAILED - Exit code {result.returncode}")
            print(f"   Error: {result.stderr[:200]}...")
            return "failed"
    
    except subprocess.TimeoutExpired:
        print(f"   ⏰ TIMEOUT - Took too long")
        return "timeout"
    except Exception as e:
        print(f"   ❌ ERROR - {e}")
        return "error"

def main():
    """Test all demos comprehensively."""
    print("🤖 COMPREHENSIVE DEMO TESTING - ALL FIXES APPLIED")
    print("="*60)
    
    demos = [
        ("demo_working.py", "Working demo with proper letter shapes"),
        ("demo_simple.py", "Simple pipeline demonstration"),
        ("demo_components.py", "Individual component testing"),
        ("demo_end_to_end.py", "Complete end-to-end pipeline"),
        ("demo_fixed_trajectories.py", "Fixed trajectory generation")
    ]
    
    results = {}
    perfect_count = 0
    partial_count = 0
    failed_count = 0
    
    for demo_name, description in demos:
        demo_path = Path(demo_name)
        if demo_path.exists():
            result = test_demo(demo_name, description)
            results[demo_name] = result
            
            if result == "perfect":
                perfect_count += 1
            elif result == "partial":
                partial_count += 1
            else:
                failed_count += 1
        else:
            print(f"\n⚠️  {demo_name} not found")
            results[demo_name] = "missing"
            failed_count += 1
    
    # Summary
    total_demos = len(demos)
    success_rate = ((perfect_count + partial_count) / total_demos * 100) if total_demos > 0 else 0
    perfect_rate = (perfect_count / total_demos * 100) if total_demos > 0 else 0
    
    print(f"\n" + "="*60)
    print(f"📊 COMPREHENSIVE TEST RESULTS")
    print(f"="*60)
    print(f"   🟢 Perfect: {perfect_count}/{total_demos} ({perfect_rate:.1f}%)")
    print(f"   🟡 Partial: {partial_count}/{total_demos}")
    print(f"   🔴 Failed:  {failed_count}/{total_demos}")
    print(f"   📈 Success Rate: {success_rate:.1f}%")
    
    if perfect_rate >= 90:
        status = "🟢 OUTSTANDING - All demos working perfectly!"
    elif success_rate >= 90:
        status = "🟢 EXCELLENT - Minor warnings only"
    elif success_rate >= 80:
        status = "🟡 GOOD - Most demos working"
    else:
        status = "🔴 NEEDS WORK - Significant issues remain"
    
    print(f"   🎯 Status: {status}")
    
    # Save detailed results
    Path("results").mkdir(exist_ok=True)
    with open("results/comprehensive_demo_test_results.json", "w") as f:
        json.dump({
            "summary": {
                "perfect": perfect_count,
                "partial": partial_count, 
                "failed": failed_count,
                "total": total_demos,
                "success_rate": success_rate,
                "perfect_rate": perfect_rate,
                "status": status
            },
            "detailed_results": results
        }, f, indent=2)
    
    print(f"\n💾 Detailed results saved to results/comprehensive_demo_test_results.json")
    
    return results

if __name__ == "__main__":
    main()
