#!/usr/bin/env python3
"""
Test runner for all tests in the tests directory.
"""

import sys
import subprocess
from pathlib import Path

def run_test(test_file):
    """Run a single test file."""
    print(f"\n🧪 Running {test_file.name}...")
    print("=" * 50)
    
    try:
        result = subprocess.run([sys.executable, str(test_file)], 
                              capture_output=False, 
                              cwd=test_file.parent.parent)
        if result.returncode == 0:
            print(f"✅ {test_file.name} passed")
            return True
        else:
            print(f"❌ {test_file.name} failed")
            return False
    except Exception as e:
        print(f"❌ {test_file.name} failed with error: {e}")
        return False

def main():
    """Run all tests."""
    tests_dir = Path(__file__).parent
    test_files = list(tests_dir.glob("test_*.py"))
    
    if not test_files:
        print("No test files found!")
        return False
    
    print(f"🔍 Found {len(test_files)} test files")
    
    passed = 0
    failed = 0
    
    for test_file in sorted(test_files):
        if run_test(test_file):
            passed += 1
        else:
            failed += 1
    
    print("\n" + "=" * 50)
    print(f"📊 Test Results: {passed} passed, {failed} failed")
    
    if failed == 0:
        print("🎉 All tests passed!")
        return True
    else:
        print("💥 Some tests failed!")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
