#!/usr/bin/env python3
"""
Test runner for AI Brain Python.

This script provides convenient ways to run different types of tests
with appropriate configurations and reporting.
"""

import sys
import os
import argparse
import subprocess
from pathlib import Path


def run_command(cmd, description=""):
    """Run a command and handle output."""
    print(f"\n{'='*60}")
    if description:
        print(f"Running: {description}")
    print(f"Command: {' '.join(cmd)}")
    print(f"{'='*60}")
    
    try:
        result = subprocess.run(cmd, check=True, capture_output=False)
        print(f"\n✅ {description or 'Command'} completed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"\n❌ {description or 'Command'} failed with exit code {e.returncode}")
        return False
    except FileNotFoundError:
        print(f"\n❌ Command not found: {cmd[0]}")
        print("Make sure pytest is installed: pip install pytest")
        return False


def run_unit_tests(verbose=False, coverage=False):
    """Run unit tests."""
    cmd = ["python", "-m", "pytest", "tests/unit/"]
    
    if verbose:
        cmd.append("-v")
    
    if coverage:
        cmd.extend(["--cov=ai_brain_python", "--cov-report=html", "--cov-report=term"])
    
    cmd.extend([
        "--tb=short",
        "-x",  # Stop on first failure
        "--disable-warnings"
    ])
    
    return run_command(cmd, "Unit Tests")


def run_integration_tests(verbose=False):
    """Run integration tests."""
    cmd = ["python", "-m", "pytest", "tests/integration/"]
    
    if verbose:
        cmd.append("-v")
    
    cmd.extend([
        "--tb=short",
        "--disable-warnings",
        "-m", "not requires_mongodb"  # Skip MongoDB tests by default
    ])
    
    return run_command(cmd, "Integration Tests")


def run_performance_tests(verbose=False):
    """Run performance tests."""
    cmd = ["python", "-m", "pytest", "tests/performance/"]
    
    if verbose:
        cmd.append("-v")
    
    cmd.extend([
        "--tb=short",
        "--disable-warnings",
        "-m", "performance and not slow"  # Skip slow tests by default
    ])
    
    return run_command(cmd, "Performance Tests")


def run_safety_tests(verbose=False):
    """Run safety system tests."""
    cmd = ["python", "-m", "pytest", "tests/unit/test_safety_systems.py"]
    
    if verbose:
        cmd.append("-v")
    
    cmd.extend([
        "--tb=short",
        "--disable-warnings"
    ])
    
    return run_command(cmd, "Safety System Tests")


def run_framework_tests(verbose=False):
    """Run framework adapter tests."""
    cmd = ["python", "-m", "pytest", "tests/integration/test_framework_adapters.py"]
    
    if verbose:
        cmd.append("-v")
    
    cmd.extend([
        "--tb=short",
        "--disable-warnings",
        "-m", "not requires_frameworks"  # Skip if frameworks not available
    ])
    
    return run_command(cmd, "Framework Adapter Tests")


def run_all_tests(verbose=False, coverage=False, include_slow=False, include_mongodb=False):
    """Run all tests."""
    cmd = ["python", "-m", "pytest", "tests/"]
    
    if verbose:
        cmd.append("-v")
    
    if coverage:
        cmd.extend(["--cov=ai_brain_python", "--cov-report=html", "--cov-report=term"])
    
    # Build marker expression
    markers = []
    if not include_slow:
        markers.append("not slow")
    if not include_mongodb:
        markers.append("not requires_mongodb")
    
    if markers:
        cmd.extend(["-m", " and ".join(markers)])
    
    cmd.extend([
        "--tb=short",
        "--disable-warnings"
    ])
    
    return run_command(cmd, "All Tests")


def run_specific_test(test_path, verbose=False):
    """Run a specific test file or test function."""
    cmd = ["python", "-m", "pytest", test_path]
    
    if verbose:
        cmd.append("-v")
    
    cmd.extend([
        "--tb=short",
        "--disable-warnings"
    ])
    
    return run_command(cmd, f"Specific Test: {test_path}")


def check_test_environment():
    """Check if the test environment is properly set up."""
    print("🔍 Checking test environment...")
    
    # Check if pytest is available
    try:
        import pytest
        print(f"✅ pytest {pytest.__version__} is available")
    except ImportError:
        print("❌ pytest is not installed")
        print("Install with: pip install pytest")
        return False
    
    # Check if AI Brain package is available
    try:
        import ai_brain_python
        print(f"✅ ai_brain_python package is available")
    except ImportError:
        print("❌ ai_brain_python package is not available")
        print("Install with: pip install -e .")
        return False
    
    # Check optional dependencies
    optional_deps = {
        "pytest-asyncio": "async test support",
        "pytest-cov": "coverage reporting",
        "pytest-mock": "mocking support",
        "mongomock-motor": "MongoDB mocking"
    }
    
    for dep, description in optional_deps.items():
        try:
            __import__(dep.replace("-", "_"))
            print(f"✅ {dep} is available ({description})")
        except ImportError:
            print(f"⚠️  {dep} is not available ({description})")
    
    # Check framework availability
    frameworks = {
        "crewai": "CrewAI framework",
        "pydantic_ai": "Pydantic AI framework", 
        "agno": "Agno framework",
        "langchain": "LangChain framework",
        "langgraph": "LangGraph framework"
    }
    
    print("\n📦 Framework availability:")
    for framework, description in frameworks.items():
        try:
            __import__(framework)
            print(f"✅ {framework} is available ({description})")
        except ImportError:
            print(f"⚠️  {framework} is not available ({description})")
    
    return True


def main():
    """Main test runner function."""
    parser = argparse.ArgumentParser(description="AI Brain Python Test Runner")
    
    parser.add_argument(
        "test_type",
        choices=["unit", "integration", "performance", "safety", "framework", "all", "check", "specific"],
        help="Type of tests to run"
    )
    
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Verbose output"
    )
    
    parser.add_argument(
        "--coverage", "-c",
        action="store_true",
        help="Generate coverage report"
    )
    
    parser.add_argument(
        "--include-slow",
        action="store_true",
        help="Include slow tests"
    )
    
    parser.add_argument(
        "--include-mongodb",
        action="store_true",
        help="Include MongoDB tests"
    )
    
    parser.add_argument(
        "--test-path",
        help="Specific test path (for 'specific' test type)"
    )
    
    args = parser.parse_args()
    
    # Change to the project root directory
    project_root = Path(__file__).parent.parent
    os.chdir(project_root)
    
    print("🧠 AI Brain Python Test Runner")
    print(f"Working directory: {os.getcwd()}")
    
    success = True
    
    if args.test_type == "check":
        success = check_test_environment()
    
    elif args.test_type == "unit":
        success = run_unit_tests(args.verbose, args.coverage)
    
    elif args.test_type == "integration":
        success = run_integration_tests(args.verbose)
    
    elif args.test_type == "performance":
        success = run_performance_tests(args.verbose)
    
    elif args.test_type == "safety":
        success = run_safety_tests(args.verbose)
    
    elif args.test_type == "framework":
        success = run_framework_tests(args.verbose)
    
    elif args.test_type == "all":
        success = run_all_tests(
            args.verbose, 
            args.coverage, 
            args.include_slow, 
            args.include_mongodb
        )
    
    elif args.test_type == "specific":
        if not args.test_path:
            print("❌ --test-path is required for 'specific' test type")
            sys.exit(1)
        success = run_specific_test(args.test_path, args.verbose)
    
    if success:
        print(f"\n🎉 Test run completed successfully!")
        sys.exit(0)
    else:
        print(f"\n💥 Test run failed!")
        sys.exit(1)


if __name__ == "__main__":
    main()
