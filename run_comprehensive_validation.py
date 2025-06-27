#!/usr/bin/env python3
"""
Universal AI Brain - Comprehensive Validation Launcher

This script launches the comprehensive validation protocol for the Universal AI Brain.
It validates all 16 cognitive systems using real MongoDB data with quantitative benchmarking.

Usage:
    python run_comprehensive_validation.py [--component COMPONENT]

Components:
    all         - Run complete validation (default)
    infra       - Infrastructure validation only
    cognitive   - Cognitive systems validation only
    agno        - Agno integration validation only
    mongodb     - MongoDB vector search validation only
"""

import argparse
import asyncio
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

def print_banner():
    """Print validation banner."""
    print("🧠" + "="*78 + "🧠")
    print("🧠" + " " * 78 + "🧠")
    print("🧠" + "  UNIVERSAL AI BRAIN - COMPREHENSIVE VALIDATION PROTOCOL".center(78) + "🧠")
    print("🧠" + " " * 78 + "🧠")
    print("🧠" + "  Validating 16 Cognitive Systems with Real MongoDB Data".center(78) + "🧠")
    print("🧠" + "  Target: 90% Performance Improvement Over Baseline".center(78) + "🧠")
    print("🧠" + " " * 78 + "🧠")
    print("🧠" + "="*78 + "🧠")
    print()


async def run_full_validation():
    """Run complete validation protocol."""
    print("🚀 Starting FULL COMPREHENSIVE VALIDATION...")
    print("This will test:")
    print("  ✅ MongoDB Atlas with Vector Search & Hybrid Search")
    print("  ✅ All 16 Cognitive Intelligence Systems")
    print("  ✅ Agno Framework Integration")
    print("  ✅ Performance Benchmarking")
    print("  ✅ Safety & Compliance Systems")
    print()
    
    try:
        from tests.comprehensive_validation.full_validation import FullValidationRunner
        runner = FullValidationRunner()
        await runner.run_full_validation()
        return True
    except Exception as e:
        print(f"❌ Full validation failed: {e}")
        return False


async def run_infrastructure_validation():
    """Run infrastructure validation only."""
    print("🔧 Starting INFRASTRUCTURE VALIDATION...")
    print("Testing MongoDB Atlas, Vector Search, and AI Brain infrastructure...")
    print()
    
    try:
        from tests.comprehensive_validation.mongodb_vector_test import MongoDBVectorSearchTester
        from tests.comprehensive_validation.test_suite import AIBrainTestSuite
        
        # Get credentials from environment variables
        import os
        mongodb_uri = os.getenv("MONGODB_URI", "mongodb+srv://demo:demo123@ai-brain-demo.mongodb.net/?retryWrites=true&w=majority&appName=ai-brain-demo")
        voyage_api_key = os.getenv("VOYAGE_API_KEY", "pa-demo-voyage-key-limited-usage")

        if "demo" in mongodb_uri:
            print("⚠️ Using demo MongoDB credentials (limited usage)")
            print("For production use, set MONGODB_URI environment variable")

        if "demo" in voyage_api_key:
            print("⚠️ Using demo Voyage AI credentials (limited usage)")
            print("For production use, set VOYAGE_API_KEY environment variable")
        
        # Test MongoDB vector search
        vector_tester = MongoDBVectorSearchTester(mongodb_uri, voyage_api_key)
        vector_results = await vector_tester.run_comprehensive_vector_tests()
        vector_tester.print_test_summary(vector_results)
        
        # Test AI Brain infrastructure
        test_suite = AIBrainTestSuite(mongodb_uri, voyage_api_key)
        infra_status = await test_suite.initialize_infrastructure()
        
        print("\n🧠 AI Brain Infrastructure Status:")
        for component, status in infra_status.items():
            status_icon = "✅" if any(x in status for x in ["connected", "available", "initialized"]) else "❌"
            print(f"{status_icon} {component}: {status}")
        
        await test_suite.cleanup()
        return True
        
    except Exception as e:
        print(f"❌ Infrastructure validation failed: {e}")
        return False


async def run_cognitive_validation():
    """Run cognitive systems validation only."""
    print("🧠 Starting COGNITIVE SYSTEMS VALIDATION...")
    print("Testing all 16 cognitive intelligence systems...")
    print()
    
    try:
        from tests.comprehensive_validation.test_suite import AIBrainTestSuite
        
        import os
        mongodb_uri = os.getenv("MONGODB_URI", "mongodb+srv://demo:demo123@ai-brain-demo.mongodb.net/?retryWrites=true&w=majority&appName=ai-brain-demo")
        voyage_api_key = os.getenv("VOYAGE_API_KEY", "pa-demo-voyage-key-limited-usage")
        
        test_suite = AIBrainTestSuite(mongodb_uri, voyage_api_key)
        
        # Initialize infrastructure
        await test_suite.initialize_infrastructure()
        
        # Run cognitive systems tests
        benchmark_report = await test_suite.generate_benchmark_report()
        test_suite.print_summary_report(benchmark_report)
        
        await test_suite.cleanup()
        return True
        
    except Exception as e:
        print(f"❌ Cognitive systems validation failed: {e}")
        return False


async def run_agno_validation():
    """Run Agno integration validation only."""
    print("🤖 Starting AGNO INTEGRATION VALIDATION...")
    print("Testing Agno framework integration with AI Brain cognitive capabilities...")
    print()
    
    try:
        from tests.comprehensive_validation.agno_integration_test import AgnoIntegrationTester
        
        import os
        mongodb_uri = os.getenv("MONGODB_URI", "mongodb+srv://demo:demo123@ai-brain-demo.mongodb.net/?retryWrites=true&w=majority&appName=ai-brain-demo")
        
        tester = AgnoIntegrationTester(mongodb_uri)
        
        if await tester.initialize():
            test_results = await tester.run_comprehensive_tests()
            report = tester.generate_test_report()
            tester.print_test_summary(report)
            await tester.cleanup()
            return True
        else:
            print("❌ Failed to initialize Agno integration tester")
            return False
            
    except Exception as e:
        print(f"❌ Agno integration validation failed: {e}")
        return False


async def run_mongodb_validation():
    """Run MongoDB vector search validation only."""
    print("🔍 Starting MONGODB VECTOR SEARCH VALIDATION...")
    print("Testing MongoDB Atlas vector search and hybrid search capabilities...")
    print()
    
    try:
        from tests.comprehensive_validation.mongodb_vector_test import MongoDBVectorSearchTester
        
        import os
        mongodb_uri = os.getenv("MONGODB_URI", "mongodb+srv://demo:demo123@ai-brain-demo.mongodb.net/?retryWrites=true&w=majority&appName=ai-brain-demo")
        voyage_api_key = os.getenv("VOYAGE_API_KEY", "pa-demo-voyage-key-limited-usage")
        
        tester = MongoDBVectorSearchTester(mongodb_uri, voyage_api_key)
        results = await tester.run_comprehensive_vector_tests()
        tester.print_test_summary(results)
        return True
        
    except Exception as e:
        print(f"❌ MongoDB validation failed: {e}")
        return False


def check_dependencies():
    """Check if required dependencies are available."""
    print("🔍 Checking dependencies...")
    
    required_packages = [
        "ai_brain_python",
        "motor",
        "pymongo", 
        "pydantic",
        "numpy"
    ]
    
    optional_packages = [
        "agno",
        "openai"
    ]
    
    missing_required = []
    missing_optional = []
    
    for package in required_packages:
        try:
            __import__(package)
            print(f"✅ {package}")
        except ImportError:
            missing_required.append(package)
            print(f"❌ {package} (REQUIRED)")
    
    for package in optional_packages:
        try:
            __import__(package)
            print(f"✅ {package}")
        except ImportError:
            missing_optional.append(package)
            print(f"⚠️ {package} (OPTIONAL)")
    
    if missing_required:
        print(f"\n❌ Missing required packages: {', '.join(missing_required)}")
        print("Install with: pip install ai-brain-python[all-frameworks]")
        return False
    
    if missing_optional:
        print(f"\n⚠️ Missing optional packages: {', '.join(missing_optional)}")
        print("Some tests may be skipped")
    
    print("✅ Dependency check completed\n")
    return True


async def main():
    """Main function."""
    parser = argparse.ArgumentParser(description="Universal AI Brain Comprehensive Validation")
    parser.add_argument(
        "--component",
        choices=["all", "infra", "cognitive", "agno", "mongodb"],
        default="all",
        help="Validation component to run (default: all)"
    )
    
    args = parser.parse_args()
    
    print_banner()
    
    # Check dependencies
    if not check_dependencies():
        sys.exit(1)
    
    # Run selected validation component
    success = False
    
    if args.component == "all":
        success = await run_full_validation()
    elif args.component == "infra":
        success = await run_infrastructure_validation()
    elif args.component == "cognitive":
        success = await run_cognitive_validation()
    elif args.component == "agno":
        success = await run_agno_validation()
    elif args.component == "mongodb":
        success = await run_mongodb_validation()
    
    # Final status
    print("\n" + "="*80)
    if success:
        print("🎉 VALIDATION COMPLETED SUCCESSFULLY!")
        print("📊 Check the detailed reports above for performance metrics")
        print("🚀 If all criteria are met, the AI Brain is ready for production!")
    else:
        print("❌ VALIDATION FAILED!")
        print("🔧 Please review the error messages and fix issues before retrying")
    print("="*80)
    
    return 0 if success else 1


if __name__ == "__main__":
    try:
        exit_code = asyncio.run(main())
        sys.exit(exit_code)
    except KeyboardInterrupt:
        print("\n⚠️ Validation interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Validation failed with error: {e}")
        sys.exit(1)
