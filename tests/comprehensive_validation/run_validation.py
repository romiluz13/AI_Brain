#!/usr/bin/env python3
"""
Universal AI Brain - Comprehensive Validation Runner

This script runs the complete validation protocol for the Universal AI Brain,
testing all 16 cognitive systems with real MongoDB data and quantitative benchmarking.

Usage:
    python run_validation.py
    
Environment Variables Required:
    MONGODB_URI: MongoDB Atlas connection string
    VOYAGE_API_KEY: Voyage AI API key for embeddings
    OPENAI_API_KEY: OpenAI API key for LLM models
"""

import asyncio
import os
import sys
import json
import logging
from pathlib import Path
from datetime import datetime

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from tests.comprehensive_validation.test_suite import AIBrainTestSuite, ComprehensiveBenchmarkReport

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(f'validation_run_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log')
    ]
)
logger = logging.getLogger(__name__)


async def main():
    """Main validation runner."""
    print("🧠 Universal AI Brain - Comprehensive Validation Protocol")
    print("=" * 70)
    
    # Check environment variables
    mongodb_uri = os.getenv("MONGODB_URI")
    voyage_api_key = os.getenv("VOYAGE_API_KEY")
    openai_api_key = os.getenv("OPENAI_API_KEY")
    
    if not mongodb_uri:
        # Use provided credentials if environment variable not set
        mongodb_uri = "mongodb+srv://romiluz:H97r3aQBnxWawZbx@aibrain.tnv45wr.mongodb.net/?retryWrites=true&w=majority&appName=aibrain"
        logger.info("Using provided MongoDB credentials")
    
    if not voyage_api_key:
        # Use provided API key if environment variable not set
        voyage_api_key = "pa-NHB7D_EtgEImAVQkjIZ6PxoGVHcTOQvUujwDeq8m9-Q"
        logger.info("Using provided Voyage AI credentials")
    
    if not openai_api_key:
        logger.warning("⚠️ OPENAI_API_KEY not set - some tests may fail")
        print("Please set OPENAI_API_KEY environment variable for full testing")
    
    # Initialize test suite
    test_suite = AIBrainTestSuite(
        mongodb_uri=mongodb_uri,
        voyage_api_key=voyage_api_key
    )
    
    try:
        print("\n🔧 Phase 1: Infrastructure Validation")
        print("-" * 50)
        
        # Validate infrastructure
        infrastructure_status = await test_suite.initialize_infrastructure()
        
        for component, status in infrastructure_status.items():
            status_icon = "✅" if "connected" in status or "available" in status or "initialized" in status else "❌"
            print(f"{status_icon} {component}: {status}")
        
        print("\n✅ Phase 2: Project Completion Validation")
        print("-" * 50)
        
        # Validate project completion
        completion_status = await test_suite.validate_project_completion()
        
        for check, passed in completion_status.items():
            status_icon = "✅" if passed else "❌"
            print(f"{status_icon} {check}: {passed}")
        
        # Check if we can proceed with full testing
        critical_components = [
            infrastructure_status.get("ai_brain") == "initialized",
            infrastructure_status.get("mongodb_connection") == "connected",
            completion_status.get("ai_brain_initialization", False)
        ]
        
        if not all(critical_components):
            print("\n❌ Critical components not available - cannot proceed with full testing")
            print("Please ensure:")
            print("- MongoDB Atlas connection is working")
            print("- AI Brain initializes successfully")
            print("- All cognitive systems are implemented")
            return
        
        print("\n🧠 Phase 3: Cognitive Systems Testing")
        print("-" * 50)
        
        # Run comprehensive benchmark
        report = await test_suite.generate_benchmark_report()
        
        # Print detailed report
        test_suite.print_summary_report(report)
        
        # Save report to file
        report_filename = f"validation_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        
        # Convert report to dict for JSON serialization
        report_dict = {
            "test_timestamp": report.test_timestamp.isoformat(),
            "total_systems_tested": report.total_systems_tested,
            "overall_improvement_percentage": report.overall_improvement_percentage,
            "systems_meeting_90_percent_target": report.systems_meeting_90_percent_target,
            "mongodb_connection_status": report.mongodb_connection_status,
            "vector_search_status": report.vector_search_status,
            "hybrid_search_status": report.hybrid_search_status,
            "agno_integration_status": report.agno_integration_status,
            "system_benchmarks": [
                {
                    "system_name": b.system_name,
                    "total_tests": b.total_tests,
                    "successful_tests": b.successful_tests,
                    "average_improvement": b.average_improvement,
                    "max_improvement": b.max_improvement,
                    "min_improvement": b.min_improvement,
                    "average_processing_time": b.average_processing_time,
                    "average_confidence": b.average_confidence
                }
                for b in report.system_benchmarks
            ],
            "infrastructure_metrics": report.infrastructure_metrics
        }
        
        with open(report_filename, 'w') as f:
            json.dump(report_dict, f, indent=2)
        
        print(f"\n📄 Detailed report saved to: {report_filename}")
        
        # Final assessment
        print("\n🎯 FINAL ASSESSMENT")
        print("=" * 70)
        
        success_criteria = {
            "Overall 90% Improvement": report.overall_improvement_percentage >= 90.0,
            "80% Systems Meeting Target": report.systems_meeting_90_percent_target >= (report.total_systems_tested * 0.8),
            "MongoDB Integration": report.mongodb_connection_status == "connected",
            "Vector Search": report.vector_search_status == "available",
            "Agno Integration": report.agno_integration_status == "available"
        }
        
        all_criteria_met = all(success_criteria.values())
        
        for criterion, met in success_criteria.items():
            status_icon = "✅" if met else "❌"
            print(f"{status_icon} {criterion}")
        
        print("\n" + "=" * 70)
        
        if all_criteria_met:
            print("🎉 SUCCESS: All validation criteria met!")
            print("🚀 Universal AI Brain is ready for production deployment!")
            print("📦 Ready for GitHub publication and distribution!")
        else:
            print("⚠️ PARTIAL SUCCESS: Some criteria not fully met")
            print("📋 Review individual system performance for optimization opportunities")
            print("🔧 Consider additional tuning before production deployment")
        
        print("=" * 70)
        
        # Performance summary
        print(f"\n📊 PERFORMANCE SUMMARY:")
        print(f"• Overall Improvement: {report.overall_improvement_percentage:.1f}%")
        print(f"• Systems Tested: {report.total_systems_tested}")
        print(f"• Systems Meeting 90% Target: {report.systems_meeting_90_percent_target}")
        print(f"• Infrastructure Status: {'Fully Operational' if all_criteria_met else 'Partial'}")
        
        # Top performing systems
        top_systems = sorted(report.system_benchmarks, key=lambda x: x.average_improvement, reverse=True)[:3]
        print(f"\n🏆 TOP PERFORMING SYSTEMS:")
        for i, system in enumerate(top_systems, 1):
            print(f"{i}. {system.system_name}: {system.average_improvement:.1f}% improvement")
        
        # Systems needing attention
        low_systems = [s for s in report.system_benchmarks if s.average_improvement < 90.0]
        if low_systems:
            print(f"\n⚠️ SYSTEMS NEEDING ATTENTION:")
            for system in low_systems:
                print(f"• {system.system_name}: {system.average_improvement:.1f}% improvement")
        
    except Exception as e:
        logger.error(f"Validation failed with error: {e}")
        print(f"\n❌ VALIDATION FAILED: {e}")
        print("Please check the logs for detailed error information")
        
    finally:
        # Cleanup
        await test_suite.cleanup()
        print("\n🧹 Cleanup completed")


def check_dependencies():
    """Check if all required dependencies are available."""
    print("🔍 Checking dependencies...")
    
    required_packages = [
        "ai_brain_python",
        "motor", 
        "pymongo",
        "pydantic",
        "asyncio"
    ]
    
    optional_packages = [
        "agno",
        "openai",
        "voyage"
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
        print("Please install with: pip install ai-brain-python[all-frameworks]")
        return False
    
    if missing_optional:
        print(f"\n⚠️ Missing optional packages: {', '.join(missing_optional)}")
        print("Some tests may be skipped")
    
    print("✅ Dependency check completed")
    return True


if __name__ == "__main__":
    print("🧠 Universal AI Brain - Validation Protocol")
    print("Starting comprehensive validation...")
    
    # Check dependencies first
    if not check_dependencies():
        sys.exit(1)
    
    # Run validation
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n⚠️ Validation interrupted by user")
    except Exception as e:
        print(f"\n❌ Validation failed: {e}")
        sys.exit(1)
