#!/usr/bin/env python3
"""
Universal AI Brain - Full Validation Protocol

This script runs the complete validation protocol including:
1. Infrastructure validation (MongoDB, Vector Search, Hybrid Search)
2. All 16 cognitive systems testing
3. Framework integration testing (Agno, CrewAI, etc.)
4. Performance benchmarking
5. Safety and compliance validation

Usage:
    python full_validation.py
"""

import asyncio
import json
import logging
import os
import sys
from datetime import datetime
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from tests.comprehensive_validation.test_suite import AIBrainTestSuite
from tests.comprehensive_validation.mongodb_vector_test import MongoDBVectorSearchTester
from tests.comprehensive_validation.agno_integration_test import AgnoIntegrationTester

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(f'full_validation_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log')
    ]
)
logger = logging.getLogger(__name__)


class FullValidationRunner:
    """Comprehensive validation runner for Universal AI Brain."""
    
    def __init__(self):
        # Use provided credentials
        self.mongodb_uri = "mongodb+srv://romiluz:H97r3aQBnxWawZbx@aibrain.tnv45wr.mongodb.net/?retryWrites=true&w=majority&appName=aibrain"
        self.voyage_api_key = "pa-NHB7D_EtgEImAVQkjIZ6PxoGVHcTOQvUujwDeq8m9-Q"
        
        # Test components
        self.test_suite = None
        self.vector_tester = None
        self.agno_tester = None
        
        # Results storage
        self.validation_results = {
            "validation_timestamp": datetime.utcnow().isoformat(),
            "infrastructure_validation": {},
            "cognitive_systems_validation": {},
            "framework_integration_validation": {},
            "vector_search_validation": {},
            "overall_assessment": {}
        }
    
    async def run_infrastructure_validation(self) -> Dict[str, Any]:
        """Run comprehensive infrastructure validation."""
        print("\n🔧 PHASE 1: INFRASTRUCTURE VALIDATION")
        print("=" * 70)
        
        infrastructure_results = {}
        
        # Test MongoDB Vector Search capabilities
        print("🔍 Testing MongoDB Atlas Vector Search...")
        self.vector_tester = MongoDBVectorSearchTester(self.mongodb_uri, self.voyage_api_key)
        vector_results = await self.vector_tester.run_comprehensive_vector_tests()
        infrastructure_results["vector_search"] = vector_results
        
        # Print vector search summary
        self.vector_tester.print_test_summary(vector_results)
        
        # Test basic AI Brain infrastructure
        print("\n🧠 Testing AI Brain Infrastructure...")
        self.test_suite = AIBrainTestSuite(self.mongodb_uri, self.voyage_api_key)
        infrastructure_status = await self.test_suite.initialize_infrastructure()
        infrastructure_results["ai_brain_infrastructure"] = infrastructure_status
        
        # Print infrastructure status
        print("\n📊 AI Brain Infrastructure Status:")
        for component, status in infrastructure_status.items():
            status_icon = "✅" if any(x in status for x in ["connected", "available", "initialized"]) else "❌"
            print(f"{status_icon} {component}: {status}")
        
        return infrastructure_results
    
    async def run_cognitive_systems_validation(self) -> Dict[str, Any]:
        """Run comprehensive cognitive systems validation."""
        print("\n🧠 PHASE 2: COGNITIVE SYSTEMS VALIDATION")
        print("=" * 70)
        
        if not self.test_suite:
            print("❌ Test suite not initialized")
            return {"error": "Test suite not available"}
        
        # Validate project completion
        print("✅ Validating project completion...")
        completion_status = await self.test_suite.validate_project_completion()
        
        print("📋 Project Completion Status:")
        for check, passed in completion_status.items():
            status_icon = "✅" if passed else "❌"
            print(f"{status_icon} {check}: {passed}")
        
        # Run comprehensive cognitive systems tests
        print("\n🧪 Running cognitive systems tests...")
        benchmark_report = await self.test_suite.generate_benchmark_report()
        
        # Print detailed report
        self.test_suite.print_summary_report(benchmark_report)
        
        return {
            "completion_status": completion_status,
            "benchmark_report": benchmark_report,
            "systems_tested": benchmark_report.total_systems_tested,
            "overall_improvement": benchmark_report.overall_improvement_percentage,
            "systems_meeting_target": benchmark_report.systems_meeting_90_percent_target
        }
    
    async def run_framework_integration_validation(self) -> Dict[str, Any]:
        """Run framework integration validation."""
        print("\n🤖 PHASE 3: FRAMEWORK INTEGRATION VALIDATION")
        print("=" * 70)
        
        framework_results = {}
        
        # Test Agno integration
        print("🔧 Testing Agno Framework Integration...")
        try:
            self.agno_tester = AgnoIntegrationTester(self.mongodb_uri)
            
            if await self.agno_tester.initialize():
                agno_test_results = await self.agno_tester.run_comprehensive_tests()
                agno_report = self.agno_tester.generate_test_report()
                self.agno_tester.print_test_summary(agno_report)
                
                framework_results["agno"] = {
                    "status": "success",
                    "report": agno_report,
                    "average_improvement": agno_report.get("average_improvement_score", 0.0)
                }
            else:
                framework_results["agno"] = {
                    "status": "initialization_failed",
                    "error": "Failed to initialize Agno integration"
                }
                
        except Exception as e:
            logger.error(f"Agno integration test failed: {e}")
            framework_results["agno"] = {
                "status": "error",
                "error": str(e)
            }
        
        # Test other frameworks (placeholder for future implementation)
        framework_results["crewai"] = {"status": "not_implemented", "note": "CrewAI integration test pending"}
        framework_results["pydantic_ai"] = {"status": "not_implemented", "note": "Pydantic AI integration test pending"}
        framework_results["langchain"] = {"status": "not_implemented", "note": "LangChain integration test pending"}
        framework_results["langgraph"] = {"status": "not_implemented", "note": "LangGraph integration test pending"}
        
        return framework_results
    
    def calculate_overall_assessment(self) -> Dict[str, Any]:
        """Calculate overall assessment based on all validation results."""
        assessment = {
            "overall_score": 0.0,
            "infrastructure_score": 0.0,
            "cognitive_systems_score": 0.0,
            "framework_integration_score": 0.0,
            "success_criteria_met": {},
            "recommendations": [],
            "production_readiness": "not_ready"
        }
        
        # Infrastructure scoring
        infra_results = self.validation_results.get("infrastructure_validation", {})
        vector_search = infra_results.get("vector_search", {})
        ai_brain_infra = infra_results.get("ai_brain_infrastructure", {})
        
        infra_score = 0.0
        if vector_search.get("overall_status") in ["ready_for_indexing", "partial_success"]:
            infra_score += 40.0
        if ai_brain_infra.get("mongodb_connection") == "connected":
            infra_score += 30.0
        if ai_brain_infra.get("ai_brain") == "initialized":
            infra_score += 30.0
        
        assessment["infrastructure_score"] = infra_score
        
        # Cognitive systems scoring
        cognitive_results = self.validation_results.get("cognitive_systems_validation", {})
        benchmark_report = cognitive_results.get("benchmark_report")
        
        cognitive_score = 0.0
        if benchmark_report:
            overall_improvement = benchmark_report.overall_improvement_percentage
            systems_meeting_target = benchmark_report.systems_meeting_90_percent_target
            total_systems = benchmark_report.total_systems_tested
            
            # Score based on improvement percentage
            cognitive_score = min(overall_improvement, 100.0)
            
            # Bonus for systems meeting target
            if total_systems > 0:
                target_ratio = systems_meeting_target / total_systems
                cognitive_score += target_ratio * 20.0  # Up to 20 bonus points
        
        assessment["cognitive_systems_score"] = min(cognitive_score, 100.0)
        
        # Framework integration scoring
        framework_results = self.validation_results.get("framework_integration_validation", {})
        framework_score = 0.0
        
        agno_result = framework_results.get("agno", {})
        if agno_result.get("status") == "success":
            agno_improvement = agno_result.get("report", {}).get("average_improvement_score", 0.0)
            framework_score += min(agno_improvement, 100.0) * 0.5  # 50% weight for Agno
        
        # Other frameworks would add to the score when implemented
        assessment["framework_integration_score"] = framework_score
        
        # Overall score calculation
        overall_score = (
            assessment["infrastructure_score"] * 0.3 +
            assessment["cognitive_systems_score"] * 0.5 +
            assessment["framework_integration_score"] * 0.2
        )
        assessment["overall_score"] = overall_score
        
        # Success criteria evaluation
        assessment["success_criteria_met"] = {
            "infrastructure_operational": infra_score >= 80.0,
            "cognitive_systems_90_percent": cognitive_score >= 90.0,
            "framework_integration_working": framework_score >= 70.0,
            "overall_90_percent": overall_score >= 90.0
        }
        
        # Production readiness assessment
        all_criteria_met = all(assessment["success_criteria_met"].values())
        if all_criteria_met:
            assessment["production_readiness"] = "ready"
        elif overall_score >= 80.0:
            assessment["production_readiness"] = "nearly_ready"
        else:
            assessment["production_readiness"] = "needs_improvement"
        
        # Generate recommendations
        recommendations = []
        if not assessment["success_criteria_met"]["infrastructure_operational"]:
            recommendations.append("Improve infrastructure setup - ensure MongoDB Atlas and vector search are properly configured")
        if not assessment["success_criteria_met"]["cognitive_systems_90_percent"]:
            recommendations.append("Optimize cognitive systems performance - review individual system benchmarks")
        if not assessment["success_criteria_met"]["framework_integration_working"]:
            recommendations.append("Complete framework integration testing for all supported frameworks")
        
        assessment["recommendations"] = recommendations
        
        return assessment
    
    def print_final_report(self):
        """Print comprehensive final validation report."""
        assessment = self.validation_results["overall_assessment"]
        
        print("\n" + "="*80)
        print("🧠 UNIVERSAL AI BRAIN - COMPREHENSIVE VALIDATION REPORT")
        print("="*80)
        
        print(f"📅 Validation Date: {self.validation_results['validation_timestamp']}")
        print(f"🎯 Overall Score: {assessment['overall_score']:.1f}/100")
        print(f"🏭 Production Readiness: {assessment['production_readiness'].upper()}")
        
        print(f"\n📊 COMPONENT SCORES:")
        print("-" * 80)
        print(f"🔧 Infrastructure: {assessment['infrastructure_score']:.1f}/100")
        print(f"🧠 Cognitive Systems: {assessment['cognitive_systems_score']:.1f}/100")
        print(f"🤖 Framework Integration: {assessment['framework_integration_score']:.1f}/100")
        
        print(f"\n✅ SUCCESS CRITERIA:")
        print("-" * 80)
        for criterion, met in assessment["success_criteria_met"].items():
            status = "✅" if met else "❌"
            print(f"{status} {criterion.replace('_', ' ').title()}")
        
        if assessment["recommendations"]:
            print(f"\n📋 RECOMMENDATIONS:")
            print("-" * 80)
            for i, rec in enumerate(assessment["recommendations"], 1):
                print(f"{i}. {rec}")
        
        print(f"\n🎯 FINAL ASSESSMENT:")
        print("-" * 80)
        
        if assessment["production_readiness"] == "ready":
            print("🎉 SUCCESS: Universal AI Brain is READY FOR PRODUCTION!")
            print("🚀 All validation criteria met - ready for GitHub publication!")
            print("📦 Package can be distributed and deployed in production environments!")
        elif assessment["production_readiness"] == "nearly_ready":
            print("⚠️ NEARLY READY: Minor improvements needed before production")
            print("🔧 Address recommendations above for full production readiness")
        else:
            print("❌ NOT READY: Significant improvements needed")
            print("🔧 Review and address all recommendations before production deployment")
        
        print("="*80)
    
    async def save_results(self):
        """Save validation results to file."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"full_validation_results_{timestamp}.json"
        
        # Convert complex objects to serializable format
        serializable_results = {}
        for key, value in self.validation_results.items():
            if hasattr(value, '__dict__'):
                serializable_results[key] = value.__dict__
            else:
                serializable_results[key] = value
        
        with open(filename, 'w') as f:
            json.dump(serializable_results, f, indent=2, default=str)
        
        print(f"\n💾 Full validation results saved to: {filename}")
    
    async def run_full_validation(self):
        """Run the complete validation protocol."""
        print("🧠 UNIVERSAL AI BRAIN - COMPREHENSIVE VALIDATION PROTOCOL")
        print("=" * 80)
        print("🚀 Starting full validation with real MongoDB data...")
        
        try:
            # Phase 1: Infrastructure
            self.validation_results["infrastructure_validation"] = await self.run_infrastructure_validation()
            
            # Phase 2: Cognitive Systems
            self.validation_results["cognitive_systems_validation"] = await self.run_cognitive_systems_validation()
            
            # Phase 3: Framework Integration
            self.validation_results["framework_integration_validation"] = await self.run_framework_integration_validation()
            
            # Calculate overall assessment
            self.validation_results["overall_assessment"] = self.calculate_overall_assessment()
            
            # Print final report
            self.print_final_report()
            
            # Save results
            await self.save_results()
            
        except Exception as e:
            logger.error(f"Full validation failed: {e}")
            print(f"\n❌ VALIDATION FAILED: {e}")
            
        finally:
            # Cleanup
            if self.test_suite:
                await self.test_suite.cleanup()
            if self.agno_tester:
                await self.agno_tester.cleanup()


async def main():
    """Main function to run full validation."""
    runner = FullValidationRunner()
    await runner.run_full_validation()


if __name__ == "__main__":
    print("🧠 Universal AI Brain - Full Validation Protocol")
    print("Starting comprehensive validation with real data...")
    
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n⚠️ Validation interrupted by user")
    except Exception as e:
        print(f"\n❌ Validation failed: {e}")
        sys.exit(1)
