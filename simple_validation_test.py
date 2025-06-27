#!/usr/bin/env python3
"""
Simple validation test to check basic functionality before running full validation.
"""

import asyncio
import sys
import traceback
from datetime import datetime

def test_imports():
    """Test if all required imports work."""
    print("🔍 Testing imports...")
    
    try:
        import motor.motor_asyncio
        print("✅ motor imported successfully")
    except ImportError as e:
        print(f"❌ motor import failed: {e}")
        return False
    
    try:
        import pymongo
        print("✅ pymongo imported successfully")
    except ImportError as e:
        print(f"❌ pymongo import failed: {e}")
        return False
    
    try:
        import agno
        print("✅ agno imported successfully")
    except ImportError as e:
        print(f"❌ agno import failed: {e}")
        return False
    
    try:
        import numpy
        print("✅ numpy imported successfully")
    except ImportError as e:
        print(f"❌ numpy import failed: {e}")
        return False
    
    return True

async def test_mongodb_connection():
    """Test MongoDB Atlas connection."""
    print("\n🗄️ Testing MongoDB Atlas connection...")
    
    mongodb_uri = "mongodb+srv://romiluz:H97r3aQBnxWawZbx@aibrain.tnv45wr.mongodb.net/?retryWrites=true&w=majority&appName=aibrain"
    
    try:
        import motor.motor_asyncio
        
        client = motor.motor_asyncio.AsyncIOMotorClient(mongodb_uri)
        
        # Test connection with ping
        await client.admin.command('ping')
        print("✅ MongoDB Atlas connection successful")
        
        # Test database access
        db = client["ai_brain_test"]
        collection = db["test_collection"]
        
        # Insert a test document
        test_doc = {
            "test": True,
            "timestamp": datetime.utcnow(),
            "message": "AI Brain validation test"
        }
        
        result = await collection.insert_one(test_doc)
        print(f"✅ Test document inserted with ID: {result.inserted_id}")
        
        # Read the document back
        found_doc = await collection.find_one({"_id": result.inserted_id})
        if found_doc:
            print("✅ Test document retrieved successfully")
        
        # Clean up
        await collection.delete_one({"_id": result.inserted_id})
        print("✅ Test document cleaned up")
        
        client.close()
        return True
        
    except Exception as e:
        print(f"❌ MongoDB connection failed: {e}")
        traceback.print_exc()
        return False

def test_agno_basic():
    """Test basic Agno functionality."""
    print("\n🤖 Testing Agno framework...")
    
    try:
        from agno.agent import Agent
        from agno.models.openai import OpenAIChat
        
        # Create a simple agent (without running it)
        agent = Agent(
            model=OpenAIChat(id="gpt-4o"),
            instructions="You are a test agent",
            markdown=True
        )
        
        print("✅ Agno agent created successfully")
        print(f"✅ Agent model: {agent.model}")
        return True
        
    except Exception as e:
        print(f"❌ Agno test failed: {e}")
        traceback.print_exc()
        return False

def test_ai_brain_imports():
    """Test AI Brain imports."""
    print("\n🧠 Testing AI Brain imports...")
    
    try:
        # Test if our AI Brain modules can be imported
        sys.path.insert(0, '.')
        
        # Try importing core models
        from ai_brain_python.core.models import CognitiveInputData, CognitiveContext
        print("✅ AI Brain core models imported")
        
        # Try creating basic objects
        context = CognitiveContext(
            user_id="test_user",
            session_id="test_session"
        )
        print("✅ CognitiveContext created")
        
        input_data = CognitiveInputData(
            text="Test input",
            input_type="test",
            context=context
        )
        print("✅ CognitiveInputData created")
        
        return True
        
    except Exception as e:
        print(f"❌ AI Brain imports failed: {e}")
        traceback.print_exc()
        return False

async def main():
    """Run simple validation tests."""
    print("🧠 AI BRAIN - SIMPLE VALIDATION TEST")
    print("=" * 50)
    
    all_passed = True
    
    # Test imports
    if not test_imports():
        all_passed = False
    
    # Test MongoDB connection
    if not await test_mongodb_connection():
        all_passed = False
    
    # Test Agno
    if not test_agno_basic():
        all_passed = False
    
    # Test AI Brain imports
    if not test_ai_brain_imports():
        all_passed = False
    
    print("\n" + "=" * 50)
    if all_passed:
        print("🎉 ALL BASIC TESTS PASSED!")
        print("✅ Ready to run comprehensive validation")
    else:
        print("❌ SOME TESTS FAILED!")
        print("🔧 Fix issues before running full validation")
    
    return all_passed

if __name__ == "__main__":
    try:
        result = asyncio.run(main())
        sys.exit(0 if result else 1)
    except Exception as e:
        print(f"❌ Test failed with error: {e}")
        traceback.print_exc()
        sys.exit(1)
