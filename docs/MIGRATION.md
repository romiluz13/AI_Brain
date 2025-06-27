# 🔄 Migration Guide: JavaScript to Python

This guide helps you migrate from the JavaScript version of AI Brain to the Python implementation.

## 📋 Overview

The Python version maintains **100% feature parity** with the JavaScript version while adding Python-specific enhancements and better framework integrations.

## 🔄 Key Differences

### Language-Specific Changes

| Aspect | JavaScript | Python |
|--------|------------|--------|
| **Async Syntax** | `async/await` | `async/await` |
| **Type System** | TypeScript | Pydantic + Type Hints |
| **Package Manager** | npm/yarn | pip |
| **Configuration** | JSON/JS objects | Pydantic models |
| **Database Driver** | MongoDB Node.js | Motor (async) |
| **Testing** | Jest | pytest |

### Framework Integrations

| Framework | JavaScript Support | Python Support |
|-----------|-------------------|----------------|
| **CrewAI** | ❌ Not available | ✅ Full integration |
| **Pydantic AI** | ❌ Not available | ✅ Full integration |
| **Agno** | ❌ Not available | ✅ Full integration |
| **LangChain** | ⚠️ Basic | ✅ Advanced integration |
| **LangGraph** | ❌ Not available | ✅ Full integration |

## 🚀 Migration Steps

### 1. Environment Setup

#### JavaScript (Before)
```bash
npm install universal-ai-brain
```

#### Python (After)
```bash
pip install ai-brain-python[all-frameworks]
```

### 2. Configuration Migration

#### JavaScript Configuration
```javascript
// config.js
const config = {
  mongodbUri: "mongodb://localhost:27017",
  databaseName: "ai_brain",
  enableSafetySystems: true,
  cognitiveSystemsConfig: {
    emotionalIntelligence: { sensitivity: 0.8 },
    goalHierarchy: { maxGoals: 10 }
  }
};
```

#### Python Configuration
```python
# config.py
from ai_brain_python import UniversalAIBrainConfig

config = UniversalAIBrainConfig(
    mongodb_uri="mongodb://localhost:27017",
    database_name="ai_brain",
    enable_safety_systems=True,
    cognitive_systems_config={
        "emotional_intelligence": {"sensitivity": 0.8},
        "goal_hierarchy": {"max_goals": 10}
    }
)
```

### 3. Initialization Migration

#### JavaScript Initialization
```javascript
// main.js
const { UniversalAIBrain } = require('universal-ai-brain');

async function main() {
  const brain = new UniversalAIBrain(config);
  await brain.initialize();
  
  // Use brain...
  
  await brain.shutdown();
}
```

#### Python Initialization
```python
# main.py
import asyncio
from ai_brain_python import UniversalAIBrain

async def main():
    brain = UniversalAIBrain(config)
    await brain.initialize()
    
    # Use brain...
    
    await brain.shutdown()

if __name__ == "__main__":
    asyncio.run(main())
```

### 4. Input Data Migration

#### JavaScript Input
```javascript
// JavaScript
const inputData = {
  text: "I'm excited about this project!",
  inputType: "user_message",
  context: {
    userId: "user123",
    sessionId: "session456"
  },
  requestedSystems: ["emotional_intelligence", "goal_hierarchy"]
};
```

#### Python Input
```python
# Python
from ai_brain_python import CognitiveInputData, CognitiveContext

input_data = CognitiveInputData(
    text="I'm excited about this project!",
    input_type="user_message",
    context=CognitiveContext(
        user_id="user123",
        session_id="session456"
    ),
    requested_systems=["emotional_intelligence", "goal_hierarchy"]
)
```

### 5. Processing Migration

#### JavaScript Processing
```javascript
// JavaScript
const response = await brain.processInput(inputData);
console.log(`Emotion: ${response.emotionalState.primaryEmotion}`);
console.log(`Goal: ${response.goalHierarchy.primaryGoal}`);
```

#### Python Processing
```python
# Python
response = await brain.process_input(input_data)
print(f"Emotion: {response.emotional_state.primary_emotion}")
print(f"Goal: {response.goal_hierarchy.primary_goal}")
```

## 🔧 Code Conversion Examples

### Basic Usage Conversion

#### JavaScript Version
```javascript
const { UniversalAIBrain } = require('universal-ai-brain');

class AIService {
  constructor() {
    this.brain = new UniversalAIBrain({
      mongodbUri: process.env.MONGODB_URI,
      enableSafetySystems: true
    });
  }

  async initialize() {
    await this.brain.initialize();
  }

  async analyzeMessage(text, userId) {
    const inputData = {
      text,
      inputType: "user_message",
      context: { userId, sessionId: `session_${Date.now()}` }
    };

    const response = await this.brain.processInput(inputData);
    
    return {
      emotion: response.emotionalState.primaryEmotion,
      confidence: response.confidence,
      recommendations: response.recommendations
    };
  }

  async shutdown() {
    await this.brain.shutdown();
  }
}
```

#### Python Version
```python
import asyncio
from datetime import datetime
from ai_brain_python import UniversalAIBrain, UniversalAIBrainConfig
from ai_brain_python import CognitiveInputData, CognitiveContext

class AIService:
    def __init__(self):
        config = UniversalAIBrainConfig(
            mongodb_uri=os.getenv("MONGODB_URI"),
            enable_safety_systems=True
        )
        self.brain = UniversalAIBrain(config)

    async def initialize(self):
        await self.brain.initialize()

    async def analyze_message(self, text: str, user_id: str) -> dict:
        input_data = CognitiveInputData(
            text=text,
            input_type="user_message",
            context=CognitiveContext(
                user_id=user_id,
                session_id=f"session_{int(datetime.now().timestamp())}"
            )
        )

        response = await self.brain.process_input(input_data)
        
        return {
            "emotion": response.emotional_state.primary_emotion,
            "confidence": response.confidence,
            "recommendations": response.recommendations or []
        }

    async def shutdown(self):
        await self.brain.shutdown()
```

### Framework Integration Migration

#### JavaScript (Limited Framework Support)
```javascript
// JavaScript - Basic LangChain integration
const { LangChainAdapter } = require('universal-ai-brain/adapters');

const adapter = new LangChainAdapter(brain);
const tool = adapter.createCognitiveTool("emotional_analysis");
```

#### Python (Full Framework Support)
```python
# Python - Advanced framework integrations

# CrewAI Integration
from ai_brain_python.adapters import CrewAIAdapter

crewai_adapter = CrewAIAdapter(ai_brain=brain)
agent = crewai_adapter.create_cognitive_agent(
    role="Emotional Intelligence Analyst",
    goal="Provide empathetic analysis",
    cognitive_systems=["emotional_intelligence", "empathy_response"]
)

# Pydantic AI Integration
from ai_brain_python.adapters import PydanticAIAdapter

pydantic_adapter = PydanticAIAdapter(ai_brain=brain)
agent = pydantic_adapter.create_cognitive_agent(
    model="gpt-4o",
    cognitive_systems=["emotional_intelligence", "goal_hierarchy"]
)

# LangChain Integration (Enhanced)
from ai_brain_python.adapters import LangChainAdapter

langchain_adapter = LangChainAdapter(ai_brain=brain)
tool = langchain_adapter.create_cognitive_tool(
    name="emotional_analysis",
    description="Advanced emotional analysis",
    cognitive_systems=["emotional_intelligence", "empathy_response"]
)
```

## 📊 Data Structure Migration

### Response Object Changes

#### JavaScript Response
```javascript
// JavaScript response structure
{
  confidence: 0.85,
  processingTimeMs: 150,
  cognitiveResults: { ... },
  emotionalState: {
    primaryEmotion: "excitement",
    emotionIntensity: 0.8,
    emotionalValence: "positive"
  },
  goalHierarchy: {
    primaryGoal: "Learn AI",
    goalPriority: 8,
    subGoals: ["Study Python", "Build projects"]
  }
}
```

#### Python Response
```python
# Python response structure (Pydantic model)
CognitiveResponse(
    confidence=0.85,
    processing_time_ms=150.0,
    cognitive_results={...},
    emotional_state=EmotionalState(
        primary_emotion="excitement",
        emotion_intensity=0.8,
        emotional_valence="positive"
    ),
    goal_hierarchy=GoalHierarchy(
        primary_goal="Learn AI",
        goal_priority=8,
        sub_goals=["Study Python", "Build projects"]
    )
)
```

## 🗄️ Database Migration

### Schema Compatibility

The Python version is **100% compatible** with existing JavaScript MongoDB schemas. No database migration is required.

#### Verification Script
```python
# verify_database_compatibility.py
import asyncio
from ai_brain_python.database.mongodb_client import MongoDBClient

async def verify_compatibility():
    """Verify that Python version can read JavaScript data."""
    client = MongoDBClient("your-mongodb-uri")
    await client.initialize()
    
    # Test reading existing memories
    memories = await client.search_memories("existing_user_id", "test query")
    print(f"Found {len(memories)} existing memories")
    
    # Test reading user profiles
    profile = await client.get_user_profile("existing_user_id")
    if profile:
        print("✅ User profile compatibility verified")
    
    await client.close()

asyncio.run(verify_compatibility())
```

## 🧪 Testing Migration

### Test Framework Changes

#### JavaScript Tests
```javascript
// JavaScript - Jest
const { UniversalAIBrain } = require('universal-ai-brain');

describe('AI Brain Tests', () => {
  let brain;

  beforeEach(async () => {
    brain = new UniversalAIBrain();
    await brain.initialize();
  });

  test('should process emotional input', async () => {
    const response = await brain.processInput({
      text: "I'm happy!",
      inputType: "test"
    });
    
    expect(response.emotionalState.primaryEmotion).toBe('happiness');
  });
});
```

#### Python Tests
```python
# Python - pytest
import pytest
from ai_brain_python import UniversalAIBrain, CognitiveInputData, CognitiveContext

class TestAIBrain:
    @pytest.fixture
    async def brain(self):
        brain = UniversalAIBrain()
        await brain.initialize()
        yield brain
        await brain.shutdown()

    @pytest.mark.asyncio
    async def test_emotional_processing(self, brain):
        input_data = CognitiveInputData(
            text="I'm happy!",
            input_type="test",
            context=CognitiveContext(user_id="test", session_id="test")
        )
        
        response = await brain.process_input(input_data)
        assert response.emotional_state.primary_emotion == "happiness"
```

## 🚨 Common Migration Issues

### 1. Naming Convention Changes

| JavaScript | Python | Fix |
|------------|--------|-----|
| `camelCase` | `snake_case` | Update all method/property names |
| `processInput` | `process_input` | Use Python naming |
| `primaryEmotion` | `primary_emotion` | Update property access |

### 2. Import Statement Changes

```python
# Old JavaScript style (won't work)
# const { UniversalAIBrain } = require('universal-ai-brain');

# New Python style
from ai_brain_python import UniversalAIBrain
```

### 3. Async Context Handling

```python
# Python requires explicit event loop management
import asyncio

# Wrap main function
if __name__ == "__main__":
    asyncio.run(main())
```

## ✅ Migration Checklist

- [ ] Install Python package: `pip install ai-brain-python[all-frameworks]`
- [ ] Convert configuration from JS objects to Pydantic models
- [ ] Update import statements to Python syntax
- [ ] Convert camelCase to snake_case naming
- [ ] Update async/await patterns for Python
- [ ] Migrate test files from Jest to pytest
- [ ] Update CI/CD pipeline for Python
- [ ] Verify database compatibility
- [ ] Test framework integrations
- [ ] Update documentation and examples

## 🎯 Benefits of Migration

### Enhanced Features in Python Version

1. **Better Framework Support**: 5 major AI frameworks vs limited JS support
2. **Type Safety**: Full Pydantic validation vs basic TypeScript
3. **Performance**: Optimized async processing with Motor
4. **Ecosystem**: Access to Python AI/ML ecosystem
5. **Production Ready**: Better monitoring and enterprise features

### New Capabilities

- **CrewAI Integration**: Build cognitive agent teams
- **Pydantic AI Integration**: Type-safe cognitive agents
- **Agno Integration**: Advanced multi-agent systems
- **Enhanced LangChain**: Better tools and memory
- **LangGraph Integration**: Cognitive state machines

## 📞 Migration Support

Need help with migration? 

1. **Check Examples**: See [examples/](../examples/) for working code
2. **Read Docs**: Full documentation in [docs/](../docs/)
3. **GitHub Issues**: Report migration issues
4. **Community**: Join our Discord for real-time help

---

**Migration Timeline Estimate**: 1-3 days for typical applications, depending on complexity and framework integrations.
