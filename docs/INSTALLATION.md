# 📦 Installation Guide

This guide provides detailed instructions for installing and setting up the Universal AI Brain Python package.

## 🚀 Quick Installation

### Basic Installation

```bash
# Install the core package
pip install ai-brain-python
```

### Framework-Specific Installation

Choose the frameworks you want to integrate with:

```bash
# CrewAI integration
pip install ai-brain-python[crewai]

# Pydantic AI integration
pip install ai-brain-python[pydantic-ai]

# Agno integration
pip install ai-brain-python[agno]

# LangChain integration
pip install ai-brain-python[langchain]

# LangGraph integration
pip install ai-brain-python[langgraph]

# All frameworks at once
pip install ai-brain-python[all-frameworks]
```

### Development Installation

For development and testing:

```bash
# Development dependencies
pip install ai-brain-python[dev]

# All dependencies (frameworks + development)
pip install ai-brain-python[all]
```

## 🔧 System Requirements

### Python Version
- **Python 3.9 or higher** (recommended: Python 3.11+)
- **64-bit architecture** (required for some dependencies)

### Operating System Support
- **Linux** (Ubuntu 20.04+, CentOS 8+, etc.)
- **macOS** (10.15+)
- **Windows** (10+, WSL2 recommended)

### Memory Requirements
- **Minimum**: 4GB RAM
- **Recommended**: 8GB+ RAM for production use
- **Heavy workloads**: 16GB+ RAM

## 🗄️ Database Setup

### MongoDB Atlas (Recommended)

1. **Create MongoDB Atlas Account**
   ```bash
   # Visit https://www.mongodb.com/atlas
   # Create a free cluster
   ```

2. **Get Connection String**
   ```bash
   # Format: mongodb+srv://username:password@cluster.mongodb.net/database
   export MONGODB_URI="mongodb+srv://your-username:your-password@your-cluster.mongodb.net/ai_brain"
   ```

3. **Configure Vector Search**
   ```javascript
   // In MongoDB Atlas, create a vector search index
   {
     "fields": [
       {
         "type": "vector",
         "path": "embedding",
         "numDimensions": 1536,
         "similarity": "cosine"
       }
     ]
   }
   ```

### Local MongoDB (Development)

```bash
# Install MongoDB locally
# Ubuntu/Debian
sudo apt-get install mongodb

# macOS
brew install mongodb-community

# Start MongoDB
sudo systemctl start mongod  # Linux
brew services start mongodb-community  # macOS

# Set connection string
export MONGODB_URI="mongodb://localhost:27017/ai_brain"
```

## 🔑 API Keys Setup

### Required API Keys

Set up environment variables for the AI services you plan to use:

```bash
# OpenAI (for GPT models)
export OPENAI_API_KEY="your-openai-api-key"

# Anthropic (for Claude models)
export ANTHROPIC_API_KEY="your-anthropic-api-key"

# Voyage AI (for embeddings)
export VOYAGE_API_KEY="your-voyage-api-key"

# Google AI (optional)
export GOOGLE_API_KEY="your-google-api-key"
```

### Environment File Setup

Create a `.env` file in your project root:

```bash
# .env file
MONGODB_URI=mongodb+srv://username:password@cluster.mongodb.net/ai_brain
OPENAI_API_KEY=your-openai-api-key
ANTHROPIC_API_KEY=your-anthropic-api-key
VOYAGE_API_KEY=your-voyage-api-key

# Optional settings
AI_BRAIN_LOG_LEVEL=INFO
AI_BRAIN_SAFETY_LEVEL=moderate
AI_BRAIN_MAX_CONCURRENT_REQUESTS=100
```

## 🐳 Docker Installation

### Using Docker Compose

```yaml
# docker-compose.yml
version: '3.8'

services:
  ai-brain-app:
    build: .
    environment:
      - MONGODB_URI=mongodb://mongodb:27017/ai_brain
      - OPENAI_API_KEY=${OPENAI_API_KEY}
      - ANTHROPIC_API_KEY=${ANTHROPIC_API_KEY}
    depends_on:
      - mongodb
    ports:
      - "8000:8000"

  mongodb:
    image: mongo:7.0
    ports:
      - "27017:27017"
    volumes:
      - mongodb_data:/data/db

volumes:
  mongodb_data:
```

### Dockerfile

```dockerfile
FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Install AI Brain
RUN pip install ai-brain-python[all-frameworks]

# Copy application code
COPY . .

# Run the application
CMD ["python", "main.py"]
```

## 🧪 Verification

### Test Installation

```python
# test_installation.py
import asyncio
from ai_brain_python import UniversalAIBrain

async def test_installation():
    """Test that AI Brain is properly installed."""
    try:
        # Initialize AI Brain
        brain = UniversalAIBrain()
        await brain.initialize()
        
        # Get system status
        status = brain.get_system_status()
        print(f"✅ AI Brain Status: {status['status']}")
        print(f"✅ Cognitive Systems: {len(status['cognitive_systems'])}")
        
        await brain.shutdown()
        print("✅ Installation test passed!")
        
    except Exception as e:
        print(f"❌ Installation test failed: {e}")

if __name__ == "__main__":
    asyncio.run(test_installation())
```

### Run Test

```bash
python test_installation.py
```

### Check Framework Availability

```python
# check_frameworks.py
def check_framework_availability():
    """Check which AI frameworks are available."""
    frameworks = {
        "crewai": "CrewAI",
        "pydantic_ai": "Pydantic AI",
        "agno": "Agno",
        "langchain": "LangChain",
        "langgraph": "LangGraph"
    }
    
    print("🔍 Framework Availability Check:")
    for module, name in frameworks.items():
        try:
            __import__(module)
            print(f"✅ {name} is available")
        except ImportError:
            print(f"❌ {name} is not installed")
            print(f"   Install with: pip install ai-brain-python[{module.replace('_', '-')}]")

if __name__ == "__main__":
    check_framework_availability()
```

## 🚨 Troubleshooting

### Common Issues

#### 1. MongoDB Connection Issues

```bash
# Error: ServerSelectionTimeoutError
# Solution: Check connection string and network access

# Test MongoDB connection
python -c "
import motor.motor_asyncio
import asyncio

async def test_mongo():
    client = motor.motor_asyncio.AsyncIOMotorClient('your-mongodb-uri')
    try:
        await client.admin.command('ping')
        print('✅ MongoDB connection successful')
    except Exception as e:
        print(f'❌ MongoDB connection failed: {e}')
    finally:
        client.close()

asyncio.run(test_mongo())
"
```

#### 2. Import Errors

```bash
# Error: ModuleNotFoundError
# Solution: Ensure proper installation

# Check installation
pip list | grep ai-brain-python

# Reinstall if needed
pip uninstall ai-brain-python
pip install ai-brain-python[all-frameworks]
```

#### 3. Memory Issues

```bash
# Error: Out of memory during processing
# Solution: Increase system memory or reduce concurrent requests

# Set memory limits in configuration
export AI_BRAIN_MAX_CONCURRENT_REQUESTS=10
export AI_BRAIN_MEMORY_LIMIT_MB=1024
```

#### 4. API Key Issues

```bash
# Error: Authentication failed
# Solution: Verify API keys

# Test OpenAI API key
python -c "
import openai
import os
openai.api_key = os.getenv('OPENAI_API_KEY')
try:
    response = openai.models.list()
    print('✅ OpenAI API key is valid')
except Exception as e:
    print(f'❌ OpenAI API key error: {e}')
"
```

### Performance Optimization

#### 1. MongoDB Optimization

```javascript
// Create indexes for better performance
db.memories.createIndex({ "user_id": 1, "timestamp": -1 })
db.memories.createIndex({ "embedding": "2dsphere" })
db.conversations.createIndex({ "user_id": 1, "session_id": 1 })
```

#### 2. Python Optimization

```bash
# Use faster JSON library
pip install orjson

# Use faster HTTP client
pip install httpx[http2]

# Enable uvloop for better async performance (Linux/macOS)
pip install uvloop
```

#### 3. System Optimization

```bash
# Increase file descriptor limits
ulimit -n 65536

# Set optimal Python settings
export PYTHONOPTIMIZE=1
export PYTHONDONTWRITEBYTECODE=1
```

## 🔄 Upgrading

### Upgrade AI Brain

```bash
# Upgrade to latest version
pip install --upgrade ai-brain-python

# Upgrade with all frameworks
pip install --upgrade ai-brain-python[all-frameworks]
```

### Migration Between Versions

```python
# migration_script.py
import asyncio
from ai_brain_python import UniversalAIBrain
from ai_brain_python.database.migration import DatabaseMigration

async def migrate_database():
    """Migrate database to new version."""
    migration = DatabaseMigration()
    await migration.run_migrations()
    print("✅ Database migration completed")

if __name__ == "__main__":
    asyncio.run(migrate_database())
```

## 📞 Support

### Getting Help

1. **Documentation**: Check the [API Reference](API_REFERENCE.md)
2. **Examples**: Browse the [examples directory](../examples/)
3. **Issues**: Report bugs on [GitHub Issues](https://github.com/romiluz13/AI_Brain/issues)
4. **Discussions**: Join [GitHub Discussions](https://github.com/romiluz13/AI_Brain/discussions)

### Community Resources

- **Discord**: [AI Brain Community](https://discord.gg/ai-brain)
- **Reddit**: [r/AIBrain](https://reddit.com/r/AIBrain)
- **Stack Overflow**: Tag questions with `ai-brain-python`

---

## ✅ Next Steps

After successful installation:

1. **Read the [Quick Start Guide](../README.md#quick-start)**
2. **Try the [Getting Started Example](../examples/getting_started.py)**
3. **Explore [Framework-Specific Guides](frameworks/)**
4. **Check out [Cognitive Systems Documentation](cognitive_systems/)**

Happy coding with AI Brain! 🧠✨
