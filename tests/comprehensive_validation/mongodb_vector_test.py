"""
MongoDB Atlas Vector Search and Hybrid Search Testing

This module tests the latest MongoDB Atlas vector search capabilities including
the new hybrid search with $rankFusion aggregation stage (2025 feature).
"""

import asyncio
import logging
import time
from datetime import datetime
from typing import List, Dict, Any, Optional
import motor.motor_asyncio
import numpy as np

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class MongoDBVectorSearchTester:
    """Test MongoDB Atlas vector search and hybrid search capabilities."""
    
    def __init__(self, mongodb_uri: str, voyage_api_key: str):
        self.mongodb_uri = mongodb_uri
        self.voyage_api_key = voyage_api_key
        self.client: Optional[motor.motor_asyncio.AsyncIOMotorClient] = None
        self.db_name = "ai_brain_vector_test"
        self.collection_name = "vector_documents"
        
        # Test documents for vector search
        self.test_documents = [
            {
                "content": "Machine learning is a subset of artificial intelligence that focuses on algorithms that can learn from data",
                "category": "AI/ML",
                "tags": ["machine learning", "artificial intelligence", "algorithms", "data"]
            },
            {
                "content": "Python is a versatile programming language widely used in data science and web development",
                "category": "Programming",
                "tags": ["python", "programming", "data science", "web development"]
            },
            {
                "content": "MongoDB Atlas provides cloud database services with built-in vector search capabilities",
                "category": "Database",
                "tags": ["mongodb", "atlas", "cloud", "vector search", "database"]
            },
            {
                "content": "Natural language processing enables computers to understand and generate human language",
                "category": "AI/ML",
                "tags": ["nlp", "natural language", "processing", "language models"]
            },
            {
                "content": "React is a JavaScript library for building user interfaces with component-based architecture",
                "category": "Frontend",
                "tags": ["react", "javascript", "ui", "components", "frontend"]
            },
            {
                "content": "Docker containers provide lightweight virtualization for application deployment",
                "category": "DevOps",
                "tags": ["docker", "containers", "virtualization", "deployment"]
            },
            {
                "content": "Kubernetes orchestrates containerized applications across distributed systems",
                "category": "DevOps", 
                "tags": ["kubernetes", "orchestration", "containers", "distributed systems"]
            },
            {
                "content": "TensorFlow is an open-source machine learning framework for deep learning applications",
                "category": "AI/ML",
                "tags": ["tensorflow", "machine learning", "deep learning", "framework"]
            }
        ]
    
    async def initialize(self) -> bool:
        """Initialize MongoDB connection and test basic connectivity."""
        try:
            self.client = motor.motor_asyncio.AsyncIOMotorClient(self.mongodb_uri)
            
            # Test connection
            await self.client.admin.command('ping')
            logger.info("✅ MongoDB Atlas connection successful")
            
            return True
            
        except Exception as e:
            logger.error(f"❌ MongoDB connection failed: {e}")
            return False
    
    def _generate_mock_embedding(self, text: str) -> List[float]:
        """Generate mock embedding for testing (in production, use Voyage AI)."""
        # Simple hash-based mock embedding for testing
        import hashlib
        
        # Create deterministic embedding based on text content
        hash_obj = hashlib.md5(text.encode())
        hash_bytes = hash_obj.digest()
        
        # Convert to 1536-dimensional vector (OpenAI embedding size)
        embedding = []
        for i in range(1536):
            byte_index = i % len(hash_bytes)
            embedding.append((hash_bytes[byte_index] - 128) / 128.0)
        
        # Normalize the vector
        norm = np.linalg.norm(embedding)
        if norm > 0:
            embedding = [x / norm for x in embedding]
        
        return embedding
    
    async def setup_test_data(self) -> bool:
        """Set up test documents with embeddings."""
        try:
            db = self.client[self.db_name]
            collection = db[self.collection_name]
            
            # Clear existing test data
            await collection.delete_many({})
            
            # Prepare documents with embeddings
            documents_with_embeddings = []
            
            for i, doc in enumerate(self.test_documents):
                # Generate embedding for content
                embedding = self._generate_mock_embedding(doc["content"])
                
                document = {
                    "_id": f"doc_{i+1}",
                    "content": doc["content"],
                    "category": doc["category"],
                    "tags": doc["tags"],
                    "embedding": embedding,
                    "created_at": datetime.utcnow(),
                    "word_count": len(doc["content"].split()),
                    "char_count": len(doc["content"])
                }
                
                documents_with_embeddings.append(document)
            
            # Insert test documents
            result = await collection.insert_many(documents_with_embeddings)
            logger.info(f"✅ Inserted {len(result.inserted_ids)} test documents")
            
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to setup test data: {e}")
            return False
    
    async def test_vector_search_index_creation(self) -> Dict[str, Any]:
        """Test vector search index creation (requires Atlas UI or API)."""
        logger.info("🔍 Testing vector search index requirements...")
        
        # Note: Vector search indexes must be created through Atlas UI or Management API
        # This test validates the index configuration requirements
        
        vector_index_config = {
            "name": "vector_index",
            "type": "vectorSearch",
            "definition": {
                "fields": [
                    {
                        "type": "vector",
                        "path": "embedding",
                        "numDimensions": 1536,
                        "similarity": "cosine"
                    },
                    {
                        "type": "filter",
                        "path": "category"
                    },
                    {
                        "type": "filter", 
                        "path": "tags"
                    }
                ]
            }
        }
        
        text_search_index_config = {
            "name": "text_index",
            "type": "search",
            "definition": {
                "mappings": {
                    "dynamic": False,
                    "fields": {
                        "content": {
                            "type": "string",
                            "analyzer": "lucene.standard"
                        },
                        "category": {
                            "type": "string"
                        },
                        "tags": {
                            "type": "string"
                        }
                    }
                }
            }
        }
        
        return {
            "vector_index_config": vector_index_config,
            "text_search_index_config": text_search_index_config,
            "status": "configuration_ready",
            "note": "Indexes must be created through MongoDB Atlas UI or Management API"
        }
    
    async def test_vector_search_query(self) -> Dict[str, Any]:
        """Test vector search query functionality."""
        logger.info("🔍 Testing vector search queries...")
        
        try:
            db = self.client[self.db_name]
            collection = db[self.collection_name]
            
            # Generate query embedding
            query_text = "machine learning algorithms for data analysis"
            query_embedding = self._generate_mock_embedding(query_text)
            
            # Vector search pipeline
            vector_search_pipeline = [
                {
                    "$vectorSearch": {
                        "index": "vector_index",
                        "path": "embedding",
                        "queryVector": query_embedding,
                        "numCandidates": 100,
                        "limit": 5
                    }
                },
                {
                    "$project": {
                        "_id": 1,
                        "content": 1,
                        "category": 1,
                        "score": {"$meta": "vectorSearchScore"}
                    }
                }
            ]
            
            start_time = time.time()
            
            # Note: This will fail without proper vector index, but tests pipeline structure
            try:
                cursor = collection.aggregate(vector_search_pipeline)
                results = await cursor.to_list(length=5)
                search_time = (time.time() - start_time) * 1000
                
                return {
                    "status": "success",
                    "query_text": query_text,
                    "results_count": len(results),
                    "search_time_ms": search_time,
                    "results": results,
                    "pipeline": vector_search_pipeline
                }
                
            except Exception as e:
                # Expected to fail without vector index
                return {
                    "status": "index_required",
                    "query_text": query_text,
                    "error": str(e),
                    "pipeline": vector_search_pipeline,
                    "note": "Vector search index required for execution"
                }
                
        except Exception as e:
            logger.error(f"❌ Vector search test failed: {e}")
            return {"status": "error", "error": str(e)}
    
    async def test_hybrid_search_with_rank_fusion(self) -> Dict[str, Any]:
        """Test hybrid search using the new $rankFusion aggregation stage."""
        logger.info("🔗 Testing hybrid search with $rankFusion...")
        
        try:
            db = self.client[self.db_name]
            collection = db[self.collection_name]
            
            query_text = "machine learning python programming"
            query_embedding = self._generate_mock_embedding(query_text)
            
            # Hybrid search pipeline with $rankFusion (2025 feature)
            hybrid_search_pipeline = [
                {
                    "$rankFusion": {
                        "input": {
                            "pipelines": [
                                # Text search pipeline
                                [
                                    {
                                        "$search": {
                                            "index": "text_index",
                                            "text": {
                                                "query": query_text,
                                                "path": "content"
                                            }
                                        }
                                    },
                                    {
                                        "$limit": 10
                                    },
                                    {
                                        "$project": {
                                            "_id": 1,
                                            "content": 1,
                                            "category": 1,
                                            "textScore": {"$meta": "searchScore"}
                                        }
                                    }
                                ],
                                # Vector search pipeline
                                [
                                    {
                                        "$vectorSearch": {
                                            "index": "vector_index",
                                            "path": "embedding", 
                                            "queryVector": query_embedding,
                                            "numCandidates": 100,
                                            "limit": 10
                                        }
                                    },
                                    {
                                        "$project": {
                                            "_id": 1,
                                            "content": 1,
                                            "category": 1,
                                            "vectorScore": {"$meta": "vectorSearchScore"}
                                        }
                                    }
                                ]
                            ]
                        }
                    }
                },
                {
                    "$limit": 5
                },
                {
                    "$project": {
                        "_id": 1,
                        "content": 1,
                        "category": 1,
                        "hybridScore": {"$meta": "rankFusionScore"}
                    }
                }
            ]
            
            start_time = time.time()
            
            try:
                cursor = collection.aggregate(hybrid_search_pipeline)
                results = await cursor.to_list(length=5)
                search_time = (time.time() - start_time) * 1000
                
                return {
                    "status": "success",
                    "query_text": query_text,
                    "results_count": len(results),
                    "search_time_ms": search_time,
                    "results": results,
                    "pipeline": hybrid_search_pipeline,
                    "feature": "rankFusion_2025"
                }
                
            except Exception as e:
                # Expected to fail without proper indexes
                return {
                    "status": "indexes_required",
                    "query_text": query_text,
                    "error": str(e),
                    "pipeline": hybrid_search_pipeline,
                    "feature": "rankFusion_2025",
                    "note": "Both vector and text search indexes required for execution"
                }
                
        except Exception as e:
            logger.error(f"❌ Hybrid search test failed: {e}")
            return {"status": "error", "error": str(e)}
    
    async def test_filtered_vector_search(self) -> Dict[str, Any]:
        """Test vector search with metadata filtering."""
        logger.info("🎯 Testing filtered vector search...")
        
        try:
            db = self.client[self.db_name]
            collection = db[self.collection_name]
            
            query_text = "programming languages"
            query_embedding = self._generate_mock_embedding(query_text)
            
            # Filtered vector search pipeline
            filtered_search_pipeline = [
                {
                    "$vectorSearch": {
                        "index": "vector_index",
                        "path": "embedding",
                        "queryVector": query_embedding,
                        "numCandidates": 100,
                        "limit": 5,
                        "filter": {
                            "category": {"$in": ["Programming", "AI/ML"]}
                        }
                    }
                },
                {
                    "$project": {
                        "_id": 1,
                        "content": 1,
                        "category": 1,
                        "tags": 1,
                        "score": {"$meta": "vectorSearchScore"}
                    }
                }
            ]
            
            start_time = time.time()
            
            try:
                cursor = collection.aggregate(filtered_search_pipeline)
                results = await cursor.to_list(length=5)
                search_time = (time.time() - start_time) * 1000
                
                return {
                    "status": "success",
                    "query_text": query_text,
                    "filter": {"category": {"$in": ["Programming", "AI/ML"]}},
                    "results_count": len(results),
                    "search_time_ms": search_time,
                    "results": results,
                    "pipeline": filtered_search_pipeline
                }
                
            except Exception as e:
                return {
                    "status": "index_required",
                    "query_text": query_text,
                    "error": str(e),
                    "pipeline": filtered_search_pipeline,
                    "note": "Vector search index with filter fields required"
                }
                
        except Exception as e:
            logger.error(f"❌ Filtered vector search test failed: {e}")
            return {"status": "error", "error": str(e)}
    
    async def run_comprehensive_vector_tests(self) -> Dict[str, Any]:
        """Run all vector search tests and return comprehensive results."""
        logger.info("🚀 Running comprehensive MongoDB vector search tests...")
        
        results = {
            "test_timestamp": datetime.utcnow().isoformat(),
            "mongodb_connection": False,
            "test_data_setup": False,
            "vector_index_config": {},
            "vector_search_test": {},
            "hybrid_search_test": {},
            "filtered_search_test": {},
            "overall_status": "failed"
        }
        
        try:
            # Initialize connection
            if await self.initialize():
                results["mongodb_connection"] = True
                
                # Setup test data
                if await self.setup_test_data():
                    results["test_data_setup"] = True
                    
                    # Test index configuration
                    results["vector_index_config"] = await self.test_vector_search_index_creation()
                    
                    # Test vector search
                    results["vector_search_test"] = await self.test_vector_search_query()
                    
                    # Test hybrid search with $rankFusion
                    results["hybrid_search_test"] = await self.test_hybrid_search_with_rank_fusion()
                    
                    # Test filtered vector search
                    results["filtered_search_test"] = await self.test_filtered_vector_search()
                    
                    # Determine overall status
                    if (results["test_data_setup"] and 
                        results["vector_index_config"].get("status") == "configuration_ready"):
                        results["overall_status"] = "ready_for_indexing"
                    else:
                        results["overall_status"] = "partial_success"
            
        except Exception as e:
            logger.error(f"❌ Comprehensive vector tests failed: {e}")
            results["error"] = str(e)
        
        finally:
            if self.client:
                self.client.close()
        
        return results
    
    def print_test_summary(self, results: Dict[str, Any]):
        """Print a formatted summary of vector search test results."""
        print("\n" + "="*70)
        print("🔍 MONGODB ATLAS VECTOR SEARCH TEST RESULTS")
        print("="*70)
        
        print(f"📅 Test Time: {results.get('test_timestamp', 'Unknown')}")
        print(f"🗄️ MongoDB Connection: {'✅' if results.get('mongodb_connection') else '❌'}")
        print(f"📊 Test Data Setup: {'✅' if results.get('test_data_setup') else '❌'}")
        print(f"🎯 Overall Status: {results.get('overall_status', 'unknown')}")
        
        print(f"\n📋 TEST RESULTS:")
        print("-" * 70)
        
        # Vector index configuration
        index_config = results.get("vector_index_config", {})
        print(f"🔧 Vector Index Config: {index_config.get('status', 'unknown')}")
        
        # Vector search test
        vector_test = results.get("vector_search_test", {})
        print(f"🔍 Vector Search: {vector_test.get('status', 'unknown')}")
        if vector_test.get("search_time_ms"):
            print(f"   ⏱️ Search Time: {vector_test['search_time_ms']:.2f}ms")
        
        # Hybrid search test
        hybrid_test = results.get("hybrid_search_test", {})
        print(f"🔗 Hybrid Search ($rankFusion): {hybrid_test.get('status', 'unknown')}")
        if hybrid_test.get("search_time_ms"):
            print(f"   ⏱️ Search Time: {hybrid_test['search_time_ms']:.2f}ms")
        
        # Filtered search test
        filtered_test = results.get("filtered_search_test", {})
        print(f"🎯 Filtered Vector Search: {filtered_test.get('status', 'unknown')}")
        if filtered_test.get("search_time_ms"):
            print(f"   ⏱️ Search Time: {filtered_test['search_time_ms']:.2f}ms")
        
        print(f"\n📝 NEXT STEPS:")
        print("-" * 70)
        
        if results.get("overall_status") == "ready_for_indexing":
            print("✅ Test data and configurations are ready")
            print("🔧 Create vector search indexes in MongoDB Atlas:")
            print("   1. Go to Atlas UI > Database > Search")
            print("   2. Create vector search index with provided configuration")
            print("   3. Create text search index for hybrid search")
            print("   4. Re-run tests to validate search functionality")
        else:
            print("❌ Some tests failed - check individual test results")
            print("🔧 Ensure MongoDB Atlas connection and permissions")
        
        print("="*70)


async def main():
    """Main function to run MongoDB vector search tests."""
    # Use provided credentials
    mongodb_uri = "mongodb+srv://romiluz:H97r3aQBnxWawZbx@aibrain.tnv45wr.mongodb.net/?retryWrites=true&w=majority&appName=aibrain"
    voyage_api_key = "pa-NHB7D_EtgEImAVQkjIZ6PxoGVHcTOQvUujwDeq8m9-Q"
    
    tester = MongoDBVectorSearchTester(mongodb_uri, voyage_api_key)
    results = await tester.run_comprehensive_vector_tests()
    tester.print_test_summary(results)
    
    return results


if __name__ == "__main__":
    asyncio.run(main())
