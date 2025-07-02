"""
MongoDB Connection for AI Brain Python

Exact Python equivalent of JavaScript MongoConnection.ts with:
- Singleton pattern matching JavaScript implementation
- Identical connection options and error handling
- Same public interface and method signatures
- Matching console output and logging format
"""

import asyncio
import logging
from typing import Any, Dict, List, Optional, Union
from urllib.parse import quote_plus
from motor.motor_asyncio import AsyncIOMotorClient, AsyncIOMotorDatabase, AsyncIOMotorCollection
from pymongo.errors import ConnectionFailure, ServerSelectionTimeoutError
from pymongo import IndexModel, ASCENDING, DESCENDING, TEXT
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


class MongoDBConfig(BaseModel):
    """MongoDB Atlas configuration model."""

    # MongoDB Atlas connection (default)
    connection_string: Optional[str] = Field(default=None, description="MongoDB Atlas connection string")
    cluster_name: Optional[str] = Field(default=None, description="MongoDB Atlas cluster name")

    # Fallback for local development
    host: str = Field(default="localhost", description="MongoDB host (local dev only)")
    port: int = Field(default=27017, description="MongoDB port (local dev only)")
    username: Optional[str] = Field(default=None, description="MongoDB username")
    password: Optional[str] = Field(default=None, description="MongoDB password")
    database: str = Field(default="ai_brain", description="Database name")
    auth_source: str = Field(default="admin", description="Authentication database")

    # Atlas-specific settings
    use_atlas: bool = Field(default=True, description="Use MongoDB Atlas by default")
    atlas_app_name: str = Field(default="AI-Brain-Python", description="Atlas application name")
    
    # Connection pool settings
    max_pool_size: int = Field(default=100, description="Maximum connection pool size")
    min_pool_size: int = Field(default=10, description="Minimum connection pool size")
    max_idle_time_ms: int = Field(default=30000, description="Maximum idle time in milliseconds")
    
    # Timeout settings
    connect_timeout_ms: int = Field(default=10000, description="Connection timeout in milliseconds")
    server_selection_timeout_ms: int = Field(default=5000, description="Server selection timeout")
    socket_timeout_ms: int = Field(default=10000, description="Socket timeout in milliseconds")
    
    # Additional options
    retry_writes: bool = Field(default=True, description="Enable retry writes")
    read_preference: str = Field(default="primary", description="Read preference")
    
    def get_connection_string(self) -> str:
        """Generate MongoDB connection string with Atlas-first approach."""
        # Use Atlas connection string if provided
        if self.use_atlas and self.connection_string:
            # Add application name and additional options to Atlas connection string
            separator = "&" if "?" in self.connection_string else "?"
            atlas_options = (
                f"{separator}appName={quote_plus(self.atlas_app_name)}"
                f"&maxPoolSize={self.max_pool_size}"
                f"&minPoolSize={self.min_pool_size}"
                f"&maxIdleTimeMS={self.max_idle_time_ms}"
                f"&connectTimeoutMS={self.connect_timeout_ms}"
                f"&serverSelectionTimeoutMS={self.server_selection_timeout_ms}"
                f"&socketTimeoutMS={self.socket_timeout_ms}"
                f"&retryWrites={str(self.retry_writes).lower()}"
                f"&readPreference={self.read_preference}"
            )
            return f"{self.connection_string}{atlas_options}"

        # Fallback to local/custom MongoDB setup
        if self.username and self.password:
            credentials = f"{quote_plus(self.username)}:{quote_plus(self.password)}@"
        else:
            credentials = ""

        return (
            f"mongodb://{credentials}{self.host}:{self.port}/{self.database}"
            f"?authSource={self.auth_source}"
            f"&appName={quote_plus(self.atlas_app_name)}"
            f"&maxPoolSize={self.max_pool_size}"
            f"&minPoolSize={self.min_pool_size}"
            f"&maxIdleTimeMS={self.max_idle_time_ms}"
            f"&connectTimeoutMS={self.connect_timeout_ms}"
            f"&serverSelectionTimeoutMS={self.server_selection_timeout_ms}"
            f"&socketTimeoutMS={self.socket_timeout_ms}"
            f"&retryWrites={str(self.retry_writes).lower()}"
            f"&readPreference={self.read_preference}"
        )


class MongoDBClient:
    """Async MongoDB client for AI Brain cognitive data storage."""
    
    def __init__(self, config: MongoDBConfig):
        """Initialize MongoDB client with configuration."""
        self.config = config
        self.client: Optional[AsyncIOMotorClient] = None
        self.database: Optional[AsyncIOMotorDatabase] = None
        self._connection_lock = asyncio.Lock()
        self._is_connected = False
        
        # Collection names for cognitive systems
        self.collections = {
            "cognitive_states": "cognitive_states",
            "semantic_memory": "semantic_memory",
            "goal_hierarchy": "goal_hierarchy",
            "emotional_states": "emotional_states",
            "attention_states": "attention_states",
            "confidence_tracking": "confidence_tracking",
            "cultural_contexts": "cultural_contexts",
            "skill_assessments": "skill_assessments",
            "communication_protocols": "communication_protocols",
            "temporal_plans": "temporal_plans",
            "safety_assessments": "safety_assessments",
            "monitoring_metrics": "monitoring_metrics",
            "tool_validations": "tool_validations",
            "workflow_states": "workflow_states",
            "multimodal_data": "multimodal_data",
            "human_feedback": "human_feedback",
        }
    
    async def connect(self) -> None:
        """Establish connection to MongoDB."""
        async with self._connection_lock:
            if self._is_connected:
                return
            
            try:
                connection_string = self.config.get_connection_string()
                logger.info(f"Connecting to MongoDB: {self.config.host}:{self.config.port}")
                
                self.client = AsyncIOMotorClient(connection_string)
                self.database = self.client[self.config.database]
                
                # Test connection
                await self.client.admin.command('ping')
                self._is_connected = True
                
                logger.info("Successfully connected to MongoDB")
                
                # Initialize collections and indexes
                await self._initialize_collections()
                
            except (ConnectionFailure, ServerSelectionTimeoutError) as e:
                logger.error(f"Failed to connect to MongoDB: {e}")
                raise
            except Exception as e:
                logger.error(f"Unexpected error connecting to MongoDB: {e}")
                raise
    
    async def disconnect(self) -> None:
        """Close MongoDB connection."""
        async with self._connection_lock:
            if self.client and self._is_connected:
                self.client.close()
                self._is_connected = False
                logger.info("Disconnected from MongoDB")
    
    async def _initialize_collections(self) -> None:
        """Initialize collections and create indexes."""
        if self.database is None:
            raise RuntimeError("Database not initialized")
        
        logger.info("Initializing MongoDB collections and indexes")
        
        # Create indexes for cognitive_states
        cognitive_states_indexes = [
            IndexModel([("system_id", ASCENDING)]),
            IndexModel([("timestamp", DESCENDING)]),
            IndexModel([("user_id", ASCENDING)]),
            IndexModel([("system_id", ASCENDING), ("timestamp", DESCENDING)]),
            IndexModel([("user_id", ASCENDING), ("system_id", ASCENDING), ("timestamp", DESCENDING)]),
        ]
        await self.database[self.collections["cognitive_states"]].create_indexes(cognitive_states_indexes)
        
        # Create indexes for semantic_memory (including vector search)
        semantic_memory_indexes = [
            IndexModel([("content", "text")]),
            IndexModel([("timestamp", DESCENDING)]),
            IndexModel([("relevance_score", DESCENDING)]),
            IndexModel([("access_count", DESCENDING)]),
            IndexModel([("user_id", ASCENDING), ("relevance_score", DESCENDING)]),
        ]
        await self.database[self.collections["semantic_memory"]].create_indexes(semantic_memory_indexes)
        
        # Create indexes for goal_hierarchy
        goal_hierarchy_indexes = [
            IndexModel([("user_id", ASCENDING)]),
            IndexModel([("status", ASCENDING)]),
            IndexModel([("priority", DESCENDING)]),
            IndexModel([("created_at", DESCENDING)]),
            IndexModel([("updated_at", DESCENDING)]),
        ]
        await self.database[self.collections["goal_hierarchy"]].create_indexes(goal_hierarchy_indexes)
        
        # Create indexes for monitoring_metrics with TTL
        monitoring_indexes = [
            IndexModel([("metric_type", ASCENDING)]),
            IndexModel([("timestamp", ASCENDING)], expireAfterSeconds=2592000),  # 30 days TTL
            IndexModel([("system_id", ASCENDING)]),
            IndexModel([("value", DESCENDING)]),
        ]
        await self.database[self.collections["monitoring_metrics"]].create_indexes(monitoring_indexes)
        
        logger.info("MongoDB collections and indexes initialized successfully")
    
    def get_collection(self, collection_name: str) -> AsyncIOMotorCollection:
        """Get a collection by name."""
        if self.database is None:
            raise RuntimeError("Database not initialized. Call connect() first.")
        
        if collection_name not in self.collections:
            raise ValueError(f"Unknown collection: {collection_name}")
        
        return self.database[self.collections[collection_name]]
    
    async def insert_one(
        self, 
        collection_name: str, 
        document: Dict[str, Any]
    ) -> str:
        """Insert a single document."""
        try:
            collection = self.get_collection(collection_name)
            document["created_at"] = datetime.utcnow()
            document["updated_at"] = datetime.utcnow()
            
            result = await collection.insert_one(document)
            logger.debug(f"Inserted document in {collection_name}: {result.inserted_id}")
            return str(result.inserted_id)
            
        except DuplicateKeyError as e:
            logger.error(f"Duplicate key error in {collection_name}: {e}")
            raise
        except Exception as e:
            logger.error(f"Error inserting document in {collection_name}: {e}")
            raise
    
    async def insert_many(
        self, 
        collection_name: str, 
        documents: List[Dict[str, Any]]
    ) -> List[str]:
        """Insert multiple documents."""
        try:
            collection = self.get_collection(collection_name)
            
            # Add timestamps to all documents
            now = datetime.utcnow()
            for doc in documents:
                doc["created_at"] = now
                doc["updated_at"] = now
            
            result = await collection.insert_many(documents)
            logger.debug(f"Inserted {len(documents)} documents in {collection_name}")
            return [str(id) for id in result.inserted_ids]
            
        except Exception as e:
            logger.error(f"Error inserting documents in {collection_name}: {e}")
            raise
    
    async def find_one(
        self, 
        collection_name: str, 
        filter_dict: Dict[str, Any],
        projection: Optional[Dict[str, Any]] = None
    ) -> Optional[Dict[str, Any]]:
        """Find a single document."""
        try:
            collection = self.get_collection(collection_name)
            result = await collection.find_one(filter_dict, projection)
            return result
            
        except Exception as e:
            logger.error(f"Error finding document in {collection_name}: {e}")
            raise
    
    async def find_many(
        self, 
        collection_name: str, 
        filter_dict: Dict[str, Any],
        projection: Optional[Dict[str, Any]] = None,
        sort: Optional[List[tuple]] = None,
        limit: Optional[int] = None,
        skip: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """Find multiple documents."""
        try:
            collection = self.get_collection(collection_name)
            cursor = collection.find(filter_dict, projection)
            
            if sort:
                cursor = cursor.sort(sort)
            if skip:
                cursor = cursor.skip(skip)
            if limit:
                cursor = cursor.limit(limit)
            
            results = await cursor.to_list(length=limit)
            return results
            
        except Exception as e:
            logger.error(f"Error finding documents in {collection_name}: {e}")
            raise
    
    async def update_one(
        self, 
        collection_name: str, 
        filter_dict: Dict[str, Any],
        update_dict: Dict[str, Any],
        upsert: bool = False
    ) -> bool:
        """Update a single document."""
        try:
            collection = self.get_collection(collection_name)
            
            # Add updated timestamp
            if "$set" not in update_dict:
                update_dict["$set"] = {}
            update_dict["$set"]["updated_at"] = datetime.utcnow()
            
            result = await collection.update_one(filter_dict, update_dict, upsert=upsert)
            return result.modified_count > 0 or (upsert and result.upserted_id is not None)
            
        except Exception as e:
            logger.error(f"Error updating document in {collection_name}: {e}")
            raise
    
    async def delete_one(
        self, 
        collection_name: str, 
        filter_dict: Dict[str, Any]
    ) -> bool:
        """Delete a single document."""
        try:
            collection = self.get_collection(collection_name)
            result = await collection.delete_one(filter_dict)
            return result.deleted_count > 0
            
        except Exception as e:
            logger.error(f"Error deleting document in {collection_name}: {e}")
            raise
    
    async def count_documents(
        self, 
        collection_name: str, 
        filter_dict: Dict[str, Any]
    ) -> int:
        """Count documents matching filter."""
        try:
            collection = self.get_collection(collection_name)
            count = await collection.count_documents(filter_dict)
            return count
            
        except Exception as e:
            logger.error(f"Error counting documents in {collection_name}: {e}")
            raise
    
    async def aggregate(
        self, 
        collection_name: str, 
        pipeline: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Execute aggregation pipeline."""
        try:
            collection = self.get_collection(collection_name)
            cursor = collection.aggregate(pipeline)
            results = await cursor.to_list(length=None)
            return results
            
        except Exception as e:
            logger.error(f"Error executing aggregation in {collection_name}: {e}")
            raise
    
    async def health_check(self) -> Dict[str, Any]:
        """Perform health check on MongoDB connection."""
        try:
            if not self.client:
                return {"status": "disconnected", "error": "No client connection"}
            
            # Ping the database
            await self.client.admin.command('ping')
            
            # Get server status
            server_status = await self.client.admin.command('serverStatus')
            
            return {
                "status": "healthy",
                "database": self.config.database,
                "host": self.config.host,
                "port": self.config.port,
                "uptime": server_status.get("uptime", 0),
                "connections": server_status.get("connections", {}),
            }
            
        except Exception as e:
            logger.error(f"MongoDB health check failed: {e}")
            return {"status": "unhealthy", "error": str(e)}
    
    async def __aenter__(self):
        """Async context manager entry."""
        await self.connect()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        await self.disconnect()
