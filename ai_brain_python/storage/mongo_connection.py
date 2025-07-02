"""
MongoDB Connection for AI Brain Python

Exact Python equivalent of JavaScript MongoConnection.ts with:
- Singleton pattern matching JavaScript implementation
- Identical connection options and error handling
- Same public interface and method signatures
- Matching console output and logging format
"""

import asyncio
from typing import Any, Dict, Optional
from motor.motor_asyncio import AsyncIOMotorClient, AsyncIOMotorDatabase
from pymongo.errors import ConnectionFailure, ServerSelectionTimeoutError
from pydantic import BaseModel, Field


class MongoConnectionConfig(BaseModel):
    """MongoDB connection configuration matching JavaScript MongoConnectionConfig interface."""
    
    uri: str = Field(..., description="MongoDB connection URI")
    db_name: str = Field(..., description="Database name")
    options: Optional[Dict[str, Any]] = Field(default=None, description="Additional MongoDB client options")


class MongoConnection:
    """
    Singleton MongoDB connection class - exact Python equivalent of JavaScript MongoConnection.ts
    
    Features:
    - Singleton pattern matching JavaScript implementation
    - Identical connection options and timeouts
    - Same error handling and logging format
    - Exact same public interface
    """
    
    _instance: Optional['MongoConnection'] = None
    
    def __init__(self, config: MongoConnectionConfig):
        """Private constructor - use get_instance() instead."""
        self._config = config
        self._is_connected = False
        self._client: Optional[AsyncIOMotorClient] = None
        self._db: Optional[AsyncIOMotorDatabase] = None
        
        # Optimized connection options for Atlas - matching JavaScript exactly
        default_options = {
            "maxPoolSize": 10,
            "serverSelectionTimeoutMS": 5000,
            "socketTimeoutMS": 45000,
            "retryWrites": True,
            "retryReads": True,
        }
        
        # Merge with user options
        if config.options:
            default_options.update(config.options)
        
        self._client = AsyncIOMotorClient(config.uri, **default_options)
        self._db = self._client[config.db_name]
    
    @classmethod
    def get_instance(cls, config: MongoConnectionConfig) -> 'MongoConnection':
        """Get singleton instance - exact equivalent of JavaScript getInstance()."""
        if cls._instance is None:
            cls._instance = cls(config)
        return cls._instance
    
    async def connect(self) -> None:
        """Connect to MongoDB - exact equivalent of JavaScript connect()."""
        if self._is_connected:
            return
        
        try:
            await self._client.connect()
            
            # Test the connection - exact same as JavaScript
            await self._client.admin.command({"ping": 1})
            
            self._is_connected = True
            print(f"✅ Connected to MongoDB Atlas: {self._config.db_name}")
            
        except Exception as error:
            self._is_connected = False
            print(f"❌ MongoDB connection failed: {error}")
            
            # Match JavaScript error message format exactly
            error_message = str(error) if error else "Unknown error"
            raise Exception(f"Failed to connect to MongoDB: {error_message}")
    
    async def disconnect(self) -> None:
        """Disconnect from MongoDB - exact equivalent of JavaScript disconnect()."""
        if not self._is_connected:
            return
        
        try:
            self._client.close()
            self._is_connected = False
            print("✅ Disconnected from MongoDB Atlas")
            
        except Exception as error:
            print(f"❌ Error disconnecting from MongoDB: {error}")
            raise error
    
    def get_db(self) -> AsyncIOMotorDatabase:
        """Get database instance - exact equivalent of JavaScript getDb()."""
        if not self._is_connected:
            raise Exception("MongoDB not connected. Call connect() first.")
        return self._db
    
    def get_client(self) -> AsyncIOMotorClient:
        """Get client instance - exact equivalent of JavaScript getClient()."""
        if not self._is_connected:
            raise Exception("MongoDB not connected. Call connect() first.")
        return self._client
    
    def is_connection_active(self) -> bool:
        """Check if connection is active - exact equivalent of JavaScript isConnectionActive()."""
        return self._is_connected
    
    async def health_check(self) -> bool:
        """Perform health check - exact equivalent of JavaScript healthCheck()."""
        try:
            await self._client.admin.command({"ping": 1})
            return True
        except:
            return False
