"""
Unit tests for MongoDB client.
"""

import pytest
import asyncio
from datetime import datetime
from unittest.mock import AsyncMock, MagicMock, patch

from ai_brain_python.storage.mongodb_client import MongoDBClient, MongoDBConfig


class TestMongoDBConfig:
    """Test MongoDB configuration."""
    
    def test_default_config(self):
        """Test default configuration values."""
        config = MongoDBConfig()
        
        assert config.host == "localhost"
        assert config.port == 27017
        assert config.database == "ai_brain"
        assert config.max_pool_size == 100
        assert config.min_pool_size == 10
    
    def test_connection_string_without_auth(self):
        """Test connection string generation without authentication."""
        config = MongoDBConfig(
            host="localhost",
            port=27017,
            database="test_db"
        )
        
        connection_string = config.get_connection_string()
        
        assert "mongodb://localhost:27017/test_db" in connection_string
        assert "authSource=admin" in connection_string
        assert "maxPoolSize=100" in connection_string
    
    def test_connection_string_with_auth(self):
        """Test connection string generation with authentication."""
        config = MongoDBConfig(
            host="localhost",
            port=27017,
            username="testuser",
            password="testpass",
            database="test_db"
        )
        
        connection_string = config.get_connection_string()
        
        assert "mongodb://testuser:testpass@localhost:27017/test_db" in connection_string
        assert "authSource=admin" in connection_string


@pytest.mark.unit
@pytest.mark.mongodb
class TestMongoDBClient:
    """Test MongoDB client functionality."""
    
    @pytest.fixture
    def config(self):
        """Create test configuration."""
        return MongoDBConfig(
            host="localhost",
            port=27017,
            database="test_ai_brain",
            username="test_user",
            password="test_pass"
        )
    
    @pytest.fixture
    def mongodb_client(self, config):
        """Create MongoDB client for testing."""
        return MongoDBClient(config)
    
    @pytest.mark.asyncio
    async def test_initialization(self, mongodb_client):
        """Test client initialization."""
        assert mongodb_client.config is not None
        assert mongodb_client.client is None
        assert mongodb_client.database is None
        assert not mongodb_client._is_connected
    
    @pytest.mark.asyncio
    @patch('ai_brain_python.storage.mongodb_client.AsyncIOMotorClient')
    async def test_connect_success(self, mock_motor_client, mongodb_client):
        """Test successful connection."""
        # Mock the motor client
        mock_client_instance = AsyncMock()
        mock_database = AsyncMock()
        mock_motor_client.return_value = mock_client_instance
        mock_client_instance.__getitem__.return_value = mock_database
        
        # Mock the ping command
        mock_client_instance.admin.command = AsyncMock(return_value={"ok": 1})
        
        # Mock collection operations for initialization
        mock_collection = AsyncMock()
        mock_database.__getitem__.return_value = mock_collection
        mock_collection.create_indexes = AsyncMock()
        
        await mongodb_client.connect()
        
        assert mongodb_client._is_connected
        assert mongodb_client.client is not None
        assert mongodb_client.database is not None
        
        # Verify ping was called
        mock_client_instance.admin.command.assert_called_once_with('ping')
    
    @pytest.mark.asyncio
    @patch('ai_brain_python.storage.mongodb_client.AsyncIOMotorClient')
    async def test_connect_failure(self, mock_motor_client, mongodb_client):
        """Test connection failure."""
        # Mock connection failure
        mock_motor_client.side_effect = Exception("Connection failed")
        
        with pytest.raises(Exception, match="Connection failed"):
            await mongodb_client.connect()
        
        assert not mongodb_client._is_connected
    
    @pytest.mark.asyncio
    async def test_disconnect(self, mongodb_client):
        """Test disconnection."""
        # Mock connected state
        mongodb_client._is_connected = True
        mongodb_client.client = MagicMock()
        
        await mongodb_client.disconnect()
        
        assert not mongodb_client._is_connected
        mongodb_client.client.close.assert_called_once()
    
    @pytest.mark.asyncio
    @patch('ai_brain_python.storage.mongodb_client.AsyncIOMotorClient')
    async def test_get_collection(self, mock_motor_client, mongodb_client):
        """Test getting a collection."""
        # Setup mocks
        mock_client_instance = AsyncMock()
        mock_database = AsyncMock()
        mock_collection = AsyncMock()
        
        mock_motor_client.return_value = mock_client_instance
        mock_client_instance.__getitem__.return_value = mock_database
        mock_database.__getitem__.return_value = mock_collection
        mock_client_instance.admin.command = AsyncMock(return_value={"ok": 1})
        
        # Mock collection initialization
        mock_collection.create_indexes = AsyncMock()
        
        await mongodb_client.connect()
        
        # Test getting valid collection
        collection = mongodb_client.get_collection("cognitive_states")
        assert collection is not None
        
        # Test getting invalid collection
        with pytest.raises(ValueError, match="Unknown collection"):
            mongodb_client.get_collection("invalid_collection")
    
    @pytest.mark.asyncio
    @patch('ai_brain_python.storage.mongodb_client.AsyncIOMotorClient')
    async def test_insert_one(self, mock_motor_client, mongodb_client):
        """Test inserting a single document."""
        # Setup mocks
        mock_client_instance = AsyncMock()
        mock_database = AsyncMock()
        mock_collection = AsyncMock()
        mock_result = MagicMock()
        mock_result.inserted_id = "test_id_123"
        
        mock_motor_client.return_value = mock_client_instance
        mock_client_instance.__getitem__.return_value = mock_database
        mock_database.__getitem__.return_value = mock_collection
        mock_client_instance.admin.command = AsyncMock(return_value={"ok": 1})
        mock_collection.create_indexes = AsyncMock()
        mock_collection.insert_one = AsyncMock(return_value=mock_result)
        
        await mongodb_client.connect()
        
        # Test document insertion
        test_doc = {"test_field": "test_value"}
        result_id = await mongodb_client.insert_one("cognitive_states", test_doc)
        
        assert result_id == "test_id_123"
        mock_collection.insert_one.assert_called_once()
        
        # Verify timestamps were added
        call_args = mock_collection.insert_one.call_args[0][0]
        assert "created_at" in call_args
        assert "updated_at" in call_args
        assert isinstance(call_args["created_at"], datetime)
    
    @pytest.mark.asyncio
    @patch('ai_brain_python.storage.mongodb_client.AsyncIOMotorClient')
    async def test_find_one(self, mock_motor_client, mongodb_client):
        """Test finding a single document."""
        # Setup mocks
        mock_client_instance = AsyncMock()
        mock_database = AsyncMock()
        mock_collection = AsyncMock()
        
        mock_motor_client.return_value = mock_client_instance
        mock_client_instance.__getitem__.return_value = mock_database
        mock_database.__getitem__.return_value = mock_collection
        mock_client_instance.admin.command = AsyncMock(return_value={"ok": 1})
        mock_collection.create_indexes = AsyncMock()
        
        # Mock find_one result
        expected_doc = {"_id": "test_id", "test_field": "test_value"}
        mock_collection.find_one = AsyncMock(return_value=expected_doc)
        
        await mongodb_client.connect()
        
        # Test finding document
        filter_dict = {"test_field": "test_value"}
        result = await mongodb_client.find_one("cognitive_states", filter_dict)
        
        assert result == expected_doc
        mock_collection.find_one.assert_called_once_with(filter_dict, None)
    
    @pytest.mark.asyncio
    @patch('ai_brain_python.storage.mongodb_client.AsyncIOMotorClient')
    async def test_update_one(self, mock_motor_client, mongodb_client):
        """Test updating a single document."""
        # Setup mocks
        mock_client_instance = AsyncMock()
        mock_database = AsyncMock()
        mock_collection = AsyncMock()
        mock_result = MagicMock()
        mock_result.modified_count = 1
        
        mock_motor_client.return_value = mock_client_instance
        mock_client_instance.__getitem__.return_value = mock_database
        mock_database.__getitem__.return_value = mock_collection
        mock_client_instance.admin.command = AsyncMock(return_value={"ok": 1})
        mock_collection.create_indexes = AsyncMock()
        mock_collection.update_one = AsyncMock(return_value=mock_result)
        
        await mongodb_client.connect()
        
        # Test document update
        filter_dict = {"_id": "test_id"}
        update_dict = {"$set": {"test_field": "new_value"}}
        result = await mongodb_client.update_one("cognitive_states", filter_dict, update_dict)
        
        assert result is True
        mock_collection.update_one.assert_called_once()
        
        # Verify updated_at timestamp was added
        call_args = mock_collection.update_one.call_args[0]
        assert "updated_at" in call_args[1]["$set"]
    
    @pytest.mark.asyncio
    @patch('ai_brain_python.storage.mongodb_client.AsyncIOMotorClient')
    async def test_health_check_healthy(self, mock_motor_client, mongodb_client):
        """Test health check when healthy."""
        # Setup mocks
        mock_client_instance = AsyncMock()
        mock_database = AsyncMock()
        
        mock_motor_client.return_value = mock_client_instance
        mock_client_instance.__getitem__.return_value = mock_database
        mock_client_instance.admin.command = AsyncMock(return_value={"ok": 1})
        mock_database.__getitem__.return_value.create_indexes = AsyncMock()
        
        # Mock server status
        mock_server_status = {
            "uptime": 12345,
            "connections": {"current": 10, "available": 90}
        }
        mock_client_instance.admin.command.side_effect = [
            {"ok": 1},  # ping
            mock_server_status  # serverStatus
        ]
        
        await mongodb_client.connect()
        
        # Test health check
        health = await mongodb_client.health_check()
        
        assert health["status"] == "healthy"
        assert health["database"] == mongodb_client.config.database
        assert health["uptime"] == 12345
        assert "connections" in health
    
    @pytest.mark.asyncio
    async def test_health_check_unhealthy(self, mongodb_client):
        """Test health check when unhealthy."""
        # Test with no client connection
        health = await mongodb_client.health_check()
        
        assert health["status"] == "disconnected"
        assert "error" in health
    
    @pytest.mark.asyncio
    @patch('ai_brain_python.storage.mongodb_client.AsyncIOMotorClient')
    async def test_context_manager(self, mock_motor_client, mongodb_client):
        """Test async context manager."""
        # Setup mocks
        mock_client_instance = AsyncMock()
        mock_database = AsyncMock()
        
        mock_motor_client.return_value = mock_client_instance
        mock_client_instance.__getitem__.return_value = mock_database
        mock_client_instance.admin.command = AsyncMock(return_value={"ok": 1})
        mock_database.__getitem__.return_value.create_indexes = AsyncMock()
        
        # Test context manager
        async with mongodb_client as client:
            assert client._is_connected
            assert client is mongodb_client
        
        # Verify disconnect was called
        mock_client_instance.close.assert_called_once()
