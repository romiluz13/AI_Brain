"""
Base Collection Classes for Universal AI Brain Python

Exact Python equivalent of JavaScript BaseCollection.ts with:
- Same CRUD operations and method signatures
- Identical indexing and schema validation
- Matching error handling and logging
- Same pagination and aggregation utilities
"""

import json
import os
from abc import ABC, abstractmethod
from datetime import datetime
from typing import Any, Dict, List, Optional, TypeVar, Generic, Union
from bson import ObjectId
from motor.motor_asyncio import AsyncIOMotorDatabase, AsyncIOMotorCollection
from pymongo import ASCENDING, DESCENDING
from pydantic import BaseModel, Field
import jsonschema
from jsonschema import validate, ValidationError as JSONSchemaValidationError

from ..utils.logger import logger
from ..core.exceptions import ValidationError, BrainError


# ============================================================================
# BASE TYPES - Exact TypeScript Equivalents
# ============================================================================

class BaseDocument(BaseModel):
    """Exact equivalent of JavaScript BaseDocument interface."""
    id_: Optional[str] = Field(None, alias='_id')
    created_at: Optional[datetime] = Field(None, alias='createdAt')
    updated_at: Optional[datetime] = Field(None, alias='updatedAt')
    
    class Config:
        allow_population_by_field_name = True
        arbitrary_types_allowed = True


class PaginationOptions(BaseModel):
    """Exact equivalent of JavaScript PaginationOptions interface."""
    limit: Optional[int] = None
    skip: Optional[int] = None
    sort: Optional[Dict[str, int]] = None  # 1 for ascending, -1 for descending


class PaginatedResult(BaseModel, Generic[TypeVar('T')]):
    """Exact equivalent of JavaScript PaginatedResult interface."""
    documents: List[Any]
    total: int
    has_more: bool = Field(..., alias='hasMore')
    page: int
    total_pages: int = Field(..., alias='totalPages')
    
    class Config:
        allow_population_by_field_name = True


# ============================================================================
# BASE COLLECTION CLASS - Exact JavaScript Equivalent
# ============================================================================

T = TypeVar('T', bound=BaseDocument)


class BaseCollection(ABC, Generic[T]):
    """
    BaseCollection - Abstract base class for MongoDB collections
    
    Exact Python equivalent of JavaScript BaseCollection with:
    - Same CRUD operations and method signatures
    - Identical validation and error handling
    - Same pagination and aggregation utilities
    - Matching index management
    """
    
    def __init__(self, db: AsyncIOMotorDatabase):
        """Initialize base collection - exact equivalent of JavaScript constructor."""
        self.db = db
        self.collection: Optional[AsyncIOMotorCollection] = None
        self.schema: Optional[Dict[str, Any]] = None
        self._collection_name: Optional[str] = None
    
    @property
    @abstractmethod
    def collection_name(self) -> str:
        """Abstract property for collection name."""
        pass
    
    def initialize_collection(self) -> None:
        """Initialize the collection - exact equivalent of JavaScript initializeCollection()."""
        if not self.collection:
            self.collection = self.db[self.collection_name]
            self._collection_name = self.collection_name
            self.load_schema()
    
    def load_schema(self) -> None:
        """Load JSON schema for validation - exact equivalent of JavaScript loadSchema()."""
        try:
            # Try to load schema from schemas directory
            schema_path = os.path.join(
                os.path.dirname(__file__), 
                '..', 'schemas', 
                f'{self.collection_name}.json'
            )
            
            if os.path.exists(schema_path):
                with open(schema_path, 'r') as f:
                    self.schema = json.load(f)
            else:
                logger.warn(f"⚠️ No schema found for collection {self.collection_name}")
        except Exception as error:
            logger.warn(f"⚠️ No schema found for collection {self.collection_name}")
    
    async def validate_document(self, document: Dict[str, Any]) -> None:
        """Validate document against JSON schema - exact equivalent of JavaScript validateDocument()."""
        if not self.schema:
            return  # No validation if no schema
        
        try:
            validate(instance=document, schema=self.schema)
        except JSONSchemaValidationError as e:
            error_message = f"Document validation failed: {e.message}"
            raise ValidationError("document", error_message, {"schema_error": str(e)})
    
    async def find_paginated(
        self,
        filter_dict: Dict[str, Any] = None,
        options: PaginationOptions = None
    ) -> PaginatedResult[T]:
        """Generic find with pagination - exact equivalent of JavaScript findPaginated()."""
        if filter_dict is None:
            filter_dict = {}
        if options is None:
            options = PaginationOptions()
        
        limit = options.limit or 50
        skip = options.skip or 0
        sort = options.sort or {"createdAt": -1}
        page = (skip // limit) + 1
        
        # Convert sort dict to list of tuples for Motor
        sort_list = [(k, v) for k, v in sort.items()]
        
        # Execute find and count in parallel
        cursor = self.collection.find(filter_dict).sort(sort_list).skip(skip).limit(limit)
        documents = await cursor.to_list(length=limit)
        total = await self.collection.count_documents(filter_dict)
        
        total_pages = (total + limit - 1) // limit  # Ceiling division
        has_more = page < total_pages
        
        return PaginatedResult(
            documents=documents,
            total=total,
            hasMore=has_more,
            page=page,
            totalPages=total_pages
        )
    
    async def find_many(
        self,
        filter_dict: Dict[str, Any] = None,
        options: Dict[str, Any] = None
    ) -> List[Dict[str, Any]]:
        """Find many documents - exact equivalent of JavaScript findMany()."""
        if filter_dict is None:
            filter_dict = {}
        if options is None:
            options = {}
        
        limit = options.get('limit', 50)
        sort = options.get('sort', {"createdAt": -1})
        
        # Convert sort dict to list of tuples for Motor
        sort_list = [(k, v) for k, v in sort.items()]
        
        cursor = self.collection.find(filter_dict).sort(sort_list).limit(limit)
        return await cursor.to_list(length=limit)
    
    async def find_one(self, filter_dict: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Generic find one - exact equivalent of JavaScript findOne()."""
        return await self.collection.find_one(filter_dict)
    
    async def find_by_id(self, id_value: Union[str, ObjectId]) -> Optional[Dict[str, Any]]:
        """Generic find by ID - exact equivalent of JavaScript findById()."""
        object_id = ObjectId(id_value) if isinstance(id_value, str) else id_value
        return await self.collection.find_one({"_id": object_id})
    
    async def insert_one(self, document: Dict[str, Any]) -> Dict[str, Any]:
        """Generic insert one - exact equivalent of JavaScript insertOne()."""
        now = datetime.utcnow()
        doc_with_timestamps = {
            **document,
            "_id": ObjectId(),
            "createdAt": now,
            "updatedAt": now
        }
        
        await self.validate_document(doc_with_timestamps)
        
        result = await self.collection.insert_one(doc_with_timestamps)
        
        if not result.acknowledged:
            raise BrainError(
                f"Failed to insert document into {self.collection_name}",
                "INSERT_FAILED"
            )
        
        return doc_with_timestamps
    
    async def insert_many(self, documents: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Generic insert many - exact equivalent of JavaScript insertMany()."""
        now = datetime.utcnow()
        docs_with_timestamps = []
        
        for doc in documents:
            doc_with_timestamps = {
                **doc,
                "_id": ObjectId(),
                "createdAt": now,
                "updatedAt": now
            }
            await self.validate_document(doc_with_timestamps)
            docs_with_timestamps.append(doc_with_timestamps)
        
        result = await self.collection.insert_many(docs_with_timestamps)
        
        if not result.acknowledged:
            raise BrainError(
                f"Failed to insert documents into {self.collection_name}",
                "INSERT_MANY_FAILED"
            )
        
        return docs_with_timestamps
    
    async def update_one(
        self,
        filter_dict: Dict[str, Any],
        update: Dict[str, Any],
        options: Dict[str, Any] = None
    ) -> Optional[Dict[str, Any]]:
        """Generic update one - exact equivalent of JavaScript updateOne()."""
        if options is None:
            options = {}
        
        # Add updatedAt to $set operation
        update_doc = {**update}
        if "$set" not in update_doc:
            update_doc["$set"] = {}
        update_doc["$set"]["updatedAt"] = datetime.utcnow()
        
        result = await self.collection.find_one_and_update(
            filter_dict,
            update_doc,
            return_document=True,  # Return updated document
            upsert=options.get('upsert', False)
        )
        
        return result
    
    async def update_by_id(
        self,
        id_value: Union[str, ObjectId],
        update: Dict[str, Any],
        options: Dict[str, Any] = None
    ) -> Optional[Dict[str, Any]]:
        """Generic update by ID - exact equivalent of JavaScript updateById()."""
        object_id = ObjectId(id_value) if isinstance(id_value, str) else id_value
        return await self.update_one({"_id": object_id}, update, options)
    
    async def delete_one(self, filter_dict: Dict[str, Any]) -> bool:
        """Generic delete one - exact equivalent of JavaScript deleteOne()."""
        result = await self.collection.delete_one(filter_dict)
        return result.deleted_count > 0
    
    async def delete_by_id(self, id_value: Union[str, ObjectId]) -> bool:
        """Generic delete by ID - exact equivalent of JavaScript deleteById()."""
        object_id = ObjectId(id_value) if isinstance(id_value, str) else id_value
        return await self.delete_one({"_id": object_id})
    
    async def delete_many(self, filter_dict: Dict[str, Any]) -> int:
        """Generic delete many - exact equivalent of JavaScript deleteMany()."""
        result = await self.collection.delete_many(filter_dict)
        return result.deleted_count
    
    async def count(self, filter_dict: Dict[str, Any] = None) -> int:
        """Count documents - exact equivalent of JavaScript count()."""
        if filter_dict is None:
            filter_dict = {}
        return await self.collection.count_documents(filter_dict)
    
    async def exists(self, filter_dict: Dict[str, Any]) -> bool:
        """Check if document exists - exact equivalent of JavaScript exists()."""
        count = await self.collection.count_documents(filter_dict, limit=1)
        return count > 0

    async def distinct(self, field: str, filter_dict: Dict[str, Any] = None) -> List[Any]:
        """Get distinct values - exact equivalent of JavaScript distinct()."""
        if filter_dict is None:
            filter_dict = {}
        return await self.collection.distinct(field, filter_dict)

    async def aggregate(self, pipeline: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Aggregate pipeline - exact equivalent of JavaScript aggregate()."""
        cursor = self.collection.aggregate(pipeline)
        return await cursor.to_list(length=None)

    async def create_text_index(
        self,
        fields: Dict[str, str],
        options: Dict[str, Any] = None
    ) -> None:
        """Create text search index - exact equivalent of JavaScript createTextIndex()."""
        if options is None:
            options = {}

        index_options = {
            "name": f"{self.collection_name}_text_search",
            **options
        }

        await self.collection.create_index(
            [(field, "text") for field in fields.keys()],
            **index_options
        )

    async def text_search(
        self,
        query: str,
        options: Dict[str, Any] = None
    ) -> List[Dict[str, Any]]:
        """Text search - exact equivalent of JavaScript textSearch()."""
        if options is None:
            options = {}

        limit = options.get('limit', 20)
        skip = options.get('skip', 0)
        filter_dict = options.get('filter', {})

        search_filter = {
            "$text": {"$search": query},
            **filter_dict
        }

        cursor = self.collection.find(search_filter).sort([
            ("score", {"$meta": "textScore"})
        ]).skip(skip).limit(limit)

        return await cursor.to_list(length=limit)

    async def bulk_write(self, operations: List[Dict[str, Any]]) -> Any:
        """Bulk write operations - exact equivalent of JavaScript bulkWrite()."""
        return await self.collection.bulk_write(operations)

    async def get_stats(self) -> Dict[str, Any]:
        """Get collection statistics - exact equivalent of JavaScript getStats()."""
        try:
            stats = await self.db.command({"collStats": self.collection_name})

            return {
                "documentCount": stats.get("count", 0),
                "avgDocumentSize": stats.get("avgObjSize", 0),
                "totalSize": stats.get("size", 0),
                "indexCount": stats.get("nindexes", 0)
            }
        except Exception as e:
            logger.error(f"Failed to get stats for collection {self.collection_name}: {e}")
            return {
                "documentCount": 0,
                "avgDocumentSize": 0,
                "totalSize": 0,
                "indexCount": 0
            }

    async def create_common_indexes(self) -> None:
        """Create common indexes - exact equivalent of JavaScript createCommonIndexes()."""
        await self.collection.create_index([("createdAt", DESCENDING)])
        await self.collection.create_index([("updatedAt", DESCENDING)])

    async def initialize(self) -> None:
        """Initialize collection - exact equivalent of JavaScript initialize()."""
        try:
            await self.create_indexes()
            logger.info(f"✅ Collection {self.collection_name} initialized successfully")
        except Exception as error:
            logger.error(f"❌ Failed to initialize collection {self.collection_name}: {error}")
            raise error

    @abstractmethod
    async def create_indexes(self) -> None:
        """Abstract method for creating collection-specific indexes."""
        pass

    async def drop(self) -> None:
        """Drop collection - exact equivalent of JavaScript drop()."""
        await self.collection.drop()

    def get_collection_name(self) -> str:
        """Get collection name - exact equivalent of JavaScript getCollectionName()."""
        return self.collection_name

    def get_collection(self) -> AsyncIOMotorCollection:
        """Get MongoDB collection instance - exact equivalent of JavaScript getCollection()."""
        return self.collection
