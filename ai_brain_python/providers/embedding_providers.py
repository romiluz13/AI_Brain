"""
Embedding Providers - Production-ready embedding implementations

Exact Python equivalents of JavaScript embedding providers with:
- VoyageAI embedding provider with state-of-the-art embeddings
- OpenAI embedding provider with Azure OpenAI support
- Automatic retry with exponential backoff
- Batch processing for efficiency
- Rate limiting and error handling
- Token counting and cost tracking

Features:
- Same API interfaces as JavaScript versions
- Identical error handling and retry logic
- Matching configuration options
- Production-ready implementations
"""

import os
import asyncio
import aiohttp
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Union
from datetime import datetime

from ..utils.logger import logger


class HybridSearchEmbeddingProvider(ABC):
    """Abstract base class for embedding providers."""
    
    @abstractmethod
    async def generate_embedding(self, text: str) -> List[float]:
        """Generate embedding for a single text."""
        pass
    
    @abstractmethod
    def get_dimensions(self) -> int:
        """Get embedding dimensions."""
        pass
    
    @abstractmethod
    def get_model(self) -> str:
        """Get model name."""
        pass


@dataclass
class VoyageAIConfig:
    """VoyageAI configuration interface."""
    api_key: str
    model: str
    base_url: str = "https://api.voyageai.com/v1"
    max_retries: int = 3
    timeout: int = 30000
    batch_size: int = 128
    input_type: Optional[str] = None  # 'query' | 'document' | None
    output_dimension: Optional[int] = None
    output_dtype: str = "float"  # 'float' | 'int8' | 'uint8' | 'binary' | 'ubinary'


@dataclass
class VoyageEmbeddingResponse:
    """Voyage AI embedding response interface."""
    object: str
    data: List[Dict[str, Any]]
    model: str
    usage: Dict[str, int]


class VoyageAIEmbeddingProvider(HybridSearchEmbeddingProvider):
    """
    VoyageAIEmbeddingProvider - Production-ready Voyage AI embedding implementation
    
    Exact Python equivalent of JavaScript VoyageAIEmbeddingProvider with:
    - Support for all Voyage AI models (voyage-3.5, voyage-3-large, voyage-code-3, etc.)
    - Automatic retry with exponential backoff
    - Batch processing for efficiency
    - Rate limiting and error handling
    - Token counting and cost tracking
    - Flexible dimensions and quantization support
    """
    
    def __init__(self, config: VoyageAIConfig):
        self.config = config
        self.request_count = 0
        self.total_tokens = 0
        self._validate_config()
    
    async def generate_embedding(self, text: str) -> List[float]:
        """Generate embedding for a single text."""
        if not text or not text.strip():
            raise ValueError("Text cannot be empty")
        
        try:
            embeddings = await self.generate_embeddings([text])
            return embeddings[0]
        except Exception as error:
            logger.error(f"Error generating embedding: {error}")
            raise Exception(f"Failed to generate embedding: {str(error)}")
    
    async def generate_embeddings(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings for multiple texts with batch processing."""
        if not texts:
            return []
        
        # Validate all texts
        for i, text in enumerate(texts):
            if not text or not text.strip():
                raise ValueError(f"Text at index {i} cannot be empty")
        
        try:
            # Process in batches
            all_embeddings = []
            for i in range(0, len(texts), self.config.batch_size):
                batch = texts[i:i + self.config.batch_size]
                batch_embeddings = await self._call_voyage_api(batch)
                all_embeddings.extend(batch_embeddings)
            
            return all_embeddings
        except Exception as error:
            logger.error(f"Error generating embeddings: {error}")
            raise Exception(f"Failed to generate embeddings: {str(error)}")
    
    def get_dimensions(self) -> int:
        """Get embedding dimensions based on model."""
        model_dimensions = {
            "voyage-3.5": 1024,
            "voyage-3.5-lite": 512,
            "voyage-3-large": 1024,
            "voyage-code-3": 1024,
            "voyage-finance-2": 1024,
            "voyage-multilingual-2": 1024,
            "voyage-law-2": 1024,
            "voyage-large-2": 1536,
            "voyage-large-2-instruct": 1024
        }
        
        return self.config.output_dimension or model_dimensions.get(self.config.model, 1024)
    
    def get_model(self) -> str:
        """Get model name."""
        return self.config.model
    
    def get_usage_stats(self) -> Dict[str, Any]:
        """Get usage statistics."""
        return {
            "request_count": self.request_count,
            "total_tokens": self.total_tokens,
            "model": self.config.model
        }
    
    async def _call_voyage_api(self, texts: List[str]) -> List[List[float]]:
        """Call Voyage AI API with retry logic."""
        last_error = Exception("Unknown error occurred")
        
        for attempt in range(self.config.max_retries + 1):
            try:
                response = await self._make_api_request(texts)
                
                self.request_count += 1
                self.total_tokens += response.usage["total_tokens"]
                
                return [item["embedding"] for item in response.data]
                
            except Exception as error:
                last_error = error
                
                if attempt == self.config.max_retries:
                    break
                
                # Check if error is retryable
                if not self._is_retryable_error(error):
                    raise error
                
                # Exponential backoff
                delay = min(1.0 * (2 ** attempt), 10.0)
                await asyncio.sleep(delay)
        
        raise last_error
    
    async def _make_api_request(self, texts: List[str]) -> VoyageEmbeddingResponse:
        """Make the actual API request to Voyage AI."""
        request_body = {
            "input": texts[0] if len(texts) == 1 else texts,
            "model": self.config.model
        }
        
        # Add optional parameters
        if self.config.input_type:
            request_body["input_type"] = self.config.input_type
        if self.config.output_dimension:
            request_body["output_dimension"] = self.config.output_dimension
        if self.config.output_dtype != "float":
            request_body["output_dtype"] = self.config.output_dtype
        
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.config.api_key}"
        }
        
        timeout = aiohttp.ClientTimeout(total=self.config.timeout / 1000)
        
        async with aiohttp.ClientSession(timeout=timeout) as session:
            try:
                async with session.post(
                    f"{self.config.base_url}/embeddings",
                    json=request_body,
                    headers=headers
                ) as response:
                    
                    if not response.ok:
                        error_data = {}
                        try:
                            error_data = await response.json()
                        except:
                            pass
                        
                        raise Exception(f"Voyage AI API error: {response.status} - {error_data.get('message', response.reason)}")
                    
                    data = await response.json()
                    return VoyageEmbeddingResponse(
                        object=data["object"],
                        data=data["data"],
                        model=data["model"],
                        usage=data["usage"]
                    )
                    
            except asyncio.TimeoutError:
                raise Exception(f"Voyage AI API request timeout after {self.config.timeout}ms")
    
    def _is_retryable_error(self, error: Exception) -> bool:
        """Check if error is retryable."""
        error_str = str(error).lower()
        
        # Rate limiting
        if "rate limit" in error_str or "429" in error_str:
            return True
        
        # Server errors
        if any(code in error_str for code in ["500", "502", "503", "504"]):
            return True
        
        # Network errors
        if any(err in error_str for err in ["timeout", "connection", "network"]):
            return True
        
        return False
    
    def _validate_config(self) -> None:
        """Validate configuration."""
        if not self.config.api_key:
            raise ValueError("Voyage AI API key is required")
        
        if not self.config.model:
            raise ValueError("Model name is required")
        
        # Validate model (warn if unknown)
        valid_models = [
            "voyage-3.5", "voyage-3.5-lite", "voyage-3-large", "voyage-code-3",
            "voyage-finance-2", "voyage-multilingual-2", "voyage-law-2",
            "voyage-large-2", "voyage-large-2-instruct"
        ]
        
        if self.config.model not in valid_models:
            logger.warning(f"Unknown Voyage AI model: {self.config.model}. Proceeding anyway...")
    
    @classmethod
    def for_general_purpose(cls, api_key: str) -> "VoyageAIEmbeddingProvider":
        """Create a Voyage AI provider with recommended settings for general purpose."""
        return cls(VoyageAIConfig(
            api_key=api_key,
            model="voyage-3.5",  # Best general-purpose model
            input_type="document"
        ))
    
    @classmethod
    def for_code(cls, api_key: str) -> "VoyageAIEmbeddingProvider":
        """Create a Voyage AI provider optimized for code."""
        return cls(VoyageAIConfig(
            api_key=api_key,
            model="voyage-code-3",  # Optimized for code
            input_type="document"
        ))
    
    @classmethod
    def for_query(cls, api_key: str) -> "VoyageAIEmbeddingProvider":
        """Create a Voyage AI provider optimized for queries."""
        return cls(VoyageAIConfig(
            api_key=api_key,
            model="voyage-3.5",
            input_type="query"
        ))
    
    @classmethod
    def for_high_performance(cls, api_key: str) -> "VoyageAIEmbeddingProvider":
        """Create a Voyage AI provider with best quality."""
        return cls(VoyageAIConfig(
            api_key=api_key,
            model="voyage-3-large",  # Best quality
            input_type="document"
        ))
    
    @classmethod
    def for_low_latency(cls, api_key: str) -> "VoyageAIEmbeddingProvider":
        """Create a Voyage AI provider optimized for speed."""
        return cls(VoyageAIConfig(
            api_key=api_key,
            model="voyage-3.5-lite",  # Optimized for speed
            input_type="document"
        ))


@dataclass
class OpenAIConfig:
    """OpenAI configuration interface."""
    api_key: str
    model: str
    base_url: str = "https://api.openai.com/v1"
    max_retries: int = 3
    timeout: int = 30000
    batch_size: int = 100


@dataclass
class EmbeddingResponse:
    """OpenAI embedding response interface."""
    object: str
    data: List[Dict[str, Any]]
    model: str
    usage: Dict[str, int]


class OpenAIEmbeddingProvider(HybridSearchEmbeddingProvider):
    """
    OpenAIEmbeddingProvider - Production-ready OpenAI embedding implementation

    Exact Python equivalent of JavaScript OpenAIEmbeddingProvider with:
    - Support for OpenAI and Azure OpenAI
    - Automatic retry with exponential backoff
    - Batch processing for efficiency
    - Rate limiting and error handling
    - Token counting and cost tracking
    """

    def __init__(self, config: OpenAIConfig):
        self.config = config
        self.request_count = 0
        self.total_tokens = 0
        self._validate_config()

    async def generate_embedding(self, text: str) -> List[float]:
        """Generate embedding for a single text."""
        if not text or not text.strip():
            raise ValueError("Text cannot be empty")

        try:
            embeddings = await self.generate_embeddings([text])
            return embeddings[0]
        except Exception as error:
            logger.error(f"Error generating embedding: {error}")
            raise Exception(f"Failed to generate embedding: {str(error)}")

    async def generate_embeddings(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings for multiple texts with batch processing."""
        if not texts:
            return []

        # Validate all texts
        for i, text in enumerate(texts):
            if not text or not text.strip():
                raise ValueError(f"Text at index {i} cannot be empty")

        try:
            return await self._process_batches(texts)
        except Exception as error:
            logger.error(f"Error generating embeddings: {error}")
            raise Exception(f"Failed to generate embeddings: {str(error)}")

    def get_dimensions(self) -> int:
        """Get embedding dimensions based on model."""
        model_dimensions = {
            "text-embedding-3-small": 1536,
            "text-embedding-3-large": 3072,
            "text-embedding-ada-002": 1536,
            "text-embedding-002": 1536
        }

        return model_dimensions.get(self.config.model, 1536)

    def get_model(self) -> str:
        """Get model name."""
        return self.config.model

    def get_usage_stats(self) -> Dict[str, Any]:
        """Get usage statistics."""
        return {
            "request_count": self.request_count,
            "total_tokens": self.total_tokens,
            "model": self.config.model
        }

    async def _process_batches(self, texts: List[str]) -> List[List[float]]:
        """Process texts in batches."""
        results = []

        for i in range(0, len(texts), self.config.batch_size):
            batch = texts[i:i + self.config.batch_size]
            batch_results = await self._call_embedding_api(batch)
            results.extend(batch_results)

        return results

    async def _call_embedding_api(self, texts: List[str]) -> List[List[float]]:
        """Call OpenAI embedding API with retry logic."""
        url = f"{self.config.base_url}/embeddings"

        request_body = {
            "input": texts,
            "model": self.config.model,
            "encoding_format": "float"
        }

        last_error = None

        for attempt in range(self.config.max_retries + 1):
            try:
                response = await self._make_request(url, request_body)

                if not response.ok:
                    error_data = {}
                    try:
                        error_data = await response.json()
                    except:
                        pass

                    raise Exception(f"API request failed: {response.status} {response.reason} - {error_data}")

                data = await response.json()

                # Update usage statistics
                self.request_count += 1
                self.total_tokens += data["usage"]["total_tokens"]

                # Extract embeddings in the correct order
                embeddings = sorted(data["data"], key=lambda x: x["index"])
                return [item["embedding"] for item in embeddings]

            except Exception as error:
                last_error = error

                if attempt == self.config.max_retries:
                    break

                # Check if error is retryable
                if not self._is_retryable_error(error):
                    raise error

                # Exponential backoff
                delay = min(1.0 * (2 ** attempt), 10.0)
                await asyncio.sleep(delay)

        raise last_error or Exception("Unknown error occurred")

    async def _make_request(self, url: str, body: Dict[str, Any]) -> aiohttp.ClientResponse:
        """Make HTTP request to OpenAI API."""
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.config.api_key}"
        }

        # Add Azure OpenAI specific headers if using Azure
        if "openai.azure.com" in self.config.base_url:
            headers["api-key"] = self.config.api_key
            del headers["Authorization"]

        timeout = aiohttp.ClientTimeout(total=self.config.timeout / 1000)

        async with aiohttp.ClientSession(timeout=timeout) as session:
            try:
                response = await session.post(url, json=body, headers=headers)
                return response
            except asyncio.TimeoutError:
                raise Exception(f"OpenAI API request timeout after {self.config.timeout}ms")

    def _is_retryable_error(self, error: Exception) -> bool:
        """Check if error is retryable."""
        error_str = str(error).lower()

        # Rate limiting
        if "rate limit" in error_str or "429" in error_str:
            return True

        # Server errors
        if any(code in error_str for code in ["500", "502", "503", "504"]):
            return True

        # Network errors
        if any(err in error_str for err in ["timeout", "connection", "network"]):
            return True

        return False

    def _validate_config(self) -> None:
        """Validate configuration."""
        if not self.config.api_key:
            raise ValueError("OpenAI API key is required")

        if not self.config.model:
            raise ValueError("Model name is required")

        # Validate model (warn if unknown)
        valid_models = [
            "text-embedding-3-small", "text-embedding-3-large",
            "text-embedding-ada-002", "text-embedding-002"
        ]

        if self.config.model not in valid_models:
            logger.warning(f"Unknown OpenAI model: {self.config.model}. Proceeding anyway...")

    @classmethod
    def for_azure(cls, config: Dict[str, str]) -> "OpenAIEmbeddingProvider":
        """Create an Azure OpenAI provider instance."""
        api_key = config["api_key"]
        endpoint = config["endpoint"]
        deployment_name = config["deployment_name"]
        api_version = config.get("api_version", "2024-02-01")

        base_url = f"{endpoint}/openai/deployments/{deployment_name}"

        return cls(OpenAIConfig(
            api_key=api_key,
            model=deployment_name,
            base_url=f"{base_url}?api-version={api_version}"
        ))
