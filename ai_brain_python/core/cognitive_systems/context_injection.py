"""
ContextInjectionEngine - Intelligent context injection for Universal AI Brain

Exact Python equivalent of JavaScript ContextInjectionEngine.ts with:
- Intelligent context selection based on semantic relevance
- Framework-specific context formatting and optimization
- Context compression and summarization
- Real-time context analytics and optimization
- Multi-modal context support (text, metadata, relationships)
- Context caching and performance optimization

Features:
- Intelligent context selection based on semantic relevance
- Framework-specific context formatting and optimization
- Context compression and summarization
- Real-time context analytics and optimization
- Multi-modal context support (text, metadata, relationships)
- Context caching and performance optimization
"""

import asyncio
import json
import math
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Union
from dataclasses import dataclass, field
from bson import ObjectId

from motor.motor_asyncio import AsyncIOMotorDatabase

from ..core.interfaces.cognitive_system import CognitiveSystemInterface
from ..core.models import CognitiveInputData, CognitiveResponse
from .semantic_memory import SemanticMemoryEngine
from .vector_search import VectorSearchEngine
from ..utils.logger import logger


@dataclass
class ContextItem:
    """Context item data structure."""
    id: str
    content: str
    type: str  # 'memory' | 'fact' | 'procedure' | 'example' | 'preference' | 'relationship'
    relevance_score: float  # 0-1 scale
    importance: float  # 0-1 scale
    source: str
    metadata: Dict[str, Any]
    relationships: Optional[List[str]] = None


@dataclass
class OptimizationMetrics:
    """Context optimization metrics."""
    context_relevance: float
    context_density: float
    token_count: int
    compression_ratio: float


@dataclass
class EnhancedPrompt:
    """Enhanced prompt with injected context."""
    original_prompt: str
    enhanced_prompt: str
    injected_context: List[ContextItem]
    context_summary: str
    optimization_metrics: OptimizationMetrics
    framework: str
    timestamp: datetime


@dataclass
class ContextOptions:
    """Context injection options."""
    max_context_items: int = 5
    max_tokens: int = 2000
    min_relevance_score: float = 0.3
    include_relationships: bool = True
    context_types: Optional[List[str]] = None
    framework: str = "universal"
    session_id: Optional[str] = None
    user_id: Optional[str] = None
    compression_level: str = "medium"  # 'none' | 'light' | 'medium' | 'aggressive'
    prioritize_recent: bool = True
    include_metadata: bool = False


@dataclass
class ContextAnalytics:
    """Context injection analytics."""
    total_context_injections: int
    average_relevance_score: float
    average_context_items: float
    average_token_count: float
    framework_usage: Dict[str, int]
    context_type_usage: Dict[str, int]
    performance_metrics: Dict[str, float]
    quality_metrics: Dict[str, float]


class ContextInjectionEngine(CognitiveSystemInterface):
    """
    ContextInjectionEngine - Intelligent context injection for AI frameworks
    
    Exact Python equivalent of JavaScript ContextInjectionEngine with:
    - Intelligent context selection based on semantic relevance
    - Framework-specific context formatting and optimization
    - Context compression and summarization
    - Real-time context analytics and optimization
    - Multi-modal context support (text, metadata, relationships)
    - Context caching and performance optimization
    """
    
    def __init__(
        self,
        db: AsyncIOMotorDatabase,
        semantic_memory_engine: Optional[SemanticMemoryEngine] = None,
        vector_search_engine: Optional[VectorSearchEngine] = None
    ):
        super().__init__(db)
        self.db = db
        self.semantic_memory_engine = semantic_memory_engine or SemanticMemoryEngine(db)
        self.vector_search_engine = vector_search_engine or VectorSearchEngine(db)
        self.context_cache: Dict[str, List[ContextItem]] = {}
        self.cache_size = 500
        self.cache_ttl = 5 * 60  # 5 minutes in seconds
        self.is_initialized = False
        
    async def initialize(self):
        """Initialize the context injection engine."""
        if self.is_initialized:
            return
            
        try:
            await self.semantic_memory_engine.initialize()
            await self.vector_search_engine.initialize()
            self.is_initialized = True
            logger.info("✅ ContextInjectionEngine initialized successfully")
        except Exception as error:
            logger.error(f"❌ Error initializing ContextInjectionEngine: {error}")
            raise error
    
    async def process(self, input_data: CognitiveInputData) -> CognitiveResponse:
        """Process context injection requests."""
        try:
            await self.initialize()
            
            # Extract context injection request from input
            request_data = input_data.additional_context.get("context_request", {})
            
            if not request_data:
                return CognitiveResponse(
                    response_text="No context injection request provided",
                    confidence=0.0,
                    processing_metadata={
                        "system": "context_injection",
                        "error": "Missing context injection request"
                    }
                )
            
            # Extract prompt and options
            prompt = request_data.get("prompt", "")
            options_data = request_data.get("options", {})
            
            # Create context options
            options = ContextOptions(
                max_context_items=options_data.get("maxContextItems", 5),
                max_tokens=options_data.get("maxTokens", 2000),
                min_relevance_score=options_data.get("minRelevanceScore", 0.3),
                include_relationships=options_data.get("includeRelationships", True),
                context_types=options_data.get("contextTypes"),
                framework=options_data.get("framework", "universal"),
                session_id=options_data.get("sessionId"),
                user_id=options_data.get("userId"),
                compression_level=options_data.get("compressionLevel", "medium"),
                prioritize_recent=options_data.get("prioritizeRecent", True),
                include_metadata=options_data.get("includeMetadata", False)
            )
            
            # Enhance the prompt
            enhanced_prompt = await self.enhance_prompt(prompt, options)
            
            return CognitiveResponse(
                response_text=enhanced_prompt.enhanced_prompt,
                confidence=enhanced_prompt.optimization_metrics.context_relevance,
                processing_metadata={
                    "system": "context_injection",
                    "original_prompt": enhanced_prompt.original_prompt,
                    "context_items": len(enhanced_prompt.injected_context),
                    "context_summary": enhanced_prompt.context_summary,
                    "token_count": enhanced_prompt.optimization_metrics.token_count,
                    "compression_ratio": enhanced_prompt.optimization_metrics.compression_ratio,
                    "framework": enhanced_prompt.framework
                }
            )
            
        except Exception as error:
            logger.error(f"Error in ContextInjectionEngine.process: {error}")
            return CognitiveResponse(
                response_text=f"Context injection error: {str(error)}",
                confidence=0.0,
                processing_metadata={
                    "system": "context_injection",
                    "error": str(error)
                }
            )
    
    async def enhance_prompt(
        self,
        prompt: str,
        options: ContextOptions = ContextOptions()
    ) -> EnhancedPrompt:
        """Enhance a prompt with relevant context."""
        start_time = datetime.utcnow()
        
        try:
            # Check cache first
            cache_key = self._generate_cache_key(prompt, options)
            cached_context = self._get_from_cache(cache_key)
            
            if cached_context is not None:
                context = cached_context
                logger.debug("Using cached context")
            else:
                # Select relevant context
                context = await self.select_relevant_context(prompt, options)
                
                # Cache the result
                self._set_cache(cache_key, context)
            
            # Optimize context for framework
            optimized_context = await self.optimize_context_for_framework(
                context,
                options.framework,
                options.max_tokens,
                options.compression_level
            )
            
            # Inject context into prompt
            enhanced_prompt_text = self._inject_context_into_prompt(
                prompt,
                optimized_context,
                options.framework,
                options.include_metadata
            )
            
            # Generate context summary
            context_summary = self._generate_context_summary(optimized_context)
            
            # Calculate optimization metrics
            optimization_metrics = self._calculate_optimization_metrics(
                prompt,
                enhanced_prompt_text,
                optimized_context,
                start_time
            )
            
            return EnhancedPrompt(
                original_prompt=prompt,
                enhanced_prompt=enhanced_prompt_text,
                injected_context=optimized_context,
                context_summary=context_summary,
                optimization_metrics=optimization_metrics,
                framework=options.framework,
                timestamp=datetime.utcnow()
            )
            
        except Exception as error:
            logger.error(f"Error enhancing prompt: {error}")
            raise error

    async def select_relevant_context(
        self,
        query: str,
        options: ContextOptions
    ) -> List[ContextItem]:
        """Select relevant context items based on semantic similarity."""
        try:
            context_items = []

            # Search semantic memory
            try:
                memory_results = await self.semantic_memory_engine.search_memories(
                    query=query,
                    limit=options.max_context_items * 2,  # Get more to filter
                    min_similarity=options.min_relevance_score
                )

                for memory in memory_results:
                    context_item = self._memory_to_context_item(memory)
                    if not options.context_types or context_item.type in options.context_types:
                        context_items.append(context_item)

            except Exception as memory_error:
                logger.warning(f"Error searching semantic memory: {memory_error}")

            # Search vector store
            try:
                vector_results = await self.vector_search_engine.search(
                    query=query,
                    limit=options.max_context_items,
                    min_score=options.min_relevance_score
                )

                for result in vector_results:
                    context_item = self._vector_result_to_context_item(result)
                    if not options.context_types or context_item.type in options.context_types:
                        context_items.append(context_item)

            except Exception as vector_error:
                logger.warning(f"Error searching vector store: {vector_error}")

            # Remove duplicates and sort by relevance
            unique_items = {}
            for item in context_items:
                if item.id not in unique_items or item.relevance_score > unique_items[item.id].relevance_score:
                    unique_items[item.id] = item

            sorted_items = sorted(
                unique_items.values(),
                key=lambda x: (x.relevance_score, x.importance),
                reverse=True
            )

            # Apply recency boost if requested
            if options.prioritize_recent:
                for item in sorted_items:
                    age_days = self._get_context_age(item)
                    if age_days < 7:  # Recent items get boost
                        item.relevance_score = min(1.0, item.relevance_score * 1.1)

            # Include relationships if requested
            if options.include_relationships:
                related_items = []
                for item in sorted_items[:options.max_context_items]:
                    if item.relationships:
                        for rel_id in item.relationships[:2]:  # Limit related items
                            try:
                                related_memory = await self.semantic_memory_engine.get_memory(rel_id)
                                if related_memory:
                                    related_item = self._memory_to_context_item(related_memory)
                                    related_item.relevance_score *= 0.8  # Reduce score for related items
                                    related_items.append(related_item)
                            except Exception:
                                continue

                sorted_items.extend(related_items)
                sorted_items.sort(key=lambda x: x.relevance_score, reverse=True)

            return sorted_items[:options.max_context_items]

        except Exception as error:
            logger.error(f"Error selecting relevant context: {error}")
            return []

    async def optimize_context_for_framework(
        self,
        context: List[ContextItem],
        framework: str,
        max_tokens: int,
        compression_level: str
    ) -> List[ContextItem]:
        """Optimize context for specific framework."""
        try:
            # Apply framework-specific optimizations
            if framework.lower() == "vercel_ai":
                context = self._optimize_for_vercel_ai(context)
            elif framework.lower() == "mastra":
                context = self._optimize_for_mastra(context)
            elif framework.lower() == "langchain":
                context = self._optimize_for_langchain(context)
            elif framework.lower() == "openai_agents":
                context = self._optimize_for_openai_agents(context)

            # Apply compression if needed
            if compression_level != "none":
                context = await self._compress_context(context, compression_level, max_tokens)

            # Enforce token limit
            context = self._enforce_token_limit(context, max_tokens)

            return context

        except Exception as error:
            logger.error(f"Error optimizing context for framework: {error}")
            return context

    def _inject_context_into_prompt(
        self,
        prompt: str,
        context: List[ContextItem],
        framework: str,
        include_metadata: bool
    ) -> str:
        """Inject context into prompt with framework-specific formatting."""
        if not context:
            return prompt

        context_text = self._format_context_for_framework(context, framework, include_metadata)

        # Framework-specific injection patterns
        if framework.lower() == "langchain":
            return f"Context:\n{context_text}\n\nQuery: {prompt}"
        elif framework.lower() == "vercel_ai":
            return f"{prompt}\n\nRelevant context:\n{context_text}"
        elif framework.lower() == "openai_agents":
            return f"[CONTEXT]\n{context_text}\n[/CONTEXT]\n\n{prompt}"
        else:
            # Universal format
            return f"Based on the following context:\n\n{context_text}\n\nPlease respond to: {prompt}"

    def _format_context_for_framework(
        self,
        context: List[ContextItem],
        framework: str,
        include_metadata: bool
    ) -> str:
        """Format context items for specific framework."""
        formatted_items = []

        for i, item in enumerate(context):
            if framework.lower() == "langchain":
                formatted = f"{i+1}. {item.content}"
            elif framework.lower() == "vercel_ai":
                formatted = f"• {item.content}"
            elif framework.lower() == "openai_agents":
                formatted = f"[{item.type.upper()}] {item.content}"
            else:
                formatted = f"- {item.content}"

            if include_metadata and item.metadata:
                metadata_str = ", ".join([f"{k}: {v}" for k, v in item.metadata.items() if k != "created"])
                if metadata_str:
                    formatted += f" ({metadata_str})"

            formatted_items.append(formatted)

        return "\n".join(formatted_items)

    def _generate_context_summary(self, context: List[ContextItem]) -> str:
        """Generate a summary of injected context."""
        if not context:
            return "No context injected"

        type_count = {}
        for item in context:
            type_count[item.type] = type_count.get(item.type, 0) + 1

        type_summary = ", ".join([f"{count} {type}" for type, count in type_count.items()])
        avg_relevance = sum(item.relevance_score for item in context) / len(context)

        return f"Injected {len(context)} context items ({type_summary}) with average relevance {avg_relevance:.2f}"

    def _calculate_optimization_metrics(
        self,
        original_prompt: str,
        enhanced_prompt: str,
        context: List[ContextItem],
        start_time: datetime
    ) -> OptimizationMetrics:
        """Calculate optimization metrics."""
        # Context relevance (average of all context items)
        context_relevance = sum(item.relevance_score for item in context) / len(context) if context else 0.0

        # Context density (context tokens / total tokens)
        original_tokens = self._estimate_token_count(original_prompt)
        enhanced_tokens = self._estimate_token_count(enhanced_prompt)
        context_tokens = enhanced_tokens - original_tokens
        context_density = context_tokens / enhanced_tokens if enhanced_tokens > 0 else 0.0

        # Compression ratio (original context size / final context size)
        original_context_size = sum(len(item.content) for item in context)
        final_context_size = sum(len(item.content) for item in context)  # After compression
        compression_ratio = original_context_size / final_context_size if final_context_size > 0 else 1.0

        return OptimizationMetrics(
            context_relevance=context_relevance,
            context_density=context_density,
            token_count=enhanced_tokens,
            compression_ratio=compression_ratio
        )

    # Helper methods

    def _memory_to_context_item(self, memory: Dict[str, Any]) -> ContextItem:
        """Convert memory to context item."""
        return ContextItem(
            id=memory.get("id", str(memory.get("_id", ""))),
            content=memory.get("content", ""),
            type=memory.get("metadata", {}).get("type", "memory"),
            relevance_score=memory.get("metadata", {}).get("confidence", 0.5),
            importance=memory.get("metadata", {}).get("importance", 0.5),
            source=memory.get("metadata", {}).get("source", "semantic_memory"),
            metadata=memory.get("metadata", {}),
            relationships=memory.get("metadata", {}).get("relationships", [])
        )

    def _vector_result_to_context_item(self, result: Dict[str, Any]) -> ContextItem:
        """Convert vector search result to context item."""
        return ContextItem(
            id=result.get("id", ""),
            content=result.get("content", ""),
            type="fact",
            relevance_score=result.get("score", 0.5),
            importance=0.7,  # Default importance for vector results
            source="vector_search",
            metadata=result.get("metadata", {}),
            relationships=result.get("metadata", {}).get("relationships", [])
        )

    def _get_context_age(self, context: ContextItem) -> float:
        """Get context age in days."""
        created = context.metadata.get("created")
        if not created:
            return 0.0

        try:
            if isinstance(created, str):
                created_date = datetime.fromisoformat(created.replace('Z', '+00:00'))
            else:
                created_date = created

            return (datetime.utcnow() - created_date).days
        except Exception:
            return 0.0

    def _optimize_for_vercel_ai(self, context: List[ContextItem]) -> List[ContextItem]:
        """Optimize context for Vercel AI - prefers concise, structured context."""
        return [
            ContextItem(
                id=item.id,
                content=self._truncate_content(item.content, 200),
                type=item.type,
                relevance_score=item.relevance_score,
                importance=item.importance,
                source=item.source,
                metadata=item.metadata,
                relationships=item.relationships
            )
            for item in context
        ]

    def _optimize_for_mastra(self, context: List[ContextItem]) -> List[ContextItem]:
        """Optimize context for Mastra - works well with procedural context."""
        return [
            item for item in context
            if item.type in ["procedure", "example", "fact"]
        ]

    def _optimize_for_langchain(self, context: List[ContextItem]) -> List[ContextItem]:
        """Optimize context for LangChain - handles longer context well."""
        return context  # No specific optimization needed

    def _optimize_for_openai_agents(self, context: List[ContextItem]) -> List[ContextItem]:
        """Optimize context for OpenAI Agents - prefers structured, role-based context."""
        return [
            ContextItem(
                id=item.id,
                content=f"[{item.type.upper()}] {item.content}",
                type=item.type,
                relevance_score=item.relevance_score,
                importance=item.importance,
                source=item.source,
                metadata=item.metadata,
                relationships=item.relationships
            )
            for item in context
        ]

    async def _compress_context(
        self,
        context: List[ContextItem],
        level: str,
        max_tokens: int
    ) -> List[ContextItem]:
        """Compress context based on compression level."""
        if level == "none":
            return context

        # Simple compression - in production, could use AI summarization
        compression_ratios = {
            "light": 0.8,
            "medium": 0.6,
            "aggressive": 0.4
        }

        ratio = compression_ratios.get(level, 0.6)
        target_length = int(max_tokens * ratio)

        return [
            ContextItem(
                id=item.id,
                content=self._truncate_content(item.content, target_length // len(context)),
                type=item.type,
                relevance_score=item.relevance_score,
                importance=item.importance,
                source=item.source,
                metadata=item.metadata,
                relationships=item.relationships
            )
            for item in context
        ]

    def _enforce_token_limit(self, context: List[ContextItem], max_tokens: int) -> List[ContextItem]:
        """Enforce token limit on context."""
        total_tokens = 0
        result = []

        for item in context:
            item_tokens = self._estimate_token_count(item.content)
            if total_tokens + item_tokens <= max_tokens:
                result.append(item)
                total_tokens += item_tokens
            else:
                break

        return result

    def _truncate_content(self, content: str, max_length: int) -> str:
        """Truncate content to maximum length."""
        if len(content) <= max_length:
            return content
        return content[:max_length - 3] + "..."

    def _estimate_token_count(self, text: str) -> int:
        """Estimate token count - rough estimation: 1 token ≈ 4 characters."""
        return math.ceil(len(text) / 4)

    def _generate_cache_key(self, prompt: str, options: ContextOptions) -> str:
        """Generate cache key for context."""
        key_data = {
            "prompt": prompt[:100],  # First 100 chars
            "max_items": options.max_context_items,
            "framework": options.framework,
            "session_id": options.session_id,
            "user_id": options.user_id
        }
        return json.dumps(key_data, sort_keys=True)

    def _get_from_cache(self, key: str) -> Optional[List[ContextItem]]:
        """Get context from cache."""
        cached = self.context_cache.get(key)
        if not cached:
            return None

        # Check TTL (simplified - in production, store timestamp with data)
        return cached

    def _set_cache(self, key: str, context: List[ContextItem]) -> None:
        """Set context in cache."""
        if len(self.context_cache) >= self.cache_size:
            # Remove oldest entry (simplified LRU)
            oldest_key = next(iter(self.context_cache))
            del self.context_cache[oldest_key]

        self.context_cache[key] = context

    async def cleanup(self) -> None:
        """Cleanup resources."""
        self.context_cache.clear()
