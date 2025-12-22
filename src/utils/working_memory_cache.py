"""
Working Memory Cache (v1.4)
===========================
Persistent storage of original messages with round numbers for provenance tracking.
Inspired by O-Mem's wm_cache_episodic_memory pattern.

This module provides:
- CachedMessage: Data class for storing original messages with metadata
- WorkingMemoryCache: Persistent JSON-based cache for conversation history

Key Features:
- Stores BOTH user AND assistant messages
- Preserves original text before any LLM processing
- Enables linking facts back to source messages via round number
- Supports keyword search on original text for retrieval enhancement
"""

import json
import os
import re
from dataclasses import dataclass, asdict
from datetime import datetime
from typing import Optional, List, Dict, Any
from pathlib import Path


@dataclass
class CachedMessage:
    """
    A cached message with provenance information.

    Attributes:
        index: Round/turn number in conversation
        raw_message: Original unprocessed text
        speaker: 'user' or 'assistant'
        timestamp: When message occurred (ISO format)
        processed_summary: Optional LLM-generated summary (for reference)
        topics: Optional list of extracted topics
        entities: Optional list of extracted entities
    """
    index: int
    raw_message: str
    speaker: str
    timestamp: str
    processed_summary: Optional[str] = None
    topics: Optional[List[str]] = None
    entities: Optional[List[str]] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "CachedMessage":
        """Create from dictionary."""
        return cls(**data)

    def contains_keyword(self, keyword: str, case_insensitive: bool = True) -> bool:
        """Check if raw_message contains the keyword."""
        text = self.raw_message
        if case_insensitive:
            text = text.lower()
            keyword = keyword.lower()
        return keyword in text


class WorkingMemoryCache:
    """
    Persistent cache for original conversation messages.

    This cache stores all messages (both user and assistant) with their
    original text, enabling:
    - Provenance tracking: Link extracted facts back to source messages
    - Keyword search: Search original text for retrieval enhancement
    - Context recovery: Get surrounding messages for a given round

    Storage format: JSON file with list of CachedMessage dictionaries.
    """

    def __init__(self, storage_path: str = "data/wm_cache.json"):
        """
        Initialize the working memory cache.

        Args:
            storage_path: Path to the JSON file for persistence
        """
        self.storage_path = Path(storage_path)
        self.cache: List[CachedMessage] = []
        self._index_map: Dict[int, int] = {}  # index -> position in cache list
        self._load()

    def add_message(
        self,
        index: int,
        raw_message: str,
        speaker: str,
        timestamp: str,
        processed_summary: Optional[str] = None,
        topics: Optional[List[str]] = None,
        entities: Optional[List[str]] = None
    ) -> CachedMessage:
        """
        Store original message with round number.

        Args:
            index: Round/turn number in conversation
            raw_message: Original unprocessed text
            speaker: 'user' or 'assistant'
            timestamp: When message occurred
            processed_summary: Optional LLM summary
            topics: Optional extracted topics
            entities: Optional extracted entities

        Returns:
            The created CachedMessage
        """
        message = CachedMessage(
            index=index,
            raw_message=raw_message,
            speaker=speaker,
            timestamp=timestamp,
            processed_summary=processed_summary,
            topics=topics,
            entities=entities
        )

        # Update index map
        self._index_map[index] = len(self.cache)
        self.cache.append(message)

        # Auto-save after adding
        self._save()

        return message

    def get_by_index(self, index: int) -> Optional[CachedMessage]:
        """
        Retrieve original message by round number.

        Args:
            index: Round/turn number

        Returns:
            CachedMessage if found, None otherwise
        """
        if index in self._index_map:
            position = self._index_map[index]
            if 0 <= position < len(self.cache):
                return self.cache[position]

        # Fallback: linear search (for backward compatibility)
        for msg in self.cache:
            if msg.index == index:
                return msg
        return None

    def get_by_indices(self, indices: List[int]) -> List[CachedMessage]:
        """
        Batch retrieve messages by round numbers.

        Args:
            indices: List of round/turn numbers

        Returns:
            List of found CachedMessages (in order of input indices)
        """
        results = []
        for idx in indices:
            msg = self.get_by_index(idx)
            if msg:
                results.append(msg)
        return results

    def get_conversation_context(
        self,
        center_index: int,
        window: int = 2
    ) -> List[CachedMessage]:
        """
        Get surrounding messages for context.

        Args:
            center_index: The round number to center on
            window: Number of messages before and after to include

        Returns:
            List of CachedMessages around the center (sorted by index)
        """
        min_idx = center_index - window
        max_idx = center_index + window

        context = []
        for msg in self.cache:
            if min_idx <= msg.index <= max_idx:
                context.append(msg)

        return sorted(context, key=lambda m: m.index)

    def search_keyword(
        self,
        keyword: str,
        case_insensitive: bool = True,
        speaker_filter: Optional[str] = None
    ) -> List[CachedMessage]:
        """
        Search raw messages for keyword matches.

        Args:
            keyword: The keyword to search for
            case_insensitive: Whether to ignore case (default True)
            speaker_filter: Optional filter by speaker ('user' or 'assistant')

        Returns:
            List of CachedMessages containing the keyword
        """
        results = []
        for msg in self.cache:
            if speaker_filter and msg.speaker != speaker_filter:
                continue
            if msg.contains_keyword(keyword, case_insensitive):
                results.append(msg)
        return results

    def search_keywords(
        self,
        keywords: List[str],
        match_all: bool = False,
        case_insensitive: bool = True
    ) -> List[CachedMessage]:
        """
        Search for messages containing multiple keywords.

        Args:
            keywords: List of keywords to search for
            match_all: If True, message must contain ALL keywords.
                      If False, message must contain ANY keyword.
            case_insensitive: Whether to ignore case

        Returns:
            List of matching CachedMessages
        """
        results = []
        for msg in self.cache:
            if match_all:
                # Must contain all keywords
                if all(msg.contains_keyword(kw, case_insensitive) for kw in keywords):
                    results.append(msg)
            else:
                # Must contain at least one keyword
                if any(msg.contains_keyword(kw, case_insensitive) for kw in keywords):
                    results.append(msg)
        return results

    def get_all_by_speaker(self, speaker: str) -> List[CachedMessage]:
        """Get all messages from a specific speaker."""
        return [msg for msg in self.cache if msg.speaker == speaker]

    def get_all(self) -> List[CachedMessage]:
        """Get all cached messages."""
        return list(self.cache)

    def get_latest(self, n: int = 10) -> List[CachedMessage]:
        """Get the n most recent messages."""
        return self.cache[-n:] if len(self.cache) >= n else list(self.cache)

    def size(self) -> int:
        """Get the number of cached messages."""
        return len(self.cache)

    def clear(self) -> None:
        """Clear all cached messages."""
        self.cache = []
        self._index_map = {}
        self._save()

    def _save(self) -> None:
        """Persist cache to JSON file."""
        # Ensure directory exists
        self.storage_path.parent.mkdir(parents=True, exist_ok=True)

        data = [msg.to_dict() for msg in self.cache]
        with open(self.storage_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

    def _load(self) -> None:
        """Load cache from JSON file."""
        if self.storage_path.exists():
            try:
                with open(self.storage_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                self.cache = [CachedMessage.from_dict(d) for d in data]
                # Rebuild index map
                self._index_map = {msg.index: i for i, msg in enumerate(self.cache)}
            except (json.JSONDecodeError, KeyError) as e:
                print(f"Warning: Failed to load WM cache: {e}. Starting with empty cache.")
                self.cache = []
                self._index_map = {}
        else:
            self.cache = []
            self._index_map = {}

    def export_for_retrieval(self) -> Dict[int, str]:
        """
        Export cache as index -> raw_message mapping.
        Useful for keyword index integration.
        """
        return {msg.index: msg.raw_message for msg in self.cache}

    def get_statistics(self) -> Dict[str, Any]:
        """Get statistics about the cache."""
        user_count = sum(1 for m in self.cache if m.speaker == "user")
        assistant_count = sum(1 for m in self.cache if m.speaker == "assistant")

        return {
            "total_messages": len(self.cache),
            "user_messages": user_count,
            "assistant_messages": assistant_count,
            "index_range": (
                min(m.index for m in self.cache) if self.cache else None,
                max(m.index for m in self.cache) if self.cache else None
            ),
            "storage_path": str(self.storage_path)
        }


# Factory function for creating cache instances
def create_working_memory_cache(
    storage_path: str = "data/wm_cache.json"
) -> WorkingMemoryCache:
    """
    Create a WorkingMemoryCache instance.

    Args:
        storage_path: Path to the JSON file for persistence

    Returns:
        Configured WorkingMemoryCache instance
    """
    return WorkingMemoryCache(storage_path=storage_path)


# Singleton instance (optional, for convenience)
_default_cache: Optional[WorkingMemoryCache] = None


def get_working_memory_cache(storage_path: str = "data/wm_cache.json") -> WorkingMemoryCache:
    """
    Get or create the default WorkingMemoryCache instance.

    Args:
        storage_path: Path to the JSON file for persistence

    Returns:
        WorkingMemoryCache singleton instance
    """
    global _default_cache
    if _default_cache is None:
        _default_cache = WorkingMemoryCache(storage_path=storage_path)
    return _default_cache
