"""
Raw Data Store
==============
Stores original dialogue data with embeddings for hybrid retrieval.

This module provides a way to preserve and index the original conversation
data, allowing the system to reference raw text during retrieval to avoid
information loss from processing.
"""

import os
import json
import uuid
import numpy as np
from typing import List, Dict, Optional, Any, Tuple
from datetime import datetime
from dataclasses import dataclass, asdict, field
from sentence_transformers import SentenceTransformer


@dataclass
class RawDialogueRecord:
    """
    A single raw dialogue record with full metadata.

    Preserves all original information from the conversation.
    """
    id: str
    session_id: str  # Which conversation session this belongs to
    turn_index: int  # Position in the conversation
    speaker: str  # Who said this
    text: str  # The original message text
    timestamp: Optional[str] = None  # Original timestamp if available
    embedding: List[float] = field(default_factory=list)

    # Additional metadata for retrieval
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'RawDialogueRecord':
        return cls(
            id=data['id'],
            session_id=data['session_id'],
            turn_index=data['turn_index'],
            speaker=data['speaker'],
            text=data['text'],
            timestamp=data.get('timestamp'),
            embedding=data.get('embedding', []),
            metadata=data.get('metadata', {})
        )

    def get_formatted_text(self) -> str:
        """Get text formatted with timestamp and speaker."""
        if self.timestamp:
            return f"[{self.timestamp}] {self.speaker}: {self.text}"
        return f"{self.speaker}: {self.text}"


class RawDataStore:
    """
    Raw Data Store for preserving original dialogue information.

    Features:
    - Stores complete original dialogue text
    - Maintains embeddings for semantic search
    - Preserves all metadata (speaker, timestamp, session, etc.)
    - Supports hybrid retrieval with processed knowledge graph

    This addresses the information loss problem where processed/summarized
    data in the knowledge graph loses important details from original text.
    """

    def __init__(
        self,
        storage_path: str,
        collection_name: str = "raw_dialogues",
        embedding_model: Optional[SentenceTransformer] = None
    ):
        """
        Initialize the raw data store.

        Args:
            storage_path: Base directory for storage
            collection_name: Name of the collection
            embedding_model: Shared embedding model (to avoid loading multiple times)
        """
        self.storage_path = storage_path
        self.collection_name = collection_name
        self.collection_path = os.path.join(storage_path, f"{collection_name}.json")

        # In-memory storage
        self._records: Dict[str, RawDialogueRecord] = {}
        self._embeddings_matrix: Optional[np.ndarray] = None
        self._id_list: List[str] = []

        # Session index for quick lookup
        self._session_index: Dict[str, List[str]] = {}  # session_id -> list of record_ids

        # Speaker index
        self._speaker_index: Dict[str, List[str]] = {}  # speaker -> list of record_ids

        # Embedding model
        if embedding_model is None:
            import sys
            sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
            import config
            device = 'cuda' if __import__('torch').cuda.is_available() else 'cpu'
            self._embedding_model = SentenceTransformer(config.EMBEDDING_MODEL_ID).to(device)
        else:
            self._embedding_model = embedding_model

        # Ensure directory exists
        os.makedirs(storage_path, exist_ok=True)

        # Load existing data
        self._load()

    def _load(self) -> None:
        """Load records from JSON file."""
        if os.path.exists(self.collection_path):
            try:
                with open(self.collection_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    for record_data in data.get('records', []):
                        record = RawDialogueRecord.from_dict(record_data)
                        self._records[record.id] = record

                        # Build indices
                        if record.session_id not in self._session_index:
                            self._session_index[record.session_id] = []
                        self._session_index[record.session_id].append(record.id)

                        if record.speaker not in self._speaker_index:
                            self._speaker_index[record.speaker] = []
                        self._speaker_index[record.speaker].append(record.id)

                self._rebuild_index()
                print(f"[RawDataStore] Loaded {len(self._records)} raw dialogue records")
            except Exception as e:
                print(f"[RawDataStore] Failed to load: {e}")
                self._records = {}

    def _save(self) -> None:
        """Save records to JSON file."""
        data = {
            'collection_name': self.collection_name,
            'record_count': len(self._records),
            'session_count': len(self._session_index),
            'speaker_count': len(self._speaker_index),
            'updated_at': datetime.now().isoformat(),
            'records': [r.to_dict() for r in self._records.values()]
        }
        with open(self.collection_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

    def _rebuild_index(self) -> None:
        """Rebuild the numpy matrix index for fast search."""
        if not self._records:
            self._embeddings_matrix = None
            self._id_list = []
            return

        self._id_list = list(self._records.keys())
        embeddings = [self._records[id].embedding for id in self._id_list]

        # Filter out records without embeddings
        valid_pairs = [(id, emb) for id, emb in zip(self._id_list, embeddings) if emb]

        if valid_pairs:
            self._id_list = [p[0] for p in valid_pairs]
            self._embeddings_matrix = np.array([p[1] for p in valid_pairs])
        else:
            self._embeddings_matrix = None
            self._id_list = []

    def add_dialogue(
        self,
        session_id: str,
        turn_index: int,
        speaker: str,
        text: str,
        timestamp: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
        record_id: Optional[str] = None
    ) -> str:
        """
        Add a single dialogue turn to the store.

        Args:
            session_id: Conversation session identifier
            turn_index: Position in the conversation
            speaker: Who said this
            text: The original message text
            timestamp: Original timestamp
            metadata: Additional metadata
            record_id: Optional custom ID

        Returns:
            Record ID
        """
        if record_id is None:
            record_id = str(uuid.uuid4())

        # Generate embedding
        embedding = self._embedding_model.encode(text).tolist()

        record = RawDialogueRecord(
            id=record_id,
            session_id=session_id,
            turn_index=turn_index,
            speaker=speaker,
            text=text,
            timestamp=timestamp,
            embedding=embedding,
            metadata=metadata or {}
        )

        self._records[record_id] = record

        # Update indices
        if session_id not in self._session_index:
            self._session_index[session_id] = []
        self._session_index[session_id].append(record_id)

        if speaker not in self._speaker_index:
            self._speaker_index[speaker] = []
        self._speaker_index[speaker].append(record_id)

        return record_id

    def add_dialogues_batch(
        self,
        dialogues: List[Dict[str, Any]],
        auto_save: bool = True
    ) -> List[str]:
        """
        Add multiple dialogues in batch.

        Args:
            dialogues: List of dialogue dicts with keys:
                       session_id, turn_index, speaker, text, timestamp, metadata
            auto_save: Whether to save after adding

        Returns:
            List of record IDs
        """
        record_ids = []

        # Batch encode all texts
        texts = [d.get('text', '') for d in dialogues]
        embeddings = self._embedding_model.encode(texts).tolist()

        for dialogue, embedding in zip(dialogues, embeddings):
            record_id = dialogue.get('id', str(uuid.uuid4()))

            record = RawDialogueRecord(
                id=record_id,
                session_id=dialogue.get('session_id', 'default'),
                turn_index=dialogue.get('turn_index', 0),
                speaker=dialogue.get('speaker', 'Unknown'),
                text=dialogue.get('text', ''),
                timestamp=dialogue.get('timestamp'),
                embedding=embedding,
                metadata=dialogue.get('metadata', {})
            )

            self._records[record_id] = record
            record_ids.append(record_id)

            # Update indices
            if record.session_id not in self._session_index:
                self._session_index[record.session_id] = []
            self._session_index[record.session_id].append(record_id)

            if record.speaker not in self._speaker_index:
                self._speaker_index[record.speaker] = []
            self._speaker_index[record.speaker].append(record_id)

        self._rebuild_index()

        if auto_save:
            self._save()

        return record_ids

    def search(
        self,
        query: str,
        top_k: int = 10,
        speaker_filter: Optional[str] = None,
        session_filter: Optional[str] = None
    ) -> List[Tuple[RawDialogueRecord, float]]:
        """
        Search for similar dialogues using semantic similarity.

        Args:
            query: Query text
            top_k: Number of results
            speaker_filter: Only return dialogues from this speaker
            session_filter: Only return dialogues from this session

        Returns:
            List of (record, similarity_score) tuples
        """
        if self._embeddings_matrix is None or len(self._id_list) == 0:
            return []

        # Get query embedding
        query_embedding = self._embedding_model.encode(query)
        query_norm = query_embedding / (np.linalg.norm(query_embedding) + 1e-10)

        # Compute similarities
        matrix_norms = self._embeddings_matrix / (
            np.linalg.norm(self._embeddings_matrix, axis=1, keepdims=True) + 1e-10
        )
        similarities = np.dot(matrix_norms, query_norm)

        # Get all results sorted by similarity
        sorted_indices = np.argsort(similarities)[::-1]

        results = []
        for idx in sorted_indices:
            if len(results) >= top_k:
                break

            record_id = self._id_list[idx]
            record = self._records[record_id]

            # Apply filters
            if speaker_filter and record.speaker != speaker_filter:
                continue
            if session_filter and record.session_id != session_filter:
                continue

            results.append((record, float(similarities[idx])))

        return results

    def search_with_context(
        self,
        query: str,
        top_k: int = 5,
        context_window: int = 2
    ) -> List[Dict[str, Any]]:
        """
        Search and return results with surrounding conversation context.

        Args:
            query: Query text
            top_k: Number of main results
            context_window: Number of turns before/after to include

        Returns:
            List of dicts with 'main' (matched record) and 'context' (surrounding turns)
        """
        main_results = self.search(query, top_k=top_k)

        enriched_results = []
        for record, score in main_results:
            # Get surrounding context from same session
            session_records = self._session_index.get(record.session_id, [])
            session_records_sorted = sorted(
                [self._records[rid] for rid in session_records],
                key=lambda r: r.turn_index
            )

            # Find position of current record
            current_idx = next(
                (i for i, r in enumerate(session_records_sorted) if r.id == record.id),
                -1
            )

            if current_idx >= 0:
                start_idx = max(0, current_idx - context_window)
                end_idx = min(len(session_records_sorted), current_idx + context_window + 1)
                context = session_records_sorted[start_idx:end_idx]
            else:
                context = [record]

            enriched_results.append({
                'main': record,
                'score': score,
                'context': context,
                'context_text': '\n'.join([r.get_formatted_text() for r in context])
            })

        return enriched_results

    def get_by_session(self, session_id: str) -> List[RawDialogueRecord]:
        """Get all dialogues from a session, sorted by turn index."""
        record_ids = self._session_index.get(session_id, [])
        records = [self._records[rid] for rid in record_ids]
        return sorted(records, key=lambda r: r.turn_index)

    def get_by_speaker(self, speaker: str) -> List[RawDialogueRecord]:
        """Get all dialogues from a speaker."""
        record_ids = self._speaker_index.get(speaker, [])
        return [self._records[rid] for rid in record_ids]

    def clear(self) -> None:
        """Clear all records."""
        self._records = {}
        self._session_index = {}
        self._speaker_index = {}
        self._embeddings_matrix = None
        self._id_list = []
        self._save()

    def count(self) -> int:
        """Return total number of records."""
        return len(self._records)

    def get_statistics(self) -> Dict[str, Any]:
        """Get storage statistics."""
        return {
            'total_records': len(self._records),
            'total_sessions': len(self._session_index),
            'total_speakers': len(self._speaker_index),
            'speakers': list(self._speaker_index.keys()),
            'avg_text_length': sum(len(r.text) for r in self._records.values()) / max(1, len(self._records))
        }

    def save(self) -> None:
        """Explicitly save to disk."""
        self._rebuild_index()
        self._save()


class HybridRetriever:
    """
    Hybrid retriever that combines knowledge graph and raw data store.

    This addresses the information loss problem by retrieving from both:
    1. Processed knowledge graph (for semantic relationships)
    2. Raw dialogue store (for original text details)
    """

    def __init__(
        self,
        knowledge_graph,  # WeightedKnowledgeGraph instance
        raw_store: RawDataStore,
        kg_weight: float = 0.5,
        raw_weight: float = 0.5
    ):
        """
        Initialize hybrid retriever.

        Args:
            knowledge_graph: The processed knowledge graph
            raw_store: Raw dialogue data store
            kg_weight: Weight for knowledge graph results
            raw_weight: Weight for raw data results
        """
        self.kg = knowledge_graph
        self.raw_store = raw_store
        self.kg_weight = kg_weight
        self.raw_weight = raw_weight

    def retrieve(
        self,
        query: str,
        top_k: int = 10,
        include_context: bool = True,
        context_window: int = 2
    ) -> Dict[str, Any]:
        """
        Hybrid retrieval from both knowledge graph and raw data.

        Args:
            query: Query text
            top_k: Number of results from each source
            include_context: Whether to include surrounding dialogue context
            context_window: Number of turns before/after for context

        Returns:
            Dict with 'kg_results', 'raw_results', and 'merged_context'
        """
        # Get results from knowledge graph
        kg_nodes, kg_scores = self.kg.retrieve(query, top_k=top_k)
        kg_contents = [node.content for node in kg_nodes]

        # Get results from raw data store
        if include_context:
            raw_results = self.raw_store.search_with_context(
                query, top_k=top_k, context_window=context_window
            )
            raw_contents = [r['context_text'] for r in raw_results]
            raw_scores = [r['score'] for r in raw_results]
        else:
            raw_results = self.raw_store.search(query, top_k=top_k)
            raw_contents = [r[0].get_formatted_text() for r in raw_results]
            raw_scores = [r[1] for r in raw_results]

        # Merge and deduplicate results
        merged_context = self._merge_results(
            kg_contents, kg_scores,
            raw_contents, raw_scores
        )

        return {
            'kg_results': list(zip(kg_contents, kg_scores)),
            'raw_results': list(zip(raw_contents, raw_scores)),
            'merged_context': merged_context,
            'kg_count': len(kg_contents),
            'raw_count': len(raw_contents)
        }

    def _merge_results(
        self,
        kg_contents: List[str],
        kg_scores: List[float],
        raw_contents: List[str],
        raw_scores: List[float]
    ) -> List[str]:
        """
        Merge results from both sources, removing duplicates.

        Uses weighted scoring and deduplication based on text similarity.
        """
        all_items = []
        seen_texts = set()

        # Add knowledge graph results
        for content, score in zip(kg_contents, kg_scores):
            normalized = content.strip().lower()[:100]  # First 100 chars for dedup
            if normalized not in seen_texts:
                seen_texts.add(normalized)
                all_items.append((content, score * self.kg_weight, 'kg'))

        # Add raw results (prioritize unique ones)
        for content, score in zip(raw_contents, raw_scores):
            # Check if this is substantially new
            normalized = content.strip().lower()[:100]
            is_new = True
            for seen in seen_texts:
                # Simple overlap check
                overlap = len(set(normalized.split()) & set(seen.split()))
                if overlap > 10:  # If too much overlap, skip
                    is_new = False
                    break

            if is_new:
                seen_texts.add(normalized)
                all_items.append((content, score * self.raw_weight, 'raw'))

        # Sort by score and return contents
        all_items.sort(key=lambda x: x[1], reverse=True)
        return [item[0] for item in all_items]

    def get_context_for_question(
        self,
        question: str,
        max_tokens: int = 2000
    ) -> str:
        """
        Get combined context for answering a question.

        Returns a single string with the most relevant context,
        limited by max_tokens.
        """
        results = self.retrieve(question, top_k=10, include_context=True)

        context_parts = []
        current_tokens = 0

        for content in results['merged_context']:
            # Rough token estimate (1 token ~ 4 chars)
            content_tokens = len(content) // 4
            if current_tokens + content_tokens > max_tokens:
                break
            context_parts.append(content)
            current_tokens += content_tokens

        return '\n\n'.join(context_parts)