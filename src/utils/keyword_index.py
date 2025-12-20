"""
Keyword Index with BM25 Scoring
===============================
Inverted index for keyword-based retrieval with BM25 ranking.

This module addresses the problem of embedding-only retrieval missing
exact keyword matches, especially for rare terms like proper nouns
(e.g., "Sweden", "violin").

Key Features:
- Inverted index for O(1) keyword lookup
- BM25 scoring for relevance ranking
- TF-IDF fallback for simpler use cases
- Integration with existing MemoryNode structure
"""

import re
import math
import json
import os
from collections import defaultdict
from typing import List, Dict, Set, Tuple, Optional, Any
from dataclasses import dataclass, field

import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))


# =============================================================================
# Constants
# =============================================================================

# English stopwords for filtering
STOPWORDS: Set[str] = {
    "a", "an", "and", "are", "as", "at", "be", "been", "being", "but", "by",
    "can", "could", "did", "do", "does", "doing", "done", "for", "from",
    "had", "has", "have", "having", "he", "her", "here", "hers", "him",
    "his", "how", "i", "if", "in", "into", "is", "it", "its", "just",
    "let", "may", "me", "might", "must", "my", "no", "nor", "not", "of",
    "off", "on", "or", "our", "ours", "out", "own", "say", "says", "she",
    "should", "so", "some", "such", "than", "that", "the", "their", "them",
    "then", "there", "these", "they", "this", "those", "through", "to",
    "too", "under", "until", "up", "us", "very", "was", "we", "were",
    "what", "when", "where", "which", "while", "who", "whom", "why",
    "will", "with", "would", "you", "your", "yours", "yourself",
    # Common conversation fillers
    "oh", "well", "yeah", "yes", "ok", "okay", "hey", "hi", "hello",
    "thanks", "thank", "please", "really", "actually", "basically",
    "like", "just", "gonna", "gotta", "wanna",
}


# =============================================================================
# Data Structures
# =============================================================================

@dataclass
class IndexedDocument:
    """A document in the keyword index."""
    doc_id: str
    content: str
    tokens: List[str] = field(default_factory=list)
    token_count: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class SearchResult:
    """A search result with score."""
    doc_id: str
    score: float
    matched_terms: List[str] = field(default_factory=list)
    content: str = ""


# =============================================================================
# Tokenizer
# =============================================================================

class Tokenizer:
    """Simple tokenizer with normalization and stopword removal."""

    def __init__(self, remove_stopwords: bool = True, min_token_length: int = 2):
        self.remove_stopwords = remove_stopwords
        self.min_token_length = min_token_length

    def tokenize(self, text: str) -> List[str]:
        """
        Tokenize text into normalized tokens.

        Args:
            text: Input text

        Returns:
            List of normalized tokens
        """
        # Lowercase and extract word tokens
        text = text.lower()
        tokens = re.findall(r'\b[a-z0-9]+\b', text)

        # Filter tokens
        filtered = []
        for token in tokens:
            if len(token) < self.min_token_length:
                continue
            if self.remove_stopwords and token in STOPWORDS:
                continue
            filtered.append(token)

        return filtered

    def tokenize_query(self, query: str) -> List[str]:
        """
        Tokenize a query, keeping important terms.

        For queries, we're less aggressive with filtering.
        """
        text = query.lower()
        tokens = re.findall(r'\b[a-z0-9]+\b', text)

        # Only remove very common words for queries
        query_stopwords = {"the", "a", "an", "is", "are", "was", "were", "to", "of"}
        filtered = [t for t in tokens if t not in query_stopwords and len(t) >= 2]

        return filtered


# =============================================================================
# Inverted Index
# =============================================================================

class InvertedIndex:
    """
    Inverted index for fast keyword lookup.

    Maps terms to document IDs for O(1) lookup.
    """

    def __init__(self):
        # term -> list of (doc_id, term_frequency)
        self.index: Dict[str, List[Tuple[str, int]]] = defaultdict(list)

        # doc_id -> document data
        self.documents: Dict[str, IndexedDocument] = {}

        # Term document frequencies (for IDF calculation)
        self.doc_freq: Dict[str, int] = defaultdict(int)

        # Total number of documents
        self.doc_count: int = 0

        # Average document length (for BM25)
        self.avg_doc_length: float = 0.0

        # Tokenizer
        self.tokenizer = Tokenizer()

    def add_document(
        self,
        doc_id: str,
        content: str,
        metadata: Optional[Dict[str, Any]] = None
    ) -> None:
        """
        Add a document to the index.

        Args:
            doc_id: Unique document identifier
            content: Document content
            metadata: Optional metadata
        """
        # Remove existing document if present
        if doc_id in self.documents:
            self._remove_from_index(doc_id)

        # Tokenize
        tokens = self.tokenizer.tokenize(content)

        # Create document
        doc = IndexedDocument(
            doc_id=doc_id,
            content=content,
            tokens=tokens,
            token_count=len(tokens),
            metadata=metadata or {}
        )
        self.documents[doc_id] = doc

        # Count term frequencies
        term_counts: Dict[str, int] = defaultdict(int)
        for token in tokens:
            term_counts[token] += 1

        # Update index
        seen_terms: Set[str] = set()
        for term, count in term_counts.items():
            self.index[term].append((doc_id, count))
            if term not in seen_terms:
                self.doc_freq[term] += 1
                seen_terms.add(term)

        # Update statistics
        self.doc_count += 1
        self._update_avg_length()

    def _remove_from_index(self, doc_id: str) -> None:
        """Remove a document from the index."""
        if doc_id not in self.documents:
            return

        doc = self.documents[doc_id]

        # Get unique terms in document
        unique_terms = set(doc.tokens)

        # Remove from posting lists
        for term in unique_terms:
            self.index[term] = [(d, c) for d, c in self.index[term] if d != doc_id]
            self.doc_freq[term] -= 1
            if self.doc_freq[term] <= 0:
                del self.doc_freq[term]
                if term in self.index:
                    del self.index[term]

        del self.documents[doc_id]
        self.doc_count -= 1
        self._update_avg_length()

    def _update_avg_length(self) -> None:
        """Update average document length."""
        if self.doc_count == 0:
            self.avg_doc_length = 0.0
        else:
            total_length = sum(doc.token_count for doc in self.documents.values())
            self.avg_doc_length = total_length / self.doc_count

    def get_documents_with_term(self, term: str) -> List[Tuple[str, int]]:
        """
        Get all documents containing a term.

        Args:
            term: Search term (will be normalized)

        Returns:
            List of (doc_id, term_frequency) tuples
        """
        normalized = term.lower()
        return self.index.get(normalized, [])

    def search(
        self,
        query: str,
        top_k: int = 10
    ) -> List[SearchResult]:
        """
        Search the index using BM25 scoring.

        Args:
            query: Search query
            top_k: Number of results to return

        Returns:
            List of SearchResult objects sorted by score
        """
        return self.bm25_search(query, top_k)

    def bm25_search(
        self,
        query: str,
        top_k: int = 10,
        k1: float = 1.2,
        b: float = 0.75
    ) -> List[SearchResult]:
        """
        BM25 ranking search.

        BM25 formula:
        score(D, Q) = Σ IDF(qi) * (f(qi, D) * (k1 + 1)) / (f(qi, D) + k1 * (1 - b + b * |D|/avgdl))

        Args:
            query: Search query
            top_k: Number of results
            k1: Term frequency saturation parameter
            b: Length normalization parameter

        Returns:
            Sorted list of SearchResult
        """
        query_tokens = self.tokenizer.tokenize_query(query)

        if not query_tokens:
            return []

        # Calculate scores for each document
        scores: Dict[str, float] = defaultdict(float)
        matched_terms: Dict[str, List[str]] = defaultdict(list)

        for term in query_tokens:
            postings = self.index.get(term, [])
            if not postings:
                continue

            # IDF calculation
            df = self.doc_freq.get(term, 0)
            if df == 0:
                continue

            idf = math.log((self.doc_count - df + 0.5) / (df + 0.5) + 1.0)

            for doc_id, tf in postings:
                doc = self.documents[doc_id]
                doc_len = doc.token_count

                # BM25 term score
                numerator = tf * (k1 + 1)
                denominator = tf + k1 * (1 - b + b * doc_len / max(self.avg_doc_length, 1))
                term_score = idf * numerator / denominator

                scores[doc_id] += term_score
                if term not in matched_terms[doc_id]:
                    matched_terms[doc_id].append(term)

        # Create results
        results = []
        for doc_id, score in scores.items():
            doc = self.documents[doc_id]
            results.append(SearchResult(
                doc_id=doc_id,
                score=score,
                matched_terms=matched_terms[doc_id],
                content=doc.content
            ))

        # Sort by score and limit
        results.sort(key=lambda x: x.score, reverse=True)
        return results[:top_k]

    def tfidf_search(
        self,
        query: str,
        top_k: int = 10
    ) -> List[SearchResult]:
        """
        Simple TF-IDF search (alternative to BM25).

        Args:
            query: Search query
            top_k: Number of results

        Returns:
            Sorted list of SearchResult
        """
        query_tokens = self.tokenizer.tokenize_query(query)

        if not query_tokens:
            return []

        scores: Dict[str, float] = defaultdict(float)
        matched_terms: Dict[str, List[str]] = defaultdict(list)

        for term in query_tokens:
            postings = self.index.get(term, [])
            if not postings:
                continue

            # IDF
            df = self.doc_freq.get(term, 0)
            if df == 0:
                continue

            idf = math.log(self.doc_count / df)

            for doc_id, tf in postings:
                # TF-IDF score
                tf_normalized = 1 + math.log(tf) if tf > 0 else 0
                scores[doc_id] += tf_normalized * idf
                if term not in matched_terms[doc_id]:
                    matched_terms[doc_id].append(term)

        results = []
        for doc_id, score in scores.items():
            doc = self.documents[doc_id]
            results.append(SearchResult(
                doc_id=doc_id,
                score=score,
                matched_terms=matched_terms[doc_id],
                content=doc.content
            ))

        results.sort(key=lambda x: x.score, reverse=True)
        return results[:top_k]

    def exact_match_search(
        self,
        query: str,
        top_k: int = 10
    ) -> List[SearchResult]:
        """
        Exact match search - prioritizes documents containing all query terms.

        Useful for specific entity lookups.

        Args:
            query: Search query
            top_k: Number of results

        Returns:
            Results sorted by number of matched terms, then by TF-IDF
        """
        query_tokens = set(self.tokenizer.tokenize_query(query))

        if not query_tokens:
            return []

        # Find documents with each term
        doc_term_matches: Dict[str, Set[str]] = defaultdict(set)

        for term in query_tokens:
            postings = self.index.get(term, [])
            for doc_id, _ in postings:
                doc_term_matches[doc_id].add(term)

        # Score by coverage then TF-IDF
        results = []
        for doc_id, matched in doc_term_matches.items():
            coverage = len(matched) / len(query_tokens)
            tfidf_results = self.tfidf_search(" ".join(matched), top_k=1)
            base_score = tfidf_results[0].score if tfidf_results else 0

            # Boost by coverage
            final_score = base_score * (1 + coverage)

            doc = self.documents[doc_id]
            results.append(SearchResult(
                doc_id=doc_id,
                score=final_score,
                matched_terms=list(matched),
                content=doc.content
            ))

        results.sort(key=lambda x: (len(x.matched_terms), x.score), reverse=True)
        return results[:top_k]

    def save(self, path: str) -> None:
        """Save the index to disk."""
        data = {
            "index": {k: list(v) for k, v in self.index.items()},
            "documents": {
                doc_id: {
                    "content": doc.content,
                    "tokens": doc.tokens,
                    "token_count": doc.token_count,
                    "metadata": doc.metadata
                }
                for doc_id, doc in self.documents.items()
            },
            "doc_freq": dict(self.doc_freq),
            "doc_count": self.doc_count,
            "avg_doc_length": self.avg_doc_length
        }

        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

    def load(self, path: str) -> bool:
        """Load the index from disk."""
        if not os.path.exists(path):
            return False

        try:
            with open(path, 'r', encoding='utf-8') as f:
                data = json.load(f)

            self.index = defaultdict(list)
            for term, postings in data.get("index", {}).items():
                self.index[term] = [tuple(p) for p in postings]

            self.documents = {}
            for doc_id, doc_data in data.get("documents", {}).items():
                self.documents[doc_id] = IndexedDocument(
                    doc_id=doc_id,
                    content=doc_data.get("content", ""),
                    tokens=doc_data.get("tokens", []),
                    token_count=doc_data.get("token_count", 0),
                    metadata=doc_data.get("metadata", {})
                )

            self.doc_freq = defaultdict(int, data.get("doc_freq", {}))
            self.doc_count = data.get("doc_count", 0)
            self.avg_doc_length = data.get("avg_doc_length", 0.0)

            return True

        except Exception as e:
            print(f"Failed to load keyword index: {e}")
            return False

    def get_statistics(self) -> Dict[str, Any]:
        """Get index statistics."""
        return {
            "document_count": self.doc_count,
            "unique_terms": len(self.index),
            "avg_doc_length": self.avg_doc_length,
            "total_postings": sum(len(v) for v in self.index.values())
        }


# =============================================================================
# Factory Functions
# =============================================================================

def create_index_from_nodes(
    nodes: List[Any],
    content_field: str = "content"
) -> InvertedIndex:
    """
    Create an inverted index from a list of MemoryNodes.

    Args:
        nodes: List of MemoryNode objects
        content_field: Field name containing the content

    Returns:
        Populated InvertedIndex
    """
    index = InvertedIndex()

    for node in nodes:
        doc_id = getattr(node, 'id', str(id(node)))
        content = getattr(node, content_field, str(node))

        metadata = {}
        for field in ['node_type', 'domain', 'weight', 'created_at']:
            if hasattr(node, field):
                value = getattr(node, field)
                if hasattr(value, 'value'):  # Enum
                    value = value.value
                elif hasattr(value, 'isoformat'):  # datetime
                    value = value.isoformat()
                metadata[field] = value

        index.add_document(doc_id, content, metadata)

    return index


def merge_with_semantic_scores(
    keyword_results: List[SearchResult],
    semantic_results: List[Tuple[str, float]],
    alpha: float = 0.5
) -> List[Tuple[str, float]]:
    """
    Merge keyword and semantic search results.

    Args:
        keyword_results: Results from keyword search
        semantic_results: Results from semantic search [(doc_id, score), ...]
        alpha: Weight for semantic scores (1-alpha for keyword)

    Returns:
        Merged and re-ranked results
    """
    # Normalize scores
    keyword_scores: Dict[str, float] = {}
    if keyword_results:
        max_kw = max(r.score for r in keyword_results)
        for r in keyword_results:
            keyword_scores[r.doc_id] = r.score / max_kw if max_kw > 0 else 0

    semantic_scores: Dict[str, float] = {}
    if semantic_results:
        max_sem = max(s for _, s in semantic_results)
        for doc_id, score in semantic_results:
            semantic_scores[doc_id] = score / max_sem if max_sem > 0 else 0

    # Combine all document IDs
    all_docs = set(keyword_scores.keys()) | set(semantic_scores.keys())

    # Calculate combined scores
    combined = []
    for doc_id in all_docs:
        kw_score = keyword_scores.get(doc_id, 0)
        sem_score = semantic_scores.get(doc_id, 0)
        combined_score = alpha * sem_score + (1 - alpha) * kw_score
        combined.append((doc_id, combined_score))

    # Sort by combined score
    combined.sort(key=lambda x: x[1], reverse=True)
    return combined
