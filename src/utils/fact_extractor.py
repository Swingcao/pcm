"""
Structured Fact Extractor
=========================
Extracts atomic, structured facts from dialogue messages for improved retrieval.

This module addresses the problem of storing only raw dialogue without structured
facts, which makes needle-in-haystack retrieval difficult.

Key Features:
- Extracts subject-predicate-object triples
- Resolves relative temporal references (e.g., "last year" -> "2022")
- Extracts entity attributes
- Maintains link to original message for context
"""

import re
import asyncio
from datetime import datetime
from dateutil.relativedelta import relativedelta
from typing import List, Optional, Dict, Any, Tuple
from dataclasses import dataclass, field
from pydantic import BaseModel, Field

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
import config
from src.utils.llm_client import get_llm_client


# =============================================================================
# Data Structures
# =============================================================================

@dataclass
class StructuredFact:
    """
    A structured fact extracted from dialogue.

    Represents atomic, queryable information in subject-predicate-object form.
    """
    subject: str                          # Entity (e.g., "Melanie", "Caroline")
    predicate: str                        # Relation (e.g., "plays", "moved_from", "likes")
    object: str                           # Object (e.g., "violin", "Sweden", "hiking")
    temporal: Optional[str] = None        # Resolved absolute time (e.g., "2022", "May 2023")
    location: Optional[str] = None        # Location if mentioned
    raw_message: str = ""                 # Original message for context
    source_timestamp: str = ""            # When the message was received
    confidence: float = 0.9               # Extraction confidence
    fact_type: str = "general"            # Type: "attribute", "preference", "event", "general"
    keywords: List[str] = field(default_factory=list)  # Important keywords for indexing

    def to_searchable_text(self) -> str:
        """Convert to searchable text for embedding."""
        parts = [f"{self.subject} {self.predicate} {self.object}"]
        if self.temporal:
            parts.append(f"({self.temporal})")
        if self.location:
            parts.append(f"at {self.location}")
        return " ".join(parts)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for storage."""
        return {
            "subject": self.subject,
            "predicate": self.predicate,
            "object": self.object,
            "temporal": self.temporal,
            "location": self.location,
            "raw_message": self.raw_message,
            "source_timestamp": self.source_timestamp,
            "confidence": self.confidence,
            "fact_type": self.fact_type,
            "keywords": self.keywords
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "StructuredFact":
        """Create from dictionary."""
        return cls(
            subject=data.get("subject", ""),
            predicate=data.get("predicate", ""),
            object=data.get("object", ""),
            temporal=data.get("temporal"),
            location=data.get("location"),
            raw_message=data.get("raw_message", ""),
            source_timestamp=data.get("source_timestamp", ""),
            confidence=data.get("confidence", 0.9),
            fact_type=data.get("fact_type", "general"),
            keywords=data.get("keywords", [])
        )


class LLMFactExtractionResult(BaseModel):
    """Pydantic model for LLM structured output."""
    facts: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="List of extracted facts"
    )
    entities: List[str] = Field(
        default_factory=list,
        description="Named entities mentioned"
    )
    temporal_expressions: List[Dict[str, str]] = Field(
        default_factory=list,
        description="Temporal expressions with resolved values"
    )


# =============================================================================
# Prompts
# =============================================================================

FACT_EXTRACTION_PROMPT = """You are a fact extraction system. Extract structured facts from the dialogue message.

Current timestamp: {timestamp}
Message: {message}

Extract ALL factual information as subject-predicate-object triples. Include:
1. Personal attributes (e.g., "Melanie plays violin")
2. Events (e.g., "Caroline attended support group")
3. Preferences (e.g., "Melanie likes painting")
4. Relationships (e.g., "Caroline has Swedish grandmother")
5. Locations (e.g., "Melanie lives in Boston")

For temporal expressions:
- Convert relative times to absolute: "last year" relative to {timestamp} → calculate actual year
- "yesterday" → calculate actual date
- "next month" → calculate actual month/year

Output JSON format:
{{
    "facts": [
        {{
            "subject": "entity name",
            "predicate": "action/relation verb",
            "object": "object of the relation",
            "temporal": "resolved absolute time or null",
            "location": "location or null",
            "fact_type": "attribute|preference|event|general",
            "confidence": 0.0-1.0,
            "keywords": ["important", "keywords", "for", "search"]
        }}
    ],
    "entities": ["list", "of", "named", "entities"],
    "temporal_expressions": [
        {{"original": "last year", "resolved": "2022"}}
    ]
}}

Extract facts even from casual conversation. Be thorough - every piece of factual information matters."""


# =============================================================================
# Temporal Resolution Utilities
# =============================================================================

class TemporalResolver:
    """Resolves relative temporal expressions to absolute dates."""

    RELATIVE_PATTERNS = [
        (r"yesterday", lambda ref: ref - relativedelta(days=1)),
        (r"today", lambda ref: ref),
        (r"tomorrow", lambda ref: ref + relativedelta(days=1)),
        (r"last week", lambda ref: ref - relativedelta(weeks=1)),
        (r"next week", lambda ref: ref + relativedelta(weeks=1)),
        (r"last month", lambda ref: ref - relativedelta(months=1)),
        (r"next month", lambda ref: ref + relativedelta(months=1)),
        (r"last year", lambda ref: ref - relativedelta(years=1)),
        (r"next year", lambda ref: ref + relativedelta(years=1)),
        (r"(\d+) days? ago", lambda ref, n: ref - relativedelta(days=int(n))),
        (r"(\d+) weeks? ago", lambda ref, n: ref - relativedelta(weeks=int(n))),
        (r"(\d+) months? ago", lambda ref, n: ref - relativedelta(months=int(n))),
        (r"(\d+) years? ago", lambda ref, n: ref - relativedelta(years=int(n))),
        (r"in (\d+) days?", lambda ref, n: ref + relativedelta(days=int(n))),
        (r"in (\d+) weeks?", lambda ref, n: ref + relativedelta(weeks=int(n))),
        (r"in (\d+) months?", lambda ref, n: ref + relativedelta(months=int(n))),
    ]

    @classmethod
    def resolve(cls, text: str, reference_time: datetime) -> Tuple[str, Optional[str]]:
        """
        Resolve relative temporal expressions in text.

        Args:
            text: Text potentially containing relative temporal expressions
            reference_time: Reference datetime for resolution

        Returns:
            Tuple of (original_expression, resolved_date_string)
        """
        text_lower = text.lower()

        for pattern, resolver in cls.RELATIVE_PATTERNS:
            match = re.search(pattern, text_lower)
            if match:
                try:
                    if match.groups():
                        # Pattern with capture group (e.g., "3 days ago")
                        resolved = resolver(reference_time, match.group(1))
                    else:
                        # Simple pattern (e.g., "yesterday")
                        resolved = resolver(reference_time)

                    return match.group(0), resolved.strftime("%B %d, %Y")
                except Exception:
                    continue

        return "", None

    @classmethod
    def extract_year(cls, text: str, reference_time: datetime) -> Optional[str]:
        """Extract just the year from relative expressions."""
        _, resolved = cls.resolve(text, reference_time)
        if resolved:
            try:
                return datetime.strptime(resolved, "%B %d, %Y").strftime("%Y")
            except Exception:
                pass
        return None


# =============================================================================
# Fact Extractor
# =============================================================================

class FactExtractor:
    """
    Extracts structured facts from dialogue messages.

    Uses LLM for semantic extraction and rule-based temporal resolution.
    """

    def __init__(self, use_llm: bool = True):
        """
        Initialize the fact extractor.

        Args:
            use_llm: Whether to use LLM for extraction (vs. rule-based only)
        """
        self.use_llm = use_llm
        self.llm_client = get_llm_client() if use_llm else None
        self.temporal_resolver = TemporalResolver()

    async def extract(
        self,
        message: str,
        timestamp: str,
        speaker: Optional[str] = None
    ) -> List[StructuredFact]:
        """
        Extract structured facts from a dialogue message.

        Args:
            message: The dialogue message text
            timestamp: Timestamp of the message (e.g., "1:56 pm on 8 May, 2023")
            speaker: Optional speaker name for context

        Returns:
            List of extracted StructuredFact objects
        """
        # Parse reference timestamp
        reference_time = self._parse_timestamp(timestamp)

        # Extract facts using LLM or rules
        if self.use_llm and self.llm_client:
            facts = await self._extract_with_llm(message, timestamp, reference_time)
        else:
            facts = self._extract_with_rules(message, reference_time)

        # Post-process: add raw message, resolve any remaining temporal expressions
        for fact in facts:
            fact.raw_message = message
            fact.source_timestamp = timestamp

            # Resolve temporal expressions in the object field
            if not fact.temporal and fact.object:
                year = self.temporal_resolver.extract_year(fact.object, reference_time)
                if year:
                    fact.temporal = year

            # Extract keywords if not already present
            if not fact.keywords:
                fact.keywords = self._extract_keywords(fact)

        return facts

    async def _extract_with_llm(
        self,
        message: str,
        timestamp: str,
        reference_time: datetime
    ) -> List[StructuredFact]:
        """Extract facts using LLM."""
        prompt = FACT_EXTRACTION_PROMPT.format(
            message=message,
            timestamp=timestamp
        )

        try:
            result = await self.llm_client.structured_output(
                messages=[{"role": "user", "content": prompt}],
                output_schema=LLMFactExtractionResult,
                temperature=0.1
            )

            facts = []
            for fact_dict in result.facts:
                fact = StructuredFact(
                    subject=fact_dict.get("subject", ""),
                    predicate=fact_dict.get("predicate", ""),
                    object=fact_dict.get("object", ""),
                    temporal=fact_dict.get("temporal"),
                    location=fact_dict.get("location"),
                    confidence=fact_dict.get("confidence", 0.9),
                    fact_type=fact_dict.get("fact_type", "general"),
                    keywords=fact_dict.get("keywords", [])
                )
                if fact.subject and fact.predicate and fact.object:
                    facts.append(fact)

            return facts

        except Exception as e:
            print(f"LLM extraction failed, falling back to rules: {e}")
            return self._extract_with_rules(message, reference_time)

    def _extract_with_rules(
        self,
        message: str,
        reference_time: datetime
    ) -> List[StructuredFact]:
        """
        Rule-based fact extraction (fallback).

        Extracts basic facts using pattern matching.
        """
        facts = []

        # Pattern: "[Speaker] [verb] [object]"
        # This is a simple fallback - LLM extraction is preferred

        # Extract speaker from message format like "[timestamp] Speaker: message"
        speaker_match = re.search(r'\] (\w+):', message)
        speaker = speaker_match.group(1) if speaker_match else "Unknown"

        # Extract the actual message content
        content_match = re.search(r'\] \w+: (.+)$', message)
        content = content_match.group(1) if content_match else message

        # Simple verb patterns for common fact types
        verb_patterns = [
            (r"(?:I|i) (?:play|plays|played) (\w+)", "plays"),
            (r"(?:I|i) (?:like|likes|liked|love|loves) (\w+)", "likes"),
            (r"(?:I|i) (?:went|go|goes) to (?:a |an |the )?(.+?)(?:\.|$|,)", "attended"),
            (r"(?:I|i) (?:am|'m) (?:a |an )?(\w+)", "is"),
            (r"(?:I|i) (?:have|has) (\d+ \w+)", "has"),
            (r"(?:I|i) (?:moved|move) from (\w+)", "moved_from"),
            (r"(?:I|i) (?:live|lives) in (\w+)", "lives_in"),
        ]

        for pattern, predicate in verb_patterns:
            match = re.search(pattern, content, re.IGNORECASE)
            if match:
                facts.append(StructuredFact(
                    subject=speaker,
                    predicate=predicate,
                    object=match.group(1).strip(),
                    confidence=0.7  # Lower confidence for rule-based extraction
                ))

        return facts

    def _parse_timestamp(self, timestamp: str) -> datetime:
        """Parse timestamp string to datetime object."""
        # Handle format like "1:56 pm on 8 May, 2023"
        patterns = [
            r"(\d{1,2}:\d{2} [ap]m) on (\d{1,2} \w+ \d{4})",
            r"(\d{1,2}:\d{2} [ap]m) on (\w+ \d{1,2}, \d{4})",
            r"(\d{4}-\d{2}-\d{2})",
        ]

        for pattern in patterns:
            match = re.search(pattern, timestamp)
            if match:
                try:
                    if len(match.groups()) == 2:
                        # Format: "1:56 pm on 8 May, 2023"
                        time_str, date_str = match.groups()
                        # Try parsing different date formats
                        for date_fmt in ["%d %B %Y", "%d %B, %Y", "%B %d, %Y"]:
                            try:
                                return datetime.strptime(date_str, date_fmt)
                            except ValueError:
                                continue
                    else:
                        # Format: "2023-05-08"
                        return datetime.strptime(match.group(1), "%Y-%m-%d")
                except Exception:
                    continue

        # Default to now if parsing fails
        return datetime.now()

    def _extract_keywords(self, fact: StructuredFact) -> List[str]:
        """Extract important keywords from a fact for indexing."""
        # Combine all text fields
        text_parts = [
            fact.subject,
            fact.predicate,
            fact.object,
            fact.temporal or "",
            fact.location or ""
        ]
        text = " ".join(text_parts).lower()

        # Simple tokenization and stopword removal
        stopwords = {
            "a", "an", "the", "is", "are", "was", "were", "be", "been",
            "has", "have", "had", "do", "does", "did", "will", "would",
            "could", "should", "may", "might", "must", "shall", "can",
            "to", "of", "in", "for", "on", "with", "at", "by", "from",
            "and", "or", "but", "if", "then", "so", "that", "this",
            "it", "its", "i", "my", "me", "we", "our", "you", "your"
        }

        tokens = re.findall(r'\b\w+\b', text)
        keywords = [t for t in tokens if t not in stopwords and len(t) > 2]

        return list(set(keywords))


# =============================================================================
# Batch Processing
# =============================================================================

async def extract_facts_from_dialogue(
    messages: List[Dict[str, str]],
    use_llm: bool = True
) -> List[StructuredFact]:
    """
    Extract facts from a list of dialogue messages.

    Args:
        messages: List of dicts with 'content' and 'timestamp' keys
        use_llm: Whether to use LLM for extraction

    Returns:
        List of all extracted facts
    """
    extractor = FactExtractor(use_llm=use_llm)
    all_facts = []

    for msg in messages:
        facts = await extractor.extract(
            message=msg.get("content", ""),
            timestamp=msg.get("timestamp", ""),
            speaker=msg.get("speaker")
        )
        all_facts.extend(facts)

    return all_facts


def extract_facts_sync(
    message: str,
    timestamp: str,
    use_llm: bool = True
) -> List[StructuredFact]:
    """
    Synchronous wrapper for fact extraction.

    Args:
        message: Dialogue message text
        timestamp: Message timestamp
        use_llm: Whether to use LLM for extraction

    Returns:
        List of extracted facts
    """
    extractor = FactExtractor(use_llm=use_llm)
    return asyncio.run(extractor.extract(message, timestamp))
