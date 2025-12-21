"""
Entity-Centric Index
====================
Provides entity-to-fact mappings for improved retrieval of entity-specific information.

This module addresses the issue where queries about specific entities (e.g., "What instruments
does Melanie play?") fail to retrieve the correct information because the entity name
is not properly linked to the relevant facts.

Key Features:
- Entity extraction from text content
- Entity-predicate-fact mappings
- Integration with BM25 keyword index
- Boost entity matches in retrieval

Design Philosophy:
- Entity names are indexed separately with higher weight
- Facts are linked to entities with predicate relationships
- Entity queries get boosted scores for entity-matching documents
"""

import re
from collections import defaultdict
from typing import List, Dict, Set, Tuple, Optional, Any
from dataclasses import dataclass, field


@dataclass
class EntityFact:
    """A fact associated with an entity."""
    entity: str           # e.g., "Melanie"
    predicate: str        # e.g., "plays"
    value: str            # e.g., "violin"
    doc_id: str           # Source document ID
    confidence: float = 1.0


@dataclass
class EntityInfo:
    """Information about an entity in the index."""
    name: str
    aliases: Set[str] = field(default_factory=set)
    doc_ids: Set[str] = field(default_factory=set)  # Documents mentioning this entity
    facts: List[EntityFact] = field(default_factory=list)


class EntityExtractorSimple:
    """
    Simple rule-based entity extractor.

    Focuses on extracting:
    - Person names (capitalized words, especially at start of sentences)
    - Pronouns resolved to speaker names
    - Common entity patterns
    """

    # Common speaker patterns in dialogue
    SPEAKER_PATTERN = re.compile(
        r'\[[\d:]+\s*(?:am|pm)?\s*(?:on\s+)?\d+\s+\w+,?\s*\d*\]\s*(\w+):'
    )

    # Name patterns (capitalized words not at start of sentence)
    NAME_PATTERN = re.compile(r'\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)?)\b')

    # Possessive patterns (X's something)
    POSSESSIVE_PATTERN = re.compile(r"(\w+)'s\s+(\w+)")

    # Common predicate patterns for fact extraction
    PREDICATE_PATTERNS = [
        (re.compile(r'(\w+)\s+plays?\s+(?:the\s+)?(\w+)', re.I), 'plays'),
        (re.compile(r'(\w+)\s+likes?\s+(\w+)', re.I), 'likes'),
        (re.compile(r'(\w+)\s+loves?\s+(\w+)', re.I), 'loves'),
        (re.compile(r'(\w+)\s+(?:has|have)\s+(?:a\s+)?(\w+(?:\s+\w+)?)\s+named\s+(\w+)', re.I), 'has_named'),
        (re.compile(r'(\w+)\s+moved?\s+from\s+(\w+)', re.I), 'moved_from'),
        (re.compile(r'(\w+)\s+(?:is|are)\s+(?:a\s+)?(\w+)', re.I), 'is'),
        (re.compile(r"(\w+)'s\s+(?:pet|dog|cat)(?:s)?\s+(?:is|are)?\s*named?\s*(\w+)", re.I), 'pet_named'),
    ]

    # Common entity types to extract
    ENTITY_INDICATORS = {
        'locations': ['sweden', 'france', 'germany', 'italy', 'spain', 'usa', 'uk', 'china', 'japan'],
        'instruments': ['violin', 'piano', 'guitar', 'drums', 'clarinet', 'flute', 'trumpet', 'cello'],
        'activities': ['painting', 'camping', 'swimming', 'hiking', 'pottery', 'running', 'cycling'],
        'pets': ['dog', 'cat', 'bird', 'fish', 'rabbit', 'hamster'],
    }

    def extract_speaker(self, content: str) -> Optional[str]:
        """Extract speaker name from dialogue format."""
        match = self.SPEAKER_PATTERN.search(content)
        if match:
            return match.group(1).strip()
        return None

    def extract_entities(self, content: str) -> Set[str]:
        """Extract all entity mentions from content."""
        entities = set()

        # Extract speaker
        speaker = self.extract_speaker(content)
        if speaker:
            entities.add(speaker.lower())

        # Extract capitalized names
        for match in self.NAME_PATTERN.finditer(content):
            name = match.group(1)
            # Filter out common words that might be capitalized
            if name.lower() not in {'the', 'a', 'an', 'i', 'am', 'is', 'are', 'was', 'were',
                                      'have', 'has', 'do', 'does', 'did', 'will', 'would',
                                      'could', 'should', 'may', 'might', 'must', 'shall'}:
                entities.add(name.lower())

        # Extract possessives
        for match in self.POSSESSIVE_PATTERN.finditer(content):
            entities.add(match.group(1).lower())

        # Extract known entity types
        content_lower = content.lower()
        for category, items in self.ENTITY_INDICATORS.items():
            for item in items:
                if item in content_lower:
                    entities.add(item)

        return entities

    def extract_facts(self, content: str, doc_id: str) -> List[EntityFact]:
        """Extract structured facts from content."""
        facts = []
        speaker = self.extract_speaker(content)

        for pattern, predicate in self.PREDICATE_PATTERNS:
            for match in pattern.finditer(content):
                groups = match.groups()
                if len(groups) >= 2:
                    entity = groups[0].lower()
                    # Use speaker name if entity is 'I' or similar
                    if entity in ('i', 'me', 'my') and speaker:
                        entity = speaker.lower()

                    if predicate == 'has_named' and len(groups) >= 3:
                        # "X has a pet named Y"
                        facts.append(EntityFact(
                            entity=entity,
                            predicate=f'has_{groups[1].lower()}',
                            value=groups[2].lower(),
                            doc_id=doc_id
                        ))
                    else:
                        facts.append(EntityFact(
                            entity=entity,
                            predicate=predicate,
                            value=groups[1].lower() if len(groups) > 1 else '',
                            doc_id=doc_id
                        ))

        return facts


class EntityCentricIndex:
    """
    Index that maps entities to their associated facts and documents.

    This enables:
    1. Fast lookup of all documents mentioning an entity
    2. Retrieval of specific facts about an entity
    3. Boosting of entity-relevant documents in search
    """

    def __init__(self):
        # Entity name -> EntityInfo
        self.entities: Dict[str, EntityInfo] = {}

        # (entity, predicate) -> list of EntityFact
        self.fact_index: Dict[Tuple[str, str], List[EntityFact]] = defaultdict(list)

        # doc_id -> list of entities mentioned
        self.doc_entities: Dict[str, Set[str]] = defaultdict(set)

        # Entity extractor
        self.extractor = EntityExtractorSimple()

    def add_document(
        self,
        doc_id: str,
        content: str,
        metadata: Optional[Dict[str, Any]] = None
    ) -> Set[str]:
        """
        Add a document to the entity index.

        Args:
            doc_id: Document ID
            content: Document content
            metadata: Optional metadata

        Returns:
            Set of extracted entity names
        """
        # Extract entities
        entities = self.extractor.extract_entities(content)

        # Register entities
        for entity_name in entities:
            if entity_name not in self.entities:
                self.entities[entity_name] = EntityInfo(name=entity_name)
            self.entities[entity_name].doc_ids.add(doc_id)

        # Track doc -> entities mapping
        self.doc_entities[doc_id] = entities

        # Extract and index facts
        facts = self.extractor.extract_facts(content, doc_id)
        for fact in facts:
            # Add to entity info
            if fact.entity in self.entities:
                self.entities[fact.entity].facts.append(fact)
            else:
                self.entities[fact.entity] = EntityInfo(
                    name=fact.entity,
                    doc_ids={doc_id},
                    facts=[fact]
                )

            # Add to fact index
            self.fact_index[(fact.entity, fact.predicate)].append(fact)

        return entities

    def get_entity_documents(self, entity_name: str) -> Set[str]:
        """Get all document IDs that mention an entity."""
        entity_name = entity_name.lower()
        if entity_name in self.entities:
            return self.entities[entity_name].doc_ids
        return set()

    def get_entity_facts(
        self,
        entity_name: str,
        predicate: Optional[str] = None
    ) -> List[EntityFact]:
        """Get facts about an entity, optionally filtered by predicate."""
        entity_name = entity_name.lower()
        if entity_name not in self.entities:
            return []

        facts = self.entities[entity_name].facts

        if predicate:
            predicate = predicate.lower()
            facts = [f for f in facts if f.predicate == predicate]

        return facts

    def extract_query_entities(self, query: str) -> Set[str]:
        """Extract entity mentions from a query."""
        return self.extractor.extract_entities(query)

    def compute_entity_boost(
        self,
        doc_id: str,
        query_entities: Set[str],
        boost_factor: float = 1.5
    ) -> float:
        """
        Compute a boost score for a document based on entity overlap.

        Args:
            doc_id: Document ID
            query_entities: Entities mentioned in the query
            boost_factor: Multiplier for entity matches

        Returns:
            Boost multiplier (1.0 = no boost)
        """
        if not query_entities or doc_id not in self.doc_entities:
            return 1.0

        doc_entities = self.doc_entities[doc_id]
        overlap = query_entities & doc_entities

        if not overlap:
            return 1.0

        # Boost based on number of matching entities
        # More matches = higher boost
        boost = 1.0 + (boost_factor - 1.0) * (len(overlap) / len(query_entities))
        return boost

    def get_statistics(self) -> Dict[str, Any]:
        """Get index statistics."""
        total_facts = sum(len(info.facts) for info in self.entities.values())
        avg_docs_per_entity = (
            sum(len(info.doc_ids) for info in self.entities.values()) / len(self.entities)
            if self.entities else 0
        )

        return {
            "entity_count": len(self.entities),
            "total_facts": total_facts,
            "documents_indexed": len(self.doc_entities),
            "avg_docs_per_entity": round(avg_docs_per_entity, 2),
            "unique_predicates": len(set(k[1] for k in self.fact_index.keys()))
        }

    def save(self, path: str) -> None:
        """Save the index to disk."""
        import json
        import os

        os.makedirs(os.path.dirname(path), exist_ok=True)

        data = {
            "entities": {
                name: {
                    "name": info.name,
                    "aliases": list(info.aliases),
                    "doc_ids": list(info.doc_ids),
                    "facts": [
                        {
                            "entity": f.entity,
                            "predicate": f.predicate,
                            "value": f.value,
                            "doc_id": f.doc_id,
                            "confidence": f.confidence
                        }
                        for f in info.facts
                    ]
                }
                for name, info in self.entities.items()
            },
            "doc_entities": {
                doc_id: list(entities)
                for doc_id, entities in self.doc_entities.items()
            }
        }

        with open(path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

    def load(self, path: str) -> bool:
        """Load the index from disk."""
        import json
        import os

        if not os.path.exists(path):
            return False

        try:
            with open(path, 'r', encoding='utf-8') as f:
                data = json.load(f)

            # Load entities
            self.entities = {}
            for name, info_data in data.get("entities", {}).items():
                facts = [
                    EntityFact(
                        entity=f["entity"],
                        predicate=f["predicate"],
                        value=f["value"],
                        doc_id=f["doc_id"],
                        confidence=f.get("confidence", 1.0)
                    )
                    for f in info_data.get("facts", [])
                ]
                self.entities[name] = EntityInfo(
                    name=info_data["name"],
                    aliases=set(info_data.get("aliases", [])),
                    doc_ids=set(info_data.get("doc_ids", [])),
                    facts=facts
                )

            # Rebuild fact index
            self.fact_index = defaultdict(list)
            for info in self.entities.values():
                for fact in info.facts:
                    self.fact_index[(fact.entity, fact.predicate)].append(fact)

            # Load doc_entities
            self.doc_entities = defaultdict(set)
            for doc_id, entities in data.get("doc_entities", {}).items():
                self.doc_entities[doc_id] = set(entities)

            return True

        except Exception as e:
            print(f"Failed to load entity index: {e}")
            return False


# =============================================================================
# Integration with HybridRetriever
# =============================================================================

def create_entity_index_from_nodes(nodes: List[Any]) -> EntityCentricIndex:
    """
    Create an entity index from a list of MemoryNodes.

    Args:
        nodes: List of MemoryNode objects

    Returns:
        Populated EntityCentricIndex
    """
    index = EntityCentricIndex()

    for node in nodes:
        doc_id = getattr(node, 'id', str(id(node)))
        content = getattr(node, 'content', str(node))
        index.add_document(doc_id, content)

    return index


# =============================================================================
# Testing
# =============================================================================

if __name__ == "__main__":
    # Test entity extraction
    test_content = """
    [1:56 pm on 8 May, 2023] Melanie: I play violin and clarinet!
    [2:00 pm on 8 May, 2023] Caroline: Nice! I moved from Sweden 4 years ago.
    [2:05 pm on 8 May, 2023] Melanie: My dog is named Oliver, and my cats are Luna and Bailey.
    """

    index = EntityCentricIndex()

    # Add test documents
    lines = test_content.strip().split('\n')
    for i, line in enumerate(lines):
        if line.strip():
            entities = index.add_document(f"doc_{i}", line.strip())
            print(f"Doc {i}: Extracted entities: {entities}")

    print("\n" + "=" * 50)
    print("Index Statistics:")
    print(index.get_statistics())

    print("\n" + "=" * 50)
    print("Entity Facts:")
    for entity_name in ['melanie', 'caroline']:
        print(f"\n{entity_name.title()}:")
        for fact in index.get_entity_facts(entity_name):
            print(f"  - {fact.predicate}: {fact.value}")

    print("\n" + "=" * 50)
    print("Query Entity Extraction:")
    test_queries = [
        "What instruments does Melanie play?",
        "Where did Caroline move from?",
        "What are Melanie's pets' names?"
    ]

    for query in test_queries:
        entities = index.extract_query_entities(query)
        print(f"Query: {query}")
        print(f"  Entities: {entities}")
