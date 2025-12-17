"""
LoComo Dataset Loader and Utilities
====================================
Load and process the LoComo benchmark dataset for memory evaluation.

Dataset Structure:
- qa: List of QA pairs with question, answer, evidence, category
- conversation: Multi-session dialogues between speaker_a and speaker_b
- observation: Evidence summaries provided by benchmark authors

Categories:
- Category 1: Single-hop factual questions
- Category 2: Temporal reasoning questions
- Category 3: Multi-hop reasoning questions
- Category 4: Adversarial/unanswerable questions
"""

import json
import os
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime

import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
import config


@dataclass
class QAPair:
    """A single QA pair from the LoComo dataset."""
    question: str
    answer: str
    evidence: List[str]
    category: int
    adversarial_answer: Optional[str] = None

    @property
    def category_name(self) -> str:
        """Get human-readable category name."""
        names = {
            1: "single-hop",
            2: "temporal",
            3: "multi-hop",
            4: "adversarial"
        }
        return names.get(self.category, "unknown")


@dataclass
class DialogueTurn:
    """A single turn in a conversation."""
    speaker: str
    text: str
    dialogue_id: str
    timestamp: str = ""
    image_url: Optional[List[str]] = None
    blip_caption: Optional[str] = None


@dataclass
class ConversationSession:
    """A single conversation session."""
    session_id: str
    date_time: str
    turns: List[DialogueTurn]


@dataclass
class LoCoMoSample:
    """A complete sample from the LoComo dataset."""
    sample_id: str
    speaker_a: str
    speaker_b: str
    qa_pairs: List[QAPair]
    sessions: List[ConversationSession]
    observations: Dict[str, List[Any]] = field(default_factory=dict)

    @property
    def total_turns(self) -> int:
        """Total number of dialogue turns."""
        return sum(len(s.turns) for s in self.sessions)

    @property
    def total_qa(self) -> int:
        """Total number of QA pairs."""
        return len(self.qa_pairs)

    def get_qa_by_category(self, category: int) -> List[QAPair]:
        """Get QA pairs filtered by category."""
        return [qa for qa in self.qa_pairs if qa.category == category]


class LoCoMoDataset:
    """
    LoComo Dataset Loader.

    Provides easy access to the LoComo benchmark data for evaluation.
    """

    def __init__(self, dataset_path: Optional[str] = None):
        """
        Initialize the dataset loader.

        Args:
            dataset_path: Path to locomo10.json file
        """
        self.dataset_path = dataset_path or config.DATASET_PATH
        self._samples: List[LoCoMoSample] = []
        self._loaded = False

    def load(self) -> None:
        """Load the dataset from file."""
        if self._loaded:
            return

        if not os.path.exists(self.dataset_path):
            raise FileNotFoundError(f"Dataset not found: {self.dataset_path}")

        with open(self.dataset_path, 'r', encoding='utf-8') as f:
            raw_data = json.load(f)

        self._samples = [self._parse_sample(item, idx) for idx, item in enumerate(raw_data)]
        self._loaded = True
        print(f"Loaded {len(self._samples)} samples from LoComo dataset")

    def _parse_sample(self, raw: Dict[str, Any], idx: int) -> LoCoMoSample:
        """Parse a raw JSON sample into a LoCoMoSample object."""
        # Parse QA pairs
        qa_pairs = []
        for qa in raw.get("qa", []):
            qa_pairs.append(QAPair(
                question=qa["question"],
                answer=str(qa.get("answer", qa.get("adversarial_answer", ""))),
                evidence=qa.get("evidence", []),
                category=qa.get("category", 1),
                adversarial_answer=qa.get("adversarial_answer")
            ))

        # Parse conversation sessions
        conv = raw.get("conversation", {})
        speaker_a = conv.get("speaker_a", "User")
        speaker_b = conv.get("speaker_b", "Assistant")

        sessions = []
        session_idx = 1
        while f"session_{session_idx}" in conv:
            session_key = f"session_{session_idx}"
            datetime_key = f"session_{session_idx}_date_time"

            turns = []
            for turn in conv.get(session_key, []):
                turns.append(DialogueTurn(
                    speaker=turn["speaker"],
                    text=turn["text"],
                    dialogue_id=turn.get("dia_id", ""),
                    timestamp=conv.get(datetime_key, ""),
                    image_url=turn.get("img_url"),
                    blip_caption=turn.get("blip_caption")
                ))

            sessions.append(ConversationSession(
                session_id=session_key,
                date_time=conv.get(datetime_key, ""),
                turns=turns
            ))
            session_idx += 1

        # Parse observations
        observations = {}
        for key, value in raw.items():
            if "observation" in key.lower():
                observations[key] = value

        return LoCoMoSample(
            sample_id=raw.get("sample_id", f"sample_{idx}"),
            speaker_a=speaker_a,
            speaker_b=speaker_b,
            qa_pairs=qa_pairs,
            sessions=sessions,
            observations=observations
        )

    def __len__(self) -> int:
        """Return number of samples."""
        self.load()
        return len(self._samples)

    def __getitem__(self, idx: int) -> LoCoMoSample:
        """Get a sample by index."""
        self.load()
        return self._samples[idx]

    def __iter__(self):
        """Iterate over samples."""
        self.load()
        return iter(self._samples)

    def get_all_dialogues(self, sample_idx: int) -> List[Dict[str, str]]:
        """
        Get all dialogues from a sample in chronological order.

        Returns:
            List of dicts with 'speaker', 'text', 'timestamp' keys
        """
        self.load()
        sample = self._samples[sample_idx]

        dialogues = []
        for session in sample.sessions:
            for turn in session.turns:
                dialogues.append({
                    "speaker": turn.speaker,
                    "text": turn.text,
                    "timestamp": turn.timestamp or session.date_time,
                    "dialogue_id": turn.dialogue_id
                })
        return dialogues

    def get_statistics(self) -> Dict[str, Any]:
        """Get dataset statistics."""
        self.load()

        total_qa = sum(s.total_qa for s in self._samples)
        total_turns = sum(s.total_turns for s in self._samples)

        category_counts = {1: 0, 2: 0, 3: 0, 4: 0}
        for sample in self._samples:
            for qa in sample.qa_pairs:
                category_counts[qa.category] = category_counts.get(qa.category, 0) + 1

        return {
            "num_samples": len(self._samples),
            "total_qa_pairs": total_qa,
            "total_dialogue_turns": total_turns,
            "qa_by_category": {
                "single-hop": category_counts.get(1, 0),
                "temporal": category_counts.get(2, 0),
                "multi-hop": category_counts.get(3, 0),
                "adversarial": category_counts.get(4, 0)
            },
            "avg_qa_per_sample": total_qa / len(self._samples) if self._samples else 0,
            "avg_turns_per_sample": total_turns / len(self._samples) if self._samples else 0
        }


def process_dialogues_for_memory(sample: LoCoMoSample) -> List[Dict[str, Any]]:
    """
    Process dialogues into a format suitable for memory ingestion.

    Args:
        sample: LoCoMoSample object

    Returns:
        List of processed dialogue entries
    """
    processed = []
    speaker_a = sample.speaker_a
    speaker_b = sample.speaker_b

    for session in sample.sessions:
        for turn in session.turns:
            # Combine image caption with text if available
            text = turn.text
            if turn.blip_caption:
                text = f"{text} [Image: {turn.blip_caption}]"

            processed.append({
                "speaker": turn.speaker,
                "message": text,
                "timestamp": turn.timestamp or session.date_time,
                "is_user": turn.speaker == speaker_a,
                "dialogue_id": turn.dialogue_id
            })

    return processed


# Convenience function
def load_locomo(path: Optional[str] = None) -> LoCoMoDataset:
    """Load the LoComo dataset."""
    dataset = LoCoMoDataset(path)
    dataset.load()
    return dataset


if __name__ == "__main__":
    # Demo: Load and display statistics
    dataset = load_locomo()
    stats = dataset.get_statistics()

    print("\nLoComo Dataset Statistics:")
    print("=" * 40)
    for key, value in stats.items():
        print(f"  {key}: {value}")

    print("\nSample 0 Info:")
    sample = dataset[0]
    print(f"  Speaker A: {sample.speaker_a}")
    print(f"  Speaker B: {sample.speaker_b}")
    print(f"  Sessions: {len(sample.sessions)}")
    print(f"  Total Turns: {sample.total_turns}")
    print(f"  QA Pairs: {sample.total_qa}")

    print("\nFirst 3 QA pairs:")
    for i, qa in enumerate(sample.qa_pairs[:3]):
        print(f"  [{qa.category_name}] Q: {qa.question[:50]}...")
        print(f"            A: {qa.answer}")
