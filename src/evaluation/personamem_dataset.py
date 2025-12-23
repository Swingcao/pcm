"""
PersonaMem Dataset Loader
=========================
Load and process the PersonaMem benchmark dataset for memory system evaluation.

The PersonaMem dataset contains:
- questions_32k.csv: Question data with multiple-choice options
- shared_contexts_32k.jsonl: Dialogue history contexts

Data structure:
- Each shared_context_id corresponds to a complete dialogue session
- Multiple questions may share the same context
- Questions are multiple-choice (4 options: a, b, c, d)
"""

import csv
import json
import os
from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional, Iterator
import pandas as pd


@dataclass
class PersonaMemQuestion:
    """A single question from the PersonaMem dataset."""
    question_id: str
    question_type: str          # 6 types: recall_user_shared_facts, etc.
    topic: str                  # 5 topics: musicRecommendation, etc.
    user_question: str          # The question text
    correct_answer: str         # Correct option: (a), (b), (c), (d)
    all_options: List[str]      # List of 4 options
    context_length_tokens: int  # Context length in tokens
    distance_to_ref_tokens: int # Distance to reference in tokens
    distance_proportion: str    # Distance proportion in context
    end_index: int              # End index in shared context

    @property
    def correct_option_index(self) -> int:
        """Get the index of correct option (0-3)."""
        mapping = {'(a)': 0, '(b)': 1, '(c)': 2, '(d)': 3}
        return mapping.get(self.correct_answer, -1)


@dataclass
class PersonaMemSample:
    """
    A sample from PersonaMem dataset.

    Each sample represents a persona with their dialogue history and questions.
    Questions are grouped by shared_context_id.
    """
    context_id: str                    # shared_context_id
    persona_id: int                    # User/persona identifier
    persona_info: str                  # Persona description from system message
    context_messages: List[Dict]       # Dialogue history [{"role": "...", "content": "..."}]
    questions: List[PersonaMemQuestion]  # Questions for this context

    @property
    def num_questions(self) -> int:
        return len(self.questions)

    @property
    def num_dialogue_turns(self) -> int:
        """Count user/assistant turns (excluding system)."""
        return sum(1 for m in self.context_messages if m['role'] != 'system')

    def get_questions_by_type(self, question_type: str) -> List[PersonaMemQuestion]:
        """Filter questions by type."""
        return [q for q in self.questions if q.question_type == question_type]

    def get_questions_by_topic(self, topic: str) -> List[PersonaMemQuestion]:
        """Filter questions by topic."""
        return [q for q in self.questions if q.topic == topic]


class PersonaMemDataset:
    """
    PersonaMem Dataset loader and manager.

    Usage:
        dataset = PersonaMemDataset(data_dir="./data/personamem")
        dataset.load()

        for sample in dataset:
            print(sample.persona_info)
            for q in sample.questions:
                print(q.user_question)
    """

    # Question type names
    QUESTION_TYPES = [
        'recall_user_shared_facts',
        'provide_preference_aligned_recommendations',
        'suggest_new_ideas',
        'recalling_the_reasons_behind_previous_updates',
        'track_full_preference_evolution',
        'generalizing_to_new_scenarios'
    ]

    # Topic names
    TOPICS = [
        'musicRecommendation',
        'bookRecommendation',
        'datingConsultation',
        'studyConsultation',
        'medicalConsultation'
    ]

    def __init__(
        self,
        data_dir: str = "../../data/personamem",
        questions_file: str = "questions_32k.csv",
        contexts_file: str = "shared_contexts_32k.jsonl"
    ):
        """
        Initialize the dataset loader.

        Args:
            data_dir: Directory containing dataset files
            questions_file: CSV file with questions
            contexts_file: JSONL file with dialogue contexts
        """
        self.data_dir = data_dir
        self.questions_path = os.path.join(data_dir, questions_file)
        self.contexts_path = os.path.join(data_dir, contexts_file)

        self.samples: List[PersonaMemSample] = []
        self._contexts: Dict[str, List[Dict]] = {}
        self._questions: List[Dict] = []  # Raw question data from CSV
        self._loaded = False

    def load(self) -> 'PersonaMemDataset':
        """
        Load the dataset from files.

        Returns:
            self for chaining
        """
        if self._loaded:
            return self

        # Load contexts
        self._load_contexts()

        # Load questions
        self._load_questions()

        # Build samples
        self._build_samples()

        self._loaded = True
        return self

    def _load_contexts(self) -> None:
        """Load dialogue contexts from JSONL file."""
        self._contexts = {}

        with open(self.contexts_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                data = json.loads(line)
                # Each line is {context_id: [messages]}
                for context_id, messages in data.items():
                    self._contexts[context_id] = messages

    def _load_questions(self) -> None:
        """Load questions from CSV file."""
        self._questions_df = pd.read_csv(self.questions_path)

    def _build_samples(self) -> None:
        """Build PersonaMemSample objects from loaded data."""
        self.samples = []

        # Group questions by shared_context_id
        grouped = self._questions_df.groupby('shared_context_id')

        for context_id, group in grouped:
            if context_id not in self._contexts:
                print(f"Warning: Context {context_id} not found, skipping")
                continue

            messages = self._contexts[context_id]

            # Extract persona info from system message
            persona_info = ""
            if messages and messages[0]['role'] == 'system':
                persona_info = messages[0]['content']

            # Get persona_id from first question
            persona_id = int(group.iloc[0]['persona_id'])

            # Build questions list
            questions = []
            for _, row in group.iterrows():
                # Parse options JSON
                try:
                    options = json.loads(row['all_options'])
                except (json.JSONDecodeError, TypeError):
                    options = []

                q = PersonaMemQuestion(
                    question_id=str(row['question_id']),
                    question_type=row['question_type'],
                    topic=row['topic'],
                    user_question=row['user_question_or_message'],
                    correct_answer=row['correct_answer'],
                    all_options=options,
                    context_length_tokens=int(row['context_length_in_tokens']),
                    distance_to_ref_tokens=int(row['distance_to_ref_in_tokens']),
                    distance_proportion=str(row['distance_to_ref_proportion_in_context']),
                    end_index=int(row['end_index_in_shared_context'])
                )
                questions.append(q)

            sample = PersonaMemSample(
                context_id=context_id,
                persona_id=persona_id,
                persona_info=persona_info,
                context_messages=messages,
                questions=questions
            )
            self.samples.append(sample)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> PersonaMemSample:
        if not self._loaded:
            self.load()
        return self.samples[idx]

    def __iter__(self) -> Iterator[PersonaMemSample]:
        if not self._loaded:
            self.load()
        return iter(self.samples)

    def get_statistics(self) -> Dict[str, Any]:
        """Get dataset statistics."""
        if not self._loaded:
            self.load()

        total_questions = sum(s.num_questions for s in self.samples)
        total_turns = sum(s.num_dialogue_turns for s in self.samples)

        # Count by question type
        by_type = {t: 0 for t in self.QUESTION_TYPES}
        for sample in self.samples:
            for q in sample.questions:
                if q.question_type in by_type:
                    by_type[q.question_type] += 1

        # Count by topic
        by_topic = {t: 0 for t in self.TOPICS}
        for sample in self.samples:
            for q in sample.questions:
                if q.topic in by_topic:
                    by_topic[q.topic] += 1

        return {
            'num_samples': len(self.samples),
            'total_questions': total_questions,
            'total_dialogue_turns': total_turns,
            'avg_questions_per_sample': total_questions / len(self.samples) if self.samples else 0,
            'avg_turns_per_sample': total_turns / len(self.samples) if self.samples else 0,
            'questions_by_type': by_type,
            'questions_by_topic': by_topic
        }


def load_personamem(data_dir: str = None) -> PersonaMemDataset:
    """
    Convenience function to load the PersonaMem dataset.

    Args:
        data_dir: Optional data directory path

    Returns:
        Loaded PersonaMemDataset instance
    """
    import config

    if data_dir is None:
        data_dir = os.path.join(config.DATA_DIR, "personamem")

    dataset = PersonaMemDataset(data_dir=data_dir)
    dataset.load()
    return dataset


def process_messages_for_memory(sample: PersonaMemSample) -> List[Dict[str, Any]]:
    """
    Process dialogue messages for memory ingestion.

    Args:
        sample: PersonaMemSample instance

    Returns:
        List of processed messages with metadata
    """
    processed = []

    for idx, msg in enumerate(sample.context_messages):
        role = msg['role']
        content = msg['content']

        if role == 'system':
            # Skip system message (handled separately as persona)
            continue

        processed.append({
            'index': idx,
            'role': role,
            'content': content,
            'is_user': role == 'user',
            'speaker': 'User' if role == 'user' else 'Assistant'
        })

    return processed


if __name__ == "__main__":
    # Test loading
    print("Loading PersonaMem dataset...")
    dataset = load_personamem()

    print(f"\nDataset Statistics:")
    stats = dataset.get_statistics()
    for key, value in stats.items():
        if isinstance(value, dict):
            print(f"  {key}:")
            for k, v in value.items():
                print(f"    {k}: {v}")
        else:
            print(f"  {key}: {value}")

    if len(dataset) > 0:
        print(f"\nFirst sample:")
        sample = dataset[0]
        print(f"  Context ID: {sample.context_id}")
        print(f"  Persona ID: {sample.persona_id}")
        print(f"  Persona Info: {sample.persona_info[:200]}...")
        print(f"  Num Questions: {sample.num_questions}")
        print(f"  Num Dialogue Turns: {sample.num_dialogue_turns}")

        if sample.questions:
            print(f"\n  First Question:")
            q = sample.questions[0]
            print(f"    Type: {q.question_type}")
            print(f"    Topic: {q.topic}")
            print(f"    Question: {q.user_question[:100]}...")
            print(f"    Correct Answer: {q.correct_answer}")
            print(f"    Options count: {len(q.all_options)}")