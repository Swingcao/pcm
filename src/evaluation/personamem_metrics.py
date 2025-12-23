"""
PersonaMem Evaluation Metrics
=============================
Evaluation metrics for multiple-choice question answering on PersonaMem dataset.

Main metrics:
- Accuracy: Exact match of predicted option vs correct option
- Per-type accuracy: Accuracy broken down by question type
- Per-topic accuracy: Accuracy broken down by topic
"""

import re
import json
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass, field
from collections import defaultdict


def extract_option(text: str) -> Optional[str]:
    """
    Extract option letter from LLM response.

    Handles various formats:
    - "(a)", "(b)", "(c)", "(d)"
    - "a", "b", "c", "d"
    - "A", "B", "C", "D"
    - "Option (a)", "Answer: (b)", etc.

    Args:
        text: LLM response text

    Returns:
        Normalized option string like "(a)" or None if not found
    """
    if not text:
        return None

    text = text.strip().lower()

    # Try to find pattern like (a), (b), (c), (d)
    match = re.search(r'\(([abcd])\)', text)
    if match:
        return f"({match.group(1)})"

    # Try to find standalone letter at the beginning
    match = re.match(r'^([abcd])[\.\)\s:]', text)
    if match:
        return f"({match.group(1)})"

    # Try to find "option a" or "answer a" pattern
    match = re.search(r'(?:option|answer|choice)[\s:]*([abcd])', text)
    if match:
        return f"({match.group(1)})"

    # Try to find just the letter
    match = re.search(r'\b([abcd])\b', text)
    if match:
        return f"({match.group(1)})"

    return None


def normalize_option(option: str) -> str:
    """
    Normalize option string to standard format.

    Args:
        option: Option string in any format

    Returns:
        Normalized option like "(a)"
    """
    if not option:
        return ""

    option = option.strip().lower()

    # Already in correct format
    if option in ['(a)', '(b)', '(c)', '(d)']:
        return option

    # Single letter
    if option in ['a', 'b', 'c', 'd']:
        return f"({option})"

    # Try to extract
    extracted = extract_option(option)
    return extracted if extracted else ""


@dataclass
class QuestionResult:
    """Result of a single question evaluation."""
    question_id: str
    question_type: str
    topic: str
    question_text: str
    correct_answer: str
    predicted_answer: str
    is_correct: bool
    option_scores: Optional[Dict[str, float]] = None  # Scores for each option
    retrieved_context: Optional[List[str]] = None


class PersonaMemEvaluator:
    """
    Evaluator for PersonaMem multiple-choice questions.

    Usage:
        evaluator = PersonaMemEvaluator()

        # Add results
        evaluator.add_result(
            question_id="...",
            question_type="recall_user_shared_facts",
            topic="musicRecommendation",
            question_text="...",
            correct_answer="(a)",
            predicted_answer="(b)",
            option_scores={...}
        )

        # Get metrics
        metrics = evaluator.get_aggregate_metrics()
        evaluator.print_summary()
    """

    # Question type display names
    TYPE_NAMES = {
        'recall_user_shared_facts': 'Recall Facts',
        'provide_preference_aligned_recommendations': 'Preference Recommendations',
        'suggest_new_ideas': 'Suggest Ideas',
        'recalling_the_reasons_behind_previous_updates': 'Recall Reasons',
        'track_full_preference_evolution': 'Track Preference',
        'generalizing_to_new_scenarios': 'Generalize'
    }

    # Topic display names
    TOPIC_NAMES = {
        'musicRecommendation': 'Music',
        'bookRecommendation': 'Book',
        'datingConsultation': 'Dating',
        'studyConsultation': 'Study',
        'medicalConsultation': 'Medical'
    }

    def __init__(self):
        self.results: List[QuestionResult] = []

    def add_result(
        self,
        question_id: str,
        question_type: str,
        topic: str,
        question_text: str,
        correct_answer: str,
        predicted_answer: str,
        option_scores: Optional[Dict[str, float]] = None,
        retrieved_context: Optional[List[str]] = None
    ) -> QuestionResult:
        """
        Add a question result.

        Args:
            question_id: Question identifier
            question_type: Type of question
            topic: Topic category
            question_text: The question text
            correct_answer: Correct option (a)/(b)/(c)/(d)
            predicted_answer: Predicted option
            option_scores: Optional scores for each option
            retrieved_context: Optional retrieved context snippets

        Returns:
            The created QuestionResult
        """
        # Normalize answers
        correct_norm = normalize_option(correct_answer)
        predicted_norm = normalize_option(predicted_answer)

        is_correct = correct_norm == predicted_norm and correct_norm != ""

        result = QuestionResult(
            question_id=question_id,
            question_type=question_type,
            topic=topic,
            question_text=question_text,
            correct_answer=correct_norm,
            predicted_answer=predicted_norm,
            is_correct=is_correct,
            option_scores=option_scores,
            retrieved_context=retrieved_context
        )

        self.results.append(result)
        return result

    def evaluate_single(
        self,
        prediction: str,
        correct_answer: str
    ) -> Dict[str, Any]:
        """
        Evaluate a single prediction.

        Args:
            prediction: Predicted option or text
            correct_answer: Correct option

        Returns:
            Dict with accuracy and normalized options
        """
        pred_norm = normalize_option(prediction)
        correct_norm = normalize_option(correct_answer)

        is_correct = pred_norm == correct_norm and pred_norm != ""

        return {
            'accuracy': 1.0 if is_correct else 0.0,
            'is_correct': is_correct,
            'predicted_option': pred_norm,
            'correct_option': correct_norm,
            'extraction_success': pred_norm != ""
        }

    def get_aggregate_metrics(self) -> Dict[str, Any]:
        """
        Calculate aggregate metrics.

        Returns:
            Dict with overall and per-category metrics
        """
        if not self.results:
            return {
                'overall': {'accuracy': 0.0, 'count': 0},
                'by_question_type': {},
                'by_topic': {},
                'total_questions': 0
            }

        # Overall accuracy
        correct_count = sum(1 for r in self.results if r.is_correct)
        total_count = len(self.results)
        overall_accuracy = correct_count / total_count if total_count > 0 else 0.0

        # By question type
        by_type = defaultdict(lambda: {'correct': 0, 'total': 0})
        for r in self.results:
            by_type[r.question_type]['total'] += 1
            if r.is_correct:
                by_type[r.question_type]['correct'] += 1

        type_metrics = {}
        for qtype, counts in by_type.items():
            type_metrics[qtype] = {
                'accuracy': counts['correct'] / counts['total'] if counts['total'] > 0 else 0.0,
                'correct': counts['correct'],
                'total': counts['total']
            }

        # By topic
        by_topic = defaultdict(lambda: {'correct': 0, 'total': 0})
        for r in self.results:
            by_topic[r.topic]['total'] += 1
            if r.is_correct:
                by_topic[r.topic]['correct'] += 1

        topic_metrics = {}
        for topic, counts in by_topic.items():
            topic_metrics[topic] = {
                'accuracy': counts['correct'] / counts['total'] if counts['total'] > 0 else 0.0,
                'correct': counts['correct'],
                'total': counts['total']
            }

        # Extraction success rate
        extraction_success = sum(1 for r in self.results if r.predicted_answer != "")
        extraction_rate = extraction_success / total_count if total_count > 0 else 0.0

        return {
            'overall': {
                'accuracy': overall_accuracy,
                'correct': correct_count,
                'total': total_count
            },
            'by_question_type': type_metrics,
            'by_topic': topic_metrics,
            'extraction_success_rate': extraction_rate,
            'total_questions': total_count
        }

    def get_detailed_results(self) -> List[Dict[str, Any]]:
        """Get detailed results for all questions."""
        return [
            {
                'question_id': r.question_id,
                'question_type': r.question_type,
                'topic': r.topic,
                'question_text': r.question_text,
                'correct_answer': r.correct_answer,
                'predicted_answer': r.predicted_answer,
                'is_correct': r.is_correct,
                'option_scores': r.option_scores,
                'retrieved_context': r.retrieved_context
            }
            for r in self.results
        ]

    def save_results(self, path: str) -> None:
        """Save detailed results to JSON file."""
        data = {
            'aggregate_metrics': self.get_aggregate_metrics(),
            'detailed_results': self.get_detailed_results()
        }
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

    def print_summary(self) -> None:
        """Print evaluation summary."""
        metrics = self.get_aggregate_metrics()

        print("\n" + "=" * 60)
        print("PersonaMem Evaluation Summary")
        print("=" * 60)

        # Overall
        overall = metrics['overall']
        print(f"\nOverall Accuracy: {overall['accuracy']:.2%} ({overall['correct']}/{overall['total']})")
        print(f"Extraction Success Rate: {metrics['extraction_success_rate']:.2%}")

        # By question type
        print("\nAccuracy by Question Type:")
        print("-" * 50)
        for qtype, data in metrics['by_question_type'].items():
            display_name = self.TYPE_NAMES.get(qtype, qtype)
            print(f"  {display_name:40} {data['accuracy']:6.2%} ({data['correct']:3}/{data['total']:3})")

        # By topic
        print("\nAccuracy by Topic:")
        print("-" * 50)
        for topic, data in metrics['by_topic'].items():
            display_name = self.TOPIC_NAMES.get(topic, topic)
            print(f"  {display_name:20} {data['accuracy']:6.2%} ({data['correct']:3}/{data['total']:3})")

        print("=" * 60)


async def score_option_with_llm(
    option: str,
    context: str,
    question: str,
    llm_client,
    max_score: int = 10
) -> float:
    """
    Score a single option using LLM.

    Args:
        option: The option text to score
        context: Retrieved context from memory
        question: The question being asked
        llm_client: LLM client for scoring
        max_score: Maximum score value

    Returns:
        Score from 0 to max_score
    """
    prompt = f"""Based on the following memory context, evaluate how well this response option answers the question.

Memory Context:
{context}

Question: {question}

Response Option:
{option}

Score this response from 0 to {max_score}:
- {max_score}: Perfect answer that directly addresses the question using information from the context
- {max_score//2}: Partially correct or relevant answer
- 0: Completely wrong or irrelevant answer

Output ONLY the numeric score (integer from 0 to {max_score}), nothing else."""

    try:
        response = await llm_client.chat_completion(
            messages=[{"role": "user", "content": prompt}],
            temperature=0.1,
            max_tokens=10
        )
        score_text = response["content"].strip()

        # Extract number
        match = re.search(r'(\d+)', score_text)
        if match:
            score = int(match.group(1))
            return min(max(score, 0), max_score)
        return 0.0
    except Exception as e:
        print(f"Error scoring option: {e}")
        return 0.0


async def score_all_options(
    options: List[str],
    context: str,
    question: str,
    llm_client
) -> List[Tuple[int, str, float]]:
    """
    Score all options and rank them.

    Args:
        options: List of option texts
        context: Retrieved context from memory
        question: The question being asked
        llm_client: LLM client for scoring

    Returns:
        List of (index, option, score) sorted by score descending
    """
    scored = []

    for idx, option in enumerate(options):
        score = await score_option_with_llm(option, context, question, llm_client)
        scored.append((idx, option, score))

    # Sort by score descending
    scored.sort(key=lambda x: x[2], reverse=True)
    return scored


def index_to_option(idx: int) -> str:
    """Convert index (0-3) to option string like (a)."""
    mapping = {0: '(a)', 1: '(b)', 2: '(c)', 3: '(d)'}
    return mapping.get(idx, '')


def option_to_index(option: str) -> int:
    """Convert option string like (a) to index (0-3)."""
    mapping = {'(a)': 0, '(b)': 1, '(c)': 2, '(d)': 3}
    return mapping.get(normalize_option(option), -1)


if __name__ == "__main__":
    # Test extraction
    test_cases = [
        ("(a)", "(a)"),
        ("(B)", "(b)"),
        ("c", "(c)"),
        ("Answer: (d)", "(d)"),
        ("I think the answer is a", "(a)"),
        ("Option (b) is correct", "(b)"),
        ("The best choice would be C.", "(c)"),
        ("", None),
        ("No clear answer", None),
    ]

    print("Testing option extraction:")
    for text, expected in test_cases:
        result = extract_option(text)
        status = "✓" if result == expected else "✗"
        print(f"  {status} '{text}' -> {result} (expected: {expected})")

    # Test evaluator
    print("\nTesting evaluator:")
    evaluator = PersonaMemEvaluator()

    evaluator.add_result(
        question_id="q1",
        question_type="recall_user_shared_facts",
        topic="musicRecommendation",
        question_text="Test question 1",
        correct_answer="(a)",
        predicted_answer="(a)"
    )

    evaluator.add_result(
        question_id="q2",
        question_type="recall_user_shared_facts",
        topic="bookRecommendation",
        question_text="Test question 2",
        correct_answer="(b)",
        predicted_answer="(c)"
    )

    evaluator.add_result(
        question_id="q3",
        question_type="suggest_new_ideas",
        topic="musicRecommendation",
        question_text="Test question 3",
        correct_answer="(c)",
        predicted_answer="(c)"
    )

    evaluator.print_summary()