"""
Evaluation Metrics for LoComo Benchmark
=======================================
Implements various answer matching metrics for evaluating memory-augmented QA.

Metrics:
- Exact Match (EM): Binary match after normalization
- Token F1: Token-level precision/recall/F1
- Contains Match: Check if reference is contained in prediction
- BLEU Score: N-gram based similarity (optional)
- LLM-as-Judge: Use LLM to evaluate semantic correctness
"""

import re
import string
from typing import List, Dict, Any, Tuple, Optional
from collections import Counter
import json

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))


def normalize_answer(text: str) -> str:
    """
    Normalize answer text for comparison.

    Operations:
    - Lowercase
    - Remove punctuation
    - Remove articles (a, an, the)
    - Remove extra whitespace
    """
    if not isinstance(text, str):
        text = str(text)

    # Lowercase
    text = text.lower()

    # Remove punctuation
    text = text.translate(str.maketrans('', '', string.punctuation))

    # Remove articles
    articles = ['a', 'an', 'the']
    words = text.split()
    words = [w for w in words if w not in articles]

    # Join and strip
    text = ' '.join(words).strip()

    return text


def exact_match(prediction: str, reference: str) -> float:
    """
    Exact match after normalization.

    Returns:
        1.0 if exact match, 0.0 otherwise
    """
    pred_norm = normalize_answer(prediction)
    ref_norm = normalize_answer(reference)
    return 1.0 if pred_norm == ref_norm else 0.0


def contains_match(prediction: str, reference: str) -> float:
    """
    Check if the reference answer is contained in the prediction.

    More lenient than exact match - useful when model gives verbose answers.

    Returns:
        1.0 if reference is contained in prediction, 0.0 otherwise
    """
    pred_norm = normalize_answer(prediction)
    ref_norm = normalize_answer(reference)

    # Check both directions for flexibility
    if ref_norm in pred_norm or pred_norm in ref_norm:
        return 1.0
    return 0.0


def token_f1(prediction: str, reference: str) -> Tuple[float, float, float]:
    """
    Compute token-level precision, recall, and F1 score.

    Returns:
        Tuple of (precision, recall, f1)
    """
    pred_tokens = normalize_answer(prediction).split()
    ref_tokens = normalize_answer(reference).split()

    if not pred_tokens or not ref_tokens:
        return (0.0, 0.0, 0.0)

    # Count common tokens
    pred_counter = Counter(pred_tokens)
    ref_counter = Counter(ref_tokens)

    # Intersection count
    common = sum((pred_counter & ref_counter).values())

    if common == 0:
        return (0.0, 0.0, 0.0)

    precision = common / len(pred_tokens)
    recall = common / len(ref_tokens)
    f1 = 2 * precision * recall / (precision + recall)

    return (precision, recall, f1)


def word_overlap_score(prediction: str, reference: str) -> float:
    """
    Compute word overlap ratio (Jaccard-like).

    Returns:
        Overlap ratio between 0 and 1
    """
    pred_tokens = set(normalize_answer(prediction).split())
    ref_tokens = set(normalize_answer(reference).split())

    if not pred_tokens or not ref_tokens:
        return 0.0

    intersection = pred_tokens & ref_tokens
    union = pred_tokens | ref_tokens

    return len(intersection) / len(union)


def numeric_match(prediction: str, reference: str) -> float:
    """
    Extract and compare numeric values (for temporal/quantity questions).

    Useful for dates, years, durations, etc.

    Returns:
        1.0 if numeric values match, 0.0 otherwise
    """
    # Extract numbers from both strings
    pred_numbers = re.findall(r'\d+', prediction)
    ref_numbers = re.findall(r'\d+', reference)

    if not pred_numbers or not ref_numbers:
        return 0.0

    # Check if any reference number appears in prediction numbers
    for ref_num in ref_numbers:
        if ref_num in pred_numbers:
            return 1.0

    return 0.0


async def llm_judge_match(
    prediction: str,
    reference: str,
    question: str,
    llm_client
) -> Tuple[float, str]:
    """
    Use LLM to judge if the prediction is semantically correct.

    This is more expensive but handles paraphrases and semantic equivalence.

    Returns:
        Tuple of (score 0.0-1.0, explanation)
    """
    prompt = f"""You are evaluating answer correctness for a memory-based QA task.

Question: {question}
Reference Answer: {reference}
Predicted Answer: {prediction}

Evaluate if the predicted answer is semantically correct compared to the reference.
Consider:
- Factual accuracy (most important)
- Semantic equivalence (different wording is OK)
- Completeness (partial credit for partial answers)

Respond with JSON:
{{
    "score": 0.0-1.0,
    "explanation": "brief explanation"
}}
"""

    try:
        response = await llm_client.chat_completion(
            messages=[{"role": "user", "content": prompt}],
            temperature=0.1,
            max_tokens=200
        )

        content = response["content"]
        if "```json" in content:
            start = content.find("```json") + 7
            end = content.find("```", start)
            content = content[start:end].strip()

        result = json.loads(content)
        return (result.get("score", 0.0), result.get("explanation", ""))

    except Exception as e:
        return (0.0, f"LLM judge failed: {str(e)}")


class LoCoMoEvaluator:
    """
    Comprehensive evaluator for LoComo benchmark.

    Computes multiple metrics and aggregates by category.
    """

    def __init__(self, use_llm_judge: bool = False, llm_client=None):
        """
        Initialize the evaluator.

        Args:
            use_llm_judge: Whether to use LLM-based semantic matching
            llm_client: LLM client for judge (required if use_llm_judge=True)
        """
        self.use_llm_judge = use_llm_judge
        self.llm_client = llm_client
        self._results: List[Dict[str, Any]] = []

    def evaluate_single(
        self,
        prediction: str,
        reference: str,
        question: str = "",
        category: int = 1
    ) -> Dict[str, float]:
        """
        Evaluate a single prediction against reference.

        Returns:
            Dict with all metric scores
        """
        metrics = {
            "exact_match": exact_match(prediction, reference),
            "contains_match": contains_match(prediction, reference),
            "word_overlap": word_overlap_score(prediction, reference),
            "numeric_match": numeric_match(prediction, reference),
        }

        # Token F1
        precision, recall, f1 = token_f1(prediction, reference)
        metrics["token_precision"] = precision
        metrics["token_recall"] = recall
        metrics["token_f1"] = f1

        return metrics

    async def evaluate_single_with_llm(
        self,
        prediction: str,
        reference: str,
        question: str,
        category: int = 1
    ) -> Dict[str, float]:
        """
        Evaluate with all metrics including LLM judge.
        """
        metrics = self.evaluate_single(prediction, reference, question, category)

        if self.use_llm_judge and self.llm_client:
            llm_score, explanation = await llm_judge_match(
                prediction, reference, question, self.llm_client
            )
            metrics["llm_judge"] = llm_score
            metrics["llm_explanation"] = explanation

        return metrics

    def add_result(
        self,
        sample_id: str,
        question: str,
        prediction: str,
        reference: str,
        category: int,
        metrics: Dict[str, float],
        metadata: Optional[Dict[str, Any]] = None
    ) -> None:
        """Add an evaluation result."""
        self._results.append({
            "sample_id": sample_id,
            "question": question,
            "prediction": prediction,
            "reference": reference,
            "category": category,
            "metrics": metrics,
            "metadata": metadata or {}
        })

    def get_aggregate_metrics(self) -> Dict[str, Any]:
        """
        Compute aggregate metrics across all results.

        Returns:
            Dict with overall and per-category metrics
        """
        if not self._results:
            return {"error": "No results to aggregate"}

        # Overall metrics
        overall = self._aggregate_metrics(self._results)

        # Per-category metrics
        by_category = {}
        category_names = {1: "single-hop", 2: "temporal", 3: "multi-hop", 4: "adversarial"}

        for cat_id, cat_name in category_names.items():
            cat_results = [r for r in self._results if r["category"] == cat_id]
            if cat_results:
                by_category[cat_name] = self._aggregate_metrics(cat_results)

        return {
            "overall": overall,
            "by_category": by_category,
            "total_samples": len(self._results)
        }

    def _aggregate_metrics(self, results: List[Dict]) -> Dict[str, float]:
        """Aggregate metrics from a list of results."""
        metric_keys = ["exact_match", "contains_match", "token_f1", "word_overlap", "numeric_match"]

        aggregated = {}
        for key in metric_keys:
            values = [r["metrics"].get(key, 0.0) for r in results]
            aggregated[key] = sum(values) / len(values) if values else 0.0

        aggregated["count"] = len(results)
        return aggregated

    def get_detailed_results(self) -> List[Dict[str, Any]]:
        """Get all detailed results."""
        return self._results

    def save_results(self, path: str) -> None:
        """Save results to a JSON file."""
        output = {
            "aggregate": self.get_aggregate_metrics(),
            "detailed": self._results
        }
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(output, f, ensure_ascii=False, indent=2)
        print(f"Results saved to {path}")

    def print_summary(self) -> None:
        """Print a summary of evaluation results."""
        agg = self.get_aggregate_metrics()

        print("\n" + "=" * 60)
        print("LoComo Evaluation Summary")
        print("=" * 60)

        print(f"\nTotal Samples: {agg['total_samples']}")

        print("\nOverall Metrics:")
        for key, value in agg["overall"].items():
            if key != "count":
                print(f"  {key}: {value:.4f}")

        print("\nBy Category:")
        for cat_name, metrics in agg.get("by_category", {}).items():
            print(f"\n  {cat_name} (n={metrics.get('count', 0)}):")
            for key, value in metrics.items():
                if key != "count":
                    print(f"    {key}: {value:.4f}")


def compute_accuracy(predictions: List[str], references: List[str]) -> Dict[str, float]:
    """
    Convenience function to compute accuracy metrics for lists.

    Returns:
        Dict with average metrics
    """
    if len(predictions) != len(references):
        raise ValueError("Predictions and references must have same length")

    metrics_sum = {
        "exact_match": 0.0,
        "contains_match": 0.0,
        "token_f1": 0.0
    }

    for pred, ref in zip(predictions, references):
        metrics_sum["exact_match"] += exact_match(pred, ref)
        metrics_sum["contains_match"] += contains_match(pred, ref)
        _, _, f1 = token_f1(pred, ref)
        metrics_sum["token_f1"] += f1

    n = len(predictions)
    return {k: v / n for k, v in metrics_sum.items()}


if __name__ == "__main__":
    # Demo: Test metrics
    test_cases = [
        ("Python", "python", "What programming language?"),
        ("I think it's Python", "Python", "What language?"),
        ("15 July 2023", "July 15, 2023", "When did it happen?"),
        ("machine learning", "ML and AI", "What field?"),
        ("4 years", "four years", "How long?"),
    ]

    print("Metric Test Cases:")
    print("=" * 60)

    for pred, ref, q in test_cases:
        print(f"\nPrediction: '{pred}'")
        print(f"Reference:  '{ref}'")
        print(f"  EM: {exact_match(pred, ref):.2f}")
        print(f"  Contains: {contains_match(pred, ref):.2f}")
        print(f"  F1: {token_f1(pred, ref)[2]:.2f}")
        print(f"  Numeric: {numeric_match(pred, ref):.2f}")
