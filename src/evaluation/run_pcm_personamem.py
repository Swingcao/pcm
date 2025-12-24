"""
PCM Memory System Evaluation on PersonaMem Dataset
===================================================

This script evaluates the PCM (Proactive Cognitive Memory) system on the
PersonaMem benchmark dataset.

Workflow:
1. Load questions from CSV and contexts from JSONL
2. For each question:
   - Truncate dialogue context to end_index_in_shared_context
   - Ingest truncated dialogue into PCM memory system
   - Query PCM for relevant context
   - Use retrieved context + question to prompt LLM
   - Extract answer and evaluate against ground truth
3. Save results to CSV

Usage:
    python src/evaluation/run_pcm_personamem.py
    python src/evaluation/run_pcm_personamem.py --mock
    python src/evaluation/run_pcm_personamem.py --max_questions 10
"""

import asyncio
import argparse
import csv
import json
import os
import sys
import re
from datetime import datetime
from tqdm import tqdm

# Add project root to path
_project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

import config
from src.core.orchestrator import create_pcm_system
from src.core.types import NodeType


def generate_experiment_suffix(experiment_name: str = None) -> str:
    """
    Generate a unique experiment suffix.

    Args:
        experiment_name: User-provided experiment name. If None, generates timestamp.

    Returns:
        Suffix string for experiment identification
    """
    if experiment_name:
        return experiment_name
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def build_jsonl_index(jsonl_path: str) -> dict:
    """
    Build an index mapping context IDs to file offsets for fast lookup.

    Args:
        jsonl_path: Path to the JSONL file

    Returns:
        Dict mapping context_id to file offset
    """
    index = {}
    with open(jsonl_path, 'r', encoding='utf-8') as f:
        while True:
            offset = f.tell()
            line = f.readline()
            if not line:
                break
            try:
                key = next(iter(json.loads(line).keys()))
                index[key] = offset
            except (json.JSONDecodeError, StopIteration):
                continue
    return index


def load_context_by_id(jsonl_path: str, offset: int) -> list:
    """
    Load a context by seeking to its file offset.

    Args:
        jsonl_path: Path to the JSONL file
        offset: File offset to seek to

    Returns:
        List of message dicts
    """
    with open(jsonl_path, 'r', encoding='utf-8') as f:
        f.seek(offset)
        item = json.loads(f.readline())
        return next(iter(item.values()))


def load_rows_with_context(csv_path: str, jsonl_path: str, max_rows: int = None):
    """
    Iterator yielding (row_data, context) pairs.

    Efficiently handles shared contexts by caching the previous context.

    Args:
        csv_path: Path to questions CSV
        jsonl_path: Path to contexts JSONL
        max_rows: Maximum number of rows to yield (None for all)

    Yields:
        Tuples of (row_data dict, context list)
    """
    jsonl_index = build_jsonl_index(jsonl_path)

    with open(csv_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        prev_sid, prev_context = None, None
        count = 0

        for row in reader:
            if max_rows is not None and count >= max_rows:
                break

            sid = row["shared_context_id"]
            if sid != prev_sid:
                if sid in jsonl_index:
                    prev_context = load_context_by_id(jsonl_path, jsonl_index[sid])
                else:
                    print(f"Warning: Context {sid} not found in index")
                    continue
                prev_sid = sid

            yield row, prev_context
            count += 1


def count_csv_rows(csv_path: str) -> int:
    """Count rows in CSV file (excluding header)."""
    with open(csv_path, 'r', encoding='utf-8') as f:
        return sum(1 for _ in f) - 1


def extract_answer(response: str, correct_answer: str) -> tuple:
    """
    Extract answer from LLM response.

    Compatible with inference_personamem.py logic:
    - Looks for <final_answer> token
    - Extracts (a), (b), (c), or (d)

    Args:
        response: Full LLM response text
        correct_answer: Ground truth answer e.g., "(c)"

    Returns:
        Tuple of (is_correct: bool, extracted_answer: str)
    """
    def _extract_options(text: str) -> set:
        text = text.lower()
        # First try to find options in parentheses
        in_parens = re.findall(r'\(([a-d])\)', text)
        if in_parens:
            return set(in_parens)
        # Fallback to standalone letters
        return set(re.findall(r'\b([a-d])\b', text))

    correct = correct_answer.lower().strip("() ")

    # Extract from <final_answer> if present
    predicted = response.strip()
    if "<final_answer>" in predicted:
        predicted = predicted.split("<final_answer>")[-1].strip()
    if predicted.endswith("</final_answer>"):
        predicted = predicted[:-len("</final_answer>")].strip()

    pred_options = _extract_options(predicted)

    # Check if prediction matches correct answer
    if pred_options == {correct}:
        return True, predicted

    # Fallback: check full response
    full_options = _extract_options(response)
    if full_options == {correct}:
        return True, predicted

    return False, predicted


async def ingest_context_to_pcm(pcm, context_messages: list) -> int:
    """
    Ingest dialogue messages into PCM memory system.

    Args:
        pcm: PCM system instance
        context_messages: List of message dicts with 'role' and 'content'

    Returns:
        Number of messages ingested
    """
    count = 0
    for msg in context_messages:
        role = msg.get('role', '')
        content = msg.get('content', '')

        if not content:
            continue

        if role == 'system':
            # Inject persona as high-weight node
            await pcm.add_knowledge(
                content=f"[Persona] {content}",
                node_type=NodeType.FACT,
                domain="Persona",
                weight=1.0
            )
        else:
            # Inject dialogue with speaker tag
            speaker = "User" if role == "user" else "Assistant"
            await pcm.add_knowledge(
                content=f"[{speaker}] {content}",
                node_type=NodeType.FACT,
                domain="Dialogue",
                weight=0.7
            )
        count += 1

    return count


async def run_evaluation(args):
    """
    Main evaluation loop.

    Args:
        args: Command line arguments
    """
    question_path = args.question_path
    context_path = args.context_path
    result_path = args.result_path

    print(f"\n{'=' * 60}")
    print(f"PCM PersonaMem Evaluation")
    print(f"{'=' * 60}")
    print(f"Questions: {question_path}")
    print(f"Contexts: {context_path}")
    print(f"Results: {result_path}")
    print(f"Top-K: {args.top_k}")
    print(f"Mock LLM: {args.mock}")
    print(f"{'=' * 60}\n")

    # Ensure results directory exists (no longer deleting existing results)
    os.makedirs(os.path.dirname(result_path) if os.path.dirname(result_path) else '.', exist_ok=True)
    os.makedirs(args.results_dir, exist_ok=True)

    # Count total rows
    total_rows = count_csv_rows(question_path)
    if args.max_questions:
        total_rows = min(total_rows, args.max_questions)

    print(f"Total questions to evaluate: {total_rows}")

    errors = []
    correct_count = 0
    total_count = 0

    for row_data, full_context in tqdm(
        load_rows_with_context(question_path, context_path, args.max_questions),
        total=total_rows,
        desc="Evaluating"
    ):
        try:
            # Extract row data
            question_id = row_data["question_id"]
            question = row_data["user_question_or_message"]
            correct_answer = row_data["correct_answer"]
            all_options = row_data["all_options"]
            end_index = int(row_data["end_index_in_shared_context"])

            # Truncate context to end_index (key evaluation logic)
            context = full_context[:end_index]

            # Create fresh PCM for this question
            pcm = create_pcm_system(
                results_dir=os.path.join(args.results_dir, f"q_{question_id[:8]}"),
                use_mock=args.mock
            )

            # Ingest truncated dialogue into PCM
            await ingest_context_to_pcm(pcm, context)

            # Query relevant context from memory
            retrieved = pcm.query_knowledge(question, top_k=args.top_k)

            # Build retrieved context string
            if retrieved:
                retrieved_str = "\n".join([
                    f"- {item['content'][:300]}"
                    for item in retrieved[:5]
                ])
            else:
                retrieved_str = "No relevant context found in memory."

            # Build prompt (compatible with inference_personamem.py)
            instructions = (
                "Based on the memory context above, select the most appropriate response. "
                "Give your final answer as (a), (b), (c), or (d) after the token <final_answer>."
            )

            prompt = f"""Memory Context:
{retrieved_str}

Question: {question}

{instructions}

Options:
{all_options}"""

            # Query LLM
            response = await pcm.llm_client.chat_completion(
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1,
                max_tokens=500
            )
            model_response = response.get("content", "")

            # Extract answer
            score, predicted = extract_answer(model_response, correct_answer)

            if score:
                correct_count += 1
            total_count += 1

            # Save result to CSV
            with open(result_path, 'a', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)

                # Write header if file is empty
                if os.stat(result_path).st_size == 0:
                    writer.writerow([
                        "score", "persona_id", "question_id",
                        "user_question_or_message", "question_type", "topic",
                        "context_length_in_tokens", "distance_to_ref_in_tokens",
                        "distance_to_ref_proportion_in_context",
                        "model_response", "len_of_model_response",
                        "predicted_answer", "correct_answer",
                        "num_retrieved", "retrieved_context_preview"
                    ])

                writer.writerow([
                    score,
                    row_data["persona_id"],
                    question_id,
                    question[:100],
                    row_data["question_type"],
                    row_data["topic"],
                    row_data["context_length_in_tokens"],
                    row_data["distance_to_ref_in_tokens"],
                    row_data["distance_to_ref_proportion_in_context"],
                    model_response[:500],
                    len(model_response),
                    predicted,
                    correct_answer,
                    len(retrieved) if retrieved else 0,
                    retrieved_str[:200]
                ])

            # Cleanup PCM to free memory
            del pcm

        except Exception as e:
            errors.append({
                "question_id": row_data.get("question_id", "unknown"),
                "error": str(e)
            })
            import traceback
            if args.verbose:
                traceback.print_exc()
            continue

    # Print summary
    print(f"\n{'=' * 60}")
    print(f"Evaluation Complete")
    print(f"{'=' * 60}")
    print(f"Total Questions: {total_count}")
    print(f"Correct: {correct_count}")
    print(f"Accuracy: {correct_count / total_count:.2%}" if total_count > 0 else "N/A")
    print(f"Errors: {len(errors)}")
    print(f"Results saved to: {result_path}")

    # Report errors
    if errors:
        print(f"\nErrors encountered:")
        for err in errors[:10]:
            print(f"  {err['question_id']}: {err['error']}")
        if len(errors) > 10:
            print(f"  ... and {len(errors) - 10} more errors")


async def main():
    parser = argparse.ArgumentParser(
        description="PCM Memory System Evaluation on PersonaMem Dataset"
    )

    # Data paths
    parser.add_argument(
        "--question_path",
        default="data/personamem/questions_32k.csv",
        help="Path to questions CSV file"
    )
    parser.add_argument(
        "--context_path",
        default="data/personamem/shared_contexts_32k.jsonl",
        help="Path to shared contexts JSONL file"
    )
    parser.add_argument(
        "--result_path",
        default="results/pcm_personamem_results.csv",
        help="Path to save results CSV"
    )
    parser.add_argument(
        "--results_dir",
        default="results/pcm_personamem",
        help="Directory for intermediate PCM results"
    )

    # Evaluation settings
    parser.add_argument(
        "--top_k",
        type=int,
        default=10,
        help="Number of contexts to retrieve from memory"
    )
    parser.add_argument(
        "--max_questions",
        type=int,
        default=None,
        help="Maximum number of questions to evaluate (for testing)"
    )

    # Debug settings
    parser.add_argument(
        "--mock",
        action="store_true",
        help="Use mock LLM for testing"
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print detailed error traces"
    )

    # Experiment versioning
    parser.add_argument(
        "--experiment_name",
        type=str,
        default=None,
        help="Experiment identifier (e.g., 'v1', 'baseline'). If not specified, uses timestamp."
    )

    args = parser.parse_args()

    # Generate experiment suffix and update paths
    experiment_suffix = generate_experiment_suffix(args.experiment_name)

    # Update result_path with experiment suffix
    base_name, extension = args.result_path.rsplit('.', 1)
    args.result_path = f"{base_name}_{experiment_suffix}.{extension}"

    # Update results_dir with experiment suffix
    args.results_dir = f"{args.results_dir}_{experiment_suffix}"

    # Run evaluation
    await run_evaluation(args)


if __name__ == "__main__":
    asyncio.run(main())
