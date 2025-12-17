"""
LoComo Experiment Runner
========================
Run evaluation experiments on the LoComo benchmark using the PCM memory system.

Usage:
    python run_experiment.py                    # Run full experiment
    python run_experiment.py --sample 0         # Run single sample
    python run_experiment.py --mock             # Run with mock LLM
    python run_experiment.py --skip-ingest      # Skip memory ingestion (use cached)
"""

import asyncio
import argparse
import json
import os
from datetime import datetime
from typing import Dict, Any, List, Optional
from tqdm import tqdm

import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import config
from src.core.orchestrator import PCMSystem, create_pcm_system
from src.core.types import NodeType
from src.evaluation.dataset import LoCoMoDataset, LoCoMoSample, load_locomo, process_dialogues_for_memory
from src.evaluation.metrics import LoCoMoEvaluator


class LoCoMoExperiment:
    """
    Experiment runner for LoComo benchmark evaluation.

    Workflow:
    1. Load dataset
    2. For each sample:
       a. Ingest dialogues into memory
       b. Answer each QA question
       c. Evaluate answers
    3. Aggregate and report results
    """

    def __init__(
        self,
        output_dir: str = "./experiment_results",
        use_mock: bool = False
    ):
        """
        Initialize the experiment.

        Args:
            output_dir: Directory to save results
            use_mock: Use mock LLM for testing
        """
        self.output_dir = output_dir
        self.use_mock = use_mock
        self.dataset: Optional[LoCoMoDataset] = None
        self.evaluator = LoCoMoEvaluator()

        os.makedirs(output_dir, exist_ok=True)

    async def run_single_sample(
        self,
        sample_idx: int,
        skip_ingest: bool = False,
        verbose: bool = True
    ) -> Dict[str, Any]:
        """
        Run experiment on a single sample.

        Args:
            sample_idx: Index of the sample to process
            skip_ingest: Skip memory ingestion (for cached experiments)
            verbose: Print progress

        Returns:
            Dict with sample results
        """
        # Load dataset if needed
        if self.dataset is None:
            self.dataset = load_locomo()

        sample = self.dataset[sample_idx]

        if verbose:
            print(f"\n{'=' * 60}")
            print(f"Processing Sample {sample_idx}: {sample.sample_id}")
            print(f"Speakers: {sample.speaker_a} <-> {sample.speaker_b}")
            print(f"Sessions: {len(sample.sessions)}, Turns: {sample.total_turns}")
            print(f"QA Pairs: {sample.total_qa}")
            print(f"{'=' * 60}")

        # Create PCM system for this sample
        sample_data_dir = os.path.join(self.output_dir, f"sample_{sample_idx}")
        os.makedirs(sample_data_dir, exist_ok=True)

        pcm = create_pcm_system(
            data_dir=sample_data_dir,
            use_mock=self.use_mock
        )

        # Ingest dialogues
        if not skip_ingest:
            await self._ingest_dialogues(pcm, sample, verbose)

        # Answer questions
        results = await self._answer_questions(pcm, sample, verbose)

        # Save sample results
        sample_result_path = os.path.join(sample_data_dir, "results.json")
        with open(sample_result_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)

        if verbose:
            print(f"\nSample {sample_idx} results saved to {sample_result_path}")

        return results

    async def _ingest_dialogues(
        self,
        pcm: PCMSystem,
        sample: LoCoMoSample,
        verbose: bool
    ) -> None:
        """Ingest all dialogues from a sample into the memory system."""
        if verbose:
            print("\nIngesting dialogues into memory...")

        dialogues = process_dialogues_for_memory(sample)
        speaker_a = sample.speaker_a

        for i, dialogue in enumerate(tqdm(dialogues, desc="Ingesting", disable=not verbose)):
            # Add to world model
            content = f"[{dialogue['timestamp']}] {dialogue['speaker']}: {dialogue['message']}"

            node_type = NodeType.FACT
            domain = "Personal"  # Could use intent classification here

            await pcm.add_knowledge(
                content=content,
                node_type=node_type,
                domain=domain,
                weight=0.7
            )

            # Also process through perception layer for context
            if dialogue['is_user']:
                # This builds the working memory context
                await pcm.process_input(
                    dialogue['message'],
                    generate_response=False
                )

        # Save the world model
        pcm.world_model.save()

        if verbose:
            stats = pcm.world_model.get_statistics()
            print(f"Memory ingestion complete. Nodes: {stats['num_nodes']}")

    async def _answer_questions(
        self,
        pcm: PCMSystem,
        sample: LoCoMoSample,
        verbose: bool
    ) -> Dict[str, Any]:
        """Answer all QA questions for a sample."""
        if verbose:
            print("\nAnswering questions...")

        results = {
            "sample_id": sample.sample_id,
            "speaker_a": sample.speaker_a,
            "speaker_b": sample.speaker_b,
            "qa_results": [],
            "timestamp": datetime.now().isoformat()
        }

        for i, qa in enumerate(tqdm(sample.qa_pairs, desc="QA", disable=not verbose)):
            # Retrieve relevant context
            relevant = pcm.query_knowledge(qa.question, top_k=10)

            # Build context for answer generation
            context_str = "\n".join([
                f"- {item['content'][:200]}"
                for item in relevant[:5]
            ])

            # Generate answer using PCM
            answer_prompt = f"""Based on the memory context, answer this question concisely.

Memory Context:
{context_str}

Question: {qa.question}

Provide ONLY the answer, no explanation. Be concise.
For dates, use format like "15 July 2023".
For durations, use format like "4 years" or "3 months".
"""

            try:
                response = await pcm.llm_client.chat_completion(
                    messages=[{"role": "user", "content": answer_prompt}],
                    temperature=0.3,
                    max_tokens=100
                )
                prediction = response["content"].strip()
            except Exception as e:
                prediction = f"Error: {str(e)}"

            # Evaluate
            metrics = self.evaluator.evaluate_single(
                prediction=prediction,
                reference=qa.answer,
                question=qa.question,
                category=qa.category
            )

            # Record result
            qa_result = {
                "question": qa.question,
                "reference_answer": qa.answer,
                "predicted_answer": prediction,
                "category": qa.category,
                "category_name": qa.category_name,
                "evidence": qa.evidence,
                "metrics": metrics,
                "retrieved_context": [item['content'][:100] for item in relevant[:3]]
            }
            results["qa_results"].append(qa_result)

            # Add to evaluator
            self.evaluator.add_result(
                sample_id=sample.sample_id,
                question=qa.question,
                prediction=prediction,
                reference=qa.answer,
                category=qa.category,
                metrics=metrics
            )

            if verbose and i < 3:  # Show first 3 QA pairs
                print(f"\n  Q: {qa.question[:60]}...")
                print(f"  Ref: {qa.answer}")
                print(f"  Pred: {prediction}")
                print(f"  EM: {metrics['exact_match']:.2f}, F1: {metrics['token_f1']:.2f}")

        return results

    async def run_all_samples(
        self,
        max_samples: Optional[int] = None,
        skip_ingest: bool = False
    ) -> Dict[str, Any]:
        """
        Run experiment on all samples.

        Args:
            max_samples: Maximum samples to process (None for all)
            skip_ingest: Skip memory ingestion

        Returns:
            Aggregate results
        """
        if self.dataset is None:
            self.dataset = load_locomo()

        n_samples = min(len(self.dataset), max_samples) if max_samples else len(self.dataset)

        print(f"\n{'=' * 60}")
        print(f"Running LoComo Experiment")
        print(f"Samples: {n_samples}, Mock: {self.use_mock}")
        print(f"Output: {self.output_dir}")
        print(f"{'=' * 60}")

        all_results = []

        for idx in range(n_samples):
            try:
                result = await self.run_single_sample(
                    sample_idx=idx,
                    skip_ingest=skip_ingest,
                    verbose=True
                )
                all_results.append(result)
            except Exception as e:
                print(f"Error processing sample {idx}: {e}")
                import traceback
                traceback.print_exc()

        # Aggregate results
        aggregate = self.evaluator.get_aggregate_metrics()

        # Save aggregate results
        aggregate_path = os.path.join(self.output_dir, "aggregate_results.json")
        with open(aggregate_path, 'w', encoding='utf-8') as f:
            json.dump({
                "aggregate": aggregate,
                "config": {
                    "use_mock": self.use_mock,
                    "n_samples": n_samples,
                    "timestamp": datetime.now().isoformat()
                }
            }, f, ensure_ascii=False, indent=2)

        # Print summary
        self.evaluator.print_summary()

        # Save detailed results
        self.evaluator.save_results(
            os.path.join(self.output_dir, "detailed_results.json")
        )

        return aggregate


async def main():
    parser = argparse.ArgumentParser(description="LoComo Benchmark Experiment")
    parser.add_argument("--sample", type=int, default=None, help="Run single sample by index")
    parser.add_argument("--max-samples", type=int, default=None, help="Maximum samples to process")
    parser.add_argument("--mock", action="store_true", help="Use mock LLM")
    parser.add_argument("--skip-ingest", action="store_true", help="Skip memory ingestion")
    parser.add_argument("--output", type=str, default="./experiment_results", help="Output directory")
    args = parser.parse_args()

    # Set mock mode
    if args.mock:
        config.USE_MOCK_LLM = True

    experiment = LoCoMoExperiment(
        output_dir=args.output,
        use_mock=args.mock
    )

    if args.sample is not None:
        # Run single sample
        await experiment.run_single_sample(
            sample_idx=args.sample,
            skip_ingest=args.skip_ingest
        )
    else:
        # Run all samples
        await experiment.run_all_samples(
            max_samples=args.max_samples,
            skip_ingest=args.skip_ingest
        )


if __name__ == "__main__":
    asyncio.run(main())
