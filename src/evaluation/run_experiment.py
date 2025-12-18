"""
LoComo Experiment Runner
========================
Run evaluation experiments on the LoComo benchmark using the PCM memory system.

All results are saved to the unified results folder in JSON format.

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
import sys
from datetime import datetime
from typing import Dict, Any, List, Optional
from tqdm import tqdm

# Add project root to path only if not already there
_project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

import config
from src.core.orchestrator import PCMSystem, create_pcm_system
from src.core.types import NodeType
from src.evaluation.dataset import LoCoMoDataset, LoCoMoSample, load_locomo, process_dialogues_for_memory
from src.evaluation.metrics import LoCoMoEvaluator
from src.utils.json_storage import ResultsManager
from src.utils.raw_data_store import RawDataStore, HybridRetriever
from src.evaluation.statistics import ExperimentStatistics, analyze_experiment_results


class LoCoMoExperiment:
    """
    Experiment runner for LoComo benchmark evaluation.

    All results are saved to the unified results folder:
    - results/experiments/{experiment_name}/
      - sample_{idx}/results.json
      - aggregate_results.json
      - detailed_results.json

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
        experiment_name: str = "locomo_experiment",
        results_base: str = None,
        use_mock: bool = False
    ):
        """
        Initialize the experiment.

        Args:
            experiment_name: Name of the experiment
            results_base: Base directory for results (default: ./results)
            use_mock: Use mock LLM for testing
        """
        self.experiment_name = experiment_name
        self.use_mock = use_mock
        self.dataset: Optional[LoCoMoDataset] = None
        self.evaluator = LoCoMoEvaluator()

        # Initialize results manager
        self.results_manager = ResultsManager(results_base or config.RESULTS_DIR)
        self.experiment_dir = os.path.join(
            self.results_manager.folders['experiments'],
            experiment_name
        )
        os.makedirs(self.experiment_dir, exist_ok=True)

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

        # Create sample-specific results directory
        sample_results_dir = os.path.join(self.experiment_dir, f"sample_{sample_idx}")
        os.makedirs(sample_results_dir, exist_ok=True)

        # Create PCM system for this sample with its own results folder
        pcm = create_pcm_system(
            results_dir=sample_results_dir,
            use_mock=self.use_mock
        )

        # Create RawDataStore for preserving original dialogue data
        raw_store_path = os.path.join(sample_results_dir, "raw_data_store")
        os.makedirs(raw_store_path, exist_ok=True)
        raw_store = RawDataStore(
            storage_path=raw_store_path,
            collection_name="raw_dialogues",
            embedding_model=pcm.world_model.embedding_model
        )

        # Create HybridRetriever
        hybrid_retriever = HybridRetriever(
            knowledge_graph=pcm.world_model,
            raw_store=raw_store,
            kg_weight=0.4,  # Give more weight to raw data for accuracy
            raw_weight=0.6
        )

        # Ingest dialogues
        if not skip_ingest:
            await self._ingest_dialogues(pcm, sample, raw_store, verbose)

        # Answer questions
        results = await self._answer_questions(pcm, sample, hybrid_retriever, verbose)

        # Save sample results to JSON
        sample_result_path = os.path.join(sample_results_dir, "results.json")
        with open(sample_result_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)

        # Calculate and save sample statistics
        sample_statistics = ExperimentStatistics()
        sample_stats = sample_statistics.analyze_sample(results)

        # Add statistics to results
        results['statistics'] = sample_stats.to_dict()

        # Save updated results with statistics
        with open(sample_result_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)

        # Save separate statistics file
        stats_path = os.path.join(sample_results_dir, "sample_statistics.json")
        with open(stats_path, 'w', encoding='utf-8') as f:
            json.dump(sample_stats.to_dict(), f, ensure_ascii=False, indent=2)

        if verbose:
            print(f"\nSample {sample_idx} results saved to {sample_result_path}")
            print(f"  Exact Match: {sample_stats.exact_match*100:.2f}%")
            print(f"  Contains Match: {sample_stats.contains_match*100:.2f}%")
            print(f"  Token F1: {sample_stats.token_f1*100:.2f}%")
            print(f"  No-Info Rate: {sample_stats.no_info_count}/{sample_stats.total_questions}")

        return results

    async def _ingest_dialogues(
        self,
        pcm: PCMSystem,
        sample: LoCoMoSample,
        raw_store: RawDataStore,
        verbose: bool
    ) -> None:
        """Ingest all dialogues from a sample into both memory systems."""
        if verbose:
            print("\nIngesting dialogues into memory...")

        dialogues = process_dialogues_for_memory(sample)
        speaker_a = sample.speaker_a

        # Prepare raw dialogues for batch insertion
        raw_dialogues = []
        for i, dialogue in enumerate(dialogues):
            raw_dialogues.append({
                'session_id': dialogue.get('session_id', 'default'),
                'turn_index': i,
                'speaker': dialogue['speaker'],
                'text': dialogue['message'],
                'timestamp': dialogue.get('timestamp'),
                'metadata': {
                    'is_user': dialogue['is_user'],
                    'sample_id': sample.sample_id,
                    'session_idx': dialogue.get('session_idx', 0)
                }
            })

        # Batch add to raw store
        raw_store.add_dialogues_batch(raw_dialogues, auto_save=True)
        if verbose:
            print(f"Added {len(raw_dialogues)} raw dialogues to RawDataStore")

        # Also add to knowledge graph
        for i, dialogue in enumerate(tqdm(dialogues, desc="Ingesting to KG", disable=not verbose)):
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
        hybrid_retriever: HybridRetriever,
        verbose: bool
    ) -> Dict[str, Any]:
        """Answer all QA questions for a sample using hybrid retrieval."""
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
            # Use hybrid retrieval for better context
            retrieval_result = hybrid_retriever.retrieve(
                query=qa.question,
                top_k=10,
                include_context=True,
                context_window=2
            )

            # Build context from merged results
            context_str = "\n".join([
                f"- {content[:300]}"
                for content in retrieval_result['merged_context'][:8]
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

            # Record result with both KG and raw retrieval info
            qa_result = {
                "question": qa.question,
                "reference_answer": qa.answer,
                "predicted_answer": prediction,
                "category": qa.category,
                "category_name": qa.category_name,
                "evidence": qa.evidence,
                "metrics": metrics,
                "retrieved_context": retrieval_result['merged_context'][:5],
                "retrieval_stats": {
                    "kg_count": retrieval_result['kg_count'],
                    "raw_count": retrieval_result['raw_count']
                }
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
        print(f"Running LoComo Experiment: {self.experiment_name}")
        print(f"Samples: {n_samples}, Mock: {self.use_mock}")
        print(f"Results directory: {self.experiment_dir}")
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

        # Save aggregate results to JSON
        aggregate_path = os.path.join(self.experiment_dir, "aggregate_results.json")
        with open(aggregate_path, 'w', encoding='utf-8') as f:
            json.dump({
                "experiment_name": self.experiment_name,
                "aggregate": aggregate,
                "config": {
                    "use_mock": self.use_mock,
                    "n_samples": n_samples,
                    "timestamp": datetime.now().isoformat()
                }
            }, f, ensure_ascii=False, indent=2)

        # Print summary
        self.evaluator.print_summary()

        # Save detailed results to JSON
        detailed_path = os.path.join(self.experiment_dir, "detailed_results.json")
        self.evaluator.save_results(detailed_path)

        # Generate comprehensive statistics report
        print("\n" + "=" * 60)
        print("GENERATING COMPREHENSIVE STATISTICS REPORT")
        print("=" * 60)
        overall_stats = analyze_experiment_results(self.experiment_dir, save_report=True)

        print(f"\nResults saved to: {self.experiment_dir}")

        return aggregate


async def main():
    parser = argparse.ArgumentParser(description="LoComo Benchmark Experiment")
    parser.add_argument("--sample", type=int, default=None, help="Run single sample by index")
    parser.add_argument("--max-samples", type=int, default=None, help="Maximum samples to process")
    parser.add_argument("--mock", action="store_true", help="Use mock LLM")
    parser.add_argument("--skip-ingest", action="store_true", help="Skip memory ingestion")
    parser.add_argument("--stats-only", action="store_true", help="Only run statistics on existing results")
    parser.add_argument("--name", type=str, default="locomo_experiment", help="Experiment name")
    parser.add_argument("--results-dir", type=str, default=None, help="Results base directory")
    args = parser.parse_args()

    # Set mock mode
    if args.mock:
        config.USE_MOCK_LLM = True

    # Handle stats-only mode
    if args.stats_only:
        results_base = args.results_dir or config.RESULTS_DIR
        experiment_dir = os.path.join(results_base, 'experiments', args.name)
        if not os.path.exists(experiment_dir):
            print(f"Error: Experiment directory not found: {experiment_dir}")
            return
        print(f"Running statistics on: {experiment_dir}")
        analyze_experiment_results(experiment_dir, save_report=True)
        return

    experiment = LoCoMoExperiment(
        experiment_name=args.name,
        results_base=args.results_dir,
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
