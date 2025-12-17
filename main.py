"""
Proactive Cognitive Memory (PCM) System - Main Entry Point
==========================================================

This script demonstrates the PCM system with example interactions.

Usage:
    python main.py              # Run interactive demo
    python main.py --mock       # Run with mock LLM (no API calls)
    python main.py --scenario   # Run predefined scenario
"""

import asyncio
import argparse
import json
from datetime import datetime

from src.core.orchestrator import PCMSystem, create_pcm_system
from src.core.types import NodeType
import config


async def run_scenario_demo(pcm: PCMSystem):
    """
    Run the predefined scenario from the framework document.

    Scenario: User transitions from Python Web Development to AI Research.
    """
    print("\n" + "=" * 60)
    print("PCM System - Scenario Demo")
    print("Scenario: User transitioning from Web Dev to AI")
    print("=" * 60 + "\n")

    # Pre-seed some knowledge
    print("Seeding initial knowledge...")
    await pcm.add_knowledge(
        "User has experience with Python web development",
        NodeType.FACT,
        "Coding",
        0.8
    )
    await pcm.add_knowledge(
        "User frequently uses Django framework",
        NodeType.FACT,
        "Coding",
        0.7
    )
    await pcm.add_knowledge(
        "User prefers backend development",
        NodeType.ATTRIBUTE,
        "Coding",
        0.6
    )
    print("Initial knowledge seeded.\n")

    # Scenario interactions
    interactions = [
        # Phase 1: Low Surprise - Confirming existing knowledge
        "How do I configure Django's database settings for PostgreSQL?",

        # Phase 2: Medium Surprise - Novel but not conflicting
        "I've been reading about PyTorch tensor operations. How can I optimize memory usage?",

        # Phase 3: More evidence for hypothesis
        "Can you explain how the Transformer attention mechanism works?",

        # Phase 4: High Surprise - Direct contradiction
        "Actually, I've decided to stop doing web development entirely. I want to focus only on AI and machine learning from now on.",

        # Phase 5: Verify updated model
        "What deep learning frameworks would you recommend for a beginner in AI?"
    ]

    for i, user_input in enumerate(interactions, 1):
        print(f"\n{'─' * 50}")
        print(f"Interaction {i}")
        print(f"{'─' * 50}")
        print(f"User: {user_input}\n")

        result = await pcm.process_input(user_input, generate_response=True)

        # Display results
        print(f"Intent: {result['intent']['label']} (confidence: {result['intent']['confidence']:.2f})")
        print(f"Surprisal: {result['surprisal']['effective']:.2f} ({result['surprisal']['level']})")
        print(f"Agent: {result['evolution']['agent'].capitalize()}")

        if result['response']:
            print(f"\nAssistant: {result['response'][:300]}...")

        # Brief pause for readability
        await asyncio.sleep(0.5)

    # Show final statistics
    print("\n" + "=" * 60)
    print("Final System Statistics")
    print("=" * 60)
    stats = pcm.get_statistics()
    print(json.dumps(stats, indent=2, default=str))


async def run_interactive_demo(pcm: PCMSystem):
    """
    Run an interactive demo where users can type messages.
    """
    print("\n" + "=" * 60)
    print("PCM System - Interactive Demo")
    print("=" * 60)
    print("\nCommands:")
    print("  /quit or /exit  - Exit the demo")
    print("  /stats          - Show system statistics")
    print("  /query <text>   - Query the knowledge graph")
    print("  /add <text>     - Add knowledge directly")
    print("  /context        - Show current working memory")
    print("\nStart chatting! The system will learn from your interactions.\n")

    while True:
        try:
            user_input = input("You: ").strip()
        except EOFError:
            break

        if not user_input:
            continue

        # Handle commands
        if user_input.lower() in ["/quit", "/exit"]:
            print("Goodbye!")
            break

        if user_input.lower() == "/stats":
            stats = pcm.get_statistics()
            print("\nSystem Statistics:")
            print(json.dumps(stats, indent=2, default=str))
            continue

        if user_input.lower().startswith("/query "):
            query = user_input[7:]
            results = pcm.query_knowledge(query)
            print("\nQuery Results:")
            for r in results:
                print(f"  - {r['content'][:80]}... (score: {r['score']:.2f})")
            continue

        if user_input.lower().startswith("/add "):
            content = user_input[5:]
            node_id = await pcm.add_knowledge(content)
            print(f"Added knowledge with ID: {node_id}")
            continue

        if user_input.lower() == "/context":
            context = pcm.get_context()
            print(f"\nWorking Memory Context:\n{context}")
            continue

        # Process regular input
        result = await pcm.process_input(user_input, generate_response=True)

        # Display surprisal info
        level = result['surprisal']['level']
        level_indicator = {"low": "🟢", "medium": "🟡", "high": "🔴"}[level]
        print(f"\n{level_indicator} Surprisal: {result['surprisal']['effective']:.2f} ({level})")
        print(f"   Agent: {result['evolution']['agent']}")

        # Display response
        if result['response']:
            print(f"\nAssistant: {result['response']}\n")


async def main():
    parser = argparse.ArgumentParser(description="PCM System Demo")
    parser.add_argument("--mock", action="store_true", help="Use mock LLM (no API calls)")
    parser.add_argument("--scenario", action="store_true", help="Run predefined scenario")
    args = parser.parse_args()

    # Override config if mock mode
    if args.mock:
        config.USE_MOCK_LLM = True
        print("Running in mock mode (no API calls)")

    # Create PCM system
    print("\nInitializing PCM System...")
    pcm = create_pcm_system(use_mock=args.mock)

    if args.scenario:
        await run_scenario_demo(pcm)
    else:
        await run_interactive_demo(pcm)


if __name__ == "__main__":
    asyncio.run(main())
