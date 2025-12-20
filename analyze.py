#!/usr/bin/env python
# coding=utf-8
"""
Analyze Results - Convenience Entry Point
==========================================
Analyze experiment results from the command line.

Usage:
    python analyze.py                              # Basic analysis
    python analyze.py --experiment my_exp          # Analyze specific experiment
    python analyze.py --compare exp1 exp2          # Compare experiments
    python analyze.py --surprisal                  # Analyze surprisal distribution
    python analyze.py --errors                     # Detailed error analysis
    python analyze.py --output ./report.json       # Save results to file
    python analyze.py --format csv                 # Output as CSV

For more options, run: python analyze.py --help
"""

import sys
import os

# Add src to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def main():
    """Main entry point that routes to the appropriate analysis module."""
    import argparse

    # Check if advanced features are requested
    advanced_args = ["--compare", "-c", "--surprisal", "--errors", "--visualize"]
    use_advanced = any(arg in sys.argv for arg in advanced_args)

    if use_advanced:
        from src.evaluation.advanced_analysis import main as advanced_main
        return advanced_main()
    else:
        from src.evaluation.analyze_results import main as basic_main
        return basic_main()


if __name__ == "__main__":
    main()
