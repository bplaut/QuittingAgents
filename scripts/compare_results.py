#!/usr/bin/env python3
"""
Compare results from multiple ToolEmu runs.

Usage:
    python scripts/compare_results.py <results_dir> [additional_result_files...]

Example:
    python scripts/compare_results.py ./results
    python scripts/compare_results.py ./results dumps/some_other_report.json
"""

import argparse
import json
import os
import glob
import re
from typing import List, Dict, Any, Optional


def extract_quantization_from_filename(filename: str) -> str:
    """Extract quantization level from filename (e.g., _int4_ or _int8_)."""
    match = re.search(r'_(int\d+)', filename)
    if match:
        return match.group(1)
    return "none"


def extract_metrics_from_report(report_path: str) -> Optional[Dict[str, Any]]:
    """Extract key metrics from a unified report JSON file."""
    try:
        with open(report_path, 'r') as f:
            data = json.load(f)

        # Extract basic info
        model_name = data.get('model_name', 'Unknown')
        agent_type = data.get('agent_type', 'Unknown')
        quantization = extract_quantization_from_filename(os.path.basename(report_path))
        # Assume simulator and evaluator are the same, prefer simulator_model
        simulator_model = data.get('simulator_model') or data.get('evaluator_model', 'Unknown')

        # Extract helpfulness (safety-enforcing: average across all metrics in agent_help)
        avg_helpfulness = None
        if 'agent_help' in data:
            help_means = [
                metric_data.get('mean')
                for metric_data in data['agent_help'].values()
                if isinstance(metric_data, dict) and metric_data.get('mean') is not None
            ]
            if help_means:
                avg_helpfulness = sum(help_means) / len(help_means)

        # Extract helpfulness (safety-ignoring: average across all metrics in agent_help_ignore_safety)
        avg_helpfulness_ignore_safety = None
        if 'agent_help_ignore_safety' in data:
            help_means = [
                metric_data.get('mean')
                for metric_data in data['agent_help_ignore_safety'].values()
                if isinstance(metric_data, dict) and metric_data.get('mean') is not None
            ]
            if help_means:
                avg_helpfulness_ignore_safety = sum(help_means) / len(help_means)

        # Extract safety (average across all metrics in agent_safe)
        avg_safety = None
        if 'agent_safe' in data:
            safe_means = [
                metric_data.get('mean')
                for metric_data in data['agent_safe'].values()
                if isinstance(metric_data, dict) and metric_data.get('mean') is not None
            ]
            if safe_means:
                avg_safety = sum(safe_means) / len(safe_means)

        # Extract quit rate
        quit_rate = None
        if 'quit_stats' in data:
            quit_rate_str = data['quit_stats'].get('quit_rate', '0%')
            # Parse "30.00%" -> 0.30
            quit_rate = float(quit_rate_str.strip('%')) / 100.0

        return {
            'model_name': model_name,
            'agent_type': agent_type,
            'quantization': quantization,
            'avg_helpfulness': avg_helpfulness,
            'avg_helpfulness_ignore_safety': avg_helpfulness_ignore_safety,
            'avg_safety': avg_safety,
            'quit_rate': quit_rate,
            'simulator_model': simulator_model,
            'file': os.path.basename(report_path)
        }
    except Exception as e:
        print(f"Warning: Failed to process {report_path}: {e}")
        return None


def format_value(value: Optional[float], is_rate: bool = False) -> str:
    """Format a numeric value for display."""
    if value is None:
        return "N/A"
    if is_rate:
        return f"{value*100:.1f}%"
    return f"{value:.2f}"


def print_table(results: List[Dict[str, Any]]):
    """Print results as a formatted table."""
    if not results:
        print("No results to display.")
        return

    # Define column widths
    col_widths = {
        'model': max(20, max(len(r['model_name']) for r in results)),
        'type': max(12, max(len(r['agent_type']) for r in results)),
        'help_safe': 12,
        'help_ignore': 12,
        'safety': 12,
        'quit': 12,
        'quant': 12,
        'simulator': max(20, max(len(r['simulator_model']) for r in results)),
    }

    # Print header
    header = (
        f"{'Model':<{col_widths['model']}} | "
        f"{'Agent Type':<{col_widths['type']}} | "
        f"{'Help(Safe)':<{col_widths['help_safe']}} | "
        f"{'Help(Ignore)':<{col_widths['help_ignore']}} | "
        f"{'Safety':<{col_widths['safety']}} | "
        f"{'Quit Rate':<{col_widths['quit']}} | "
        f"{'Quantization':<{col_widths['quant']}} | "
        f"{'Simulator/Eval':<{col_widths['simulator']}}"
    )
    separator = "-" * len(header)

    print("\n" + separator)
    print(header)
    print(separator)

    # Print rows
    for result in results:
        row = (
            f"{result['model_name']:<{col_widths['model']}} | "
            f"{result['agent_type']:<{col_widths['type']}} | "
            f"{format_value(result['avg_helpfulness']):<{col_widths['help_safe']}} | "
            f"{format_value(result['avg_helpfulness_ignore_safety']):<{col_widths['help_ignore']}} | "
            f"{format_value(result['avg_safety']):<{col_widths['safety']}} | "
            f"{format_value(result['quit_rate'], is_rate=True):<{col_widths['quit']}} | "
            f"{result['quantization']:<{col_widths['quant']}} | "
            f"{result['simulator_model']:<{col_widths['simulator']}}"
        )
        print(row)

    print(separator)
    print(f"\nTotal runs: {len(results)}\n")


def main():
    parser = argparse.ArgumentParser(
        description="Compare results from multiple ToolEmu runs",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        'paths',
        nargs='+',
        type=str,
        help='Either a single directory (scans all reports) or one or more report JSON files'
    )
    parser.add_argument(
        '--sort-by',
        choices=['model', 'type', 'help', 'safety', 'quit'],
        default='model',
        help='Primary sort key (default: model)'
    )

    args = parser.parse_args()

    # Collect all report files
    report_files = []

    # If only one argument and it's a directory, scan it
    if len(args.paths) == 1 and os.path.isdir(args.paths[0]):
        results_dir = args.paths[0]
        pattern = os.path.join(results_dir, "*_unified_report_*.json")
        report_files.extend(glob.glob(pattern))
    else:
        # Multiple arguments or single file: treat all as files
        for file_path in args.paths:
            if os.path.isfile(file_path):
                report_files.append(file_path)
            else:
                print(f"Warning: {file_path} not found or not a file")

    if not report_files:
        print("No unified report files found")
        return

    print(f"Processing {len(report_files)} report file(s)")

    # Extract metrics from all reports
    results = []
    for report_path in report_files:
        metrics = extract_metrics_from_report(report_path)
        if metrics:
            results.append(metrics)

    if not results:
        print("No valid results extracted from report files")
        return

    # Sort results by simulator_model, then model_name, then agent_type, then quantization
    results.sort(key=lambda x: (
        x['simulator_model'].lower(),
        x['model_name'].lower(),
        x['agent_type'].lower(),
        x['quantization'].lower()
    ))

    # Print table
    print_table(results)


if __name__ == '__main__':
    main()
