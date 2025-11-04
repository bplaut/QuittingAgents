#!/usr/bin/env python3
"""
Aggregate results from multiple parallel ToolEmu runs into a unified report.

Usage:
    python scripts/aggregate_results.py --pattern "dumps/*_r*_unified_report_*.json"
    python scripts/aggregate_results.py --files file1.json file2.json file3.json
    python scripts/aggregate_results.py --pattern "..." --output aggregated_report.json
"""

import argparse
import collections
import glob
import json
import os
import re
from typing import List, Dict, Any, Tuple, Optional


def parse_range_from_filename(filename: str) -> Optional[Tuple[int, int]]:
    """Extract task index range from filename (e.g., _r0-48_ -> (0, 48))."""
    match = re.search(r'_r(\d+)-(\d+)_', filename)
    if match:
        return (int(match.group(1)), int(match.group(2)))
    return None


def extract_base_config(filename: str) -> str:
    """
    Extract base configuration from filename (everything except range and timestamp).
    E.g., "gpt-5_quit_sim-gpt-5_int4_r0-48_20251103_143022.json"
      -> "gpt-5_quit_sim-gpt-5_int4"
    """
    # Remove range suffix (_r\d+-\d+)
    base = re.sub(r'_r\d+-\d+', '', filename)
    # Remove timestamp suffix (_\d{8}_\d{6})
    base = re.sub(r'_\d{8}_\d{6}', '', base)
    # Remove file extension
    base = re.sub(r'\.json$', '', base)
    # Remove unified_report suffix if present
    base = re.sub(r'_unified_report$', '', base)
    return base


def validate_completeness(reports: List[Dict[str, Any]], expected_total: int = 144) -> Tuple[bool, str]:
    """
    Validate that ranges are complete and non-overlapping.

    Returns:
        (is_valid, message)
    """
    if not reports:
        return False, "No reports provided"

    # Extract ranges
    ranges = []
    for report in reports:
        if 'range' not in report:
            return False, f"Report missing range information: {report.get('file', 'unknown')}"
        ranges.append(report['range'])

    # Sort by start index
    ranges.sort(key=lambda r: r[0])

    # Check for gaps and overlaps
    expected_start = 0
    for i, (start, end) in enumerate(ranges):
        if start != expected_start:
            if start > expected_start:
                return False, f"Gap detected: expected range starting at {expected_start}, found {start}"
            else:
                return False, f"Overlap detected: range {i} starts at {start}, but previous range ended at {expected_start}"
        expected_start = end

    # Check total coverage
    if expected_start != expected_total:
        return False, f"Incomplete coverage: ranges cover 0-{expected_start}, expected 0-{expected_total}"

    return True, "All ranges are complete and non-overlapping"


def aggregate_quit_stats(reports: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Aggregate quit statistics from multiple reports."""
    total_cases = 0
    quit_count = 0
    all_quit_reasons = []

    for report in reports:
        if 'quit_stats' not in report:
            continue
        qs = report['quit_stats']
        total_cases += qs.get('total_cases', 0)
        quit_count += qs.get('quit_count', 0)
        all_quit_reasons.extend(qs.get('quit_reasons', []))

    # Calculate aggregated quit rate
    quit_rate = f"{(quit_count / total_cases * 100):.2f}%" if total_cases > 0 else "0.00%"

    # Count quit reasons
    reason_counts = collections.Counter(all_quit_reasons)
    top_quit_reasons = [
        {"reason": reason, "count": count}
        for reason, count in reason_counts.most_common(5)
    ]

    return {
        'total_cases': total_cases,
        'quit_count': quit_count,
        'quit_rate': quit_rate,
        'quit_reasons': all_quit_reasons,
        'top_quit_reasons': top_quit_reasons
    }


def aggregate_cost_summary(reports: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Aggregate cost summaries from multiple reports."""
    total_cost = 0.0
    component_costs = collections.defaultdict(float)
    model_costs = collections.defaultdict(lambda: {
        'input_tokens': 0,
        'output_tokens': 0,
        'total_cost': 0.0
    })

    for report in reports:
        if 'cost_summary' not in report:
            continue
        cs = report['cost_summary']

        total_cost += cs.get('total_cost', 0.0)

        for comp, cost in cs.get('component_costs', {}).items():
            component_costs[comp] += cost

        for model, mdata in cs.get('model_costs', {}).items():
            model_costs[model]['input_tokens'] += mdata.get('input_tokens', 0)
            model_costs[model]['output_tokens'] += mdata.get('output_tokens', 0)
            model_costs[model]['total_cost'] += mdata.get('total_cost', 0.0)

    return {
        'total_cost': total_cost,
        'component_costs': dict(component_costs),
        'model_costs': dict(model_costs)
    }


def aggregate_evaluation_metrics(reports: List[Dict[str, Any]], metric_group: str) -> Dict[str, Any]:
    """
    Aggregate evaluation metrics (agent_safe, agent_help, etc.) from multiple reports.

    Combines histograms and recalculates mean/std across all data points.
    """
    if not reports:
        return {}

    # Collect all metrics from first report
    first_report = next((r for r in reports if metric_group in r), None)
    if not first_report or metric_group not in first_report:
        return {}

    metric_names = list(first_report[metric_group].keys())
    aggregated = {}

    for metric_name in metric_names:
        # Collect all histograms
        combined_histogram = collections.Counter()
        combined_binarized_histogram = collections.Counter()

        for report in reports:
            if metric_group not in report or metric_name not in report[metric_group]:
                continue

            metric_data = report[metric_group][metric_name]

            # Combine histograms
            if 'histogram' in metric_data:
                for score, count in metric_data['histogram'].items():
                    # Handle both int and float score keys
                    score_key = int(float(score))
                    combined_histogram[score_key] += count

            if 'binarized_histogram' in metric_data:
                for score, count in metric_data['binarized_histogram'].items():
                    # Handle both int and float score keys
                    score_key = int(float(score))
                    combined_binarized_histogram[score_key] += count

        # Recalculate mean and std from combined histogram
        all_scores = []
        for score, count in combined_histogram.items():
            all_scores.extend([score] * count)

        if all_scores:
            import numpy as np
            mean = float(np.mean(all_scores))
            std = float(np.std(all_scores))
        else:
            mean = None
            std = None

        aggregated[metric_name] = {
            'mean': mean,
            'std': std,
            'histogram': dict(combined_histogram),
            'binarized_histogram': dict(combined_binarized_histogram)
        }

    return aggregated


def aggregate_reports(reports: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Aggregate multiple reports into a unified report."""
    if not reports:
        raise ValueError("No reports to aggregate")

    # Use first report as template for metadata
    template = reports[0]

    unified = {
        'model_name': template.get('model_name'),
        'agent_type': template.get('agent_type'),
        'simulator_model': template.get('simulator_model'),
        'evaluator_model': template.get('evaluator_model'),
    }

    # Aggregate quit stats
    unified['quit_stats'] = aggregate_quit_stats(reports)

    # Aggregate cost summary
    unified['cost_summary'] = aggregate_cost_summary(reports)

    # Aggregate evaluation metrics
    for metric_group in ['agent_safe', 'agent_help', 'agent_help_ignore_safety']:
        if any(metric_group in r for r in reports):
            unified[metric_group] = aggregate_evaluation_metrics(reports, metric_group)

    return unified


def main():
    parser = argparse.ArgumentParser(
        description="Aggregate results from multiple parallel ToolEmu runs",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        '--pattern',
        type=str,
        help='Glob pattern for report files (e.g., "dumps/*_r*_unified_report_*.json")'
    )
    parser.add_argument(
        '--files',
        nargs='+',
        type=str,
        help='Explicit list of report files to aggregate'
    )
    parser.add_argument(
        '--output',
        type=str,
        default=None,
        help='Output path for aggregated report (default: auto-generated)'
    )
    parser.add_argument(
        '--expected-total',
        type=int,
        default=144,
        help='Expected total number of tasks (default: 144)'
    )

    args = parser.parse_args()

    # Collect report files
    if args.pattern:
        report_files = glob.glob(args.pattern)
    elif args.files:
        report_files = args.files
    else:
        parser.error("Must specify either --pattern or --files")

    if not report_files:
        print("Error: No report files found")
        return 1

    print(f"Found {len(report_files)} report file(s)")

    # Load reports
    reports_with_ranges = []
    base_configs = set()

    for report_file in report_files:
        try:
            with open(report_file, 'r') as f:
                report = json.load(f)

            # Extract range from filename
            range_tuple = parse_range_from_filename(report_file)
            if range_tuple is None:
                print(f"Warning: Could not parse range from {report_file}, skipping")
                continue

            # Extract base config
            base_config = extract_base_config(os.path.basename(report_file))
            base_configs.add(base_config)

            report['range'] = range_tuple
            report['file'] = report_file
            reports_with_ranges.append(report)

        except Exception as e:
            print(f"Error loading {report_file}: {e}")
            continue

    if not reports_with_ranges:
        print("Error: No valid reports loaded")
        return 1

    # Check that all reports are for the same configuration
    if len(base_configs) > 1:
        print(f"Error: Reports have different configurations: {base_configs}")
        print("All reports must be for the same model/agent-type/quantization")
        return 1

    # Validate completeness
    is_valid, message = validate_completeness(reports_with_ranges, args.expected_total)
    if not is_valid:
        print(f"Error: {message}")
        return 1

    print(f"✓ Validation passed: {message}")

    # Aggregate
    print("Aggregating reports...")
    unified_report = aggregate_reports(reports_with_ranges)

    # Determine output path
    if args.output:
        output_path = args.output
    else:
        # Auto-generate output path
        base_config = list(base_configs)[0]
        output_path = f"dumps/{base_config}_unified_report_aggregated.json"

    # Write output
    os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(unified_report, f, indent=2)

    print(f"\n✓ Aggregated report saved to: {output_path}")

    # Print summary
    print("\n=== Aggregation Summary ===")
    print(f"Model: {unified_report.get('model_name', 'N/A')}")
    print(f"Agent Type: {unified_report.get('agent_type', 'N/A')}")
    print(f"Total Cases: {unified_report.get('quit_stats', {}).get('total_cases', 'N/A')}")
    print(f"Quit Rate: {unified_report.get('quit_stats', {}).get('quit_rate', 'N/A')}")
    if 'cost_summary' in unified_report:
        print(f"Total Cost: ${unified_report['cost_summary'].get('total_cost', 0):.4f}")
    print("=" * 30)

    return 0


if __name__ == '__main__':
    exit(main())
