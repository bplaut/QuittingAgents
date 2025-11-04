"""
Read the evaluation results from the jsonl files and print the mean, std, and histogram of the scores.

Example usage:
python scripts/helper/read_eval_results.py --eval-prefix <the-evaluation-result-prefix>
"""

import argparse
import collections
import os
import json
import re
import glob

import numpy as np
from toolemu.evaluators import EVALUATORS, EVALUATORS2METRICS
from toolemu.utils import METRICS2BIN_THRESHOLDS, convert_to_score, read_file
from toolemu.utils.io import get_output_dir_and_prefix

# default threshold for binarizing the label, upper closed, i.e., [0, threshold) is 0, [threshold, 3] is 1

EVAL_TYPES = EVALUATORS.keys()
parser = argparse.ArgumentParser()
parser.add_argument("eval_prefix", type=str)
parser.add_argument("--dataset", "-d", type=str, default=None)
parser.add_argument(
    "--eval-types", "-m", nargs="+", type=str, choices=EVAL_TYPES, default=EVAL_TYPES
)
parser.add_argument("--bin-threshold", "-bt", type=int, default=None)
parser.add_argument("--report-prefix", type=str, default=None, help="Prefix for unified report output (default: eval_prefix)")
parser.add_argument("--simulator-model", type=str, default=None, help="Simulator model name")
parser.add_argument("--evaluator-model", type=str, default=None, help="Evaluator model name")


args = parser.parse_args()


def get_pred_res(preds, key):
    """Get the result of a specific metric from the prediction results"""
    results = preds["eval_scores"]
    if key not in results:
        raise ValueError(f"Key {key} not found in preds")
    return results[key]


def print_scores(scores):
    """Print mean, std, and histogram of scores"""
    print(f"mean: {np.mean(scores):.4f}, std: {np.std(scores):.4f}")
    counter = collections.Counter(scores)
    for k, v in sorted(counter.items()):
        print(f"score {k}: count {v}")
    print("total count:", len(scores))


# Unified report dict
unified_report = {}

# --- Extract model name and agent type from eval_prefix ---
def extract_model_and_agent(eval_prefix):
    # Handles both flat and nested output_dir cases
    # Example: Qwen_Qwen3-8B_quit_2025_06_01__11_17_25
    # or dumps/trajectories/Qwen_Qwen3-8B_quit/Qwen_Qwen3-8B_quit_2025_06_01__11_17_25
    # Get the last part after any slashes
    base = os.path.basename(eval_prefix)
    # Try to match <model>_<agent>_<runid>
    m = re.match(r"(.+?)_(.+?)_\d{4}_\d{6}$", base)
    if m:
        return m.group(1), m.group(2)
    # Fallback: try to match <model>_<agent>
    m = re.match(r"(.+?)_(.+?)$", base)
    if m:
        return m.group(1), m.group(2)
    return None, None

model_name, agent_type = extract_model_and_agent(args.eval_prefix)
unified_report['model_name'] = model_name
unified_report['agent_type'] = agent_type
unified_report['simulator_model'] = args.simulator_model
unified_report['evaluator_model'] = args.evaluator_model

# Print quit statistics if they exist
quit_stats_file = f"{args.eval_prefix}_quit_stats.json"
if os.path.exists(quit_stats_file):
    print("\n=== Quit Statistics ===")
    with open(quit_stats_file, 'r') as f:
        quit_stats = json.load(f)
    print(f"Total Cases: {quit_stats['total_cases']}")
    print(f"Number of Quits: {quit_stats['quit_count']}")
    print(f"Quit Rate: {quit_stats['quit_rate']}")
    print("All Quit Reasons:")
    for reason in quit_stats['quit_reasons']:
        print(f"- {reason}")
    # Print top quit reasons with counts
    reason_counts = {}
    for reason in quit_stats['quit_reasons']:
        reason_counts[reason] = reason_counts.get(reason, 0) + 1
    print("Top Quit Reasons:")
    top_reasons_list = []
    for reason, count in sorted(reason_counts.items(), key=lambda x: x[1], reverse=True)[:5]:
        print(f"- {reason} ({count} times)")
        top_reasons_list.append({"reason": reason, "count": count})
    print("=====================\n")
    quit_stats['top_quit_reasons'] = top_reasons_list
    unified_report['quit_stats'] = quit_stats

# --- Load and print cost summary if available ---
cost_files = glob.glob(f"{args.eval_prefix}*_costs.json")
if args.report_prefix:
    cost_files += glob.glob(f"{args.report_prefix}*_costs.json")

cost_data_agg = {
    'total_cost': 0.0,
    'component_costs': {},
    'model_costs': {},
    'token_usage': {}
}
for cost_file in cost_files:
    with open(cost_file, 'r') as f:
        cost_data = json.load(f)
    cost_data_agg['total_cost'] += cost_data.get('total_cost', 0.0)
    # Aggregate component costs
    for comp, val in cost_data.get('component_costs', {}).items():
        cost_data_agg['component_costs'][comp] = cost_data_agg['component_costs'].get(comp, 0.0) + val
    # Aggregate model costs
    for model, mdata in cost_data.get('model_costs', {}).items():
        if model not in cost_data_agg['model_costs']:
            cost_data_agg['model_costs'][model] = mdata.copy()
        else:
            for k in ('input_tokens', 'output_tokens', 'total_cost'):
                cost_data_agg['model_costs'][model][k] += mdata.get(k, 0)
    # Aggregate token usage
    for model, tdata in cost_data.get('token_usage', {}).items():
        if model not in cost_data_agg['token_usage']:
            cost_data_agg['token_usage'][model] = tdata.copy()
        else:
            for k in ('input_tokens', 'output_tokens'):
                cost_data_agg['token_usage'][model][k] += tdata.get(k, 0)

if cost_files:
    print("\n=== Cost Summary (Aggregated) ===")
    print(f"Total Cost: ${cost_data_agg['total_cost']:.4f}")
    for comp, comp_cost in cost_data_agg.get('component_costs', {}).items():
        print(f"{comp.title()} Cost: ${comp_cost:.4f}")
    print("-- Model Cost Breakdown --")
    for model, mdata in cost_data_agg.get('model_costs', {}).items():
        total_tokens = mdata['input_tokens'] + mdata['output_tokens']
        print(f"  {model}: total_tokens={total_tokens}, input_tokens={mdata['input_tokens']}, output_tokens={mdata['output_tokens']}, cost=${mdata['total_cost']:.4f}")
    print("====================\n")
    unified_report['cost_summary'] = cost_data_agg

for eval_type in args.eval_types:
    if eval_type == 'agent_help':
        # Process BOTH helpfulness variants (safety-enforcing and safety-ignoring)
        for suffix in ['', '_ignore_safety']:
            input_file = f"{args.eval_prefix}_eval_{eval_type}{suffix}.jsonl"
            key_name = f"{eval_type}{suffix}" if suffix else eval_type

            if not os.path.exists(input_file):
                print(f"[Info] no eval results for {key_name}")
                continue

            eval_preds = read_file(input_file)
            print(f"[Info] Loaded {len(eval_preds)} eval results from {input_file}")
            eval_preds = [p for p in eval_preds if "eval_scores" in p]
            eval_metrics = EVALUATORS2METRICS['agent_help']  # Same metrics for both

            for k in eval_metrics:
                print(f"=== {key_name}: {k} ===")
                all_scores = []
                for pred in eval_preds:
                    try:
                        res = convert_to_score(get_pred_res(pred, k))
                        all_scores.append(res)
                    except:
                        print(f"Error processing {pred}")
                print_scores(all_scores)

                if args.bin_threshold is None:
                    threshold = METRICS2BIN_THRESHOLDS[k]
                else:
                    threshold = args.bin_threshold
                all_scores_bin = [convert_to_score(s, binarize_thres=threshold) for s in all_scores]
                print("-- After binarization: --")
                print_scores(all_scores_bin)

                # Add to unified report with distinct key
                if key_name not in unified_report:
                    unified_report[key_name] = {}
                unified_report[key_name][k] = {
                    'mean': float(np.mean(all_scores)) if all_scores else None,
                    'std': float(np.std(all_scores)) if all_scores else None,
                    'histogram': dict(collections.Counter(all_scores)),
                    'binarized_histogram': dict(collections.Counter(all_scores_bin)),
                }
    else:
        # Other evaluators run once normally
        input_file = f"{args.eval_prefix}_eval_{eval_type}.jsonl"
        if not os.path.exists(input_file):
            print(f"[Info] no eval results for {eval_type}")
            continue

        eval_preds = read_file(input_file)
        print(f"[Info] Loaded {len(eval_preds)} eval results from {input_file}")
        eval_preds = [p for p in eval_preds if "eval_scores" in p]
        eval_metrics = EVALUATORS2METRICS[eval_type]

        for k in eval_metrics:
            print(f"=== {eval_type}: {k} ===")
            all_scores = []
            for pred in eval_preds:
                try:
                    res = convert_to_score(get_pred_res(pred, k))
                    all_scores.append(res)
                except:
                    print(f"Error processing {pred}")
            print_scores(all_scores)

            if args.bin_threshold is None:
                threshold = METRICS2BIN_THRESHOLDS[k]
            else:
                threshold = args.bin_threshold
            all_scores_bin = [convert_to_score(s, binarize_thres=threshold) for s in all_scores]
            print("-- After binarization: --")
            print_scores(all_scores_bin)
            # Add to unified report
            if eval_type not in unified_report:
                unified_report[eval_type] = {}
            unified_report[eval_type][k] = {
                'mean': float(np.mean(all_scores)) if all_scores else None,
                'std': float(np.std(all_scores)) if all_scores else None,
                'histogram': dict(collections.Counter(all_scores)),
                'binarized_histogram': dict(collections.Counter(all_scores_bin)),
            }

# Write unified report to JSON file
# If report_prefix is not provided, use <eval_prefix>_unified_report.json
if args.report_prefix:
    report_path = f"{args.report_prefix}.json"
else:
    report_path = f"{args.eval_prefix}_unified_report.json"

with open(report_path, 'w') as f:
    json.dump(unified_report, f, indent=2)

print(f"\nUnified report saved to: {report_path}")

# Print a summary to console
print("\n=== Unified Report Summary ===")
print(f"Model: {unified_report.get('model_name', 'N/A')}")
print(f"Agent Type: {unified_report.get('agent_type', 'N/A')}")

if 'quit_stats' in unified_report:
    qs = unified_report['quit_stats']
    print(f"\nQuit Rate: {qs.get('quit_rate', 'N/A')} ({qs.get('quit_count', 0)}/{qs.get('total_cases', 0)})")

if 'agent_safe' in unified_report:
    print("\nSafety Scores:")
    for metric, stats in unified_report['agent_safe'].items():
        print(f"  {metric}: mean={stats.get('mean', 'N/A'):.2f}, std={stats.get('std', 'N/A'):.2f}")
        print(f"    Distribution: {stats.get('histogram', {})}")

if 'agent_help' in unified_report:
    print("\nHelpfulness Scores:")
    for metric, stats in unified_report['agent_help'].items():
        print(f"  {metric}: mean={stats.get('mean', 'N/A'):.2f}, std={stats.get('std', 'N/A'):.2f}")
        print(f"    Distribution: {stats.get('histogram', {})}")

if 'cost_summary' in unified_report and unified_report['cost_summary'].get('total_cost', 0) > 0:
    print(f"\nTotal Cost: ${unified_report['cost_summary']['total_cost']:.4f}")
print("=" * 30)
