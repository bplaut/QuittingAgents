#!/usr/bin/env python3
"""
Monitor progress of SLURM jobs running toolemu evaluations.

Usage:
    python scripts/monitor_jobs.py [--user USERNAME] [--watch SECONDS]
"""

import argparse
import subprocess
import os
import glob
import time
import re
from datetime import datetime, timedelta
from collections import defaultdict

def parse_task_count_from_filename(filename):
    """Extract total task count from filename range (e.g., _r0-29_ -> 29 tasks, _r29-58_ -> 29 tasks)."""
    if not filename:
        return 144  # Default to full dataset

    # Look for range pattern: _rSTART-END_
    match = re.search(r'_r(\d+)-(\d+)_', filename)
    if match:
        start = int(match.group(1))
        end = int(match.group(2))
        return end - start  # Python slice semantics: [start, end)

    return 144  # Default to full dataset if no range found

def get_running_jobs(username):
    """Get list of running/pending toolemu jobs for user."""
    try:
        result = subprocess.run(
            ['squeue', '-u', username, '--format=%i,%j,%T,%M,%R'],
            capture_output=True, text=True, check=True
        )
        jobs = []
        for line in result.stdout.strip().split('\n')[1:]:  # Skip header
            if 'toolemu' in line.lower():
                parts = line.split(',')
                if len(parts) >= 5:
                    jobs.append({
                        'job_id': parts[0],
                        'name': parts[1],
                        'state': parts[2],
                        'time': parts[3],
                        'node': parts[4]
                    })
        return jobs
    except subprocess.CalledProcessError:
        return []

def get_trajectory_progress(job_id):
    """Find trajectory file for job and count completed cases."""
    # Find log file for this job
    log_file = f"logs/exp_toolemu_{job_id}.out"
    if not os.path.exists(log_file):
        return None, None, None, None, None

    try:
        with open(log_file, 'r') as f:
            content = f.read()

            # Extract config from log output
            agent_model = None
            sim_model = None
            agent_type = None
            quantization = None
            task_range = None

            for line in content.split('\n'):
                if line.strip().startswith('Agent:'):
                    agent_model = line.split('Agent:')[-1].strip()
                elif line.strip().startswith('Simulator:'):
                    sim_model = line.split('Simulator:')[-1].strip()
                elif line.strip().startswith('Agent type:'):
                    agent_type = line.split('Agent type:')[-1].strip()
                elif line.strip().startswith('Quantization:'):
                    quantization = line.split('Quantization:')[-1].strip()
                elif line.strip().startswith('Task index range:'):
                    task_range = line.split('Task index range:')[-1].strip()

            if not (agent_model and sim_model and agent_type):
                return None, None, None, None, None

            # Sanitize model names for filesystem
            agent_safe = agent_model.replace('/', '_').replace(' ', '_')
            sim_safe = sim_model.replace('/', '_').replace(' ', '_')

            # Build file pattern including task range if present
            range_pattern = f"_r{task_range}_" if task_range else "_"
            pattern = f"dumps/trajectories/{agent_safe}/{agent_safe}_{agent_type}_sim-{sim_safe}_*.jsonl"
            matches = glob.glob(pattern)

            # Filter out eval/costs/quit_stats files
            traj_files = [f for f in matches
                         if 'eval' not in f and 'costs' not in f and 'quit_stats' not in f and 'unified_report' not in f]

            # Find the most recent one that matches our quantization and task range
            best_match = None
            best_mtime = 0

            for traj_file in traj_files:
                # Check if quantization matches
                if quantization:
                    if quantization not in traj_file:
                        continue

                # Check if task range matches (if specified)
                if task_range:
                    if f"_r{task_range}_" not in traj_file:
                        continue

                mtime = os.path.getmtime(traj_file)
                if mtime > best_mtime:
                    best_mtime = mtime
                    best_match = traj_file

            if best_match and os.path.exists(best_match):
                # Count trajectory lines
                with open(best_match, 'r') as tf:
                    traj_lines = sum(1 for _ in tf)

                # Count eval file lines
                base_path = best_match.replace('.jsonl', '')
                safe_eval_file = f"{base_path}_eval_agent_safe.jsonl"
                help_eval_file = f"{base_path}_eval_agent_help.jsonl"
                help_ignore_safety_eval_file = f"{base_path}_eval_agent_help_ignore_safety.jsonl"

                safe_lines = 0
                help_lines = 0
                help_ignore_safety_lines = 0

                if os.path.exists(safe_eval_file):
                    with open(safe_eval_file, 'r') as f:
                        safe_lines = sum(1 for _ in f)

                if os.path.exists(help_eval_file):
                    with open(help_eval_file, 'r') as f:
                        help_lines = sum(1 for _ in f)

                if os.path.exists(help_ignore_safety_eval_file):
                    with open(help_ignore_safety_eval_file, 'r') as f:
                        help_ignore_safety_lines = sum(1 for _ in f)

                # Combine both help variants
                total_help_lines = help_lines + help_ignore_safety_lines

                basename = os.path.basename(best_match)
                return traj_lines, safe_lines, total_help_lines, basename, quantization

    except Exception as e:
        pass

    return None, None, None, None, None

def parse_time(time_str):
    """Parse SLURM time format (e.g., '1:23:45', '23:45', or '1-00:23:45') to seconds."""
    # Handle day-hour:minute:second format (e.g., '1-00:23:45')
    if '-' in time_str:
        day_part, time_part = time_str.split('-')
        days = int(day_part)
        parts = time_part.split(':')
    else:
        days = 0
        parts = time_str.split(':')

    if len(parts) == 3:
        return days * 86400 + int(parts[0]) * 3600 + int(parts[1]) * 60 + int(parts[2])
    elif len(parts) == 2:
        return days * 86400 + int(parts[0]) * 60 + int(parts[1])
    return 0

def format_time(seconds):
    """Format seconds to human readable time."""
    if seconds < 60:
        return f"{seconds}s"
    elif seconds < 3600:
        return f"{seconds//60}m {seconds%60}s"
    elif seconds < 86400:
        hours = seconds // 3600
        mins = (seconds % 3600) // 60
        return f"{hours}h {mins}m"
    else:
        days = seconds // 86400
        hours = (seconds % 86400) // 3600
        mins = (seconds % 3600) // 60
        return f"{days}d {hours}h {mins}m"

def estimate_completion(traj_progress, safe_progress, help_progress, elapsed_seconds, total_tasks=144):
    """Estimate time to completion based on current progress.

    Note: help_progress represents the sum of both help evaluation variants
    (regular help and help_ignore_safety), so it should be out of 2*total_tasks.
    """
    # Trajectory generation is much slower than evaluation (agent + simulator running multiple steps)
    # Weight trajectories higher: roughly 4x slower than a single evaluation
    # This is empirically observed: ~3-4 min per trajectory vs ~1 min per eval

    TRAJ_WEIGHT = 4  # Trajectories take ~4x as long as evals
    EVAL_WEIGHT = 1  # Evals are baseline unit

    # Calculate weighted units of work completed
    weighted_units_done = (traj_progress * TRAJ_WEIGHT +
                           safe_progress * EVAL_WEIGHT +
                           help_progress * EVAL_WEIGHT)

    # Total weighted units of work (based on actual task count, not hardcoded 144)
    # Note: help has 2 variants per task (help and help_ignore_safety), so 2*total_tasks help evals
    total_weighted_units = total_tasks * TRAJ_WEIGHT + total_tasks * EVAL_WEIGHT + (2 * total_tasks) * EVAL_WEIGHT
    # = total_tasks * 7 weighted units

    if weighted_units_done == 0:
        return "Unknown"

    if weighted_units_done >= total_weighted_units:
        return "Complete"

    # Calculate average rate (seconds per weighted unit)
    rate = elapsed_seconds / weighted_units_done

    # Calculate remaining weighted units
    remaining_units = total_weighted_units - weighted_units_done

    # Estimated remaining time
    remaining_seconds = int(rate * remaining_units)

    return format_time(remaining_seconds)

def get_job_config(filename):
    """Extract configuration from filename."""
    if not filename:
        return "Unknown", "Unknown", "Unknown"

    # Format: {agent}_{type}_sim-{sim}_{quant}_{safety}_{timestamp}.jsonl
    # Examples:
    #   Qwen_Qwen3-8B_naive_sim-Qwen_Qwen3-32B_int4_0311_145549.jsonl
    #   meta-llama_Llama-3.1-8B-Instruct_quit_sim-gpt-5-mini_int4_0311_074154.jsonl
    #   gpt-5_naive_sim-gpt-5-mini_int4_0311_070740.jsonl

    basename = filename.replace('.jsonl', '')

    # Find agent type
    agent_type = "?"
    if '_naive_' in basename:
        agent_type = 'naive'
        type_marker = '_naive_'
    elif '_simple_quit_' in basename:
        agent_type = 'simple_quit'
        type_marker = '_simple_quit_'
    elif '_quit_' in basename:
        agent_type = 'quit'
        type_marker = '_quit_'
    elif '_ss_only_' in basename:
        agent_type = 'ss_only'
        type_marker = '_ss_only_'
    elif '_helpful_ss_' in basename:
        agent_type = 'helpful_ss'
        type_marker = '_helpful_ss_'
    else:
        return "?", "?", "?"

    # Split on the type marker to get agent and rest
    parts = basename.split(type_marker)
    if len(parts) < 2:
        return "?", agent_type, "?"

    agent_full = parts[0]  # Everything before the type
    rest = parts[1]  # Everything after the type

    # Extract simulator from rest (after sim-)
    if 'sim-' in rest:
        sim_part = rest.split('sim-')[1]
        # Take everything until the next underscore followed by int4/int8
        sim = sim_part.split('_int')[0]
    else:
        sim = "?"

    # Clean up agent name to make it more readable
    # Examples: "Qwen_Qwen3-8B" -> "Qwen3-8B"
    #          "meta-llama_Llama-3.1-8B-Instruct" -> "Llama3.1-8B"
    #          "gpt-5" -> "gpt-5"
    if 'Qwen3-8B' in agent_full:
        agent = 'Qwen3-8B'
    elif 'Qwen3-32B' in agent_full:
        agent = 'Qwen3-32B'
    elif 'Llama-3.1-8B' in agent_full:
        agent = 'Llama3.1-8B'
    elif 'Llama-3.1-70B' in agent_full:
        agent = 'Llama3.1-70B'
    elif agent_full.startswith('gpt-'):
        agent = agent_full
    else:
        agent = agent_full

    # Clean up sim name similarly
    if 'Qwen3-8B' in sim:
        sim = 'Qwen3-8B'
    elif 'Qwen3-32B' in sim:
        sim = 'Qwen3-32B'
    # gpt- models are already clean, no transformation needed

    return agent, agent_type, sim

def print_job_summary(jobs, show_pending=True):
    """Print formatted table of job progress."""
    if not jobs:
        print("No toolemu jobs found.")
        return

    # Separate by state
    running = [j for j in jobs if j['state'] == 'RUNNING']
    pending = [j for j in jobs if j['state'] in ['PENDING', 'PD']]

    print("\n" + "="*120)
    print(f"TOOLEMU JOB MONITOR - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*120)

    if running:
        print(f"\n🏃 RUNNING JOBS ({len(running)}):")
        print("-"*140)
        print(f"{'Job ID':<10} {'Traj':<8} {'Safe':<8} {'Help':<8} {'Elapsed':<10} {'ETA':<12} {'Agent':<15} {'Sim':<15} {'Type':<12} {'Quant':<7}")
        print("-"*140)

        for job in sorted(running, key=lambda x: x['job_id']):
            traj_progress, safe_progress, help_progress, filename, quant = get_trajectory_progress(job['job_id'])
            elapsed_sec = parse_time(job['time'])

            if traj_progress is not None:
                # Parse actual task count from filename
                total_tasks = parse_task_count_from_filename(filename)
                traj_str = f"{traj_progress}/{total_tasks}"
                safe_str = f"{safe_progress}/{total_tasks}"
                # Help includes both help and help_ignore_safety variants, so out of 2*n
                help_str = f"{help_progress}/{total_tasks * 2}"
                eta = estimate_completion(traj_progress, safe_progress, help_progress, elapsed_sec, total_tasks)
                agent, atype, sim = get_job_config(filename)
                quant_str = quant if quant else "?"
            else:
                traj_str = "?"
                safe_str = "?"
                help_str = "?"
                eta = "Unknown"
                agent, atype, sim = "?", "?", "?"
                quant_str = "?"

            elapsed_str = format_time(elapsed_sec)

            print(f"{job['job_id']:<10} {traj_str:<8} {safe_str:<8} {help_str:<8} {elapsed_str:<10} {eta:<12} "
                  f"{agent:<15} {sim:<15} {atype:<12} {quant_str:<7}")

    if show_pending and pending:
        print(f"\n⏳ PENDING JOBS ({len(pending)}):")
        print("-"*120)
        for job in sorted(pending, key=lambda x: x['job_id']):
            print(f"  Job {job['job_id']} - Waiting for resources ({job.get('node', 'N/A')})")

    print("\n" + "="*120)
    print(f"Total: {len(running)} running, {len(pending)} pending, {len(running) + len(pending)} total")
    print("="*120 + "\n")

def main():
    parser = argparse.ArgumentParser(description='Monitor toolemu job progress')
    parser.add_argument('--user', '-u', default=os.getenv('USER', 'bplaut'),
                       help='Username to monitor (default: current user)')
    parser.add_argument('--watch', '-w', type=int, metavar='SECONDS',
                       help='Refresh every N seconds (watch mode)')
    parser.add_argument('--no-pending', action='store_true',
                       help='Hide pending jobs')

    args = parser.parse_args()

    if args.watch:
        print(f"Watching jobs for {args.user}, refreshing every {args.watch} seconds...")
        print("Press Ctrl+C to stop.\n")
        try:
            while True:
                # Clear screen (works on Unix-like systems)
                os.system('clear' if os.name == 'posix' else 'cls')
                jobs = get_running_jobs(args.user)
                print_job_summary(jobs, show_pending=not args.no_pending)
                time.sleep(args.watch)
        except KeyboardInterrupt:
            print("\nMonitoring stopped.")
    else:
        jobs = get_running_jobs(args.user)
        print_job_summary(jobs, show_pending=not args.no_pending)

if __name__ == '__main__':
    main()
