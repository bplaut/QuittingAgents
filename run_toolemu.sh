#!/bin/bash
#SBATCH --job-name=toolemu
#SBATCH --output=logs/exp_%x_%j.out
#SBATCH --error=logs/exp_%x_%j.err
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --time=72:00:00
#SBATCH --qos=default
# Note: --nodelist should be passed as a command-line argument to sbatch

# Exit on any error
set -e

# Print commands being executed
# set -x

# Usage (positional arguments):
#   This script is called by submit_toolemu.sh with positional arguments:
#     run_toolemu.sh <input_path> <agent_model> <simulator_model> <evaluator_model> <agent_type> <quantization> [trunc_num] [help_ignore_safety]
#
#   Example:
#     sbatch --nodes=1 --nodelist=<nodes> run_toolemu.sh \
#       ./assets/all_cases.json \
#       Qwen/Qwen3-8B \
#       Qwen/Qwen3-32B \
#       Qwen/Qwen3-32B \
#       quit \
#       int4 \
#       10 \
#       true

# Parse positional arguments
INPUT_PATH="$1"
AGENT_MODEL="$2"
SIMULATOR_MODEL="$3"
EVALUATOR_MODEL="$4"
AGENT_TYPE="$5"
QUANTIZATION="$6"
TRUNC_NUM="${7:-}"  # Optional, default to empty
HELP_IGNORE_SAFETY="${8:-false}"  # Optional, default to false

# Validate required arguments
if [ -z "$INPUT_PATH" ] || [ -z "$AGENT_MODEL" ] || [ -z "$SIMULATOR_MODEL" ] || \
   [ -z "$EVALUATOR_MODEL" ] || [ -z "$AGENT_TYPE" ] || [ -z "$QUANTIZATION" ]; then
    echo "Error: Missing required positional arguments"
    echo "Usage: run_toolemu.sh <input_path> <agent_model> <simulator_model> <evaluator_model> <agent_type> <quantization> [trunc_num] [help_ignore_safety]"
    exit 1
fi

# Initialize conda properly for the batch job
eval "$(/nas/ucb/bplaut/miniconda3/bin/conda shell.bash hook)"
conda activate llm-finetune || { echo "Failed to activate conda environment"; exit 1; }

# Load environment variables from .env file (including HuggingFace token)
if [ -f "/nas/ucb/bplaut/QuittingAgents/.env" ]; then
    export $(grep -v '^#' /nas/ucb/bplaut/QuittingAgents/.env | xargs)
fi

# Set HuggingFace token from environment variables for accessing gated models
if [ -n "${HF_KEY:-}" ]; then
    export HF_TOKEN="$HF_KEY"
    export HUGGING_FACE_HUB_TOKEN="$HF_KEY"
fi

# Change to the correct directory
cd /nas/ucb/bplaut/QuittingAgents || { echo "Failed to change directory"; exit 1; }

# Create logs directory if it doesn't exist
mkdir -p logs

# Start GPU monitoring if GPUs are requested
if [ "${SLURM_GPUS_PER_NODE:-0}" -gt 0 ] || [ "${SLURM_JOB_GPUS:-0}" -gt 0 ] || grep -q -- '--gpus=[1-9]' <<< "$(head -n 10 $0)"; then
    (while true; do nvidia-smi >> logs/gpu_usage_${SLURM_JOB_ID}.log; sleep 30; done) &
    GPU_MON_PID=$!
fi

# Build command for scripts/run.py
CMD=(
    python scripts/run.py
    --agent-model-name "$AGENT_MODEL"
    --simulator-model-name "$SIMULATOR_MODEL"
    --evaluator-model-name "$EVALUATOR_MODEL"
    --input-path "$INPUT_PATH"
    --agent-type "$AGENT_TYPE"
    --quantization "$QUANTIZATION"
    --auto
    --track-costs
    -bs 1
)

if [ -n "$TRUNC_NUM" ]; then
    CMD+=(--trunc-num "$TRUNC_NUM")
fi

if [ "$HELP_IGNORE_SAFETY" = "true" ]; then
    CMD+=(--help-ignore-safety)
fi

# Run the evaluation
echo "Running evaluation with models:"
echo "  Agent: $AGENT_MODEL"
echo "  Simulator: $SIMULATOR_MODEL"
echo "  Evaluator: $EVALUATOR_MODEL"
echo "  Agent type: $AGENT_TYPE"
echo "  Quantization: $QUANTIZATION"
if [ -n "$TRUNC_NUM" ]; then
    echo "  Trunc num: $TRUNC_NUM"
fi
echo "  Help ignore safety: $HELP_IGNORE_SAFETY"

"${CMD[@]}" || { echo "Evaluation failed"; exit 1; }

# Stop GPU monitoring if started
if [ ! -z "${GPU_MON_PID:-}" ]; then
    kill $GPU_MON_PID 2>/dev/null || true
fi

# Final GPU usage snapshot
echo "Final GPU usage snapshot saved to logs/gpu_usage_${SLURM_JOB_ID}.log"
