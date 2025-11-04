#!/bin/bash
#SBATCH --job-name=toolemu
#SBATCH --output=logs/exp_%x_%j.out
#SBATCH --error=logs/exp_%x_%j.err
#SBATCH --nodes=1
#SBATCH --time=72:00:00
#SBATCH --qos=default
#SBATCH --mem=8G
# Note: This script does NOT request GPUs. For GPU workloads, use run_toolemu.sh

# Exit on any error
set -e

# Print commands being executed
# set -x

# Usage (positional arguments):
#   This script is called by submit_toolemu.sh with positional arguments.
#     no_gpu_run_toolemu.sh <input_path> <agent_model> <simulator_model> <evaluator_model> <agent_type> <quantization> [task_index_range]
#
#   Example:
#     sbatch --nodes=1 no_gpu_run_toolemu.sh \
#       ./assets/all_cases.json \
#       gpt-4o-mini \
#       gpt-5-mini \
#       gpt-5-mini \
#       quit \
#       int4 \
#       0-48

# Parse positional arguments
INPUT_PATH="$1"
AGENT_MODEL="$2"
SIMULATOR_MODEL="$3"
EVALUATOR_MODEL="$4"
AGENT_TYPE="$5"
QUANTIZATION="$6"
TASK_INDEX_RANGE="${7:-}"  # Optional, default to empty

# Validate required arguments
if [ -z "$INPUT_PATH" ] || [ -z "$AGENT_MODEL" ] || [ -z "$SIMULATOR_MODEL" ] || \
   [ -z "$EVALUATOR_MODEL" ] || [ -z "$AGENT_TYPE" ] || [ -z "$QUANTIZATION" ]; then
    echo "Error: Missing required positional arguments"
    echo "Usage: no_gpu_run_toolemu.sh <input_path> <agent_model> <simulator_model> <evaluator_model> <agent_type> <quantization> [task_index_range]"
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

if [ -n "$TASK_INDEX_RANGE" ]; then
    CMD+=(--task-index-range "$TASK_INDEX_RANGE")
fi

# Run the evaluation
echo "Running evaluation with models (CPU-only, no GPU):"
echo "  Agent: $AGENT_MODEL"
echo "  Simulator: $SIMULATOR_MODEL"
echo "  Evaluator: $EVALUATOR_MODEL"
echo "  Agent type: $AGENT_TYPE"
echo "  Quantization: $QUANTIZATION"
if [ -n "$TASK_INDEX_RANGE" ]; then
    echo "  Task index range: $TASK_INDEX_RANGE"
fi

"${CMD[@]}" || { echo "Evaluation failed"; exit 1; }

echo "Evaluation complete!"
