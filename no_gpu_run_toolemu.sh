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
#     no_gpu_run_toolemu.sh <input_path> <agent_model> <simulator_model> <evaluator_model> <agent_type> <quantization> <help_ignore_safety> [trunc_num]
#
#   Example:
#     sbatch --nodes=1 no_gpu_run_toolemu.sh \
#       ./assets/all_cases.json \
#       gpt-4o-mini \
#       gpt-5-mini \
#       gpt-5-mini \
#       quit \
#       int4 \
#       true \
#       10

# Parse positional arguments
INPUT_PATH="$1"
AGENT_MODEL="$2"
SIMULATOR_MODEL="$3"
EVALUATOR_MODEL="$4"
AGENT_TYPE="$5"
QUANTIZATION="$6"
HELP_IGNORE_SAFETY="$7"  # Required: true or false
TRUNC_NUM="${8:-}"  # Optional, default to empty

# Validate required arguments
if [ -z "$INPUT_PATH" ] || [ -z "$AGENT_MODEL" ] || [ -z "$SIMULATOR_MODEL" ] || \
   [ -z "$EVALUATOR_MODEL" ] || [ -z "$AGENT_TYPE" ] || [ -z "$QUANTIZATION" ] || \
   [ -z "$HELP_IGNORE_SAFETY" ]; then
    echo "Error: Missing required positional arguments"
    echo "Usage: no_gpu_run_toolemu.sh <input_path> <agent_model> <simulator_model> <evaluator_model> <agent_type> <quantization> <help_ignore_safety> [trunc_num]"
    exit 1
fi

# Validate help_ignore_safety is either "true" or "false"
if [ "$HELP_IGNORE_SAFETY" != "true" ] && [ "$HELP_IGNORE_SAFETY" != "false" ]; then
    echo "Error: help_ignore_safety must be either 'true' or 'false', got: '$HELP_IGNORE_SAFETY'"
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

if [ -n "$TRUNC_NUM" ]; then
    CMD+=(--trunc-num "$TRUNC_NUM")
fi

if [ "$HELP_IGNORE_SAFETY" = "true" ]; then
    CMD+=(--help-ignore-safety)
fi

# Run the evaluation
echo "Running evaluation with models (CPU-only, no GPU):"
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

echo "Evaluation complete!"
