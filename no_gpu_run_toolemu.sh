#!/bin/bash
#SBATCH --job-name=toolemu_nogpu
#SBATCH --output=logs/exp_%x_%j.out
#SBATCH --error=logs/exp_%x_%j.err
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32GB
#SBATCH --time=24:00:00
#SBATCH --qos=default

# Exit on any error
set -e

# Print commands being executed
# set -x

# Usage: sbatch run_toolemu_no_gpu.sh <input_path> <agent_model> <simulator_model> <evaluator_model> <agent_type> <trunc_num> [additional_args]

# Check that all required arguments are provided
if [ $# -lt 6 ]; then
    echo "Error: Missing required arguments"
    echo "Usage: sbatch run_toolemu_no_gpu.sh <input_path> <agent_model> <simulator_model> <evaluator_model> <agent_type> <trunc_num> [additional_args]"
    echo "  input_path:       Path to test cases JSON file"
    echo "  agent_model:      Model name for agent (e.g., gpt-4o-mini, claude-sonnet-4)"
    echo "  simulator_model:  Model name for simulator"
    echo "  evaluator_model:  Model name for evaluator"
    echo "  agent_type:       Agent type (e.g., naive, quit, simple_quit)"
    echo "  trunc_num:        Number of test cases to run"
    echo ""
    echo "Note: This script does NOT request GPUs. Use run_toolemu_os.sh for HuggingFace models."
    exit 1
fi

INPUT_PATH=$1
AGENT_MODEL=$2
SIMULATOR_MODEL=$3
EVALUATOR_MODEL=$4
AGENT_TYPE=$5
TRUNC_NUM=$6
# Capture all additional arguments
ADDITIONAL_ARGS="${@:7}"

# Initialize conda properly for the batch job
eval "$(/nas/ucb/bplaut/miniconda3/bin/conda shell.bash hook)"
conda activate llm-finetune || { echo "Failed to activate conda environment"; exit 1; }

# Load environment variables from .env file (including API keys)
if [ -f "/nas/ucb/bplaut/QuittingAgents/.env" ]; then
    export $(grep -v '^#' /nas/ucb/bplaut/QuittingAgents/.env | xargs)
fi

# Set HuggingFace token from environment variables (in case mixing API + HF models)
if [ -n "${HF_KEY:-}" ]; then
    export HF_TOKEN="$HF_KEY"
    export HUGGING_FACE_HUB_TOKEN="$HF_KEY"
fi

# Change to the correct directory
cd /nas/ucb/bplaut/QuittingAgents || { echo "Failed to change directory"; exit 1; }

# Create logs directory if it doesn't exist
mkdir -p logs

# Run the evaluation (quantization args will be ignored by API models)
echo "Running evaluation with models (CPU-only, no GPU):"
echo "  Agent: $AGENT_MODEL"
echo "  Simulator: $SIMULATOR_MODEL"
echo "  Evaluator: $EVALUATOR_MODEL"
python scripts/run.py \
    --agent-model-name "$AGENT_MODEL" \
    --simulator-model-name "$SIMULATOR_MODEL" \
    --evaluator-model-name "$EVALUATOR_MODEL" \
    --input-path "$INPUT_PATH" \
    --agent-type "$AGENT_TYPE" \
    --auto \
    --track-costs \
    --trunc-num "$TRUNC_NUM" \
    -bs 1 \
    --agent-quantization int4 \
    --simulator-quantization int4 \
    --evaluator-quantization int4 \
    $ADDITIONAL_ARGS \
    || { echo "Evaluation failed"; exit 1; }

echo "Evaluation complete!"
