#!/bin/bash
#SBATCH --job-name=toolemu
#SBATCH --output=logs/exp_%x_%j.out
#SBATCH --error=logs/exp_%x_%j.err
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --time=24:00:00
#SBATCH --qos=default
# Note: --nodelist should be passed as a command-line argument to sbatch

# Exit on any error
set -e

# Print commands being executed
# set -x

# Usage:
#   Option 1 is to use submit_toolemu.sh, which calls this script: ./submit_toolemu.sh <input_path> <agent_model> <simulator_model> <evaluator_model> <agent_type> <trunc_num> [additional_args]
#   Option 2 is to call this script directly:      sbatch --nodes=1 --nodelist=<nodes> run_toolemu.sh <input_path> <agent_model> <simulator_model> <evaluator_model> <agent_type> <trunc_num> [additional_args]
#
# Option 1 is recommended because it automatically selects appropriate GPU nodes (80GB vs standard) based on model sizes. Argument validation is kept here in case of Option 2.

# Check that all required arguments are provided
if [ $# -lt 6 ]; then
    echo "Error: Missing required arguments"
    echo "Usage: sbatch run_toolemu_os.sh <input_path> <agent_model> <simulator_model> <evaluator_model> <agent_type> <trunc_num> [additional_args]"
    echo "  input_path:       Path to test cases JSON file"
    echo "  agent_model:      Model name for agent (e.g., meta-llama/Llama-3.1-8B-Instruct)"
    echo "  simulator_model:  Model name for simulator"
    echo "  evaluator_model:  Model name for evaluator"
    echo "  agent_type:       Agent type (e.g., naive, quit, simple_quit)"
    echo "  trunc_num:        Number of test cases to run"
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

# Run the evaluation (API models will ignore quantization args)
echo "Running evaluation with models:"
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

# Stop GPU monitoring if started
if [ ! -z "${GPU_MON_PID:-}" ]; then
    kill $GPU_MON_PID 2>/dev/null || true
fi

# Final GPU usage snapshot
echo "Final GPU usage snapshot saved to logs/gpu_usage_${SLURM_JOB_ID}.log" 
