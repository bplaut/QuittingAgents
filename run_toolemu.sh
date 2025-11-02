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

# Usage:
#   Option 1 (recommended): Use submit_toolemu.sh, which calls this script
#   Option 2: Call this script directly:
#      sbatch --nodes=1 --nodelist=<nodes> run_toolemu.sh \
#        --input-path ./assets/all_cases.json \
#        --agent-model Qwen/Qwen3-8B \
#        --simulator-model Qwen/Qwen3-32B \
#        --evaluator-model Qwen/Qwen3-32B \
#        --agent-type quit \
#        --quantization int4 \
#        [--trunc-num 10] \
#        [--help-ignore-safety]

# Parse named arguments
INPUT_PATH=""
AGENT_MODEL=""
SIMULATOR_MODEL=""
EVALUATOR_MODEL=""
AGENT_TYPE=""
QUANTIZATION=""
TRUNC_NUM=""
HELP_IGNORE_SAFETY=false

while [[ $# -gt 0 ]]; do
    case $1 in
        --input-path)
            INPUT_PATH="$2"
            shift 2
            ;;
        --agent-model)
            AGENT_MODEL="$2"
            shift 2
            ;;
        --simulator-model)
            SIMULATOR_MODEL="$2"
            shift 2
            ;;
        --evaluator-model)
            EVALUATOR_MODEL="$2"
            shift 2
            ;;
        --agent-type)
            AGENT_TYPE="$2"
            shift 2
            ;;
        --quantization)
            QUANTIZATION="$2"
            shift 2
            ;;
        --trunc-num)
            TRUNC_NUM="$2"
            shift 2
            ;;
        --help-ignore-safety)
            HELP_IGNORE_SAFETY=true
            shift
            ;;
        *)
            echo "Unknown argument: $1"
            exit 1
            ;;
    esac
done

# Validate required arguments
if [ -z "$INPUT_PATH" ] || [ -z "$AGENT_MODEL" ] || [ -z "$SIMULATOR_MODEL" ] || \
   [ -z "$EVALUATOR_MODEL" ] || [ -z "$AGENT_TYPE" ] || [ -z "$QUANTIZATION" ]; then
    echo "Error: Missing required arguments"
    echo "Required: --input-path, --agent-model, --simulator-model, --evaluator-model, --agent-type, --quantization"
    echo "Optional: --trunc-num, --help-ignore-safety"
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

if [ "$HELP_IGNORE_SAFETY" = true ]; then
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
