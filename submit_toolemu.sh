#!/bin/bash
# Smart wrapper for run_toolemu.sh that automatically selects appropriate GPU nodes based on model sizes
# and handles cross-product submission of multiple agent models, types, and quantization levels
#
# Usage:
#   ./submit_toolemu.sh \
#     --input-path ./assets/all_cases.json \
#     --agent-model Qwen/Qwen3-8B meta-llama/Llama-3.1-8B-Instruct \
#     --simulator-model Qwen/Qwen3-32B \
#     --evaluator-model Qwen/Qwen3-32B \
#     --agent-type naive quit simple_quit \
#     --quantization int4 int8 \
#     --help-ignore-safety true false \
#     [--trunc-num 10]
#
# This will submit jobs for all combinations (cross product) of agent-model, agent-type, quantization, and help-ignore-safety
# Example: 2 models × 3 types × 2 quantizations × 2 help modes = 24 jobs

set -e

# Function to extract model size in billions of parameters
# Returns 0 for API models (gpt, claude, etc.)
get_model_size() {
    local model=$1

    # API models don't use GPU memory
    if echo "$model" | grep -qiE '^(gpt-|claude-|gemini-)'; then
        echo 0
        return
    fi

    # Extract number before B (e.g., "8B" -> 8, "70B" -> 70, "32B" -> 32)
    local size=$(echo "$model" | grep -oE '[0-9]+B' | grep -oE '[0-9]+' | head -1)

    if [ -z "$size" ]; then
        echo 0
    else
        echo "$size"
    fi
}

# Parse named arguments
INPUT_PATH=""
AGENT_MODELS=()
SIMULATOR_MODEL=""
EVALUATOR_MODEL=""
AGENT_TYPES=()
QUANTIZATIONS=()
HELP_IGNORE_SAFETY_MODES=()
TRUNC_NUM=""

while [[ $# -gt 0 ]]; do
    case $1 in
        --input-path)
            INPUT_PATH="$2"
            shift 2
            ;;
        --agent-model)
            shift
            while [[ $# -gt 0 ]] && [[ ! "$1" =~ ^-- ]]; do
                AGENT_MODELS+=("$1")
                shift
            done
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
            shift
            while [[ $# -gt 0 ]] && [[ ! "$1" =~ ^-- ]]; do
                AGENT_TYPES+=("$1")
                shift
            done
            ;;
        --quantization)
            shift
            while [[ $# -gt 0 ]] && [[ ! "$1" =~ ^-- ]]; do
                QUANTIZATIONS+=("$1")
                shift
            done
            ;;
        --help-ignore-safety)
            shift
            while [[ $# -gt 0 ]] && [[ ! "$1" =~ ^-- ]]; do
                HELP_IGNORE_SAFETY_MODES+=("$1")
                shift
            done
            ;;
        --trunc-num)
            TRUNC_NUM="$2"
            shift 2
            ;;
        *)
            echo "Unknown argument: $1"
            exit 1
            ;;
    esac
done

# Validate required arguments
if [ -z "$INPUT_PATH" ]; then
    echo "Error: --input-path is required"
    exit 1
fi

if [ ${#AGENT_MODELS[@]} -eq 0 ]; then
    echo "Error: At least one --agent-model is required"
    exit 1
fi

if [ -z "$SIMULATOR_MODEL" ]; then
    echo "Error: --simulator-model is required"
    exit 1
fi

if [ -z "$EVALUATOR_MODEL" ]; then
    echo "Error: --evaluator-model is required"
    exit 1
fi

if [ ${#AGENT_TYPES[@]} -eq 0 ]; then
    echo "Error: At least one --agent-type is required"
    exit 1
fi

if [ ${#QUANTIZATIONS[@]} -eq 0 ]; then
    echo "Error: At least one --quantization is required"
    exit 1
fi

if [ ${#HELP_IGNORE_SAFETY_MODES[@]} -eq 0 ]; then
    echo "Error: At least one --help-ignore-safety value is required (use: true false)"
    exit 1
fi

# Calculate total number of jobs
TOTAL_JOBS=$((${#AGENT_MODELS[@]} * ${#AGENT_TYPES[@]} * ${#QUANTIZATIONS[@]} * ${#HELP_IGNORE_SAFETY_MODES[@]}))
echo "========================================="
echo "Submitting $TOTAL_JOBS job(s) for cross product:"
echo "  Agent models: ${AGENT_MODELS[*]}"
echo "  Agent types: ${AGENT_TYPES[*]}"
echo "  Quantizations: ${QUANTIZATIONS[*]}"
echo "  Help ignore safety: ${HELP_IGNORE_SAFETY_MODES[*]}"
echo "  Simulator: $SIMULATOR_MODEL"
echo "  Evaluator: $EVALUATOR_MODEL"
if [ -n "$TRUNC_NUM" ]; then
    echo "  Trunc num: $TRUNC_NUM"
else
    echo "  Trunc num: (full dataset)"
fi
echo "========================================="
echo ""

JOB_COUNT=0

# Generate and submit all combinations
for AGENT_MODEL in "${AGENT_MODELS[@]}"; do
    for AGENT_TYPE in "${AGENT_TYPES[@]}"; do
        for QUANTIZATION in "${QUANTIZATIONS[@]}"; do
            for HELP_MODE in "${HELP_IGNORE_SAFETY_MODES[@]}"; do
                JOB_COUNT=$((JOB_COUNT + 1))

                echo "[$JOB_COUNT/$TOTAL_JOBS] Submitting: agent=$AGENT_MODEL, type=$AGENT_TYPE, quant=$QUANTIZATION, help_ignore_safety=$HELP_MODE"

            # Calculate GPU requirements
            AGENT_SIZE=$(get_model_size "$AGENT_MODEL")
            SIMULATOR_SIZE=$(get_model_size "$SIMULATOR_MODEL")
            TOTAL_SIZE=$((AGENT_SIZE + SIMULATOR_SIZE))

            if [ "$TOTAL_SIZE" -gt 70 ]; then
                NODELIST="airl.ist.berkeley.edu,sac.ist.berkeley.edu,cirl.ist.berkeley.edu,rlhf.ist.berkeley.edu"
            else
                NODELIST="airl.ist.berkeley.edu,cirl.ist.berkeley.edu,rlhf.ist.berkeley.edu,gan.ist.berkeley.edu,ddpg.ist.berkeley.edu,dqn.ist.berkeley.edu"
            fi

                # Build arguments for run_toolemu.sh
                RUN_ARGS=(
                    --input-path "$INPUT_PATH"
                    --agent-model "$AGENT_MODEL"
                    --simulator-model "$SIMULATOR_MODEL"
                    --evaluator-model "$EVALUATOR_MODEL"
                    --agent-type "$AGENT_TYPE"
                    --quantization "$QUANTIZATION"
                )

                if [ -n "$TRUNC_NUM" ]; then
                    RUN_ARGS+=(--trunc-num "$TRUNC_NUM")
                fi

                if [ "$HELP_MODE" = "true" ]; then
                    RUN_ARGS+=(--help-ignore-safety)
                fi

                # Submit job
                sbatch --nodes=1 --nodelist="$NODELIST" run_toolemu.sh "${RUN_ARGS[@]}"

            done
        done
    done
done

echo ""
echo "========================================="
echo "Submitted $JOB_COUNT job(s) successfully!"
echo "========================================="
