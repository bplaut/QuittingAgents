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
#     [--trunc-num 10]
#
# This will submit jobs for all combinations (cross product) of agent-model, agent-type, and quantization
# Example: 2 models × 3 types × 2 quantizations = 12 jobs

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
TASK_INDEX_RANGE=""
PARALLEL_SPLITS=""

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
        --task-index-range)
            TASK_INDEX_RANGE="$2"
            shift 2
            ;;
        --parallel-splits)
            PARALLEL_SPLITS="$2"
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

# Calculate task index ranges
# If parallel-splits specified, calculate ranges automatically
# Otherwise use task-index-range or full dataset
if [ -n "$PARALLEL_SPLITS" ]; then
    TOTAL_TASKS=144
    BASE_SIZE=$((TOTAL_TASKS / PARALLEL_SPLITS))
    REMAINDER=$((TOTAL_TASKS % PARALLEL_SPLITS))

    # Generate ranges array
    RANGES=()
    START=0
    for i in $(seq 0 $((PARALLEL_SPLITS - 1))); do
        # First REMAINDER jobs get BASE_SIZE+1, rest get BASE_SIZE
        if [ $i -lt $REMAINDER ]; then
            SIZE=$((BASE_SIZE + 1))
        else
            SIZE=$BASE_SIZE
        fi
        END=$((START + SIZE))
        RANGES+=("$START-$END")
        START=$END
    done
elif [ -n "$TASK_INDEX_RANGE" ]; then
    RANGES=("$TASK_INDEX_RANGE")
else
    RANGES=("")  # Full dataset
fi

# Calculate total number of jobs
TOTAL_JOBS=$((${#RANGES[@]} * ${#AGENT_MODELS[@]} * ${#AGENT_TYPES[@]} * ${#QUANTIZATIONS[@]}))
echo "========================================="
echo "Submitting $TOTAL_JOBS job(s) for the following cross product:"
echo "  Agent models: ${AGENT_MODELS[*]}"
echo "  Agent types: ${AGENT_TYPES[*]}"
echo "  Quantizations: ${QUANTIZATIONS[*]}"
if [ ${#RANGES[@]} -gt 1 ]; then
    echo "  Task ranges: ${RANGES[*]}"
elif [ -n "${RANGES[0]}" ]; then
    echo "  Task range: ${RANGES[0]}"
else
    echo "  Task range: (full dataset)"
fi
echo "  Simulator: $SIMULATOR_MODEL"
echo "  Evaluator: $EVALUATOR_MODEL"
echo "========================================="
echo ""

JOB_COUNT=0

# Generate and submit all combinations
for RANGE in "${RANGES[@]}"; do
    for AGENT_MODEL in "${AGENT_MODELS[@]}"; do
        for AGENT_TYPE in "${AGENT_TYPES[@]}"; do
            for QUANTIZATION in "${QUANTIZATIONS[@]}"; do
                JOB_COUNT=$((JOB_COUNT + 1))

                if [ -n "$RANGE" ]; then
                    echo "[$JOB_COUNT/$TOTAL_JOBS] Submitting: agent=$AGENT_MODEL, type=$AGENT_TYPE, quant=$QUANTIZATION, range=$RANGE"
                else
                    echo "[$JOB_COUNT/$TOTAL_JOBS] Submitting: agent=$AGENT_MODEL, type=$AGENT_TYPE, quant=$QUANTIZATION"
                fi

                # Calculate GPU requirements
                AGENT_SIZE=$(get_model_size "$AGENT_MODEL")
                SIMULATOR_SIZE=$(get_model_size "$SIMULATOR_MODEL")
                TOTAL_SIZE=$((AGENT_SIZE + SIMULATOR_SIZE))

                # Build positional arguments
                # Order: input_path agent_model simulator_model evaluator_model agent_type quantization [task_index_range]
                RUN_ARGS=(
                    "$INPUT_PATH"
                    "$AGENT_MODEL"
                    "$SIMULATOR_MODEL"
                    "$EVALUATOR_MODEL"
                    "$AGENT_TYPE"
                    "$QUANTIZATION"
                )

                # Add optional task-index-range if set
                if [ -n "$RANGE" ]; then
                    RUN_ARGS+=("$RANGE")
                fi

            # If both agent and simulator are API models (size 0), use no-GPU script
            if [ "$AGENT_SIZE" -eq 0 ] && [ "$SIMULATOR_SIZE" -eq 0 ]; then
                # Both are API models - no GPU needed
                sbatch --nodes=1 no_gpu_run_toolemu.sh "${RUN_ARGS[@]}"
            else
                # At least one HuggingFace model - need GPU
                # Adjust threshold based on quantization
                # int4: ~0.5GB/B, int8: ~1GB/B, fp16: ~2GB/B
                # For int8, use stricter threshold since it uses 2x memory vs int4
                if [ "$QUANTIZATION" = "int8" ]; then
                    THRESHOLD=35  # More conservative for int8 (35B threshold ~40GB)
                else
                    THRESHOLD=70  # int4 or none (70B threshold fits on 48GB nodes)
                fi

                if [ "$TOTAL_SIZE" -gt "$THRESHOLD" ]; then
                    NODELIST="airl.ist.berkeley.edu,sac.ist.berkeley.edu,cirl.ist.berkeley.edu,rlhf.ist.berkeley.edu"
                else
                    NODELIST="airl.ist.berkeley.edu,cirl.ist.berkeley.edu,rlhf.ist.berkeley.edu,gan.ist.berkeley.edu,ddpg.ist.berkeley.edu,dqn.ist.berkeley.edu"
                fi

                sbatch --nodes=1 --nodelist="$NODELIST" run_toolemu.sh "${RUN_ARGS[@]}"
            fi

            done
        done
    done
done

echo ""
echo "========================================="
echo "Submitted $JOB_COUNT job(s) successfully!"
echo "========================================="
