#!/bin/bash
# Smart wrapper for run_toolemu.sh that automatically selects appropriate GPU nodes based on model sizes
# Usage: ./submit_toolemu.sh <input_path> <agent_model> <simulator_model> <evaluator_model> <agent_type> <trunc_num> [additional_args]

# Check that all required arguments are provided
if [ $# -lt 6 ]; then
    echo "Error: Missing required arguments"
    echo "Usage: ./submit_toolemu.sh <input_path> <agent_model> <simulator_model> <evaluator_model> <agent_type> <trunc_num> [additional_args]"
    echo "  input_path:       Path to test cases JSON file"
    echo "  agent_model:      Model name for agent (e.g., meta-llama/Llama-3.1-8B-Instruct)"
    echo "  simulator_model:  Model name for simulator"
    echo "  evaluator_model:  Model name for evaluator"
    echo "  agent_type:       Agent type (e.g., naive, quit, simple_quit)"
    echo "  trunc_num:        Number of test cases to run"
    exit 1
fi

AGENT_MODEL=$2
SIMULATOR_MODEL=$3
EVALUATOR_MODEL=$4

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

# Get model sizes
AGENT_SIZE=$(get_model_size "$AGENT_MODEL")
SIMULATOR_SIZE=$(get_model_size "$SIMULATOR_MODEL")

# Calculate total GPU memory needed (agent + simulator, since evaluator == simulator)
TOTAL_SIZE=$((AGENT_SIZE + SIMULATOR_SIZE))

echo "Model sizes: Agent=${AGENT_SIZE}B, Simulator/Evaluator=${SIMULATOR_SIZE}B, Total=${TOTAL_SIZE}B"

# Threshold for 80GB GPUs: if total > 50B parameters, use 80GB nodes
# With int4 quantization: ~50B params uses ~25GB, leaving room for activations
# 40GB GPUs can handle up to ~50B total, 80GB GPUs can handle 120B+
if [ "$TOTAL_SIZE" -gt 50 ]; then
    NODELIST="airl.ist.berkeley.edu,sac.ist.berkeley.edu,cirl.ist.berkeley.edu,rlhf.ist.berkeley.edu"
    echo "Total model size ${TOTAL_SIZE}B > 50B threshold, requesting 80GB GPU nodes"
else
    NODELIST="airl.ist.berkeley.edu,cirl.ist.berkeley.edu,rlhf.ist.berkeley.edu,gan.ist.berkeley.edu,ddpg.ist.berkeley.edu,dqn.ist.berkeley.edu"
    echo "Total model size ${TOTAL_SIZE}B <= 50B threshold, using standard GPU nodes"
fi

# Submit the job with the appropriate nodelist
sbatch --nodes=1 --nodelist="$NODELIST" run_toolemu.sh "$@"
