#!/bin/bash
#SBATCH --job-name=toolemu
#SBATCH --output=logs/exp_%x_%j.out
#SBATCH --error=logs/exp_%x_%j.err
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --nodelist=airl.ist.berkeley.edu,cirl.ist.berkeley.edu,rlhf.ist.berkeley.edu,gan.ist.berkeley.edu,ddpg.ist.berkeley.edu,dqn.ist.berkeley.edu
#SBATCH --time=24:00:00
#SBATCH --qos=high

# Exit on any error
set -e

# Print commands being executed
# set -x

# Usage: sbatch run_toolemu_os.sh <input_path> <agent_model> <simulator_model> <evaluator_model> <agent_type> <trunc_num> [additional_args]
INPUT_PATH=${1:-./assets/all_cases.json}
AGENT_MODEL=${2:-Qwen/Qwen2.5-1.5B-Instruct}
SIMULATOR_MODEL=${3:-Qwen/Qwen2.5-1.5B-Instruct}
EVALUATOR_MODEL=${4:-Qwen/Qwen2.5-1.5B-Instruct}
AGENT_TYPE=${5:-naive}
TRUNC_NUM=${6:-1000}
# Capture all additional arguments
ADDITIONAL_ARGS="${@:7}"

# Initialize conda properly for the batch job
eval "$(/nas/ucb/bplaut/miniconda3/bin/conda shell.bash hook)"
conda activate llm-finetune || { echo "Failed to activate conda environment"; exit 1; }

# Set HuggingFace token for accessing gated models

# Change to the correct directory
cd /nas/ucb/bplaut/QuittingAgents || { echo "Failed to change directory"; exit 1; }

# Create logs directory if it doesn't exist
mkdir -p logs

# Start GPU monitoring if GPUs are requested
if [ "${SLURM_GPUS_PER_NODE:-0}" -gt 0 ] || [ "${SLURM_JOB_GPUS:-0}" -gt 0 ] || grep -q -- '--gpus=[1-9]' <<< "$(head -n 10 $0)"; then
    (while true; do nvidia-smi >> logs/gpu_usage_${SLURM_JOB_ID}.log; sleep 30; done) &
    GPU_MON_PID=$!
fi

# 3. Start vLLM server for the agent model
echo "Starting vLLM server for model: $AGENT_MODEL"
python -m vllm.entrypoints.openai.api_server \
    --model "$AGENT_MODEL" \
    --port 8000 \
    --tensor-parallel-size 1 \
    --gpu-memory-utilization 0.85 \
    --disable-log-requests \
    > logs/vllm_server_${SLURM_JOB_ID}.log 2>&1 &
VLLM_PID=$!
echo "vLLM server started with PID: $VLLM_PID"

# Wait for vLLM server to be ready
echo "Waiting for vLLM server to be ready..."
for i in {1..300}; do
    if curl -s http://localhost:8000/v1/models > /dev/null 2>&1; then
        echo "vLLM server is ready!"
        break
    fi
    if [ $i -eq 300 ]; then
        echo "vLLM server failed to start within 300 seconds"
        kill $VLLM_PID 2>/dev/null || true
        exit 1
    fi
    sleep 1
done

# 4. Run the evaluation
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
    $ADDITIONAL_ARGS \
    || { echo "Evaluation failed"; kill $VLLM_PID 2>/dev/null || true; exit 1; }

# Stop vLLM server
echo "Stopping vLLM server (PID: $VLLM_PID)"
kill $VLLM_PID 2>/dev/null || true
# Give it a moment to shut down gracefully, then force kill if needed
sleep 2
kill -9 $VLLM_PID 2>/dev/null || true

# Stop GPU monitoring if started
if [ ! -z "${GPU_MON_PID:-}" ]; then
    kill $GPU_MON_PID 2>/dev/null || true
fi

# 5. Final GPU usage snapshot
echo "Final GPU usage snapshot saved to logs/gpu_usage_${SLURM_JOB_ID}.log" 
