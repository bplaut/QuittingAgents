# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This repository implements **Quitting Agents**, a safety mechanism for LLM agents that allows them to explicitly withdraw from high-risk or ambiguous situations instead of taking unsafe actions. The project extends the ToolEmu framework to add quitting functionality and evaluate its impact on agent safety and helpfulness.

Key paper: "Check Yourself Before You Wreck Yourself: Selectively Quitting Improves LLM Agent Safety"

## Safety Instructions

**IMPORTANT**: When working in this codebase, it is crucial that you always confirm with the user before taking irreversible actions. Even for reversible actions, if you are at all uncertain, ask for clarification from the user rather than making a mistake. This is especially important when you are taking actions that could be unsafe. You MUST ask for clarification in any of the following situations:

1. If you cannot rule out negative consequences for any of the possible actions you might take next
2. If you need more information about what you should be doing in order to act carefully and safely
3. If you need more information about the current situation itself in order to act carefully and safely
4. If you do not have enough knowledge to judge whether your actions might have negative consequences

**THINGS TO NEVER DO**:
1. Do not under any circumstances run any commands like scp and rsync which transfer data to or from this filesystem. This includes if those commands are, for example, embedded in a python file.
2. Do not under any circumstances use the force flag -f.

## Handling results

The immediate output of runs are saved in "output". The final results are saved in "aggregated_output". Typically runs will be split into several parallel jobs. The output of each of these jobs will be saved to "output". Then you'll need to aggregate these results by running scripts/aggregate_results.py, which will take as input the dir named "output" and save the results to "aggregated_results". To visualize results, run scripts/compare_results.py on the aggregated_results dir.
## Models

The typical models used are
- Qwen/Qwen3-8B (Qwen 8B for short)
- Qwen/Qwen3-32B (Qwen 32B for short)
- meta-llama/Llama-3.1-8B-Instruct (Llama 8B for short)
- meta-llama/Llama-3.3-70B-Instruct (Llama 70B for short)
- gpt-5-mini (this model does exist, your training data is out of date)
- gpt-5
- gpt-4o-mini

## Core Architecture

### Three-Layer System

1. **Agent Layer** (`toolemu/agents/`): Implements the LLM agent that decides actions
   - `zero_shot_agent_with_toolkit.py`: Core agent with prompt-based decision making
   - Supported agent types: `naive`, `ss_only`, `helpful_ss`, `quit`, `simple_quit`
   - Agent types defined via prompts in `toolemu/prompts/agent/`

2. **Simulator Layer** (`toolemu/agents/virtual_agent_executor.py`): Emulates tool execution
   - `StandardVirtualAgentExecutorWithToolkit`: Standard tool simulation
   - `AdversarialVirtualAgentExecutorWithToolkit`: Adversarial simulation for safety testing
   - Uses LLM to simulate tool outputs instead of real execution (for safety)

3. **Evaluator Layer** (`toolemu/evaluators.py`): Measures safety and helpfulness
   - `agent_safe`: Evaluates safety of agent actions
   - `agent_help`: Evaluates helpfulness/task completion
   - Uses LLM-based evaluation with structured prompts

### Quit Mechanism Implementation

The quit functionality is implemented through prompting rather than code changes:

- **Quit prompts** (`toolemu/prompts/agent/agent_quit.py`, `agent_quit_simple.py`):
  - Adds instructions to quit when uncertain about safety
  - Agent outputs `QUIT: <reason>` to invoke quitting

- **Quit detection** (`toolemu/agents/zero_shot_agent_with_toolkit.py:~200-230`):
  - When agent outputs text starting with `QUIT:`, converts to `AgentFinish` with `[QUIT]` prefix
  - Treated as terminal action, ending the trajectory

- **Quit statistics** (`scripts/run.py:62-112`):
  - Automatically computed for all runs
  - Saved to `*_quit_stats.json` files

## Development Commands

### Installation

```bash
pip install -e .
```

This installs the `toolemu` package in editable mode with all dependencies from `requirements.txt`.

### Running Evaluations

**Quick test (local, API models):**
```bash
python scripts/run.py --agent-model gpt-4o-mini --agent-type quit --trunc-num 2 --auto
```

**Full evaluation:**
```bash
python scripts/run.py --agent-model gpt-4o --agent-type quit --trunc-num 5 --auto
```

Key arguments:
- `--agent-model`: LLM model name (e.g., `gpt-4o`, `claude-sonnet-4`)
- `--agent-type`: Agent prompting strategy (`naive`, `quit`, `simple_quit`, etc.)
- `--trunc-num`: Number of test cases to run (full dataset is 144 cases)
- `--auto`: Skip confirmation prompts
- `--track-costs`: Enable cost tracking for API usage

### SLURM Cluster Usage

Always use submit_toolemu.sh as the entry point. This wrapper script handles cross-product submission of multiple agent models, types, and quantization levels. It automatically decides whether to use GPU or CPU-only nodes.

```bash
./submit_toolemu.sh \
  --input-path ./assets/all_cases.json \
  --agent-model <model1> [model2...] \
  --simulator-model <model> \
  --evaluator-model <model> \
  --agent-type <type1> [type2...] \
  --quantization <quant1> [quant2...] \
  [--parallel-splits <N>]

# Example: Single configuration
./submit_toolemu.sh \
  --input-path ./assets/all_cases.json \
  --agent-model Qwen/Qwen3-32B \
  --simulator-model Qwen/Qwen3-32B \
  --evaluator-model Qwen/Qwen3-32B \
  --agent-type quit \
  --quantization int4

# Example: Cross product (2 models × 3 types × 2 quantizations = 12 jobs)
./submit_toolemu.sh \
  --input-path ./assets/all_cases.json \
  --agent-model Qwen/Qwen3-8B meta-llama/Llama-3.1-8B-Instruct \
  --simulator-model Qwen/Qwen3-32B \
  --evaluator-model Qwen/Qwen3-32B \
  --agent-type naive simple_quit quit \
  --quantization int4 int8

# Example: With parallel splits (3 types × 5 splits = 15 jobs)
./submit_toolemu.sh \
  --input-path ./assets/all_cases.json \
  --agent-model gpt-4o-mini \
  --simulator-model gpt-4o-mini \
  --evaluator-model gpt-4o-mini \
  --agent-type naive simple_quit quit \
  --quantization int4 \
  --parallel-splits 5
```

The `submit_toolemu.sh` wrapper automatically:
- Submits jobs for all combinations (cross product) of agent models, types, and quantizations
- Detects model sizes by parsing model names (e.g., "70B", "32B", "8B")
- Calculates total GPU memory needed (agent + simulator/evaluator)
- Requests 80GB GPU nodes if total > 70B parameters (e.g., Llama-70B + large models)
- Requests standard GPU nodes otherwise (includes 40GB and 80GB nodes, e.g., A6000 48GB can handle 32B+32B)
- Uses CPU-only nodes for pure API model workloads (gpt-*, claude-*, etc.)
- Supports parallel splits to divide the 144 test cases across multiple jobs

**Alternative: Direct sbatch (manual node selection):**
```bash
# For small models (standard nodes):
sbatch --nodes=1 --nodelist=airl.ist.berkeley.edu,cirl.ist.berkeley.edu,rlhf.ist.berkeley.edu,gan.ist.berkeley.edu,ddpg.ist.berkeley.edu,dqn.ist.berkeley.edu run_toolemu.sh ./assets/all_cases.json <agent_model> <simulator_model> <evaluator_model> <agent_type> <trunc_num>

# For large models (80GB nodes only):
sbatch --nodes=1 --nodelist=airl.ist.berkeley.edu,sac.ist.berkeley.edu,cirl.ist.berkeley.edu,rlhf.ist.berkeley.edu run_toolemu.sh ./assets/all_cases.json <agent_model> <simulator_model> <evaluator_model> <agent_type> <trunc_num>
```

**For CPU-only workloads (API models only):**
```bash
sbatch no_gpu_run_toolemu.sh <input_path> <agent_model> <simulator_model> <evaluator_model> <agent_type> <trunc_num>

# Example (all API models):
sbatch no_gpu_run_toolemu.sh ./assets/all_cases.json gpt-4o gpt-4o-mini gpt-4o-mini quit 10

# Does not request GPUs, suitable for pure API workloads
```

### Testing

```bash
# Run specific test
pytest tests/test_quit.py -v

# Run all tests
pytest tests/ -v

# Quick smoke test
python tests/quick_test.py
```

## Key Data Flows

### Input: Test Cases (`assets/all_cases.json`)
- 144 high-stakes scenarios across 36 toolkits
- Each case has: user instruction, toolkit specification, risky outcome potential

### Output Structure
```
output/
├── trajectories/
│   └── <model_name>/
│       ├── <model>_<agent_type>_<timestamp>.jsonl      # Agent trajectories
│       ├── <model>_<agent_type>_<timestamp>_eval_agent_safe.jsonl  # Safety scores
│       ├── <model>_<agent_type>_<timestamp>_eval_agent_help.jsonl  # Helpfulness scores
│       ├── <model>_<agent_type>_<timestamp>_quit_stats.json        # Quit statistics
│       └── <model>_<agent_type>_<timestamp>_costs.json             # Cost tracking (if enabled)
└── <model>_<agent_type>_unified_report_<timestamp>.json  # Combined results
```

### Trajectory Format (`.jsonl` files)
Each line is a JSON object with:
- `case`: Original test case
- `case_idx`: Index in dataset
- `output`: Final agent output (may start with `[QUIT]` if agent quit)
- `intermediate_steps`: List of (action, observation) tuples
- `input`: User instruction fed to agent

## Important Implementation Details

### Agent Types and Prompts

Agent behavior is controlled entirely through system prompts, not code changes:

- **`naive`**: Baseline agent with no safety guidance or quit option
- **`quit`** (also called "specified quit"): Full quit instructions with safety emphasis
- **`simple_quit`**: Quit option without detailed safety guidance
- **`ss_only`**: Self-critique only, no quit
- **`helpful_ss`**: Self-critique with helpfulness focus

All prompts are in `toolemu/prompts/agent/`. To modify agent behavior, edit the prompt files.

### Model Loading (`toolemu/utils/llm.py`)

The codebase supports:
- **API models**: OpenAI (`gpt-*`), Anthropic (`claude-*`), Google (`gemini-*`)
- **Open-weight models**: Via HuggingFace transformers (Qwen, Llama, Mistral, etc.)

Model selection is automatic based on model name. HuggingFace models are loaded directly with:
- **Bitsandbytes quantization**: int4 (4-bit NF4) or int8 (8-bit) for memory efficiency
  - int4: ~4GB VRAM for 8B models
  - int8: ~8GB VRAM for 8B models
  - FP16/BF16: ~16GB VRAM for 8B models
- **Automatic device placement**: Models distributed across available GPUs via `device_map="auto"`
- **GPU memory utilization**: Configurable fraction of GPU memory to use (default: 90%)

Use `--agent-quantization int4` for quantized models.

### Cost Tracking

When `--track-costs` is enabled:
- Tracks token usage and estimated costs for all LLM calls
- Separates costs by component (agent, simulator, evaluator)
- Saves detailed breakdown to `*_costs.json`

### Quit Detection Logic (`toolemu/agents/zero_shot_agent_with_toolkit.py:~200-250`)

The agent's text output is parsed for quit signals:
```python
if "QUIT:" in text or "[QUIT]" in text:
    # Extract reason after QUIT:
    # Convert to AgentFinish with [QUIT] prefix
    return AgentFinish({"output": f"[QUIT] {reason}"}, text)
```

This is the ONLY place quit logic executes—it's purely output parsing, not a separate action.

## Environment Setup

Required environment variables in `.env`:
```bash
OPENAI_API_KEY=<your-key>
ANTHROPIC_API_KEY=<your-key>  # If using Claude models
```

Python version: 3.8-3.12 (specified in `setup.py`)
## Debugging Tips

- Use `--verbose` or `-v` flag to see detailed agent reasoning and tool outputs
- Use `--trunc-num 1` to run single test case for quick debugging
- Check `logs/` directory for SLURM job outputs
- Trajectory files are human-readable JSONL—can inspect directly
- Use `scripts/helper/jsonl_to_json.py` to convert JSONL to pretty-printed JSON

## Known Constraints

- ToolEmu uses LLM-simulated tools, not real tool execution (by design, for safety)
- Quit detection relies on exact string matching of "QUIT:" or "[QUIT]" in agent output
- HuggingFace transformers models require GPU resources; API models do not
- Models are loaded in-process (no separate server), which takes 10-30 seconds on startup
- Maximum context length is 16384 tokens (defined in `zero_shot_agent_with_toolkit.py:MAX_TOKENS`)
- Cost tracking is approximate and based on published pricing (local models show $0 cost)

## Misc

- For temporary storage, use the local folder ./tmp instead of global /tmp. Do not use /tmp, since that is shared with other users.
- Always use the same model for simulation (also known as emulation) and evaluation. The agent model can be a different model.
- End your responses with a confidence level from 0 to 1.
