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

## Results reporting

When reporting results, include the raw safety and helpfulness scores out of 3 and not the percents. First report the high-level averages (safety, helpfulness, and quit rate) in a compact format, then provide a more detailed distributional breakdown. To compute quit rate, look in the trajectories for responses matching the pattern that is indicated in the prompts.

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

**Recommended: Use the smart wrapper script for GPU workloads**
```bash
./submit_toolemu.sh <input_path> <agent_model> <simulator_model> <evaluator_model> <agent_type> <trunc_num>

# Example with HuggingFace models (auto-quantized to int4):
./submit_toolemu.sh ./assets/all_cases.json Qwen/Qwen3-32B Qwen/Qwen3-32B Qwen/Qwen3-32B quit 10

# Example with large models (automatically requests 80GB GPU nodes):
./submit_toolemu.sh ./assets/all_cases.json meta-llama/Llama-3.1-70B-Instruct Qwen/Qwen3-32B Qwen/Qwen3-32B quit 144

# Example with mixed API + HuggingFace:
./submit_toolemu.sh ./assets/all_cases.json gpt-4o-mini Qwen/Qwen3-32B Qwen/Qwen3-32B quit 10
```

The `submit_toolemu.sh` wrapper automatically:
- Detects model sizes by parsing model names (e.g., "70B", "32B", "8B")
- Calculates total GPU memory needed (agent + simulator/evaluator)
- Requests 80GB GPU nodes if total > 70B parameters (e.g., Llama-70B + large models)
- Requests standard GPU nodes otherwise (includes 40GB and 80GB nodes, e.g., A6000 48GB can handle 32B+32B)
- HuggingFace models are automatically quantized to int4 (API models ignore quantization)

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

### Individual Pipeline Stages

**Stage 1: Generate trajectories only:**
```bash
python scripts/emulate.py -inp ./assets/all_cases.json -atp quit -stp adv_thought -am gpt-4o -sm gpt-4o-mini -tn 5
```

**Stage 2: Evaluate existing trajectories:**
```bash
python scripts/evaluate.py -inp dumps/trajectories/gpt-4o/output.jsonl -ev agent_safe -bs 5
python scripts/evaluate.py -inp dumps/trajectories/gpt-4o/output.jsonl -ev agent_help -bs 5
```

**Stage 3: Read evaluation results:**
```bash
python scripts/helper/read_eval_results.py dumps/trajectories/gpt-4o/output
```

## Key Data Flows

### Input: Test Cases (`assets/all_cases.json`)
- 144 high-stakes scenarios across 36 toolkits
- Each case has: user instruction, toolkit specification, risky outcome potential

### Output Structure
```
dumps/
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

## Common Development Patterns

### Adding a New Agent Type

1. Create new prompt file in `toolemu/prompts/agent/agent_<name>.py`
2. Define `AGENT_<NAME>_PROMPT` and `AGENT_<NAME>_SYSTEM_INFO`
3. Import in `toolemu/prompts/agent/__init__.py`
4. Add type name to `AGENT_TYPES` in `toolemu/agents/zero_shot_agent_with_toolkit.py`
5. Update prompt selection logic in `zero_shot_agent_with_toolkit.py:create_prompt()`

### Adding a New Evaluator

1. Create evaluator class in `toolemu/evaluators.py`
2. Define prompts in `toolemu/prompts/evaluator/`
3. Register in `EVALUATORS` dict in `evaluators.py`

### Modifying Simulator Behavior

Simulator prompts are in `toolemu/prompts/simulator/`:
- `standard.py`: Standard tool simulation
- `adversarial.py`: Adversarial simulation for safety testing

Simulator type selected via `--simulator-type` (default: `adv_thought`)

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
