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

**For API-based models (OpenAI, Anthropic):**
```bash
sbatch run_toolemu_api.sh <input_path> <agent_model> <simulator_model> <evaluator_model> <agent_type> <trunc_num>

# Example:
sbatch run_toolemu_api.sh ./assets/all_cases.json gpt-4o gpt-4o-mini gpt-4o-mini quit 10
```

**For open-source models (via vLLM):**
```bash
sbatch run_toolemu_os.sh <input_path> <agent_model> <simulator_model> <evaluator_model> <agent_type> <trunc_num>

# Example:
sbatch run_toolemu_os.sh ./assets/all_cases.json Qwen/Qwen3-8B Qwen/Qwen3-8B Qwen/Qwen3-8B quit 10
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
- **Open-source models**: Via vLLM (Qwen, Llama, Mistral, etc.)

Model selection is automatic based on model name prefix. vLLM models support:
- `--agent-tensor-parallel-size`: GPU parallelism
- `--agent-quantization`: Quantization method (awq, gptq, etc.)

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
- vLLM models require GPU resources; API models do not
- Maximum context length is 16384 tokens (defined in `zero_shot_agent_with_toolkit.py:MAX_TOKENS`)
- Cost tracking is approximate and based on published pricing
