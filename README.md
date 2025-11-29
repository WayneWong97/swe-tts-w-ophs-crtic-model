# Trajectory Critic Toolkit

This repository contains a lightweight toolchain for turning SWE-style agent
trajectories into clean dialogue transcripts and evaluating them with a trained
critic model.  It supports both the OpenHands-run vLLM reward server and a pure
Transformers (Hugging Face) inference path.

The workflow is split into four stages:

1. **Pre-processing** raw evaluation logs and reports.
2. **Aggregating** per-run outputs into a single JSONL dataset.
3. **Converting** trajectories into `funccalloff_messages` (dialogue format).
4. **Scoring** each run with either the vLLM critic or the HF-only critic.


## 1. Pre-process raw logs (`pre_process.py`)

Each evaluation `report_run_k.json` lists the `resolved_ids` (successful
instances).  The corresponding `output_run_k.jsonl` contains the full trajectory
history but lacks the resolved flag.  Run the following to merge them and write
new JSONL files with a `resolved` boolean for every row:

```bash
python pre_process.py \
  --reports_dir /path/to/reports \
  --original_outputs_dir /path/to/raw_logs \
  --enriched_outputs_dir /path/to/logs_enriched \
  --num_runs 8
```

Edit the constants at the top of `pre_process.py` (or add CLI handling) to match
your workspace.  The script writes enriched `output_run_k.jsonl` files into the
target directory.


## 2. Aggregate multiple runs (`process_logs.py`)

After `pre_process.py`, combine all `output_run_k.jsonl` files into a single
JSONL dataset where each line corresponds to an `instance_id` and contains
`run_1`, `run_2`, … entries (each storing the patch, trajectory, and resolved
flag).  The script also prints pass@k statistics:

```bash
python process_logs.py \
  --input_dir /path/to/logs_enriched \
  --output_file qwen3-8b-welltrained_consolidated_data_8.jsonl
```


## 3. Convert to dialogue format (`convert_tts_jsonl_to_funccalloff.py`)

The critic expects conversation-style messages instead of raw JSON logs.  Use
this converter to attach `funccalloff_messages` to every `run_*` entry:

```bash
python convert_tts_jsonl_to_funccalloff.py \
  --jsonl_path qwen3-8b-welltrained_consolidated_data_8.jsonl \
  --save_file_path tts_funccalloff_all.json
```

Each `funccalloff_messages` is a list of `{"role": ..., "content": ...}`
records (system, user, assistant, tool), mirroring the format used by
OpenHands/SWE-Gym.  If you need to remove in-context examples or task-tracker
turns, adjust the script accordingly.

> **OpenHands conversion.** If you already have OpenHands evaluation outputs,
> the companion script `convert_openhands_to_funccalloff.py` can process them
> directly.  The scoring step below is identical once you have
> `funccalloff_messages` for every run.


## 4. Score each run with the critic

You can choose between two scoring scripts depending on your serving stack:

### 4.1 Token reward via vLLM (`score_and_select_vllm.py`)

This path uses a customized vLLM engine with `task="token_reward"`.  It loads
the critic once, converts `funccalloff_messages` back into text, aligns tokens to
assistant messages, and computes four scoring schemes (last token, tail average,
discounted per-message reward, attention-weighted reward).

Usage:

```bash
python score_and_select_vllm.py
```

Configure `MODEL_PATH`, `INPUT_FILE` (e.g., `tts_funccalloff_all.json`), and any
scheme selection inside the script.  Outputs are written to
`scored_results_scheme_8_{scheme}.jsonl` by default.

### 4.2 Hugging Face-only inference (`score_and_select.py`)

If you cannot or do not want to run vLLM, `score_and_select.py` performs the
same logic using `AutoModelForTokenClassification` from Transformers.  It still
uses the token-reward head but executes entirely through HF APIs.

Usage:

```bash
python score_and_select.py
```

Set `MODEL_PATH` to the critic checkpoint, point `INPUT_FILE` to
`tts_funccalloff_all.json`, and pick the schemes you want from `schemes_to_run`.
Results are written to `scored_results_scheme_hf_{scheme}.jsonl`.

Both scoring scripts output, for each `instance_id`:

- `run_i_score`: critic score for each run.
- `best_run_id`, `best_score`, `best_run_resolved`: the best-scoring run and
  whether it solved the task.

They also print aggregate stats such as the rate at which the chosen run was
resolved.


## Inspecting samples (`sample_funccalloff_viewer.py`)

To preview converted trajectories, use the viewer script.  It picks a random run
from `tts_funccalloff_all.json` and prints the first 20 dialogue turns:

```bash
python sample_funccalloff_viewer.py --jsonl_path tts_funccalloff_all.json --seed 42
```


## Summary

1. `pre_process.py` – merge reports with raw logs to add `resolved` flags.
2. `process_logs.py` – aggregate all runs per `instance_id` into
   `*_consolidated_data_*.jsonl`.
3. `convert_tts_jsonl_to_funccalloff.py` (or `convert_openhands_to_funccalloff.py`) –
   attach `funccalloff_messages`.
4. `score_and_select_vllm.py` **or** `score_and_select.py` – run the critic to
   score each run and select the best patch.

With these scripts you can rebuild the full pipeline from raw SWE trajectories
to critic-evaluated results, using either vLLM or plain Transformers for the
reward model inference.
