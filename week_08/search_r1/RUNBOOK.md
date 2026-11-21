# Assignment 5 execution runbook

This runbook separates the three code sources. Do not treat running the
Countdown starter as Search-R1 results, and do not invoke another student's
ephemeral Modal app without their agreement.

## Where the code lives

1. **Class starter (LoRA + GRPO pattern):** [`suhabe/tiny-zero-lora`](https://github.com/suhabe/tiny-zero-lora)
   - `train.py`: TRL `GRPOTrainer`, Qwen2.5-3B-Instruct, PEFT LoRA, Countdown
     data and arithmetic rewards.
   - `modal_train.py`: Modal wrapper for that Countdown trainer.
   - This is the code the Canvas prompt asks you to understand and use as a
     starting point. It has no search environment and no PPO implementation.
   - Inspected source commit: `3659acfad61dab2f90fa46294566ff6808452e87`.
     The actual entry point uses `Jiayi-Pan/Countdown-Tasks-3to4`, a Qwen
     chat prompt, format and arithmetic rewards, TRL `GRPOTrainer`, and PEFT
     LoRA (rank 16, alpha 32, dropout 0.05; q/k/v/o and gate/up/down
     projections). Its default run is 2,000 steps with 16 generations and
     batch size 16. `modal_train.py` wraps this same `train.py` on Modal.
     This source inspection is documented; an actual training run is still
     required as reproduction evidence.
2. **Search-R1 (search rollout, reward, datasets and paper scripts):**
   [`PeterGriffinJin/Search-R1`](https://github.com/PeterGriffinJin/Search-R1)
   - `scripts/nq_hotpotqa/v0.1/train_grpo.sh`
   - `scripts/nq_hotpotqa/v0.1/train_ppo.sh`
   - `scripts/nq_hotpotqa/data_process.sh`, `retrieval_launch.sh`, and
     `infer.py`
   - The v0.1 scripts are useful references for the full search training
     pipeline, but they predate the course's PEFT requirement and use
     full-model training. Do not submit a full-fine-tune run as PEFT.
3. **Current PEFT/multi-turn integration reference:**
   - [VeRL PPO LoRA guide](https://github.com/verl-project/verl/blob/main/docs/advance/ppo_lora.rst)
   - [VeRL multi-turn search-tool guide](https://verl.readthedocs.io/en/latest/sglang_multiturn/search_tool_example.html)
   - VeRL documents LoRA for its PPO trainer and multi-turn search rollouts.
     Its old in-tree `SearchTool` example has been removed, so the project must
     supply a compatible search tool implementation and pin a compatible VeRL
     revision. Do not combine arbitrary `main` branches and expect the old
     Search-R1 shell scripts to work unchanged.

## Steps

### 1. Reproduce the starter locally

Clone the starter into a separate scratch directory and inspect `train.py`,
`train_callback.py`, `requirements.txt`, and `modal_train.py`. Record the base
model, dataset, reward functions, LoRA rank/alpha/target modules, optimizer,
rollout count, and logging. Run a tiny Countdown smoke run only; do not use its
reward or checkpoint as a Search-R1 result.

```bash
git clone https://github.com/suhabe/tiny-zero-lora.git
cd tiny-zero-lora
```

On a GPU environment with its pinned dependencies, `python train.py` runs the
starter. `modal run modal_train.py` is its Modal wrapper. Both commands are
only for the starter exercise. Avoid the full default run when checking the
setup; temporarily reduce its dataset and steps in a local copy.

### 2. Verify Search-R1 inference and retrieval

Use the course Qwen2.5-3B-Instruct model and the supplied Bamboogle set. First
run a one-question direct inference check. Then start a compatible `/retrieve`
service and run a one-question Search-R1 check. The repo's
[`evaluate_bamboogle.py`](evaluate_bamboogle.py) writes predictions and
exact-match records; use `--limit 1` only for smoke checking. The full reported
evaluation must use all 125 questions.

The Bamboogle questions are evaluation-only. The Search-R1 training source is
the paper's released NQ + HotpotQA data, not the course Bamboogle sheet.

### 3. Prepare the Search-R1 + PEFT trainer

This repo now pins VeRL in `verl.pin` and contains `setup_verl.sh`,
`prepare_search_dataset.py`, `search_tool.py`, `rewards.py`, and
`train_verl.sh`. The launcher selects `algorithm.adv_estimator=grpo` for GRPO
or `gae` for PPO, uses VeRL's tool agent for retrieval, and applies LoRA rank
16 / alpha 32 / tiny-zero-lora target modules to the policy actor. The pinned
FSDP PEFT interface defaults dropout to 0.0 (the starter uses 0.05); PPO's
value critic is a separate VeRL model, not this policy adapter.
VeRL's tool agent encodes retrieved tool responses with response mask 0,
excluding observations from policy loss.

The remote Modal function uses the course NQ/HotpotQA and wiki-18 volumes and
writes checkpoints and trainer logs to the assignment's own results volume.
The first run is the required one-step integration smoke:

```bash
cd week_08/search_r1
modal run modal_train_verl.py --algorithm grpo --steps 1 --batch-size 2 --rollouts 2
```

The smoke prepares two real NQ/HotpotQA prompts, loads the E5 wiki-18 index,
probes a real retrieval result, then requires logged tool calls, logged custom
rewards, and a saved LoRA-only FSDP checkpoint before reporting success. The
Modal function terminates its retriever in `finally` and its container stops
when the function exits. Inspect the result and container list before training.
If the smoke passes, run GRPO and wait for the invocation to finish:

```bash
modal run modal_train_verl.py --algorithm grpo --steps 500 --batch-size 128 --rollouts 5
```

Then confirm the invocation ended and its Modal container is stopped. Run PPO
as a separate invocation with the same settings:

```bash
modal run modal_train_verl.py --algorithm ppo --steps 500 --batch-size 128 --rollouts 5
```

Each checkpoint is configured to save only LoRA weights; VeRL stores the
FSDP shards and `lora_train_meta.json` under the checkpoint's `actor` folder.
Convert/export that checkpoint to a PEFT-loadable adapter before Bamboogle
inference, and keep the export alongside the original checkpoint and logs.

The prototype is not GPU-validated. Before longer runs, verify tool-call
parsing with Qwen2.5-3B-Instruct, retrieval response shape, policy adapter
checkpoint export, PPO value checkpoint export, and actual step count. Save exact configs, image
definition, dataset revision, retrieval endpoint, seed, and output volume path.
An existing dashboard function named `train` is not enough to establish that
it uses your code or which algorithm it runs.

### 4. Run local smoke checks before Modal

Follow the class reminder: exercise the data loader, reward/answer extraction,
retriever request, and one short rollout locally or on a permitted local GPU.
Use a tiny batch and a single optimizer step. Confirm the GRPO reward grouping
and PPO value/advantage path separately. These checks are not benchmark
results.

### 5. Run the two algorithms one at a time

Before submitting a Modal job, check the Workspace budget and the Containers
page. The shared workspace is near its displayed limit, and the class reminder
asks students not to leave jobs running. Use an app built from your own source
or your group's source; the dashboard app `w8-searchr1-7k3q` shown in the
provided screenshot was created by another student, so confirm with its owner
before invoking it.

Run GRPO first. Wait for completion, confirm its container is stopped, save
its logs and LoRA adapter, and record its actual step count. Then run PPO with
the same base model, training split, LoRA config, search corpus, prompt,
retrieval top-k, answer reward, and comparable rollout/compute budget. Do not
run both jobs concurrently. The shared chat does not establish an instructor
approved minimum step count; report any compute-limited budget honestly.

### 6. Evaluate and submit

Run Direct Inference and each resulting adapter across all 125 Bamboogle
questions with the same evaluator settings. Save JSONL predictions, summary
JSON, training logs/curves, comparison CSV, and chart. Keep paper reference
scores clearly separate from measured results. Add every group member's full
name; every student submits their own Canvas entry and repository URL.

## Current checkpoint

- Completed: source/data review, Bamboogle PDF/CSV cross-check, syntax check,
  answer-metric smoke check, and CSV integrity check.
- Completed: VeRL-based Search-R1 training prototype, pinned source revision,
  separate GRPO/PPO selector, retrieval tool, data conversion, answer-only
  reward, local syntax/logic smoke checks, and a Modal Qwen2.5-3B LoRA GPU
  update smoke. See `results/modal_gpu_smoke.json`.
- Prepared `modal_train_verl.py` to run the one-step GRPO smoke and sequential
  GRPO/PPO jobs using the existing course data/index volumes. Modal rejected
  its earlier launch before creating an app: `Workspace has exceeded its spend limit`.
- Prepared real VeRL parquet files from the course `week08-search-r1-data`
  volume under `/tmp/searchr1-processed/`: 169,615 training rows (79,168 NQ,
  90,447 HotpotQA) and 11,015 in-domain validation rows (3,610 NQ, 7,405
  HotpotQA). The source test parquet also contains 125 Bamboogle rows; these
  and all other out-of-domain sources are excluded from validation to prevent
  benchmark leakage. Conversion and Parquet round-trip checks passed.
- Not completed: VeRL tool-rollout smoke, actual model retrieval inference,
  GRPO/PPO training, adapter evaluation, and measured result plots.
- The post-run Modal billing summary showed $1,450.34 billed this month while
  the supplied banner showed a $1,200 budget. Billing is shared; the post-run
  active-container check showed no active containers. The API's spend-limit
  rejection confirms that an Owner/Manager must reset or raise the effective
  limit before another run can start.
