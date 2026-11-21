# Re-run on a second GB300 node (adapter regeneration)

The intermediate, smoke and tiny-zero adapters from the original run were lost
when the first GB300 reservation (`pod4-gb300-2-tray10-f3`) ended before they
were uploaded. This directory holds a **re-run** on a second reservation
(`pod4-gb300-2-tray01-f3`, 4x NVIDIA GB300, 2026-11-21 01:10–07:10 PST) that
regenerated them.

**The headline results in `REPLICATION_REPORT.md` are from the original run and
are unchanged.** The re-run is reported separately; its weights are not the
original weights, and its scores differ within run-to-run noise.

## What was re-run

Same code (repo commit `c829aea`), pinned VeRL `453954b3` + the same critic
patch, same data preparation (169,615 train / 11,015 validation rows, same
512-row validation sample), same retriever (`retriever_check.json`: identical
top-3 ids to Search-R1's CPU FAISS path), same hyper-parameters and seed:

- `runs/grpo-smoke`, `runs/ppo-smoke`: one-step integration smokes (both passed).
- `runs/grpo-main`, `runs/ppo-main`: the original 300-step configuration
  (64 prompts x 5 rollouts, LoRA r=16/alpha=32, lr 2e-5), stopped at step 100
  (`planned_stop.json`). Adapters for steps 25, 50 and 100 are in
  `adapter-step*/lora_adapter/`.
- `bamboogle/`: all 125 Bamboogle questions for each re-run checkpoint, with
  the same evaluator settings as the original run (tool mode, greedy).
- `tiny_zero_starter/`: the tiny-zero-lora starter at commit `3659acfa`, again
  reduced to 60 steps (`reduced_config.diff`).

Rollouts are sampled (sglang, temperature 1.0), so a re-run with the same seed
does not reproduce the same trajectories or weights.

## Re-run vs original

Bamboogle EM (125 questions, same evaluator):

| Checkpoint | GRPO original | GRPO re-run | PPO original | PPO re-run |
| --- | ---: | ---: | ---: | ---: |
| step 25 | 0.248 | 0.232 | 0.176 | 0.184 |
| step 50 | 0.264 | 0.208 | 0.104 | 0.080 |
| step 100 | 0.280 | 0.304 | 0.168 | 0.160 |

In-domain validation EM (512 NQ/HotpotQA questions):

| Step | 0 | 25 | 50 | 75 | 100 |
| --- | ---: | ---: | ---: | ---: | ---: |
| GRPO original | 0.184 | 0.295 | 0.330 | 0.311 | 0.354 |
| GRPO re-run | 0.162 | 0.322 | 0.316 | 0.314 | 0.332 |
| PPO original | 0.180 | 0.279 | 0.283 | 0.299 | 0.303 |
| PPO re-run | 0.168 | 0.279 | 0.258 | 0.316 | 0.289 |

`RESULTS.md` and `training_curves.png` in this directory are generated from
the re-run only. The untrained Bamboogle reference line (0.184) in the plot
is the original run's untrained-model evaluation; it was not re-evaluated.

Observations:

- The main findings reproduce. GRPO ends well above the untrained model on
  Bamboogle (0.304 at step 100 vs 0.184); PPO again dips below it around step
  50 (0.080) and only partly recovers by step 100 (0.160).
- The PPO dip has the same cause as in the original run: at step 50, 113/125
  Bamboogle answers used a single search (original: 105/125), while GRPO
  kept searching two or more times on 113/125 questions. PPO's KL to the
  reference peaked at 0.378 (step 94; original peak ~0.25).
- Differences between runs of up to ~0.06 Bamboogle EM (about 7 questions) at
  the same step are run-to-run variance from sampled rollouts. Even the
  step-0 validation score of the same untrained model differs (0.184 vs
  0.162), because VeRL validation is not bit-for-bit deterministic across
  sglang runs.
- tiny-zero starter (60 steps): held-out format rate 32.5% -> 98.0% and solve
  rate 4.0% -> 4.5% (original run: 96.0% and 7.5%).

## tiny-zero adapter

`adapter_model.safetensors` is 120 MB, above GitHub's 100 MB file limit, so it
is stored in parts. Reassemble before loading:

```bash
cd tiny_zero_starter/checkpoint
cat adapter_model.safetensors.part-* > adapter_model.safetensors
sha256sum -c adapter_model.safetensors.sha256
```
