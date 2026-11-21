# Search-R1 Replication Report

## Objective

This project reproduces the Week 8 Search-R1 experiment on the course-provided
125-question Bamboogle set. The required comparison is direct inference and
Search-R1-style inference using Qwen2.5-3B-Instruct, with both GRPO and PPO;
the assignment also requires PEFT. The CSV contains question and accepted
answer aliases only. It is held out from policy training.

## What the readings contribute

### Policy gradients, baselines, and actor-critic

For a stochastic policy \(\pi_\theta(a\mid s)\), the objective is expected
return \(J(\theta)\). The policy-gradient theorem writes its gradient as an
expectation of score-function terms:

\[
\nabla_\theta J(\theta) =
  \mathbb{E}_{s,a\sim\pi_\theta}
  [\nabla_\theta\log\pi_\theta(a\mid s) Q^{\pi_\theta}(s,a)].
\]

REINFORCE estimates the return from sampled episodes. Subtracting a state-only
baseline leaves the expected gradient unchanged while reducing variance.
Actor-critic methods learn a critic to estimate the baseline/advantage, so the
actor can update from shorter bootstrapped samples. This is the connection
between the Week 5 material and the policy optimization used for language
model actions.

### Monotonic improvement and PPO

The Week 6 policy-improvement results motivate controlling how far an update
moves the policy: a large policy change can invalidate the old policy's value
estimates. TRPO expresses this as a constrained update (maximize a surrogate
objective subject to a divergence constraint). PPO's clipped surrogate is a
practical approximation:

\[
L^{clip}(\theta)=\mathbb{E}_t\left[
\min(r_t(\theta)A_t,
\operatorname{clip}(r_t(\theta),1-\epsilon,1+\epsilon)A_t)
\right],\quad
r_t(\theta)=\frac{\pi_\theta(a_t\mid s_t)}{\pi_{old}(a_t\mid s_t)}.
\]

PPO also uses a learned value critic (and commonly an entropy term). GRPO
instead normalizes rewards among multiple sampled responses for the same
prompt and does not require a separate value model. Keep prompt groups and
rollout counts fixed when comparing the algorithms.

### Search-R1's environment and reward

Search-R1 turns retrieval into an action in a multi-turn environment. A rollout
alternates model tokens and tool observations: the model reasons in `<think>`,
emits `<search>query</search>`, receives passages in
`<information>...</information>`, and eventually emits
`<answer>...</answer>`. The paper uses an outcome exact-match reward and masks
retrieved passage tokens from policy/KL loss, because those tokens are
environment observations rather than sampled policy actions. This project
therefore evaluates the exact answer block and logs search-call counts.

The paper's Bamboogle references for the same 125-question course sheet are:

| Method | EM | Correct |
| --- | ---: | ---: |
| Direct inference | 0.024 | 3/125 |
| Search-R1-Instruct (Table 2) | 0.264 | 33/125 |
| Search-R1 GRPO (Table 3) | 0.232 | 29/125 |
| Search-R1 PPO (Table 3) | 0.264 | 33/125 |

These are paper reference scores, not local measurements. The reported values
are kept separate in [`results/paper_reference.csv`](results/paper_reference.csv).

### PEFT / LoRA and why the starter is only a starting point

LoRA freezes pretrained \(W_0\) and learns a low-rank update
\(\Delta W=BA\), so a layer computes \(W_0x+BAx\). This reduces trainable
parameters and checkpoint size. The Week 8 class starter,
[`suhabe/tiny-zero-lora`](https://github.com/suhabe/tiny-zero-lora), is a
useful concrete example: it adapts Qwen2.5-3B-Instruct with PEFT LoRA and runs
TRL GRPO on a single-turn Countdown task. Its reward functions, dataset, and
training loop do not implement multi-turn search or PPO. The implementation
must borrow the adapter configuration pattern from the starter and the search
environment/trajectory logic from Search-R1; treating the Countdown notebook
as a Search-R1 trainer would be incorrect.

The paper *RL's Razor* argues that on-policy RL often changes a base policy
less (in KL) than supervised fine-tuning at comparable new-task performance,
which can help preserve prior behavior. This is useful motivation for tracking
KL and avoiding claims based only on task reward; it does not replace the
Search-R1 experiments or guarantee that a particular small run will avoid
forgetting.

## Experimental protocol

1. Evaluate `Qwen/Qwen2.5-3B-Instruct` in direct mode on all 125 rows.
2. Run Search-R1 tool inference with the published Qwen2.5-3B-Instruct GRPO
   and PPO checkpoints as reference-pipeline checks.
3. Train one GRPO and one PPO policy from the same Qwen2.5-3B-Instruct base,
   using the Search-R1 NQ + HotpotQA training data and a PEFT LoRA adapter.
   Bamboogle stays out of training and tuning.
4. Hold retrieval corpus/top-k, prompt, generation settings, reward and answer
   normalization, LoRA rank/alpha/target modules, and training budget constant
   across PPO and GRPO. Record the actual config, seed, step count, trainable
   parameter count, reward, KL, checkpoint, GPU/compute allocation, and
   validation/evaluation results.
5. Evaluate both trained adapters over all Bamboogle questions, save every
   completion and correctness flag, then create `comparison.csv` and
   `comparison.png` with `compare_results.py`.

Run commands for inference and chart generation are in [`README.md`](README.md).
The published Search-R1 v0.1 PPO/GRPO scripts provide a reference for the
multi-turn training design, but the inspected scripts full-fine-tune the
actor. They cannot be used unchanged to satisfy PEFT. The course starter
supports GRPO with LoRA but is single-turn. This repository now includes a
pinned VeRL-based integration prototype for both PPO and GRPO, with a search
function tool, outcome exact-match reward, and actor LoRA. VeRL's agent loop
sets the response mask to zero for tool observations. Its FSDP PEFT config
uses the library default LoRA dropout of 0.0 rather than the starter's 0.05;
the PPO value critic remains a separate VeRL model. See the runbook for the
implementation details and smoke-run sequence.

## Results and limitations

Local smoke checks passed for reward extraction, retrieval request formatting,
Bamboogle CSV integrity, and Python/shell syntax. The released course parquet
files were converted and round-trip checked: 169,615 NQ/HotpotQA training rows
and 11,015 NQ/HotpotQA validation rows. The source test parquet's 125
Bamboogle rows and all other datasets were excluded from validation. The class
starter's actual source at commit
`3659acfad61dab2f90fa46294566ff6808452e87` was cloned and inspected, but its
training code has not yet been run. A Modal NVIDIA A10 smoke also
loaded Qwen2.5-3B-Instruct and completed one synthetic-text LoRA optimizer
update; see `results/modal_gpu_smoke.json`. This does not validate VeRL tool
rollouts or constitute Search-R1 training. Full model retrieval inference and
PPO/GRPO runs have not been performed, so no Bamboogle EM, RL curves, or
Search-R1-trained adapter results are claimed. The post-run Modal billing
summary showed $1,450.34 billed this month versus the supplied $1,200 budget
banner. The subsequent VeRL launch was rejected because the workspace exceeded
its spend limit, and the active-container check was empty. An Owner/Manager
must reset or raise that limit before further GPU use. This remains an
uncompleted empirical replication.

## Team

Add full names of all project members before submission: **TODO**.

## References

- Sutton and Barto, *Reinforcement Learning: An Introduction*, 2nd ed., Ch. 13.
- Week 5: Policy Gradient | Notion (course PDF export).
- Week 6: Monotonic Improvement | Notion (course PDF export).
- Week 7: Proximal Policy Optimization (course PDF export).
- Week 8: RL-Train LLMs | Notion (course PDF export).
- Jin et al., [Search-R1, arXiv:2503.09516](https://arxiv.org/abs/2503.09516).
- Hu et al., [LoRA, arXiv:2106.09685](https://arxiv.org/abs/2106.09685).
- Shenfeld et al., [RL's Razor, arXiv:2509.04259](https://arxiv.org/abs/2509.04259).
- [Thinking Machines: LoRA](https://thinkingmachines.ai/blog/lora/).
- [Search-R1 official code](https://github.com/PeterGriffinJin/Search-R1).
