# Assignment 5 Search-R1 handover prompt

Paste the prompt below into the coding agent on the B200 machine after making
`week_08/search_r1` available there.

---

You are continuing my INFO7375 Assignment 5 Search-R1 replication. The target
machine has a B200 GPU but currently has no project code. Work toward the full
assignment, do not claim a run succeeded without saved logs/checkpoints and
verified outputs, and keep compute usage controlled: run one job at a time,
inspect running jobs before every launch, stop jobs cleanly, and check spend
before and after each run. Do not bypass any workspace spend limit.

## Repository access and code transfer

The repository is:

```text
https://github.com/wang-luf/neu-self-improve-ai.git
```

The old remote `main` is at `e8b87cf5c3c77e6fa104e03cfb9394fbbe71b4b7` and
does **not** contain `week_08/search_r1`. That directory currently exists only
in the previous machine's local checkout and is untracked. Do not assume a
plain clone contains the Search-R1 implementation.

Preferred transfer: on the previous machine, publish the work on a dedicated
branch (do not push to `main`):

```bash
cd /Users/luffy/neu-self-improve-ai
git switch -c assignment5-search-r1
git add week_08/search_r1
git commit -m "Add Assignment 5 Search-R1 replication pipeline"
git push -u origin assignment5-search-r1
```

Then on this B200 machine:

```bash
git clone --branch assignment5-search-r1 \
  https://github.com/wang-luf/neu-self-improve-ai.git
cd neu-self-improve-ai/week_08/search_r1
```

If that branch is not published, stop and ask the user to transfer the
`week_08/search_r1` folder (for example, as a tar archive or via secure copy).
Never claim that the clone has the implementation until `ls` confirms it.

## Existing implementation and evidence

- Read `README.md`, `RUNBOOK.md`, and `REPLICATION_REPORT.md` first.
- The course Bamboogle evaluation set is `data/bamboogle.csv` (125 rows).
- The real prepared NQ/HotpotQA data was created on the previous machine at
  `/tmp/searchr1-processed/{train,test}.parquet`; it is **not** included in
  Git. The source course parquet volume contained 169,615 train rows and a
  test file with 51,713 mixed rows. The processed data keeps 169,615 NQ +
  HotpotQA training rows and only 11,015 NQ + HotpotQA validation rows,
  excluding Bamboogle and other benchmarks.
- Search-R1 uses the official wiki-18 corpus plus E5 Flat FAISS index. Those
  large assets are not in Git. Obtain them from the course Modal volumes if
  authorized, or from the official Search-R1/Hugging Face sources documented
  in the runbook. Do not substitute a fake or synthetic retriever.
- Tiny-zero source was inspected at commit
  `3659acfad61dab2f90fa46294566ff6808452e87`; its actual training has **not**
  been run. Its `train.py` is TRL GRPO on Countdown with Qwen2.5-3B-Instruct,
  LoRA rank 16 / alpha 32 / dropout 0.05 and two reward functions. Run a real,
  short starter reproduction and save the console log and checkpoint evidence.
- A synthetic-text A10 LoRA update was run earlier, but it is not Search-R1
  or tiny-zero reproduction evidence.
- A prior attempt to launch the VeRL smoke on the class Modal workspace was
  rejected before a container started: `Workspace ... has exceeded its spend
  limit`. Latest observed billing was $1,451.28 and active containers were
  empty. If using that workspace, recheck its current limit first; B200 access
  does not bypass the workspace limit. If using an independently provisioned
  B200 host, record its allocation and costs and follow that host's policies.

## Required order of work

1. Confirm repository branch/files, GPU visibility, CUDA/driver/Python versions,
   free disk/RAM, and whether this is a bare B200 host or Modal. Do not copy the
   previous machine's Modal token/config file. Authenticate separately using
   the user's own authorized account. Never print credentials.
2. Obtain the real NQ/HotpotQA parquet data, wiki-18 corpus and E5 Flat index;
   prepare VeRL train/validation files while keeping all Bamboogle rows out.
3. Run one GRPO step with two prompts and two rollouts. Verify the actual
   retrieval result, a logged function-tool call, custom exact-match reward,
   LoRA-only checkpoint and exportable adapter. Terminate the retriever and
   verify the job/container has stopped. If the current Modal workspace still
   rejects the job due to spend limit, do not retry or switch workspaces to
   evade it; report the blocker and continue only on the authorized B200 host.
4. Train GRPO, save config/logs/checkpoints/adapter, stop and verify the job;
   only then start PPO with the same base model, data, search index, reward,
   LoRA configuration, prompt, batch/rollout budget and seed. Save its evidence
   and verify it stopped too.
5. Evaluate Direct, the published Search-R1 baseline, and both trained
   adapters on all 125 Bamboogle questions. Keep the inference protocol
   consistent with each model's tool format, save per-question JSONL plus
   summaries, then create the comparison CSV/chart and a results report.
6. Actually run the tiny-zero starter training code on a short real GPU run;
   save source commit, exact reduced config, log, and checkpoint. Keep this
   result distinct from Search-R1.
7. Update the report with actual measured results only. Include full names of
   all group members, list compute hardware and steps, and clearly label any
   incomplete runs. Every student submits individually.

## Useful commands

Start by reading the runbook. Its Modal smoke command is:

```bash
modal run modal_train_verl.py --algorithm grpo --steps 1 --batch-size 2 --rollouts 2
```

The current prototype requests two H100s (`gpu="H100:2"`) to separate the E5
retriever encoder and the training process. For a single local B200, inspect
and adapt the device placement rather than blindly launching two GPU workers;
keep the FAISS index on CPU and verify memory/process placement. A specific
B200 request on Modal would use `gpu="B200"` (or `"B200:2"` when two are
actually allocated), but only after workspace limits and authorization allow
the run.

Do not start the 500-step jobs until the real one-step retrieval/tool/reward/
checkpoint smoke passes. Keep the work on a dedicated branch and do not push
results to public services or submit to Canvas without the user's direction.

---
