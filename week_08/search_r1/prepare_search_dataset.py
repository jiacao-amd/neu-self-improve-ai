"""Convert the official Search-R1 NQ + HotpotQA parquet files to VeRL format."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd
from huggingface_hub import hf_hub_download


SYSTEM_PROMPT = (
    "You answer multi-hop questions using the search tool when evidence is needed. "
    "Think step by step. Call the search tool with a concise query; use the returned "
    "documents as evidence and search again when needed. Finish with exactly one "
    "<answer>short answer</answer> block. Do not put retrieved text in your answer."
)


def answers_from_row(row: dict[str, Any]) -> list[str]:
    reward = row.get("reward_model")
    candidates: Any = reward.get("ground_truth") if isinstance(reward, dict) else None
    if isinstance(candidates, dict):
        candidates = candidates.get("target")
        if candidates is None:
            candidates = reward.get("ground_truth", {}).get("answers")
    if hasattr(candidates, "tolist"):
        candidates = candidates.tolist()
    if candidates is None:
        candidates = row.get("golden_answers")
        if hasattr(candidates, "tolist"):
            candidates = candidates.tolist()
        if candidates is None or candidates == []:
            candidates = row.get("answer")
    if isinstance(candidates, str):
        try:
            parsed = json.loads(candidates)
            candidates = parsed if isinstance(parsed, list) else [candidates]
        except json.JSONDecodeError:
            candidates = [candidates]
    if not isinstance(candidates, (list, tuple)):
        candidates = [str(candidates)] if candidates is not None else []
    return [str(item).strip() for item in candidates if str(item).strip()]


def convert_row(row: dict[str, Any], split: str, index: int) -> dict[str, Any]:
    question = str(row.get("question", "")).strip()
    answers = answers_from_row(row)
    if not question or not answers:
        raise ValueError(f"{split}[{index}] is missing question or answer aliases")
    return {
        "data_source": "searchR1_nq_hotpotqa",
        "agent_name": "tool_agent",
        "prompt": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": question},
        ],
        "ability": "multi_hop_qa",
        "reward_model": {"style": "rule", "ground_truth": {"target": answers}},
        "extra_info": {"split": split, "index": index, "question": question},
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--hf-repo-id", default="PeterJinGo/nq_hotpotqa_train")
    parser.add_argument("--output-dir", type=Path, default=Path("data/searchr1_processed"))
    parser.add_argument("--cache-dir", type=Path, default=None)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    for split in ("train", "test"):
        local_file = hf_hub_download(
            repo_id=args.hf_repo_id,
            filename=f"{split}.parquet",
            repo_type="dataset",
            cache_dir=str(args.cache_dir) if args.cache_dir else None,
        )
        source = pd.read_parquet(local_file)
        # Keep an in-domain validation split. The released test parquet also
        # contains Bamboogle and other benchmarks, which must stay untouched
        # until final evaluation.
        if "data_source" in source.columns:
            source = source[source["data_source"].astype(str).isin({"nq", "hotpotqa"})]
        converted = [convert_row(row, split, i) for i, row in enumerate(source.to_dict("records"))]
        destination = args.output_dir / f"{split}.parquet"
        pd.DataFrame(converted).to_parquet(destination, index=False)
        print(f"{split}: {len(converted)} rows -> {destination}")


if __name__ == "__main__":
    main()
