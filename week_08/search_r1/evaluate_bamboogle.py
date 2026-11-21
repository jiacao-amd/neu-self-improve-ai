"""Evaluate direct or Search-R1-style inference on the course Bamboogle set."""

from __future__ import annotations

import argparse
import csv
import json
import re
import time
from pathlib import Path
from typing import Any

import requests
import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    StoppingCriteria,
    StoppingCriteriaList,
)

from metrics import exact_match, extract_answer


DEFAULT_DATA = Path(__file__).parent / "data" / "bamboogle.csv"
SEARCH_SYSTEM = (
    "Answer the given question. You must conduct reasoning inside <think> and </think> first "
    "every time you get new information. After reasoning, if you find you lack some knowledge, "
    "you can call a search engine by <search> query </search> and it will return the top searched "
    "results between <information> and </information>. You can search as many times as your want. "
    "If you find no further external knowledge needed, you can directly provide the answer inside "
    "<answer> and </answer>, without detailed illustrations. For example, <answer> Beijing </answer>."
)
TOOL_SEARCH_SYSTEM = (
    "Answer the question using evidence from Wikipedia. Call the search tool at least once, "
    "and use its results before answering. Finish with exactly one "
    "<answer>short answer</answer> block."
)
SEARCH_TOOL_SCHEMA = [
    {
        "type": "function",
        "function": {
            "name": "search",
            "description": "Search Wikipedia for evidence relevant to the question.",
            "parameters": {
                "type": "object",
                "properties": {"query": {"type": "string", "description": "Concise search query"}},
                "required": ["query"],
            },
        },
    }
]


class StopOnMarkers(StoppingCriteria):
    """Stop a single-sequence generation at a Search-R1 action boundary."""

    def __init__(self, tokenizer: Any, markers: tuple[str, ...]):
        self.patterns = [
            tokenizer.encode(marker, add_special_tokens=False) for marker in markers
        ]

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs: Any) -> bool:
        row = input_ids[0]
        for pattern in self.patterns:
            if pattern and row.numel() >= len(pattern):
                target = torch.tensor(pattern, device=row.device, dtype=row.dtype)
                if torch.equal(row[-len(pattern) :], target):
                    return True
        return False


def load_rows(path: Path, limit: int | None) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    return rows if limit is None else rows[:limit]


def make_prompt(tokenizer: Any, question: str, mode: str) -> str:
    if mode == "search":
        # Match Search-R1's infer.py: a single raw instruction + question
        # prompt (tokenizer adds the model's BOS token when encoding it).
        normalized_question = question.strip()
        if normalized_question and not normalized_question.endswith("?"):
            normalized_question += "?"
        return f"{SEARCH_SYSTEM} Question: {normalized_question}\n"

    messages = [
        {"role": "user", "content": f"Answer the question concisely.\nQuestion: {question.strip()}"}
    ]
    if tokenizer.chat_template:
        return tokenizer.apply_chat_template(
            messages, add_generation_prompt=True, tokenize=False
        )
    return messages[0]["content"] + "\nAnswer:"


def get_search_text(url: str, query: str, topk: int, timeout: float) -> str:
    response = requests.post(
        url,
        json={"queries": [query], "topk": topk, "return_scores": True},
        timeout=timeout,
    )
    response.raise_for_status()
    docs = response.json()["result"][0]
    formatted: list[str] = []
    for index, item in enumerate(docs, start=1):
        contents = item.get("document", {}).get("contents", "")
        if not contents:
            continue
        title, _, body = contents.partition("\n")
        formatted.append(f"Doc {index}(Title: {title}) {body}")
    return "\n".join(formatted) or "No relevant documents were found."


def generate_direct(model: Any, tokenizer: Any, prompt: str, max_new_tokens: int) -> str:
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.inference_mode():
        output = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )
    return tokenizer.decode(output[0, inputs.input_ids.shape[1] :], skip_special_tokens=True)


def generate_search(
    model: Any,
    tokenizer: Any,
    prompt: str,
    retrieval_url: str,
    topk: int,
    max_turns: int,
    max_new_tokens: int,
    temperature: float,
    timeout: float,
) -> str:
    prefix = prompt
    chunks: list[str] = []
    stop = StoppingCriteriaList([StopOnMarkers(tokenizer, ("</search>", "</answer>"))])

    for _ in range(max_turns):
        inputs = tokenizer(prefix, return_tensors="pt").to(model.device)
        with torch.inference_mode():
            output = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                stopping_criteria=stop,
                pad_token_id=tokenizer.eos_token_id,
                do_sample=temperature > 0,
                temperature=max(temperature, 1e-5),
            )
        generated = tokenizer.decode(
            output[0, inputs.input_ids.shape[1] :], skip_special_tokens=True
        )
        chunks.append(generated)

        if "</answer>" in generated or output[0, -1].item() == tokenizer.eos_token_id:
            break

        match = re.search(r"<search>(.*?)</search>", generated, flags=re.DOTALL)
        if match is None:
            break
        results = get_search_text(retrieval_url, match.group(1).strip(), topk, timeout)
        prefix += generated + f"<information>{results}</information>\n\n"

    return "".join(chunks)


def generate_tool_search(
    model: Any,
    tokenizer: Any,
    question: str,
    retrieval_url: str,
    topk: int,
    max_turns: int,
    max_new_tokens: int,
    temperature: float,
    timeout: float,
) -> tuple[str, int]:
    """Evaluate VeRL-trained policies using the same native function-tool protocol."""
    if not tokenizer.chat_template:
        raise ValueError("Native search-tool evaluation requires a tokenizer chat template")
    messages = [
        {"role": "system", "content": TOOL_SEARCH_SYSTEM},
        {"role": "user", "content": question.strip()},
    ]
    chunks: list[str] = []
    search_calls = 0
    for _ in range(max_turns):
        prompt = tokenizer.apply_chat_template(
            messages,
            tools=SEARCH_TOOL_SCHEMA,
            add_generation_prompt=True,
            tokenize=False,
        )
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        generation_args = {
            "max_new_tokens": max_new_tokens,
            "pad_token_id": tokenizer.eos_token_id,
            "do_sample": temperature > 0,
        }
        if temperature > 0:
            generation_args["temperature"] = temperature
        with torch.inference_mode():
            output = model.generate(**inputs, **generation_args)
        generated = tokenizer.decode(
            output[0, inputs.input_ids.shape[1] :], skip_special_tokens=False
        )
        chunks.append(generated)
        if "<tool_call>" not in generated:
            break
        match = re.search(r"<tool_call>\s*(\{.*?\})\s*</tool_call>", generated, flags=re.DOTALL)
        if match is None:
            break
        try:
            call = json.loads(match.group(1))
            query = str(call["arguments"]["query"]).strip()
        except (KeyError, TypeError, json.JSONDecodeError):
            break
        if not query:
            break
        messages.append({"role": "assistant", "content": generated})
        tool_result = get_search_text(retrieval_url, query, topk, timeout)
        messages.append({"role": "tool", "name": "search", "content": tool_result})
        search_calls += 1
    return "".join(chunks), search_calls


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="Qwen/Qwen2.5-3B-Instruct")
    parser.add_argument("--adapter", default=None, help="Optional PEFT adapter directory")
    parser.add_argument("--mode", choices=("direct", "search", "tool"), default="direct")
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--output", type=Path, default=Path("results/predictions.jsonl"))
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--max-turns", type=int, default=4)
    parser.add_argument("--topk", type=int, default=3)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--retrieval-url", default="http://127.0.0.1:8000/retrieve")
    parser.add_argument("--retrieval-timeout", type=float, default=30)
    args = parser.parse_args()

    rows = load_rows(args.data, args.limit)
    tokenizer = AutoTokenizer.from_pretrained(args.model, padding_side="left")
    dtype = torch.bfloat16 if torch.cuda.is_available() else "auto"
    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=dtype, device_map="auto"
    )
    if args.adapter:
        from peft import PeftModel

        model = PeftModel.from_pretrained(model, args.adapter)
    model.eval()

    args.output.parent.mkdir(parents=True, exist_ok=True)
    correct = 0
    total_search_calls = 0
    started = time.perf_counter()
    with args.output.open("w", encoding="utf-8") as handle:
        for i, row in enumerate(rows, start=1):
            prompt = make_prompt(tokenizer, row["question"], args.mode)
            if args.mode == "search":
                completion = generate_search(
                    model,
                    tokenizer,
                    prompt,
                    args.retrieval_url,
                    args.topk,
                    args.max_turns,
                    args.max_new_tokens,
                    args.temperature,
                    args.retrieval_timeout,
                )
                search_calls = len(re.findall(r"<search>.*?</search>", completion, flags=re.DOTALL))
            elif args.mode == "tool":
                completion, search_calls = generate_tool_search(
                    model,
                    tokenizer,
                    row["question"],
                    args.retrieval_url,
                    args.topk,
                    args.max_turns,
                    args.max_new_tokens,
                    args.temperature,
                    args.retrieval_timeout,
                )
            else:
                completion = generate_direct(model, tokenizer, prompt, args.max_new_tokens)
                search_calls = 0

            prediction = extract_answer(completion)
            answers = [row.get("answer", ""), row.get("answer2", "")]
            is_correct = exact_match(prediction, answers)
            correct += int(is_correct)
            total_search_calls += search_calls
            record = {
                "index": i - 1,
                "question": row["question"],
                "answers": [answer for answer in answers if answer],
                "completion": completion,
                "prediction": prediction,
                "exact_match": is_correct,
                "search_calls": search_calls,
                "mode": args.mode,
                "model": args.model,
                "adapter": args.adapter,
            }
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
            handle.flush()
            print(f"{i}/{len(rows)} EM={correct / i:.3f} latest={int(is_correct)}")

    elapsed = time.perf_counter() - started
    summary = {
        "model": args.model,
        "adapter": args.adapter,
        "mode": args.mode,
        "examples": len(rows),
        "correct": correct,
        "exact_match": correct / len(rows) if rows else 0.0,
        "search_calls": total_search_calls,
        "mean_search_calls": total_search_calls / len(rows) if rows else 0.0,
        "seconds": elapsed,
        "predictions": str(args.output),
    }
    summary_path = args.output.with_suffix(".summary.json")
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
