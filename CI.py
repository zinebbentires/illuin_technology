#!/usr/bin/env python3
"""Codeforces tag prediction CLI — Gemini zero-shot.

Requires: pip install google-genai
          export GEMINI_API_KEY='...'

Example:
    python CI.py --input data/code_classification_dataset \
        --config configs/gemini_zero_shot.json \
        --output predictions.jsonl

Accepted input: directory of JSON files, JSONL, JSON list, or CSV.
Output: JSONL or CSV with columns/fields `id`, `tags` (predicted labels).
Successful API responses are cached incrementally; reruns only call missing inputs.
"""

import argparse
import asyncio
import csv
import hashlib
import json
import os
import re
import sys
import time
from pathlib import Path

REQUIRED = (
    "prob_desc_description",
    "prob_desc_input_spec",
    "prob_desc_output_spec",
    "source_code",
)


def load_dataset(path: Path):
    if path.is_dir():
        files = sorted(path.glob("sample_*.json")) or sorted(path.glob("*.json"))
        if not files:
            raise ValueError(f"No JSON files found in {path}")
        rows = []
        for p in files:
            row = json.loads(p.read_text(encoding="utf-8"))
            if not isinstance(row, dict):
                raise ValueError(f"Expected JSON object in {p}")
            if "id" not in row:
                match = re.search(r"(\d+)$", p.stem)
                row["id"] = int(match.group(1)) if match else p.stem
            rows.append(row)
    elif path.suffix.lower() == ".jsonl":
        rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    elif path.suffix.lower() == ".json":
        data = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(data, list):
            rows = data
        elif isinstance(data, dict):
            rows = next((data[k] for k in ("data", "samples", "problems") if isinstance(data.get(k), list)), [data])
        else:
            raise ValueError("JSON input must be an object or a list of objects")
    elif path.suffix.lower() == ".csv":
        with path.open(newline="", encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
    else:
        raise ValueError("Input must be a directory, .json, .jsonl or .csv")

    seen = set()
    for i, row in enumerate(rows):
        if not isinstance(row, dict):
            raise ValueError(f"Row {i} is not a JSON object")
        row.setdefault("id", i)
        if row["id"] is None or str(row["id"]).strip() == "":
            raise ValueError(f"Missing ID at row {i}")
        # Normalize IDs to strings to avoid CSV/JSON cache mismatches.
        row["id"] = str(row["id"])
        if row["id"] in seen:
            raise ValueError(f"Duplicate ID: {row['id']}")
        seen.add(row["id"])
        for field in REQUIRED:
            if field not in row:
                raise ValueError(f"Missing field '{field}' at ID {row['id']}")
            if row[field] is None:
                row[field] = ""
            if not isinstance(row[field], str):
                raise ValueError(f"Field '{field}' must be a string at ID {row['id']}")
    return rows


def make_prompt(row, cfg):
    # Exact structure and separators from the research notebook.
    return f"""
{cfg['instruction']}

Allowed tags: {", ".join(cfg['allowed_tags'])}

Problem:
{row['prob_desc_description']}

Input:
{row['prob_desc_input_spec']}

Output:
{row['prob_desc_output_spec']}

Python solution:
{row['source_code']}
"""


def request_key(row, cfg):
    request = {
        "model": cfg["model"],
        "temperature": cfg["temperature"],
        "response_schema": cfg["response_schema"],
        "prompt": make_prompt(row, cfg),
    }
    return hashlib.sha256(json.dumps(request, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def read_cache(path):
    cache = {}
    if path.exists():
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            record = json.loads(line)
            if record.get("ok") and record.get("key"):
                cache[record["key"]] = record
    return cache


async def infer(rows, cfg, cache_path, concurrency, retries):
    from google import genai
    from google.genai import types

    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise EnvironmentError("Set GEMINI_API_KEY before running inference")

    client = genai.Client(api_key=api_key)
    api_config = types.GenerateContentConfig(
        temperature=cfg["temperature"],
        response_mime_type="application/json",
        response_schema=cfg["response_schema"],
    )
    allowed = set(cfg["allowed_tags"])
    cache = read_cache(cache_path)
    keys = {row["id"]: request_key(row, cfg) for row in rows}
    pending = [row for row in rows if keys[row["id"]] not in cache]
    print(f"Input: {len(rows)} | Cached: {len(rows) - len(pending)} | API calls needed: {len(pending)}", flush=True)

    semaphore = asyncio.Semaphore(concurrency)

    async def worker(row):
        async with semaphore:
            for attempt in range(retries):
                start = time.perf_counter()
                try:
                    response = await client.aio.models.generate_content(
                        model=cfg["model"],
                        contents=make_prompt(row, cfg),
                        config=api_config,
                    )
                    payload = json.loads(response.text)
                    tags = payload["tags"]
                    if not isinstance(tags, list) or any(not isinstance(t, str) or t not in allowed for t in tags):
                        raise ValueError(f"Invalid tags: {tags}")
                    usage = response.usage_metadata
                    return {
                        "id": row["id"], "key": keys[row["id"]], "tags": list(dict.fromkeys(tags)),
                        "ok": True, "latency_s": round(time.perf_counter() - start, 3),
                        "input_tokens": getattr(usage, "prompt_token_count", None),
                        "output_tokens": getattr(usage, "candidates_token_count", None),
                        "thinking_tokens": getattr(usage, "thoughts_token_count", None),
                        "total_tokens": getattr(usage, "total_token_count", None),
                    }
                except Exception as exc:
                    if attempt == retries - 1:
                        return {"id": row["id"], "key": keys[row["id"]], "ok": False, "error": str(exc)}
                    await asyncio.sleep(min(2 ** attempt, 30))

    failures = []
    if pending:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        tasks = [asyncio.create_task(worker(row)) for row in pending]
        with cache_path.open("a", encoding="utf-8") as f:
            for count, task in enumerate(asyncio.as_completed(tasks), 1):
                record = await task
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
                f.flush()
                if record["ok"]:
                    cache[record["key"]] = record
                else:
                    failures.append(record)
                if count % 25 == 0 or count == len(pending):
                    print(f"Completed {count}/{len(pending)}", flush=True)

    results = [cache.get(keys[row["id"]]) for row in rows]
    missing = [row["id"] for row, result in zip(rows, results) if result is None]
    if missing:
        print(f"ERROR: {len(missing)} inputs failed (IDs: {missing[:10]}). Successful responses remain cached.", file=sys.stderr)
        return None
    return results


def save_predictions(path, results):
    path.parent.mkdir(parents=True, exist_ok=True)
    records = [{"id": r["id"], "tags": r["tags"]} for r in results]
    if path.suffix.lower() == ".jsonl":
        with path.open("w", encoding="utf-8") as f:
            for row in records:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
    elif path.suffix.lower() == ".csv":
        with path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=["id", "tags"])
            writer.writeheader()
            for row in records:
                writer.writerow({"id": row["id"], "tags": json.dumps(row["tags"], ensure_ascii=False)})
    elif path.suffix.lower() == ".json":
        path.write_text(json.dumps(records, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    else:
        raise ValueError("Output extension must be .jsonl, .json or .csv")


def main():
    parser = argparse.ArgumentParser(description="Gemini zero-shot Codeforces tag inference")
    parser.add_argument("--input", required=True, type=Path, help="Dataset directory, .json, .jsonl or .csv")
    parser.add_argument("--output", required=True, type=Path, help="Predictions .jsonl, .json or .csv")
    parser.add_argument("--config", type=Path, default=Path(__file__).resolve().parent / "configs/gemini_zero_shot.json")
    parser.add_argument("--cache", type=Path, help="Resume cache (default: <output>.cache.jsonl)")
    parser.add_argument("--concurrency", type=int, default=5)
    parser.add_argument("--retries", type=int, default=3)
    parser.add_argument("--limit", type=int, help="Run on first N rows for a smoke test")
    parser.add_argument("--dry-run", action="store_true", help="Validate input and show prompt, without API calls")
    args = parser.parse_args()
    if args.concurrency < 1 or args.retries < 1 or (args.limit is not None and args.limit < 1):
        parser.error("concurrency, retries and limit must be positive")
    if args.output.suffix.lower() not in (".jsonl", ".json", ".csv"):
        parser.error("--output must end with .jsonl, .json or .csv")
    if not args.config.is_file():
        parser.error(f"Configuration missing: {args.config} (copy your original configs/gemini_zero_shot.json)")
    cfg = json.loads(args.config.read_text(encoding="utf-8"))
    for key in ("model", "instruction", "allowed_tags", "temperature", "response_schema"):
        if key not in cfg:
            parser.error(f"Configuration missing key: {key}")
    if not isinstance(cfg["allowed_tags"], list) or not cfg["allowed_tags"]:
        parser.error("allowed_tags must be a nonempty list")
    rows = load_dataset(args.input)
    if args.limit is not None:
        rows = rows[:args.limit]
    if not rows:
        parser.error("Empty input dataset")
    print(f"Loaded {len(rows)} problems | model: {cfg['model']}")
    if args.dry_run:
        print("First prompt preview:\n" + make_prompt(rows[0], cfg)[:1000])
        return
    cache_path = args.cache or Path(str(args.output) + ".cache.jsonl")
    if cache_path.resolve() == args.output.resolve():
        parser.error("--cache and --output must be different paths")
    results = asyncio.run(infer(rows, cfg, cache_path, args.concurrency, args.retries))
    if results is None:
        sys.exit(1)
    save_predictions(args.output, results)
    latencies = [r["latency_s"] for r in results if r.get("latency_s") is not None]
    tokens = [r["total_tokens"] for r in results if r.get("total_tokens") is not None]
    print(f"Saved {len(results)} predictions to {args.output}")
    if latencies:
        print(f"Mean API latency: {sum(latencies)/len(latencies):.2f} s | Total recorded tokens: {sum(tokens) if tokens else 'N/A'}")


if __name__ == "__main__":
    main()
