"""Async pipeline to label PKU responses with MJL harm metadata."""

from __future__ import annotations

import argparse
import asyncio
import csv
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Tuple
from time import perf_counter

from loguru import logger
from openai import AsyncOpenAI

ROOT_DIR = Path(__file__).resolve().parents[1]

if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from utils.openai_client import extract_output_text  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Label PKU assistant responses with MJL harm taxonomy (async)."
    )
    parser.add_argument(
        "--input-path",
        type=Path,
        default=Path("data/pku_anno_formatted_test.jsonl"),
        help="Path to source PKU JSONL file.",
    )
    parser.add_argument(
        "--prompt-path",
        type=Path,
        default=Path("prompts/pku_mjl_prompt_20251031.txt"),
        help="Prompt template to drive the classification.",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="gpt-5-mini",
        help="OpenAI model name to invoke.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Process only the first N rows (0 means all rows).",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=20,
        help="Number of rows scheduled per batch.",
    )
    parser.add_argument(
        "--concurrency",
        type=int,
        default=5,
        help="Maximum concurrent OpenAI requests.",
    )
    parser.add_argument(
        "--max-retries",
        type=int,
        default=3,
        help="Retry attempts per row on API failure.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("artifacts"),
        help="Directory where labeled JSONL will be written.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Emit full model responses for debugging.",
    )
    return parser.parse_args()


def configure_logging(verbose: bool) -> None:
    logger.remove()
    level = "DEBUG" if verbose else "INFO"
    try:
        logger.add(sys.stderr, level=level, enqueue=True)
    except PermissionError:
        logger.add(sys.stderr, level=level, enqueue=False)


def load_env_variables(env_path: Path) -> None:
    """Populate os.environ from a simple KEY=VALUE .env file if present."""
    if not env_path.exists():
        return
    for line in env_path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or "=" not in stripped:
            continue
        key, value = stripped.split("=", 1)
        key = key.strip()
        value = value.strip().strip("'").strip('"')
        if key and key not in os.environ:
            os.environ[key] = value

def load_jsonl(path: Path) -> List[Dict[str, Any]]:
    records: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as error:
                raise ValueError(f"Invalid JSON on line {line_number}: {error}") from error
    return records


def build_json_schema() -> Dict[str, Any]:
    return {
        "$schema": "https://json-schema.org/draft-07/schema#",
        "title": "pku_mjl_label",
        "type": "object",
        "required": ["mjl_harmfulness", "mjl_serverity", "mjl_harmtype"],
        "properties": {
            "mjl_harmfulness": {
                "type": "string",
                "enum": ["harmless", "harmful", "unknown"],
            },
            "mjl_serverity": {
                "type": "string",
                "enum": ["moderate", "severe", "mild", "unknown"],
            },
            "mjl_harmtype": {
                "type": "string",
                "enum": [
                    "Unproven Therapy",
                    "Triage Delay",
                    "Medication Dose Error",
                    "Diagnostic Overconfidence",
                    "Privacy/PHI Violation",
                    "Unproven Therapy;Triage Delay",
                    "Bias/Discrimination",
                    "Triage Delay;Unproven Therapy",
                    "Unproven Therapy;Medication Dose Error",
                    "Diagnostic Overconfidence;Unproven Therapy",
                    "Medication Dose Error;Unproven Therapy",
                    "Unproven Therapy;Triage Delay;Medication Dose Error;Unproven Therapy",
                    "Unproven Therapy;Unproven Therapy",
                    "Unknown"
                ],
            },
        },
        "additionalProperties": False,
    }


def extract_system_and_template(prompt_path: Path) -> Tuple[str, str]:
    text = prompt_path.read_text(encoding="utf-8")
    if "USER MESSAGE:" not in text:
        raise ValueError("Prompt must include 'USER MESSAGE:' delimiter.")
    system_part, user_part = text.split("USER MESSAGE:", 1)
    system_msg = system_part.replace("SYSTEM MESSAGE:", "").strip()
    user_template = user_part.strip()
    if "{{ response }}" not in user_template:
        raise ValueError("Prompt user template must include '{{ response }}' placeholder.")
    return system_msg, user_template


def substitute_template(template: str, response: str) -> str:
    return template.replace("{{ response }}", response)


async def call_openai_with_retry(
    client: AsyncOpenAI,
    model: str,
    system_msg: str,
    user_msg: str,
    schema: Dict[str, Any],
    max_retries: int,
) -> Dict[str, Any]:
    delay = 1
    for attempt in range(1, max_retries + 1):
        try:
            result = await client.responses.create(
                model=model,
                input=[
                    {"role": "system", "content": system_msg},
                    {"role": "user", "content": user_msg},
                ],
                text={
                    "format": {
                        "type": "json_schema",
                        "name": schema.get("title", "pku_mjl_label"),
                        "schema": schema,
                    }
                },
            )
            payload = getattr(result, "output_text", None) or extract_output_text(result)
            if not payload:
                raise ValueError("Empty response payload.")
            return json.loads(payload)
        except Exception as error:  # noqa: BLE001
            if attempt >= max_retries:
                raise
            logger.warning(
                "Retrying after error (attempt {}/{}): {}",
                attempt,
                max_retries,
                error,
            )
            await asyncio.sleep(delay)
            delay = min(delay * 2, 8)
    raise RuntimeError("Retry loop exhausted unexpectedly.")


async def process_record(
    index: int,
    total: int,
    record: Dict[str, Any],
    client: AsyncOpenAI,
    model: str,
    system_msg: str,
    template: str,
    schema: Dict[str, Any],
    semaphore: asyncio.Semaphore,
    max_retries: int,
) -> Tuple[int, Dict[str, Any], bool]:
    response_text = str(record.get("response", "") or "")
    print(f"[{index + 1}/{total}] Scheduling response classification.")

    async with semaphore:
        try:
            user_message = substitute_template(template, response_text)
            classification = await call_openai_with_retry(
                client=client,
                model=model,
                system_msg=system_msg,
                user_msg=user_message,
                schema=schema,
                max_retries=max_retries,
            )
            merged = {**record, **classification}
            print(
                f"[{index + 1}/{total}] Completed | harmfulness={classification.get('mjl_harmfulness')}"
            )
            return index, merged, True
        except Exception as error:  # noqa: BLE001
            logger.exception("Failed to classify record {}: {}", index, error)
            fallback = {
                **record,
                "mjl_harmfulness": "unknown",
                "mjl_serverity": "unknown",
                "mjl_harmtype": "Unproven Therapy;Unproven Therapy",
            }
            print(f"[{index + 1}/{total}] Failed | defaulting to unknown labels.")
            return index, fallback, False


async def run_pipeline(args: argparse.Namespace) -> Path:
    configure_logging(args.verbose)

    input_path = (ROOT_DIR / args.input_path).resolve()
    prompt_path = (ROOT_DIR / args.prompt_path).resolve()
    output_dir = (ROOT_DIR / args.output_dir).resolve()

    records = load_jsonl(input_path)
    if args.limit > 0:
        records = records[: args.limit]
    if not records:
        raise ValueError("No records available for processing.")

    system_msg, user_template = extract_system_and_template(prompt_path)
    schema = build_json_schema()

    timestamp = datetime.now(tz=timezone.utc).strftime("%Y%m%d%H%M%S")
    output_path = output_dir / f"pku_mjl_anno_{timestamp}.jsonl"
    output_dir.mkdir(parents=True, exist_ok=True)

    load_env_variables(ROOT_DIR / ".env")
    api_key = os.getenv("OPENAI_API_KEY", "").strip()
    if not api_key:
        raise EnvironmentError("OPENAI_API_KEY is not set; unable to call the Responses API.")
    client = AsyncOpenAI(api_key=api_key)
    semaphore = asyncio.Semaphore(max(1, args.concurrency))

    total = len(records)
    print(f"Loaded {total} records from {input_path}.")
    print(
        f"Processing with batch_size={args.batch_size}, concurrency={args.concurrency}, model={args.model}."
    )

    results: Dict[int, Dict[str, Any]] = {}
    success_flags: List[bool] = []

    batch_width = max(1, args.batch_size)
    total_batches = max(1, (total + batch_width - 1) // batch_width)
    for batch_index, batch_start in enumerate(range(0, total, batch_width), start=1):
        batch_records = records[batch_start : batch_start + batch_width]
        start_timestamp = datetime.now(tz=timezone.utc)
        start_perf = perf_counter()
        print(
            f"[Batch {batch_index}/{total_batches}] Start: {start_timestamp.isoformat()}"
        )
        tasks = [
            asyncio.create_task(
                process_record(
                    index=batch_start + offset,
                    total=total,
                    record=row,
                    client=client,
                    model=args.model,
                    system_msg=system_msg,
                    template=user_template,
                    schema=schema,
                    semaphore=semaphore,
                    max_retries=args.max_retries,
                )
            )
            for offset, row in enumerate(batch_records)
        ]
        batch_results = await asyncio.gather(*tasks)
        end_timestamp = datetime.now(tz=timezone.utc)
        elapsed = perf_counter() - start_perf
        print(
            f"[Batch {batch_index}/{total_batches}] End: {end_timestamp.isoformat()} | Duration={elapsed:.2f}s"
        )
        for index, record, ok in batch_results:
            results[index] = record
            success_flags.append(ok)

    ordered_records = [results[idx] for idx in sorted(results)]

    with output_path.open("w", encoding="utf-8") as handle:
        for entry in ordered_records:
            handle.write(json.dumps(entry, ensure_ascii=False))
            handle.write("\n")

    csv_path = output_path.with_suffix(".csv")
    if ordered_records:
        fieldnames = list(ordered_records[0].keys())
        with csv_path.open("w", encoding="utf-8", newline="") as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(ordered_records)
    else:
        csv_path.touch()

    completed = sum(1 for flag in success_flags if flag)
    print(
        f"Completed labeling. Success={completed} / {len(success_flags)}. "
        f"JSONL -> {output_path} | CSV -> {csv_path}"
    )
    return output_path


def main() -> None:
    args = parse_args()
    asyncio.run(run_pipeline(args))


if __name__ == "__main__":
    main()
