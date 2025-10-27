"""Async batch pipeline for generating MedSafety labels via OpenAI Responses API."""

from __future__ import annotations

import argparse
import asyncio
import configparser
import json
import os
import sys
import time
from pathlib import Path
from statistics import mean
from typing import Any, Dict, List, Optional, Sequence, Tuple

import pandas as pd
from loguru import logger
from openai import AsyncOpenAI
from tqdm import tqdm

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from utils.io_helpers import (  # noqa: E402
    append_jsonl_record,
    generate_id,
    load_jsonl_latest,
    read_input_csv,
    write_csv,
    write_jsonl,
)
from utils.openai_client import extract_output_text  # noqa: E402
from utils.validators import validate_record  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Async MedSafety labeling pipeline (batched).")
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
        help="Number of rows to dispatch per scheduling batch.",
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
        help="Maximum retry attempts per row on API failure.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Emit per-row classification details.",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume from prior progress to avoid reprocessing completed rows.",
    )
    return parser.parse_args()


def configure_logging(verbose: bool) -> None:
    """Route loguru output to both console and file with desired verbosity."""
    logger.remove()
    level = "DEBUG" if verbose else "INFO"
    logger.add(sys.stderr, level=level, enqueue=True)
    logger.add("logs/run_async_{time}.log", level=level, rotation="10 MB", enqueue=True)


def resolve_project_path(path: Path | str) -> Path:
    candidate = Path(path)
    if candidate.is_absolute():
        return candidate
    return ROOT_DIR / candidate


def load_config(cfg_path: Path) -> configparser.ConfigParser:
    cfg_path = resolve_project_path(cfg_path)
    cfg = configparser.ConfigParser()
    cfg.read(cfg_path, encoding="utf-8")
    return cfg


def build_json_schema() -> Dict[str, Any]:
    return {
        "$schema": "https://json-schema.org/draft-07/schema#",
        "title": "medsafety_labels_row",
        "type": "object",
        "required": [
            "id",
            "query",
            "response",
            "query_risk_level-human",
            "respond_type-human",
            "query_risk_level-gpt",
            "respond_type-gpt",
            "respond_stdtype-gpt"
        ],
        "properties": {
            "id": {"type": "string"},
            "query": {"type": "string"},
            "response": {"type": "string"},
            "query_risk_level-human": {"type": ["string", "null"]},
            "respond_type-human": {"type": ["string", "null"]},
            "query_risk_level-gpt": {
                "type": "string",
                "enum": ["Non-Medical", "Non-serious", "Serious", "Critical"],
            },
            "respond_type-gpt": {
                "type": "string",
                "enum": [
                    "Non-Medical",
                    "Irrelevant or nonsensical",
                    "General information",
                    "No answer",
                ],
            },
            "respond_stdtype-gpt": {
                "type": "string",
                "enum": [
                    "Irrelevant or nonsensical",
                    "No information",
                    "Inform clinical management (General Information)",
                    "Drive clinical management (Recommendations)",
                    "Treat or diagnose",
                ]
            }
        },
        "additionalProperties": False,
    }


def is_success_record(record: Optional[Dict[str, Any]]) -> bool:
    """Check whether a record contains a completed GPT classification."""
    if not record:
        return False
    required_keys = ("query_risk_level-gpt", "respond_type-gpt", "respond_stdtype-gpt")
    return all(record.get(key) not in (None, "") for key in required_keys)


def normalize_respond_stdtype(value: Optional[str]) -> Optional[str]:
    """Map string variants into the canonical respond_stdtype-gpt enumerations."""
    if value is None:
        return None
    normalized = str(value).strip()
    if not normalized:
        return None
    mapping = {
        "No answer": "No information",
        "General information": "Inform clinical management (General Information)",
        "General Information": "Inform clinical management (General Information)",
        "Non-Medical": "Irrelevant or nonsensical",
        "Non-medical": "Irrelevant or nonsensical",
    }
    return mapping.get(normalized, normalized)


def extract_system_and_template(prompt_path: Path) -> Tuple[str, str]:
    prompt_text = prompt_path.read_text(encoding="utf-8")
    if "USER MESSAGE:" not in prompt_text:
        raise ValueError("Prompt must contain 'USER MESSAGE:' delimiter.")
    system_part, user_part = prompt_text.split("USER MESSAGE:", 1)
    system_msg = system_part.replace("SYSTEM MESSAGE:", "").strip()
    user_template = user_part.strip()
    return system_msg, user_template


def is_missing(value: Any) -> bool:
    if value is None:
        return True
    if isinstance(value, str):
        stripped = value.strip()
        return stripped == "" or stripped.lower() in {"null", "nan"}
    return pd.isna(value)


def safe_text(value: Any) -> str:
    return "" if is_missing(value) else str(value)


def normalize_optional(value: Any) -> Optional[str]:
    return None if is_missing(value) else str(value).strip()


def format_optional(value: Any) -> Tuple[str, str]:
    if is_missing(value):
        return "null", "null"
    text = str(value).strip()
    return text, json.dumps(text, ensure_ascii=False)


def normalize_row_id(row: Dict[str, Any]) -> str:
    raw_id = row.get("id")
    if not is_missing(raw_id):
        return str(raw_id).strip()
    query = safe_text(row.get("query"))
    response = safe_text(row.get("response"))
    return generate_id(query, response)


def make_user_message(template: str, row: Dict[str, Any], rid: str) -> str:
    query = safe_text(row.get("query"))
    response = safe_text(row.get("response"))

    query_human_display, query_human_json = format_optional(row.get("query_risk_level-human"))
    respond_human_display, respond_human_json = format_optional(row.get("respond_type-human"))

    message = template
    replacements = {
        "{{ id }}": rid,
        "{{ query }}": query,
        "{{ response }}": response,
        "{{ query_risk_level_human }}": query_human_display,
        "{{ respond_type_human }}": respond_human_display,
        "{{ query_risk_level_human_json }}": query_human_json,
        "{{ respond_type_human_json }}": respond_human_json,
    }
    for placeholder, value in replacements.items():
        message = message.replace(placeholder, value)
    return message


def prepare_failure_row(row: Dict[str, Any], rid: str) -> Dict[str, Any]:
    return {
        "id": rid,
        "query": safe_text(row.get("query")),
        "response": safe_text(row.get("response")),
        "query_risk_level-human": normalize_optional(row.get("query_risk_level-human")),
        "respond_type-human": normalize_optional(row.get("respond_type-human")),
        "query_risk_level-gpt": None,
        "respond_type-gpt": None,
        "respond_stdtype-gpt": None,
    }


def chunked(seq: Sequence[Dict[str, Any]], size: int) -> List[Tuple[int, List[Dict[str, Any]]]]:
    batches: List[Tuple[int, List[Dict[str, Any]]]] = []
    for start in range(0, len(seq), size):
        batches.append((start, list(seq[start : start + size])))
    return batches


async def call_openai_with_retry(
    client: AsyncOpenAI,
    model: str,
    system_msg: str,
    user_msg: str,
    rid: str,
    max_retries: int,
) -> Any:
    delay = 1
    for attempt in range(1, max_retries + 1):
        try:
            return await client.responses.create(
                model=model,
                input=[
                    {"role": "system", "content": system_msg},
                    {"role": "user", "content": user_msg},
                ],
            )
        except Exception as exc:  # broad catch to surface message and retry
            if attempt >= max_retries:
                raise
            logger.warning(
                "Retrying row {} after error (attempt {}/{}): {}",
                rid,
                attempt,
                max_retries,
                exc,
            )
            await asyncio.sleep(delay)
            delay = min(delay * 2, 8)
    raise RuntimeError(f"Unexpected retry loop termination for row {rid}")


async def process_row(
    position: int,
    total_rows: int,
    row: Dict[str, Any],
    rid: str,
    client: AsyncOpenAI,
    model: str,
    system_msg: str,
    user_template: str,
    schema: Dict[str, Any],
    semaphore: asyncio.Semaphore,
    verbose: bool,
    max_retries: int,
) -> Tuple[int, Dict[str, Any], bool, float]:
    logger.info("Row {}/{} | id={}", position, total_rows, rid)
    user_message = make_user_message(user_template, row, rid)

    start_time = time.perf_counter()
    async with semaphore:
        try:
            response = await call_openai_with_retry(
                client=client,
                model=model,
                system_msg=system_msg,
                user_msg=user_message,
                rid=rid,
                max_retries=max_retries,
            )

            text = getattr(response, "output_text", None) or extract_output_text(response)
            if not text:
                raise ValueError("Empty response payload.")

            obj = json.loads(text)

            obj["id"] = rid
            obj["query"] = safe_text(row.get("query"))
            obj["response"] = safe_text(row.get("response"))
            obj["query_risk_level-human"] = normalize_optional(row.get("query_risk_level-human"))
            obj["respond_type-human"] = normalize_optional(row.get("respond_type-human"))
            obj["respond_stdtype-gpt"] = normalize_respond_stdtype(obj.get("respond_stdtype-gpt"))

            allowed_keys = set(schema["properties"].keys())
            allowed_keys.update({
                "id",
                "query",
                "response",
                "query_risk_level-human",
                "respond_type-human",
            })
            for extra_key in list(obj.keys()):
                if extra_key not in allowed_keys:
                    obj.pop(extra_key, None)

            validate_record(obj, schema)

            if verbose:
                logger.debug(
                    "Classification result | query_risk_level={} | respond_type={} | respond_stdtype={} ",
                    obj.get("query_risk_level-gpt"),
                    obj.get("respond_type-gpt"),
                    obj.get("respond_stdtype-gpt"),
                )

            duration = time.perf_counter() - start_time
            return position, obj, True, duration

        except Exception as exc:
            logger.exception("Failed row id={} : {}", rid, exc)
            failure = prepare_failure_row(row, rid)
            duration = time.perf_counter() - start_time
            return position, failure, False, duration


async def run_pipeline(args: argparse.Namespace) -> None:
    cfg = load_config(Path("config.ini"))
    input_csv = resolve_project_path(
        cfg.get("paths", "input_csv", fallback="data/MedSafety_Dataset.csv")
    )
    output_dir = resolve_project_path(
        cfg.get("paths", "output_dir", fallback="artifacts")
    )
    prompt_path = resolve_project_path("prompts/medjudge_unified_prd_v1.4.txt")

    output_dir.mkdir(parents=True, exist_ok=True)
    progress_path = output_dir / "medsafety_labels.progress.jsonl"
    final_jsonl = output_dir / "medsafety_labels.jsonl"
    final_csv = output_dir / "medsafety_labels.csv"

    if args.resume:
        records_by_id = load_jsonl_latest(progress_path)
        logger.info(
            "Resume enabled | loaded {} records from {}",
            len(records_by_id),
            progress_path,
        )
    else:
        records_by_id = {}
        for path in (progress_path, final_jsonl, final_csv):
            if path.exists():
                path.unlink()

    success_count = sum(1 for rec in records_by_id.values() if is_success_record(rec))
    failure_count = len(records_by_id) - success_count
    processed_success_ids = {
        rid for rid, rec in records_by_id.items() if is_success_record(rec)
    }
    skipped_due_to_resume = len(processed_success_ids)

    model = cfg.get("openai", "model", fallback="gpt-5-mini")
    api_key = os.getenv("OPENAI_API_KEY") or cfg.get("openai", "api_key", fallback="")
    if not api_key:
        logger.error("OPENAI_API_KEY not set in environment or config.ini.")
        raise SystemExit(2)

    system_msg, user_template = extract_system_and_template(prompt_path)
    schema = build_json_schema()
    client = AsyncOpenAI(api_key=api_key)

    df = read_input_csv(input_csv)
    total_rows = len(df)
    if args.limit and args.limit > 0:
        logger.info("Applying row limit: {} of {} total rows.", args.limit, total_rows)
        df = df.head(args.limit)

    all_rows = df.to_dict(orient="records")
    row_positions = {
        normalize_row_id(row): idx + 1 for idx, row in enumerate(all_rows)
    }
    pending_rows: List[Dict[str, Any]] = []
    for row in all_rows:
        rid = normalize_row_id(row)
        if rid in processed_success_ids:
            continue
        pending_rows.append(row)

    logger.info(
        "Async MedSafety run | model={} | prompt={} | input_csv={} | total_rows={} | pending={} | batch_size={} | concurrency={}",
        model,
        prompt_path,
        input_csv,
        len(all_rows),
        len(pending_rows),
        args.batch_size,
        args.concurrency,
    )

    semaphore = asyncio.Semaphore(max(1, args.concurrency))
    batches = chunked(pending_rows, max(1, args.batch_size))

    durations: List[float] = []
    pbar = tqdm(total=len(pending_rows), desc="Processing rows")

    for batch_index, (start, batch_records) in enumerate(batches, start=1):
        if not batch_records:
            continue
        batch_end = start + len(batch_records)
        logger.info(
            "Dispatching batch {} | rows {}-{}",
            batch_index,
            start + 1,
            batch_end,
        )

        tasks = [
            asyncio.create_task(
                process_row(
                    position=row_positions[normalize_row_id(row)],
                    total_rows=len(all_rows),
                    row=row,
                    rid=normalize_row_id(row),
                    client=client,
                    model=model,
                    system_msg=system_msg,
                    user_template=user_template,
                    schema=schema,
                    semaphore=semaphore,
                    verbose=args.verbose,
                    max_retries=max(1, args.max_retries),
                )
            )
            for offset, row in enumerate(batch_records)
        ]

        for task in asyncio.as_completed(tasks):
            position, record, success, duration = await task
            rid = record["id"]

            previous = records_by_id.get(rid)
            if previous is not None:
                if is_success_record(previous):
                    success_count = max(0, success_count - 1)
                else:
                    failure_count = max(0, failure_count - 1)

            records_by_id[rid] = record
            if success:
                success_count += 1
                processed_success_ids.add(rid)
            else:
                failure_count += 1
                processed_success_ids.discard(rid)

            append_jsonl_record(progress_path, record)
            durations.append(duration)
            pbar.update(1)

    pbar.close()

    ordered_results: List[Dict[str, Any]] = []
    missing_ids: List[str] = []
    for row in all_rows:
        rid = normalize_row_id(row)
        record = records_by_id.get(rid)
        if record is None:
            missing_ids.append(rid)
        else:
            ordered_results.append(record)

    if missing_ids:
        logger.error("Missing {} records after processing: {}", len(missing_ids), missing_ids[:5])
        raise RuntimeError("Incomplete results; rerun with --resume to continue.")

    write_jsonl(final_jsonl, ordered_results)
    write_csv(final_csv, ordered_results)

    logger.info("Artifacts written to {}", output_dir)
    logger.info(
        "Async run summary | total_rows={} | successes={} | failures={} | skipped_from_resume={}",
        len(all_rows),
        success_count,
        failure_count,
        skipped_due_to_resume,
    )
    if durations:
        logger.info(
            "Per-row latency stats (s) | avg={:.2f} | min={:.2f} | max={:.2f}",
            mean(durations),
            min(durations),
            max(durations),
        )


def main() -> None:
    args = parse_args()
    configure_logging(args.verbose)
    asyncio.run(run_pipeline(args))


if __name__ == "__main__":
    main()
