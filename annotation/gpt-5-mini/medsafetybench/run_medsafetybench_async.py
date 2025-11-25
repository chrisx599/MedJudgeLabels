#!/usr/bin/env python3
"""Async MedSafety bench annotator that appends gpt_harmfulness labels."""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

from loguru import logger
from openai import AsyncOpenAI

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from utils.io_helpers import generate_id, write_csv, write_jsonl  # noqa: E402
from utils.openai_client import extract_output_text  # noqa: E402

ALLOWED_BINARY_VALUES = {"harmless", "harmful"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Annotate MedSafety bench responses as harmful/harmless via gpt-5-mini."
    )
    parser.add_argument(
        "--input-jsonl",
        type=Path,
        required=True,
        help="Path to the source JSONL file (must contain a 'response' field).",
    )
    parser.add_argument(
        "--prompt-path",
        type=Path,
        default=Path("prompts/anno_prompt.txt"),
        help="Prompt template with {{ query }} and {{ response }} placeholders.",
    )
    parser.add_argument(
        "--output-prefix",
        type=str,
        default="medsafety_harm_labels",
        help="Basename for written artifacts (timestamp is appended automatically).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("artifacts"),
        help="Directory for JSONL/CSV artifacts.",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="gpt-5-mini",
        help="Model name to call via the OpenAI Responses API.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=8,
        help="Number of rows scheduled together before awaiting completions.",
    )
    parser.add_argument(
        "--concurrency",
        type=int,
        default=4,
        help="Maximum number of in-flight OpenAI requests.",
    )
    parser.add_argument(
        "--rate-limit-qps",
        type=float,
        default=0.0,
        help="Optional client-side QPS cap. Set to 0 to disable throttling.",
    )
    parser.add_argument(
        "--max-retries",
        type=int,
        default=3,
        help="Maximum retry attempts per row when the API errors.",
    )
    parser.add_argument(
        "--row-limit",
        type=int,
        default=0,
        help="Process only the first N rows (0 means all).",
    )
    parser.add_argument(
        "--emit-meta",
        action="store_true",
        help="Include severity/harm_types/explanation under a _meta object.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Skip OpenAI calls and emit placeholder harmless labels (useful for smoke tests).",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable debug logging.",
    )
    return parser.parse_args()


def configure_logging(verbose: bool) -> None:
    logger.remove()
    level = "DEBUG" if verbose else "INFO"
    logger.add(sys.stderr, level=level)
    logger.add("logs/run_medsafetybench_async_{time}.log", level=level, rotation="10 MB")


def timestamp_suffix() -> str:
    return datetime.now(tz=timezone.utc).strftime("%y%m%d_%H%M")


def read_jsonl_rows(path: Path, limit: int | None = None) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for idx, line in enumerate(handle):
            if limit and len(rows) >= limit:
                break
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Failed to parse JSONL line {idx + 1}: {exc}") from exc
            if "response" not in row:
                raise ValueError(f"JSON object on line {idx + 1} is missing 'response'.")
            rows.append(row)
    if not rows:
        raise ValueError(f"No records found in {path}")
    return rows


def resolve_row_id(row: Dict[str, Any]) -> str:
    existing = row.get("id")
    if isinstance(existing, str) and existing.strip():
        return existing.strip()
    if existing is not None:
        value = str(existing).strip()
        if value:
            return value
    query = str(row.get("query") or "")
    response = str(row.get("response") or "")
    return generate_id(query, response)


def load_prompt_template(path: Path) -> Tuple[str, str]:
    text = path.read_text(encoding="utf-8")
    marker = "### INPUT"
    if marker not in text:
        raise ValueError("Prompt template must contain '### INPUT' section.")
    before, after = text.split(marker, 1)
    system_prompt = before.strip()
    user_template = f"{marker}{after}".strip()
    return system_prompt, user_template


def render_user_message(user_template: str, query: str, response: str) -> str:
    replacements = {
        "{{ query }}": query,
        "{{ response }}": response,
    }
    rendered = user_template
    for placeholder, value in replacements.items():
        rendered = rendered.replace(placeholder, value)
    return rendered


def parse_model_payload(text: str) -> Dict[str, Any]:
    """Parse model text output into a JSON object, applying simple recovery heuristics."""
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        start = text.find("{")
        end = text.rfind("}")
        if start != -1 and end != -1 and end > start:
            snippet = text[start : end + 1]
            try:
                return json.loads(snippet)
            except json.JSONDecodeError:
                pass
    raise ValueError(f"Unable to parse model output as JSON: {text!r}")


def build_output_record(
    row: Dict[str, Any],
    annotation: Dict[str, Any],
    *,
    include_meta: bool,
) -> Dict[str, Any]:
    record = dict(row)
    raw_value = annotation.get("binary_harmfulness")
    if not isinstance(raw_value, str):
        raise ValueError("Model output missing 'binary_harmfulness'.")
    normalized = raw_value.strip().lower()
    if normalized not in ALLOWED_BINARY_VALUES:
        raise ValueError(f"Unexpected binary_harmfulness value: {raw_value!r}")
    record["gpt_harmfulness"] = normalized
    if include_meta:
        record["_meta"] = {
            "severity": annotation.get("severity"),
            "harm_types": annotation.get("harm_types"),
            "explanation": annotation.get("explanation"),
        }
    return record


class RateLimiter:
    def __init__(self, qps: float) -> None:
        self.qps = max(0.0, qps)
        self._min_interval = 1.0 / self.qps if self.qps > 0 else 0.0
        self._lock = asyncio.Lock()
        self._last = 0.0

    async def wait(self) -> None:
        if self._min_interval <= 0:
            return
        async with self._lock:
            now = time.perf_counter()
            wait_for = self._min_interval - (now - self._last)
            if wait_for > 0:
                await asyncio.sleep(wait_for)
            self._last = time.perf_counter()


@dataclass(slots=True)
class PipelineConfig:
    input_path: Path
    prompt_path: Path
    output_jsonl: Path
    output_csv: Path
    model: str
    batch_size: int
    concurrency: int
    max_retries: int
    emit_meta: bool
    dry_run: bool
    verbose: bool
    rate_limiter: RateLimiter


async def call_model_with_retry(
    client: AsyncOpenAI,
    cfg: PipelineConfig,
    *,
    system_prompt: str,
    user_message: str,
    rid: str,
) -> Dict[str, Any]:
    delay = 1.0
    for attempt in range(1, cfg.max_retries + 1):
        try:
            await cfg.rate_limiter.wait()
            response = await client.responses.create(
                model=cfg.model,
                input=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_message},
                ],
            )
            text = getattr(response, "output_text", None) or extract_output_text(response)
            if not text:
                raise ValueError("Empty payload from model.")
            return parse_model_payload(text)
        except Exception as exc:
            if attempt >= cfg.max_retries:
                logger.error("Row {} failed after {} attempts: {}", rid, attempt, exc)
                raise
            logger.warning(
                "Retrying row {} after error (attempt {}/{}): {}",
                rid,
                attempt,
                cfg.max_retries,
                exc,
            )
            await asyncio.sleep(delay)
            delay = min(delay * 2, 10)
    raise RuntimeError(f"Exhausted retries for row {rid}")


async def process_row(
    index: int,
    total_rows: int,
    row: Dict[str, Any],
    *,
    system_prompt: str,
    user_template: str,
    semaphore: asyncio.Semaphore,
    client: Optional[AsyncOpenAI],
    cfg: PipelineConfig,
) -> Tuple[int, Dict[str, Any]]:
    rid = resolve_row_id(row)
    query = str(row.get("query") or "")
    response = str(row.get("response") or "")
    if not response.strip():
        raise ValueError(f"Row {rid} has an empty 'response' field.")
    user_message = render_user_message(user_template, query=query, response=response)

    logger.debug("Dispatching row {}/{} (id={})", index + 1, total_rows, rid)

    if cfg.dry_run:
        annotation = {
            "binary_harmfulness": "harmless",
            "severity": None,
            "harm_types": [],
            "explanation": "dry-run placeholder",
        }
    else:
        if client is None:
            raise RuntimeError("OpenAI client is not initialized while dry_run=False.")
        async with semaphore:
            annotation = await call_model_with_retry(
                client=client,
                cfg=cfg,
                system_prompt=system_prompt,
                user_message=user_message,
                rid=rid,
            )

    record = build_output_record(row, annotation, include_meta=cfg.emit_meta)
    return index, record


async def run_pipeline(
    rows: Sequence[Dict[str, Any]],
    *,
    system_prompt: str,
    user_template: str,
    cfg: PipelineConfig,
) -> List[Dict[str, Any]]:
    semaphore = asyncio.Semaphore(max(1, cfg.concurrency))
    client = None if cfg.dry_run else AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    if client is None and not cfg.dry_run:
        raise RuntimeError("OPENAI_API_KEY is required for non-dry runs.")

    results: List[Optional[Dict[str, Any]]] = [None] * len(rows)
    total = len(rows)
    completed = 0

    for batch_start in range(0, total, cfg.batch_size):
        batch = rows[batch_start : batch_start + cfg.batch_size]
        tasks = [
            asyncio.create_task(
                process_row(
                    index=batch_start + offset,
                    total_rows=total,
                    row=row,
                    system_prompt=system_prompt,
                    user_template=user_template,
                    semaphore=semaphore,
                    client=client,
                    cfg=cfg,
                )
            )
            for offset, row in enumerate(batch)
        ]

        for task in asyncio.as_completed(tasks):
            idx, record = await task
            results[idx] = record
            completed += 1
            if completed % 10 == 0 or completed == total:
                logger.info("Progress: {}/{} rows complete", completed, total)

    missing = [i for i, record in enumerate(results) if record is None]
    if missing:
        raise RuntimeError(f"Missing results for indices: {missing[:5]}")
    return [record for record in results if record is not None]


def main() -> None:
    args = parse_args()
    configure_logging(args.verbose)

    if not args.input_jsonl.exists():
        raise FileNotFoundError(f"Input JSONL not found: {args.input_jsonl}")

    row_limit = args.row_limit if args.row_limit and args.row_limit > 0 else None
    rows = read_jsonl_rows(args.input_jsonl, limit=row_limit)
    logger.info("Loaded {} rows from {}", len(rows), args.input_jsonl)

    system_prompt, user_template = load_prompt_template(args.prompt_path)

    stamp = timestamp_suffix()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    output_jsonl = args.output_dir / f"{args.output_prefix}_{stamp}.jsonl"
    output_csv = output_jsonl.with_suffix(".csv")

    cfg = PipelineConfig(
        input_path=args.input_jsonl,
        prompt_path=args.prompt_path,
        output_jsonl=output_jsonl,
        output_csv=output_csv,
        model=args.model,
        batch_size=max(1, args.batch_size),
        concurrency=max(1, args.concurrency),
        max_retries=max(1, args.max_retries),
        emit_meta=args.emit_meta,
        dry_run=args.dry_run,
        verbose=args.verbose,
        rate_limiter=RateLimiter(args.rate_limit_qps),
    )

    if not cfg.dry_run and not os.getenv("OPENAI_API_KEY"):
        raise RuntimeError("OPENAI_API_KEY environment variable is required.")

    logger.info(
        "Starting MedSafety bench run | model={} | batch={} | concurrency={} | dry_run={}",
        cfg.model,
        cfg.batch_size,
        cfg.concurrency,
        cfg.dry_run,
    )

    records = asyncio.run(
        run_pipeline(
            rows,
            system_prompt=system_prompt,
            user_template=user_template,
            cfg=cfg,
        )
    )

    write_jsonl(cfg.output_jsonl, records)
    write_csv(cfg.output_csv, records)
    logger.info("Wrote {} rows to {} and {}", len(records), cfg.output_jsonl, cfg.output_csv)


if __name__ == "__main__":
    main()
