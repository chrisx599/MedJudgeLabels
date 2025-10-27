"""I/O utilities for the MedSafety labeling pipeline."""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List

import pandas as pd


def read_input_csv(path: Path) -> pd.DataFrame:
    """Load the MedSafety dataset, normalizing column names and optional labels."""
    df = pd.read_csv(path, dtype=str, keep_default_na=False)

    # Normalize BOM-prefixed id column (common when authoring in Excel).
    if "\ufeffid" in df.columns:
        df = df.rename(columns={"\ufeffid": "id"})

    # Standardize optional human label columns.
    if "respond_type-human" not in df.columns and "response_type-human" in df.columns:
        df["respond_type-human"] = df["response_type-human"]
    if "respond_type-human" not in df.columns:
        df["respond_type-human"] = ""
    if "query_risk_level-human" not in df.columns:
        df["query_risk_level-human"] = ""

    # Ensure consistent column order for downstream formatting.
    columns: List[str] = [
        "id",
        "query",
        "response",
        "query_risk_level-human",
        "respond_type-human",
    ]
    for col in columns:
        if col not in df.columns:
            df[col] = ""

    # Reorder columns while preserving additional metadata if any.
    other_cols = [c for c in df.columns if c not in columns]
    df = df[columns + other_cols]
    return df


def generate_id(query: str, response: str) -> str:
    """Generate a stable 16-hex identifier when the CSV omits an id."""
    material = (query or "") + (response or "")
    digest = hashlib.sha256(material.encode("utf-8")).hexdigest()
    return digest[:16]


def enforce_30_words(text: str | None) -> str:
    """Truncate explanations to 30 words while preserving original spacing."""
    if not text:
        return ""
    words = text.split()
    if len(words) <= 30:
        return text.strip()
    return " ".join(words[:30]).strip()


def write_jsonl(path: Path, records: Iterable[Dict[str, Any]]) -> None:
    """Write records to JSONL, one object per line."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False))
            f.write("\n")


def append_jsonl_record(path: Path, record: Dict[str, Any]) -> None:
    """Append a single record to a JSONL file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False))
        f.write("\n")


def load_jsonl_latest(path: Path) -> Dict[str, Dict[str, Any]]:
    """Load JSONL records, returning the latest entry per id."""
    records: Dict[str, Dict[str, Any]] = {}
    if not path.exists():
        return records
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            rid = obj.get("id")
            if rid:
                records[str(rid)] = obj
    return records


def append_csv(path: Path, records: Iterable[Dict[str, Any]]) -> None:
    """Append records to CSV, emitting a header when creating a new file."""
    records_list = list(records)
    if not records_list:
        return

    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(records_list[0].keys())
    file_exists = path.exists()

    with path.open("a", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if not file_exists:
            writer.writeheader()
        writer.writerows(records_list)


def write_csv(path: Path, records: Iterable[Dict[str, Any]]) -> None:
    """Write records to CSV from scratch."""
    records_list = list(records)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not records_list:
        with path.open("w", encoding="utf-8", newline="") as f:
            pass
        return
    fieldnames = list(records_list[0].keys())
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(records_list)
