"""Validation helpers for MedSafety labeling outputs."""

from __future__ import annotations

from typing import Any, Dict

from jsonschema import ValidationError as JsonSchemaError, validate


def validate_record(obj: Dict[str, Any], schema: Dict[str, Any]) -> None:
    """Validate the response object against the JSON Schema."""
    try:
        validate(instance=obj, schema=schema)
    except JsonSchemaError as exc:
        raise ValueError(f"JSON Schema validation failed: {exc.message}") from exc
