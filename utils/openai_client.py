"""OpenAI client helpers for the MedSafety labeling pipeline."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Dict, Optional

from openai import OpenAI
from tenacity import retry, stop_after_attempt, wait_exponential


def extract_output_text(response: Any) -> str:
    """Traverse response.output to find the textual payload."""
    output = getattr(response, "output", None)
    if not output:
        return ""
    for item in output:
        content = getattr(item, "content", None)
        if not content:
            continue
        for block in content:
            text = getattr(block, "text", None)
            if text:
                return text
    return ""


@dataclass(slots=True)
class OpenAIClient:
    """Wrapper around the OpenAI Responses API with JSON Schema enforcement."""

    api_key: str
    model: str
    json_schema: Dict[str, Any]
    _client: Optional[OpenAI] = field(init=False, repr=False, default=None)

    def __post_init__(self) -> None:
        self._client = OpenAI(api_key=self.api_key)

    @retry(wait=wait_exponential(multiplier=1, min=1, max=8), stop=stop_after_attempt(3))
    def _create_response(self, system_msg: str, user_msg: str):
        if self._client is None:
            raise RuntimeError("OpenAI client is not initialized.")
        return self._client.responses.create(
            model=self.model,
            input=[
                {"role": "system", "content": system_msg},
                {"role": "user", "content": user_msg},
            ],
        )

    def classify_row(self, system_msg: str, user_msg: str) -> Dict[str, Any]:
        """Call the Responses API and parse the JSON payload."""
        response = self._create_response(system_msg=system_msg, user_msg=user_msg)

        text = getattr(response, "output_text", None)
        if not text:
            text = extract_output_text(response)
        if not text:
            raise ValueError("No textual content returned from OpenAI response.")

        return json.loads(text)
