from __future__ import annotations

import json
from typing import Any

from anthropic import Anthropic


API_KEY = "msk-67260fae97f724abc513186a3ba35d6e5b61c5bfe262b58e9928c0d69aeb3ccf"
# import os 
# ANTHROPIC_BASE_URL = "http://claude0openai.a.pinggy.link/v1"
MODEL = "claude-3-5-sonnet-20241022"
MAX_TOKENS = 512


client = Anthropic(api_key=API_KEY)


def get_weather(location: str, unit: str = "celsius") -> dict[str, Any]:
    return {
        "location": location,
        "temperature": 22,
        "unit": unit,
        "condition": "Sunny",
    }


def calculate_sum(a: int, b: int) -> int:
    return a + b


TOOLS: list[dict[str, Any]] = [
    {
        "name": "get_weather",
        "description": "Get the current weather for a specified location",
        "input_schema": {
            "type": "object",
            "properties": {
                "location": {
                    "type": "string",
                    "description": "City name",
                },
                "unit": {
                    "type": "string",
                    "enum": ["celsius", "fahrenheit"],
                    "description": "Temperature unit",
                },
            },
            "required": ["location"],
        },
    },
    {
        "name": "calculate_sum",
        "description": "Calculate the sum of two numbers",
        "input_schema": {
            "type": "object",
            "properties": {
                "a": {"type": "integer"},
                "b": {"type": "integer"},
            },
            "required": ["a", "b"],
        },
    },
]


def pretty(title: str, payload: Any) -> None:
    print(f"\n=== {title} ===")
    print(json.dumps(payload, indent=2, ensure_ascii=False))


def run_basic_completion() -> None:
    response = client.messages.create(
        model=MODEL,
        max_tokens=MAX_TOKENS,
        messages=[{"role": "user", "content": "Who are you?"}],
    )
    pretty("Basic completion (raw)", response.model_dump())


def run_tool_call() -> None:
    response = client.messages.create(
        model=MODEL,
        max_tokens=MAX_TOKENS,
        tools=TOOLS,
        messages=[
            {"role": "user", "content": "What's the weather in Beijing and what's 5 + 3?"}
        ],
    )
    pretty("Tool call response (raw)", response.model_dump())
    for block in response.content:
        if block.type == "tool_use":
            print(f"\nTool requested: {block.name}")
            pretty("Arguments", block.input)


def main() -> None:
    run_basic_completion()
    print("\n--- Function Call Test ---")
    run_tool_call()


if __name__ == "__main__":
    main()