from __future__ import annotations

import json
from typing import Any

import litellm
from litellm.types.utils import ModelResponse


API_KEY = "msk-67260fae97f724abc513186a3ba35d6e5b61c5bfe262b58e9928c0d69aeb3ccf"
API_BASE = "http://claude0openai.a.pinggy.link/v1"
MODEL = "Claude 4.5 Sonnet"


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
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Get the current weather for a specified location",
            "parameters": {
                "type": "object",
                "properties": {
                    "location": {
                        "type": "string",
                        "description": "The city name to get weather for",
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
    },
    {
        "type": "function",
        "function": {
            "name": "calculate_sum",
            "description": "Calculate the sum of two numbers",
            "parameters": {
                "type": "object",
                "properties": {
                    "a": {
                        "type": "integer",
                        "description": "First number",
                    },
                    "b": {
                        "type": "integer",
                        "description": "Second number",
                    },
                },
                "required": ["a", "b"],
            },
        },
    },
]


def pretty(title: str, payload: Any) -> None:
    print(f"\n=== {title} ===")
    print(json.dumps(payload, indent=2, ensure_ascii=False))


def make_messages(question: str) -> list[dict[str, str]]:
    return [{"role": "user", "content": question}]


def invoke_litellm(
    messages: list[dict[str, str]],
    *,
    tools: list[dict[str, Any]] | None = None,
) -> ModelResponse:
    return litellm.completion(
        model=MODEL,
        messages=messages,
        api_base=API_BASE,
        api_key=API_KEY,
        custom_llm_provider="openai",
        temperature=0.6,
        tools=tools,
        tool_choice="auto" if tools else None,
    )


def dump_message_shape(response: ModelResponse) -> None:
    choice = response.choices[0]
    content = choice.message.content
    print("\n--- Message Introspection ---")
    print(f"type(content) = {type(content)}")
    print(f"content = {content}")
    if choice.message.tool_calls:
        pretty(
            "tool_calls",
            [call.to_dict() for call in choice.message.tool_calls],
        )
    else:
        print("tool_calls = []")


def main() -> None:
    basic_response = invoke_litellm(make_messages("who are you?"))
    pretty("Basic completion (model_dump)", basic_response.model_dump())
    dump_message_shape(basic_response)

    print("\n--- Function Call Test ---")
    function_response = invoke_litellm(
        make_messages("What's the weather in Beijing and what's 5 + 3?"),
        tools=TOOLS,
    )
    pretty("Function call response (model_dump)", function_response.model_dump())
    dump_message_shape(function_response)


if __name__ == "__main__":
    main()