from openai import OpenAI
import json

# openai_api_key = "token-abc123"
# openai_api_base = "http://publicshare.a.pinggy.link/v1"

api_key = "msk-67260fae97f724abc513186a3ba35d6e5b61c5bfe262b58e9928c0d69aeb3ccf"
# base_url = "http://publicshare.a.pinggy.link/v1"
base_url = "http://claude0openai.a.pinggy.link/v1"
model = "Claude 4.5 Sonnet"

client = OpenAI(
    api_key=api_key,
    base_url=base_url,
)

chat_response = client.chat.completions.create(
    model=model, # empty
    messages=[
        {"role": "user", "content": "who are you?"*1},
    ],
    # max_tokens=32768,
    temperature=0.6,
    # top_p=0.95,
)
print("Chat response:", chat_response)

# =============== Function Call Test ===============

def get_weather(location: str, unit: str = "celsius") -> dict:
    """Mock function to get weather information"""
    return {
        "location": location,
        "temperature": 22,
        "unit": unit,
        "condition": "Sunny"
    }

def calculate_sum(a: int, b: int) -> int:
    """Mock function to calculate sum of two numbers"""
    return a + b

# Define tools for function calling
tools = [
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
                        "description": "The city name to get weather for"
                    },
                    "unit": {
                        "type": "string",
                        "enum": ["celsius", "fahrenheit"],
                        "description": "Temperature unit"
                    }
                },
                "required": ["location"]
            }
        }
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
                        "description": "First number"
                    },
                    "b": {
                        "type": "integer",
                        "description": "Second number"
                    }
                },
                "required": ["a", "b"]
            }
        }
    }
]

# Test function calling
print("\n--- Function Call Test ---")
function_call_response = client.chat.completions.create(
    model=model,
    messages=[
        {"role": "user", "content": "What's the weather in Beijing and what's 5 + 3?"}
    ],
    tools=tools,
    tool_choice="auto",  # Let the model decide whether to use tools
    temperature=0.6,
    # top_p=0.95,
)

print("Function Call Response:")
print(json.dumps(function_call_response.model_dump(), indent=2))

# Process tool calls if any
if function_call_response.choices[0].message.tool_calls:
    print("\nTool calls detected:")
    for tool_call in function_call_response.choices[0].message.tool_calls:
        print(f"  - Function: {tool_call.function.name}")
        print(f"    Arguments: {tool_call.function.arguments}")
        
        # Parse and execute the function call
        args = json.loads(tool_call.function.arguments)
        if tool_call.function.name == "get_weather":
            result = get_weather(**args)
        elif tool_call.function.name == "calculate_sum":
            result = calculate_sum(**args)
        else:
            result = {"error": "Unknown function"}
        
        print(f"    Result: {result}\n")