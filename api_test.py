from openai import OpenAI

client = OpenAI(
  base_url="https://sd5410idmmjm9nglrb3o0.apigateway-cn-shanghai.volceapi.com/v1",
  api_key="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",
)

# ANTHROPIC_BASE_URL = "https://aigc.x-see.cn/"
# ANTHROPIC_API_KEY = "sk-3rhltYvvgQsAMDkHAdsbKoIFqrpmWr0v84dTor9zTIEAgc62"

completion = client.chat.completions.create(
  extra_headers={
    "HTTP-Referer": "<YOUR_SITE_URL>", # Optional. Site URL for rankings on openrouter.ai.
    "X-Title": "<YOUR_SITE_NAME>", # Optional. Site title for rankings on openrouter.ai.
  },
  model="qwen/qwen3-coder-30b-a3b-instruct",
  messages=[
    {
      "role": "user",
      "content": "What is the meaning of life? Answer in code."
    }
  ]
)

print(completion.choices[0].message.content)


# import anthropic

# client = anthropic.Anthropic(
#     base_url="https://aigc.x-see.cn/",
#     api_key="sk-3rhltYvvgQsAMDkHAdsbKoIFqrpmWr0v84dTor9zTIEAgc62",
# )
# message = client.messages.create(
#     model="claude-sonnet-4-20250514",
#     max_tokens=1024,
#     messages=[
#         {"role": "user", "content": "Hello, Claude"}
#     ]
# )
# print(message.content)