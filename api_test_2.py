from openai import OpenAI

# OPENAI_BASE_URL="https://ark.cn-beijing.volces.com/api/v1"
# OPENAI_API_KEY="1a6d8e05-0978-496b-87c1-fd4fb3885e7c"
# MODEL="deepseek-v3-1-terminus"

OPENAI_BASE_URL="https://openrouter.ai/api/v1"
OPENAI_API_KEY="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"
MODEL="moonshotai/kimi-k2-0905:exacto"

client = OpenAI(
  base_url=OPENAI_BASE_URL,
  api_key=OPENAI_API_KEY,
)

completion = client.chat.completions.create(
  extra_headers={
    "HTTP-Referer": "<YOUR_SITE_URL>", # Optional. Site URL for rankings on openrouter.ai.
    "X-Title": "<YOUR_SITE_NAME>", # Optional. Site title for rankings on openrouter.ai.
  },
  model=MODEL,
  messages=[
    {
      "role": "user",
      "content": "What is the meaning of life? Answer in code."
    }
  ]
)

print(completion.choices[0].message.content)





# curl -X POST \
#   "https://sd4o2qalmmjm9nglr45tg.apigateway-cn-shanghai.volceapi.com/v1/chat/completions" \
#   -H "Content-Type: application/json" \
#   -H "Authorization: Bearer <API KEY>" \
#   -d '{
#     "model": "LLM",
#     "messages": [
#       {
#         "role": "system",
#         "content": "You are a helpful AI assistant"
#       },
#       {
#         "role": "user",
#         "content": "你是谁"
#       }
#     ],
#     "temperature": 0.6,
#     "max_tokens": 8192
#   }'
