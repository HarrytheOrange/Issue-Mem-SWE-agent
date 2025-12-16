from openai import OpenAI

OPENAI_BASE_URL="https://ark.cn-beijing.volces.com/api/v1"
OPENAI_API_KEY="1a6d8e05-0978-496b-87c1-fd4fb3885e7c"


client = OpenAI(
  base_url=OPENAI_BASE_URL,
  api_key=OPENAI_API_KEY,
)

completion = client.chat.completions.create(
  extra_headers={
    "HTTP-Referer": "<YOUR_SITE_URL>", # Optional. Site URL for rankings on openrouter.ai.
    "X-Title": "<YOUR_SITE_NAME>", # Optional. Site title for rankings on openrouter.ai.
  },
  model="deepseek-v3-1-terminus",
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
