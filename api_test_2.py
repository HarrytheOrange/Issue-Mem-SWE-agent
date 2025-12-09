from openai import OpenAI

client = OpenAI(
  base_url="https://sd4o2qalmmjm9nglr45tg.apigateway-cn-shanghai.volceapi.com/v1",
  api_key="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",
)

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
