from openai import OpenAI

DEEPSEEK_API_KEY = "1a6d8e05-0978-496b-87c1-fd4fb3885e7c"
DEEPSEEK_MODEL = "deepseek-v3-1-terminus"
DEEPSEEK_BASE_URL = "https://ark.cn-beijing.volces.com/api/v1"

# client = OpenAI(
#   base_url="https://openrouter.ai/api/v1",
#   api_key="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",
# )

client = OpenAI(
  base_url=DEEPSEEK_BASE_URL,
  api_key=DEEPSEEK_API_KEY,
)
# import os 
# api_key = os.getenv("OPENAI_API_KEY")
# base_url = os.getenv("OPENAI_API_BASE_URL")
# client = OpenAI()


completion = client.chat.completions.create(
  extra_headers={
    "HTTP-Referer": "<YOUR_SITE_URL>", # Optional. Site URL for rankings on openrouter.ai.
    "X-Title": "<YOUR_SITE_NAME>", # Optional. Site title for rankings on openrouter.ai.
  },
  model="deepseek-v3-1-terminus",
  messages=[
    {
      "role": "user",
      "content": "What is the meaning of life? Answer in 100 words."
    }
  ]
)

print(completion.choices[0].message.content)