from openai import OpenAI

OPENAI_BASE_URL="https://openrouter.ai/api/v1"
OPENAI_API_KEY="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"

client = OpenAI(
  base_url=OPENAI_BASE_URL,
  api_key=OPENAI_API_KEY,
)

completion = client.chat.completions.create(
  extra_headers={
    "HTTP-Referer": "<YOUR_SITE_URL>", # Optional. Site URL for rankings on openrouter.ai.
    "X-Title": "<YOUR_SITE_NAME>", # Optional. Site title for rankings on openrouter.ai.
  },
  model="moonshotai/kimi-k2",
  messages=[
    {
      "role": "user",
      "content": "What is the meaning of life? Answer in code."
    }
  ]
)

print(completion.choices[0].message.content)