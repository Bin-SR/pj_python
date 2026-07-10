from openai import OpenAI

client = OpenAI(
    api_key="sk-97cec811f742451abc2de02f0a5eb7ed",
    base_url="https://api.deepseek.com"
)

system_prompt = """你好，请介绍一下你自己"""


response = client.chat.completions.create(
        model="deepseek-chat",
        messages=[
            {"role":"system","content":system_prompt},
            # {"role":"user","content":user_cmd}
        ]
    )

answer = response.choices[0].message.content

print(answer)