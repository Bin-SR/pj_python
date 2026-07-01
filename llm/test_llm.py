from transformers import AutoTokenizer
from transformers import AutoModelForCausalLM

model_name = "Qwen/Qwen2.5-0.5B-Instruct"

tokenizer = AutoTokenizer.from_pretrained(model_name)

model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype="auto", device_map="auto")

prompt = "你好，请介绍一下你自己"

messages = [{"role":"user","content":prompt}]

text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

inputs = tokenizer(text, return_tensors="pt").to(model.device)

outputs = model.generate(**inputs, max_new_tokens=128)

response = tokenizer.decode(outputs[0], skip_special_tokens=True)

print(response)



