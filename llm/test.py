from transformers import AutoTokenizer
from transformers import AutoModelForCausalLM

model_name = "Qwen/Qwen2.5-0.5B-Instruct"

tokenizer = AutoTokenizer.from_pretrained(model_name)
text = "机器人喜欢足球"
print(tokenizer.tokenize(text))
print(tokenizer.encode(text))


model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype="auto", device_map="auto")
print(model)
# embedding = model.model.embed_tokens
# print(embedding.weight.shape)  
# shape = torch.Size([151936, 896])
# 意思是词表大小：151936个词 每个词的维度：896
