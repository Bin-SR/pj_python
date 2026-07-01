import torch
import torch.nn as nn


class TokenEmbedding(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.embedding = nn.Embedding(config.vocab_size, config.hidden_size)

    def forward(self, input_ids):
        return self.embedding(input_ids)

Test = 0
if Test == 1:
    ###### Test ######
    from config import LlamaConfig
    conf = LlamaConfig()
    ttt = TokenEmbedding(conf)
    # 随机生成x，作为文本的token索引，对应于词汇表
    # (batch, token) = (2, 8)
    x = torch.randint(0, conf.vocab_size, (2, 8))
    print(x.shape)
    print("x: ", x)

    # 输出y为embedding
    y = ttt(x)
    print(y.shape)